import os
from typing import Dict, List, Optional, Tuple

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

import wandb
import warpconvnet.nn.functional.transforms as T
from warpconvnet.geometry.ops.voxel_ops import (
    voxel_downsample_random_indices_list_of_coords,
)
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import (
    BatchedDiscreteCoordinates,
    BatchedFeatures,
    SpatiallySparseTensor,
)
from warpconvnet.models.sparse_conv_unet import SparseConvDecoder, SparseConvEncoder
from warpconvnet.nn.activations import ReLU
from warpconvnet.nn.normalizations import LayerNorm
from warpconvnet.nn.sparse_conv import (
    SPATIALLY_SPARSE_CONV_ALGO_MODE,
    STRIDED_CONV_MODE,
    SparseConv3d,
)
from warpconvnet.nn.sparse_pool import SparseMaxPool

CONV_MODE = SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM
KERNEL_MATMUL_BATCH_SIZE = 9
SCANNET_URL = "https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip"


class ScanNetDataset(Dataset):
    """
    Dataset from the OpenScene project.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        self.prepare_data()

    def prepare_data(self):
        # If data is not downloaded, download it
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            os.system(f"wget {SCANNET_URL} -O {self.root}/scannet_3d.zip")
            os.system(f"unzip {self.root}/scannet_3d.zip -d {self.root}")
            os.system(f"mv {self.root}/scannet_3d/* {self.root}")
            os.system(f"rmdir {self.root}/scannet_3d")

        # Get split txts
        self.files = []
        with open(os.path.join(self.root, f"scannetv2_{self.split}.txt"), "r") as f:
            self.files = sorted(f.readlines())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        coords, colors, labels = torch.load(
            os.path.join(self.root, self.split, file.strip() + "_vh_clean_2.pth"),
            weights_only=False,
        )
        return {
            "coords": coords,
            "colors": colors,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, Tensor]]):
    """
    Return dict of list of tensors
    """
    keys = batch[0].keys()
    return {key: [torch.tensor(item[key]) for item in batch] for key in keys}


def dict_to_sparse_tensor(
    batch_dict: Dict[str, Tensor], voxel_size: float, device: str
) -> Tuple[SpatiallySparseTensor, Dict[str, Tensor]]:
    """
    Return sparse tensor
    """
    unique_indices, batch_offsets = voxel_downsample_random_indices_list_of_coords(
        batch_dict["coords"], voxel_size, device=device
    )
    # cat all features into a single tensor
    cat_batch_dict = {
        k: torch.cat(v, dim=0).to(device)[unique_indices] for k, v in batch_dict.items()
    }
    return (
        SpatiallySparseTensor(
            batched_coordinates=BatchedDiscreteCoordinates(
                torch.floor(cat_batch_dict["coords"] / voxel_size).int(),
                offsets=batch_offsets,
            ),
            batched_features=BatchedFeatures(
                cat_batch_dict["colors"],
                offsets=batch_offsets,
            ),
            voxel_size=voxel_size,
        ),
        cat_batch_dict,
    )


def confusion_matrix_to_metrics(conf_matrix: Tensor) -> Dict[str, float]:
    """
    Return accuracy, miou, class_iou, class_accuracy

    Rows are ground truth, columns are predictions.
    """
    conf_matrix = conf_matrix.cpu()
    accuracy = (conf_matrix.diag().sum() / conf_matrix.sum()).item() * 100
    class_accuracy = (conf_matrix.diag() / conf_matrix.sum(dim=1)) * 100
    class_iou = conf_matrix.diag() / (
        conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - conf_matrix.diag()
    )
    miou = class_iou.mean().item() * 100
    return {
        "accuracy": accuracy,
        "miou": miou,
        "class_iou": class_iou,
        "class_accuracy": class_accuracy,
    }


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, base_channels),
            nn.LayerNorm(base_channels),
            nn.ReLU(),
        )
        final_channels = base_channels * 4
        encoder_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        decoder_channels = [
            base_channels * 16,
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            final_channels,
        ]
        self.encoder = SparseConvEncoder(
            num_levels=4,
            kernel_sizes=3,
            encoder_channels=encoder_channels,
            num_blocks_per_level=[1, 1, 1, 1],
        )
        self.decoder = SparseConvDecoder(
            num_levels=4,
            kernel_sizes=3,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_blocks_per_level=[1, 1, 1, 1],
        )
        self.final_conv = nn.Sequential(
            SparseConv3d(
                final_channels,
                final_channels,
                kernel_size=1,
            ),
            LayerNorm(final_channels),
            ReLU(),
            SparseConv3d(
                final_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: SpatiallySparseTensor):
        torch.cuda.empty_cache()
        x = T.apply_feature_transform(x, self.mlp)
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        output = self.final_conv(decoder_outputs[-1])
        return output


def train(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    voxel_size: float = 0.05,
    ignore_index: int = 255,
    use_wandb: bool = True,
):
    model.train()
    bar = tqdm(train_loader)
    for batch_idx, batch_dict in enumerate(bar):
        optimizer.zero_grad()
        st, batch_dict = dict_to_sparse_tensor(batch_dict, voxel_size=voxel_size, device=device)
        output = model(st.to(device))
        loss = F.cross_entropy(
            output.feature_tensor,
            batch_dict["labels"].long(),
            reduction="mean",
            ignore_index=ignore_index,
        )
        loss.backward()
        optimizer.step()
        bar.set_description(f"Train Epoch: {epoch} \tLoss: {loss.item():.3f}")
        if use_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )


@torch.inference_mode()
def test(
    model: nn.Module,
    device: str,
    test_loader: DataLoader,
    voxel_size: float = 0.05,
    ignore_index: int = 255,
    use_wandb: bool = True,
    num_test_batches: Optional[int] = None,
):
    model.eval()
    confusion_matrix = MulticlassConfusionMatrix(num_classes=20, ignore_index=ignore_index).to(
        device
    )
    test_loss = 0
    num_batches = 0
    for batch_dict in test_loader:
        st, batch_dict = dict_to_sparse_tensor(batch_dict, voxel_size=voxel_size, device=device)
        output = model(st.to(device))
        labels = batch_dict["labels"].long()
        test_loss += F.cross_entropy(
            output.feature_tensor,
            labels,
            reduction="mean",
            ignore_index=ignore_index,
        ).item()
        pred = output.feature_tensor.argmax(dim=1)
        confusion_matrix.update(pred, labels)
        num_batches += 1
        if num_test_batches is not None and num_batches >= num_test_batches:
            break

    test_loss /= num_batches
    metrics = confusion_matrix_to_metrics(confusion_matrix.compute())

    if use_wandb:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": metrics["accuracy"],
                "test/miou": metrics["miou"],
                "test/class_iou": metrics["class_iou"],
                "test/class_accuracy": metrics["class_accuracy"],
            }
        )

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.2f}%, mIoU: {metrics['miou']:.2f}%\n"
    )
    return metrics["miou"]


def main(
    root_dir: str = "./data/scannet_3d",
    batch_size: int = 32,
    voxel_size: float = 0.05,
    ignore_index: int = 255,
    epochs: int = 50,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
    use_wandb: bool = False,
):
    if use_wandb:
        wandb.init(
            project="scannet-segmentation",
            config={
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "scheduler_step_size": scheduler_step_size,
                "gamma": gamma,
                "device": device,
            },
        )

    wp.init()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    train_dataset = ScanNetDataset(root_dir, split="train")
    test_dataset = ScanNetDataset(root_dir, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = UNet(in_channels=3, out_channels=20).to(device)  # Assuming 20 classes for ScanNet
    if use_wandb:
        wandb.watch(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    # Test before training
    miou = test(
        model, device, test_loader, voxel_size, ignore_index, use_wandb, num_test_batches=5
    )
    for epoch in range(1, epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            voxel_size,
            ignore_index,
            use_wandb,
        )
        miou = test(model, device, test_loader, voxel_size, ignore_index, use_wandb)
        scheduler.step()

    print(f"Final mIoU: {miou:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
