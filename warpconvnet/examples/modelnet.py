import os
import ssl
import urllib.request
import zipfile

import fire
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import warpconvnet.nn.functional.transforms as T
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.normalizations import LayerNorm
from warpconvnet.nn.sparse_conv import SparseConv3d

ssl._create_default_https_context = ssl._create_unverified_context

# Constants
_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
_LABELS = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Download and extract dataset if it doesn't exist
        if not os.path.exists(root_dir):
            self.download_and_extract()

        # Load file list
        file_list_path = os.path.join(root_dir, f"modelnet40_ply_hdf5_2048/{split}_files.txt")
        with open(file_list_path, "r") as f:
            self.file_list = [
                os.path.join(root_dir, line.strip().replace("data/", "")) for line in f.readlines()
            ]

        self.data = []
        self.labels = []
        self.load_data()

    def download_and_extract(self):
        print("Downloading and extracting dataset...")
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, "modelnet40.zip")
        urllib.request.urlretrieve(_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root_dir)
        os.remove(zip_path)
        print("Dataset downloaded and extracted.")

    def load_data(self):
        for filepath in self.file_list:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            with h5py.File(filepath, "r") as h5file:
                self.data.append(torch.from_numpy(h5file["data"][:]))
                self.labels.append(torch.from_numpy(h5file["label"][:]))

        if not self.data:
            raise FileNotFoundError("No valid data files were found.")

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            points = self.transform(points)

        return points, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoding channels * 3 (xyz) is the input features
        self.conv1 = SparseConv3d(24, 32, kernel_size=3, stride=1)
        self.norm1 = LayerNorm(32)
        self.conv1_stride = SparseConv3d(32, 64, kernel_size=2, stride=2)
        self.conv2 = SparseConv3d(64, 128, kernel_size=3, stride=1)
        self.norm2 = LayerNorm(128)
        self.conv2_stride = SparseConv3d(128, 256, kernel_size=2, stride=2)
        self.conv3 = SparseConv3d(256, 512, kernel_size=3, stride=1)
        self.norm3 = LayerNorm(512)
        self.conv4 = nn.Conv3d(512, 1024, kernel_size=3, stride=2)
        self.conv4_stride = nn.Conv3d(1024, 1024, kernel_size=2, stride=2)
        self.norm4 = nn.LayerNorm(1024 * 2 * 2 * 2)
        self.fc1 = nn.Linear(1024 * 2 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 40)  # 40 classes in ModelNet40

    def forward(self, x: Float[Tensor, "B N 3"]):
        pc = PointCollection.from_list_of_coordinates(x, encoding_channels=8, encoding_range=1)
        x = pc.to_sparse(voxel_size=0.05)
        x = self.conv1(x)
        x = self.norm1(x)
        x = T.relu(x)
        x = self.conv1_stride(x)
        x = T.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = T.relu(x)
        x = self.conv2_stride(x)
        x = T.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x.to_dense(channel_dim=1, min_coords=(-5, -5, -5), max_coords=(4, 4, 4))
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_stride(x)
        x = torch.flatten(x, 1)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def main(
    root_dir: str = "./data/modelnet40",
    batch_size: int = 128,
    test_batch_size: int = 100,
    epochs: int = 100,
    lr: float = 1e-3,
    scheduler_step_size: int = 10,
    gamma: float = 0.7,
    device: str = "cuda",
):
    wp.init()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    train_dataset = ModelNet40Dataset(root_dir, split="train")
    test_dataset = ModelNet40Dataset(root_dir, split="test")

    print(f"Dataset root directory: {root_dir}")
    print(f"Files in root directory: {os.listdir(root_dir)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()

    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    fire.Fire(main)
