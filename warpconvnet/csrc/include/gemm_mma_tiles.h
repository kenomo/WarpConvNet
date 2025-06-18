// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace warpconvnet {
namespace gemm {

// -----------------------------------------------------------------------------
//  Alternative tile tags and precision specialisations (runtime-selectable)
// -----------------------------------------------------------------------------
//  These must be declared *after* the default specialisations so that we can
//  inherit from them.
// -----------------------------------------------------------------------------

// Tile tag representing alternative thread-blocks / 31×32×32 warp configuration
// on Ampere Tensor Cores.
struct Tile128x128x32 {};
struct Tile128x64x32 {};
struct Tile64x128x32 {};
struct Tile64x64x32 {};

enum class MMATile : int {
  Tile128x128x32 = 0,  // 128×128×32
  Tile128x64x32 = 1,   // 128x64x32
  Tile64x128x32 = 2,   // 64x128x32
  Tile64x64x32 = 3,    //  64×64×32
};

}  // namespace gemm
}  // namespace warpconvnet
