// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

/// What the game's render layer knows about a draw before it writes the command packets: hashes of groups of the
/// engine objects bound for it (see core/native_draw.cpp). It travels in the command stream as a NOP packet in
/// front of the draw and is attached to that draw by the command processor.
struct NativeTag {
    bool valid{};
    u32 meta{}; ///< pass class (low 8 bits), draw kind (next 8 bits)
    u64 k1{};   ///< vertex shader, pixel shader, input layout, topology
    u64 k2{};   ///< render targets, depth view, blend / rasterizer / depth-stencil state, viewport
    u64 k3{};   ///< shader resource views and samplers
    u64 seq{};  ///< native record (0 = none): the register state the game's objects describe for this draw
};

constexpr u32 NativeTagMagic = 0x4E445447; ///< "NDTG"
constexpr u32 NativeTagPayloadDwords = 12; ///< magic, meta, k1, k2, k3, seq (two dwords each)

} // namespace VideoCore
