// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include "common/types.h"

namespace Common {

/// Hash for 64-bit integer keys. The standard library hashes integers byte by byte (FNV-1a), which
/// showed up in profiles of lookups done for every draw; one multiplication mixes well enough for
/// tables that mask the low bits.
struct U64Hash {
    std::size_t operator()(u64 value) const noexcept {
        return static_cast<std::size_t>((value * 0x9E3779B97F4A7C15ULL) >> 32);
    }
};

} // namespace Common
