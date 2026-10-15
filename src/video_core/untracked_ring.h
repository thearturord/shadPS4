// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include "common/types.h"

namespace VideoCore {

/// A guest memory range the game uses as a per-frame upload ring (TLG: the 64 MB ring behind AllocFrameGarlic).
/// The CPU writes it and the GPU reads it; it is never read back. Its pages are not write-protected (no faults on the
/// game threads, no protect calls on the raster thread). Instead every read of it by the GPU uploads the requested
/// range. Found by the native draw hook at run time (native_draw_mode 3, readbacks off).
class UntrackedRing {
public:
    static UntrackedRing& Get() {
        static UntrackedRing ring;
        return ring;
    }

    void Set(VAddr base, u64 size) {
        end.store(base + size, std::memory_order_release);
        start.store(base, std::memory_order_release);
    }

    [[nodiscard]] VAddr Start() const noexcept {
        return start.load(std::memory_order_acquire);
    }
    [[nodiscard]] VAddr End() const noexcept {
        return end.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool Active() const noexcept {
        return Start() != End();
    }

    /// The part of [addr, addr + size) inside the ring (lo >= hi: none).
    [[nodiscard]] std::pair<VAddr, VAddr> Clip(VAddr addr, u64 size) const noexcept {
        const VAddr lo = std::max<VAddr>(addr, Start());
        const VAddr hi = std::min<VAddr>(addr + size, End());
        return {lo, hi};
    }

private:
    std::atomic<VAddr> start{0};
    std::atomic<VAddr> end{0};
};

} // namespace VideoCore
