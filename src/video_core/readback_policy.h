// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string_view>
#include <nlohmann/json.hpp>
#include "common/types.h"

namespace VideoCore::ReadbackPolicy {

/// Readback behavior that can be changed while the game runs (debug server /set), so variants can be
/// compared in the same scene without restarting. Initialised from the settings (InitFromSettings),
/// so with nothing changed the behavior is the one of the settings. All values are plain atomics read
/// on the hot paths.
struct State {
    /// Fence kinds (bit mask, see FenceKind) whose fences prefetch the hot buffer ranges.
    std::atomic<u32> prefetch_kinds_mask{~0u};
    /// Fence kinds whose fences download the bound linear images.
    std::atomic<u32> image_download_kinds_mask{~0u};
    /// -1 = readback_prefetch_mode setting, otherwise 0 (off), 1 or 2.
    std::atomic<s32> prefetch_mode_override{-1};
    /// Fences a hot range stays prefetched after its last fault (readback_prefetch_lifetime).
    std::atomic<u32> prefetch_lifetime{600};
    /// cp_full_fence_kinds.
    std::atomic<u32> full_fence_kinds{31};
    /// readback_linear_images_async.
    std::atomic<bool> images_async{false};
    /// readback_dma_async: DMA copies into guest memory are also downloaded into a private staging buffer
    /// and written to guest memory by the priority thread (counters: dma_async_copies, dma_async_bytes).
    std::atomic<bool> dma_async{false};
    std::atomic<u64> dma_async_copies{0};
    std::atomic<u64> dma_async_bytes{0};
    /// Live switch skip_shadows: depth-only draws into a square depth target are dropped (1), or into
    /// one with this pitch (value >= 256), before any Vulkan work. skipped_draws counts them.
    std::atomic<u32> skip_shadows{0};
    std::atomic<u64> skipped_draws{0};
    /// Histogram of the depth targets of depth-only draws (key = pitch << 16 | height), whatever the switch says.
    struct DepthOnlyBucket {
        std::atomic<u32> key{0};
        std::atomic<u64> count{0};
    };
    std::array<DepthOnlyBucket, 24> depth_hist{};
    void NoteDepthOnly(u32 pitch, u32 height) {
        const u32 key = (pitch << 16) | (height & 0xffff);
        for (auto& bucket : depth_hist) {
            u32 current = bucket.key.load(std::memory_order_relaxed);
            if (current == 0) {
                bucket.key.compare_exchange_strong(current, key);
            }
            if (bucket.key.load(std::memory_order_relaxed) == key) {
                bucket.count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }
    /// -1 = readbacks_mode setting, otherwise 0 Disabled, 1 Relaxed, 2 Precise.
    std::atomic<s32> readbacks_mode_override{-1};
    /// cp_flush_commands: hand recorded commands to the encoder every N commands (0 = off).
    std::atomic<u32> flush_commands{0};
    /// cp_raster_core: logical processor of the raster thread, -1 = not pinned. Applied live.
    std::atomic<s32> raster_core{-1};
    /// native_draw_mode / native_draw_classes (see emulator_settings.h). Applied live.
    std::atomic<u32> native_draw_mode{0};
    std::atomic<u32> native_draw_classes{3};
    std::atomic<u32> native_synth_verify{0};
    std::atomic<u32> native_audit_every{64}; ///< see emulator_settings.h
    std::atomic<u32> native_sdk_strip{0};    ///< see emulator_settings.h
    std::atomic<u32> native_draw_exec{0};
};

State& Get();

/// The readbacks mode in use (setting or live override).
int ReadbacksMode();

/// True on the emulator's own threads (names starting with "shadPS4:").
bool IsEmulatorThread();

/// Copies the relevant settings into the state. Call once when the rasterizer starts.
void InitFromSettings();

/// Debug server: /set?name=value (the name is one of the fields above, values are integers or
/// 0/1). Returns false for an unknown name.
bool Set(std::string_view name, s64 value);

/// Current values as JSON.
nlohmann::json Dump();

} // namespace VideoCore::ReadbackPolicy
