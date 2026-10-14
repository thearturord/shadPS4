// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fmt/format.h>
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "video_core/readback_policy.h"

namespace VideoCore::ReadbackPolicy {

State& Get() {
    static State state;
    return state;
}

int ReadbacksMode() {
    const s32 override_mode = Get().readbacks_mode_override.load(std::memory_order_relaxed);
    return override_mode >= 0 ? override_mode : static_cast<int>(EmulatorSettings.GetReadbacksMode());
}

bool IsEmulatorThread() {
    thread_local const bool emulator = Common::GetCurrentThreadName().starts_with("shadPS4:");
    return emulator;
}

void InitFromSettings() {
    auto& s = Get();
    s.prefetch_lifetime = EmulatorSettings.GetReadbackPrefetchLifetime();
    s.full_fence_kinds = EmulatorSettings.GetCpFullFenceKinds();
    s.images_async = EmulatorSettings.IsReadbackLinearImagesAsync();
    s.dma_async = EmulatorSettings.IsReadbackDmaAsync();
    s.flush_commands = EmulatorSettings.GetCpFlushCommands();
    s.raster_core = EmulatorSettings.GetCpRasterCore();
}

bool Set(std::string_view name, s64 value) {
    auto& s = Get();
    if (name == "prefetch_kinds_mask") {
        s.prefetch_kinds_mask = static_cast<u32>(value);
    } else if (name == "image_download_kinds_mask") {
        s.image_download_kinds_mask = static_cast<u32>(value);
    } else if (name == "prefetch_mode_override") {
        s.prefetch_mode_override = static_cast<s32>(value);
    } else if (name == "prefetch_lifetime") {
        s.prefetch_lifetime = static_cast<u32>(value);
    } else if (name == "full_fence_kinds") {
        s.full_fence_kinds = static_cast<u32>(value);
    } else if (name == "images_async") {
        s.images_async = value != 0;
    } else if (name == "dma_async") {
        s.dma_async = value != 0;
    } else if (name == "skip_shadows") {
        s.skip_shadows = static_cast<u32>(value);
    } else if (name == "depth_hist_reset") {
        for (auto& bucket : s.depth_hist) {
            bucket.count = 0;
            bucket.key = 0;
        }
        s.skipped_draws = 0;
    } else if (name == "readbacks_mode") {
        s.readbacks_mode_override = static_cast<s32>(value);
    } else if (name == "flush_commands") {
        s.flush_commands = static_cast<u32>(value);
    } else if (name == "raster_core") {
        s.raster_core = static_cast<s32>(value);
    } else {
        return false;
    }
    return true;
}

nlohmann::json Dump() {
    const auto& s = Get();
    nlohmann::json hist = nlohmann::json::array();
    for (const auto& bucket : s.depth_hist) {
        const u32 key = bucket.key.load();
        if (key != 0) {
            hist.push_back({{"size", std::to_string(key >> 16) + "x" + std::to_string(key & 0xffff)},
                            {"draws", bucket.count.load()}});
        }
    }
    return {{"skip_shadows", s.skip_shadows.load()},
            {"skipped_draws", s.skipped_draws.load()},
            {"depth_only_draws_by_target", hist},
            {"prefetch_kinds_mask", s.prefetch_kinds_mask.load()},
            {"image_download_kinds_mask", s.image_download_kinds_mask.load()},
            {"prefetch_mode_override", s.prefetch_mode_override.load()},
            {"prefetch_lifetime", s.prefetch_lifetime.load()},
            {"full_fence_kinds", s.full_fence_kinds.load()},
            {"images_async", s.images_async.load()},
            {"dma_async", s.dma_async.load()},
            {"dma_async_copies", s.dma_async_copies.load()},
            {"dma_async_bytes", s.dma_async_bytes.load()},
            {"readbacks_mode_override", s.readbacks_mode_override.load()},
            {"flush_commands", s.flush_commands.load()},
            {"raster_core", s.raster_core.load()},
            {"note", "fence kinds: 1 gfx EOP, 2 gfx EOS, 4 compute release, 8 held write, "
                     "16 submit end"}};
}

} // namespace VideoCore::ReadbackPolicy
