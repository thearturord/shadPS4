// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <vector>
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace VideoCore {

/// Collects GPU to CPU copies so that they can be completed with a single GPU wait.
/// Recorders call Reserve() before recording a new copy and Add() after recording it.
/// Write back callbacks run after the wait, in the order they were added.
class ReadbackBatch {
public:
    explicit ReadbackBatch(Vulkan::Scheduler& scheduler_, u64 max_bytes_)
        : scheduler{scheduler_}, max_bytes{max_bytes_} {}

    ~ReadbackBatch() {
        Flush();
    }

    ReadbackBatch(const ReadbackBatch&) = delete;
    ReadbackBatch& operator=(const ReadbackBatch&) = delete;

    /// Must be called before recording a new copy. A batch is kept within max_bytes: a bigger one
    /// is completed first.
    void Reserve(u64 size) {
        if (!callbacks.empty() && bytes + size > max_bytes) {
            Flush();
        }
    }

    void Add(u64 size, bool is_image, Common::UniqueFunction<void>&& write_back) {
        bytes += size;
        (is_image ? num_images : num_buffers)++;
        callbacks.push_back(std::move(write_back));
    }

    [[nodiscard]] bool Empty() const noexcept {
        return callbacks.empty();
    }

    /// Hands over the recorded write backs without waiting. The caller must run them once the
    /// GPU has executed the current command buffer.
    std::vector<Common::UniqueFunction<void>> TakeCallbacks() {
        bytes = 0;
        num_buffers = 0;
        num_images = 0;
        return std::exchange(callbacks, {});
    }

    /// Waits for the GPU once and writes all recorded copies back to guest memory.
    void Flush() {
        if (callbacks.empty()) {
            return;
        }
        scheduler.Finish();
        for (auto& callback : callbacks) {
            callback();
        }
        callbacks.clear();
        bytes = 0;
        num_buffers = 0;
        num_images = 0;
    }

private:
    Vulkan::Scheduler& scheduler;
    u64 max_bytes;
    u64 bytes{};
    u32 num_buffers{};
    u32 num_images{};
    std::vector<Common::UniqueFunction<void>> callbacks;
};

} // namespace VideoCore
