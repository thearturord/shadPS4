// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include "common/types.h"

// Diagnostic counters for GPU readbacks and related stalls. Enabled per game with the
// `readback_stats_enabled` GPU setting. Writes readback_stats.csv (one line per second) and
// readback_top.txt (buffers/images/shaders ranked by stall time) to the log directory.
namespace VideoCore::ReadbackStats {

/// Pseudo stage ids used for writers that are not shaders.
constexpr u32 WriterFill = 0xFD;
constexpr u32 WriterCopy = 0xFE;

namespace Detail {
extern std::atomic<bool> enabled;
}

[[nodiscard]] inline bool IsEnabled() noexcept {
    return Detail::enabled.load(std::memory_order_relaxed);
}

[[nodiscard]] inline u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Reads the setting and starts the reporter thread when enabled.
void Start();

/// Stops the reporter thread and writes the final report.
void Stop();

void OnPresent();
void OnSubmit();
void OnDraw();
void OnDispatch();
void OnFault(bool is_write, u64 handler_ns);
void OnProtect(u64 ns);
void OnFinish(u64 wait_ns);
void OnGpuThreadIdle(u64 idle_ns);

/// Records which shader (or fill/copy) last wrote a guest memory range on the GPU.
void RecordWriter(VAddr base, u64 size, u64 shader_hash, u32 stage);

/// Records a buffer readback caused by a CPU access to GPU modified memory.
void OnBufferReadback(VAddr fault_addr, u64 bytes, u64 finish_ns, u64 blocked_ns,
                      bool write_triggered);

/// Records a batched readback that completed buffers and images with a single GPU wait.
void OnBatch(u32 num_buffers, u32 num_images, u64 bytes, u64 wait_ns);

/// Records a command stream wait on memory (WaitRegMem) that did not pass right away.
/// `shortcut` means it was satisfied by a fence write deferred behind readbacks.
void OnRegMemWait(u64 wait_ns, bool shortcut);

/// Records time spent waiting for the GPU to free space in a stream buffer.
void OnStreamWait(u64 wait_ns);

/// Records a linear image readback.
void OnImageReadback(VAddr addr, u64 bytes, u32 width, u32 height, u32 num_bits, u64 finish_ns);

} // namespace VideoCore::ReadbackStats
