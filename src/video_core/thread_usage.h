// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "common/types.h"

namespace VideoCore {

struct ThreadUsage {
    std::string name;
    double percent; // 100 = one core
};

/// Measures the CPU use of this process' threads between two Sample() calls. Windows only (empty
/// result elsewhere and on the first call, which sets the baseline). Not thread safe.
class ThreadUsageSampler {
public:
    /// Threads using at least `min_percent`, busiest first, at most `max_threads`. `total` receives
    /// the use of all threads.
    std::vector<ThreadUsage> Sample(double* total = nullptr, double min_percent = 3.0,
                                    size_t max_threads = 14);

private:
    std::unordered_map<u32, u64> last_cpu;
    u64 last_wall = 0;
};

/// Windows: pins the calling thread to logical processor `core` and restricts every other thread of
/// the process to the remaining processors except `core ^ 1` (its SMT sibling). Call again later
/// (ReapplyCoreIsolation) to cover threads created since.
void PinThreadAndIsolateCore(u32 core);
void ReapplyCoreIsolation();

/// Undoes PinThreadAndIsolateCore: every thread (also the pinned ones) may run on every processor
/// of the process again and the pins are forgotten.
void ClearCoreIsolation();

/// Id of the first thread of this process whose name (SetThreadDescription) contains `name_part`,
/// or 0. Windows only.
u32 FindThreadId(const std::string& name_part);

/// CPU use of the busiest threads of this process since the previous call, as text
/// ("name 62%, other 14%, ..."), for the log. Windows only; empty elsewhere and on the first call
/// (which sets the baseline). 100% is one core.
std::string ThreadUsageReport();

} // namespace VideoCore
