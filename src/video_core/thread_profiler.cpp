// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include "common/types.h"
#include "video_core/thread_profiler.h"
#include "video_core/thread_usage.h"

#if defined(_WIN32) && defined(_M_X64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace VideoCore {

#if defined(_WIN32) && defined(_M_X64)

namespace {

constexpr u32 MaxDepth = 32;
constexpr u32 MaxSamples = 60'000;

struct Sample {
    std::array<u64, MaxDepth> functions; // start address of each function on the stack, innermost first
    u32 depth;
};

struct Counts {
    u32 self = 0;
    u32 inclusive = 0;
};

thread_local bool unwinding = false;

/// Walks the stack of a suspended thread. Only touches memory that exists already: the target may
/// hold locks, so nothing here may allocate or wait.
void CaptureStackUnsafe(CONTEXT context, Sample& sample) {
    sample.depth = 0;
    u64 pc = context.Rip;
    while (pc != 0 && sample.depth < MaxDepth) {
        DWORD64 image_base = 0;
        const RUNTIME_FUNCTION* function = RtlLookupFunctionEntry(pc, &image_base, nullptr);
        if (!function) {
            // A leaf function without unwind data: the return address is on top of the stack.
            sample.functions[sample.depth++] = pc;
            if (context.Rsp == 0 || context.Rsp & 7) {
                break;
            }
            pc = *reinterpret_cast<const u64*>(context.Rsp);
            context.Rsp += 8;
            continue;
        }
        sample.functions[sample.depth++] = image_base + function->BeginAddress;
        void* handler_data = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, pc, const_cast<PRUNTIME_FUNCTION>(function),
                         &context, &handler_data, &establisher, nullptr);
        pc = context.Rip;
    }
}

/// Like CaptureStackUnsafe, but a fault while unwinding (bad unwind data or stack pointer) only
/// ends the walk of this sample.
void CaptureStack(const CONTEXT& context, Sample& sample) {
    unwinding = true;
    __try {
        CaptureStackUnsafe(context, sample);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    unwinding = false;
}

std::string DescribeAddress(u64 address, const char* exe_name, u64 exe_base, u64& offset) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &module) ||
        !module) {
        offset = address;
        return "unknown";
    }
    offset = address - reinterpret_cast<u64>(module);
    if (reinterpret_cast<u64>(module) == exe_base) {
        return exe_name;
    }
    char path[MAX_PATH]{};
    GetModuleFileNameA(module, path, MAX_PATH);
    std::string name = path;
    return name.substr(name.find_last_of("\\/") + 1);
}

} // namespace

bool IsProfilerUnwinding() noexcept {
    return unwinding;
}

nlohmann::json ProfileThread(const std::string& thread_name, double seconds, u32 hz, u32 top) {
    const u32 tid = FindThreadId(thread_name);
    if (tid == 0) {
        return {{"error", "no thread with a name containing '" + thread_name + "'"}};
    }
    seconds = std::clamp(seconds, 0.5, 30.0);
    hz = std::clamp<u32>(hz, 10, 2000);
    const HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                         THREAD_QUERY_LIMITED_INFORMATION,
                                     FALSE, tid);
    if (!thread) {
        return {{"error", fmt::format("cannot open thread {}", tid)}};
    }
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
    if (!timer) {
        timer = CreateWaitableTimerW(nullptr, TRUE, nullptr);
    }

    std::vector<Sample> samples;
    samples.resize(std::min<u64>(static_cast<u64>(seconds * hz) + 16, MaxSamples));
    size_t count = 0;
    LARGE_INTEGER frequency, start, now;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    const s64 period_ticks = frequency.QuadPart / hz;
    s64 next = start.QuadPart;
    u64 suspended_ticks = 0;
    const s64 end_ticks = start.QuadPart + static_cast<s64>(seconds * frequency.QuadPart);

    while (count < samples.size()) {
        next += period_ticks;
        QueryPerformanceCounter(&now);
        if (now.QuadPart >= end_ticks) {
            break;
        }
        if (next > now.QuadPart) {
            LARGE_INTEGER due;
            due.QuadPart = -(next - now.QuadPart) * 10'000'000 / frequency.QuadPart;
            if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
                WaitForSingleObject(timer, 100);
            } else {
                Sleep(1);
            }
        } else {
            next = now.QuadPart; // fell behind, do not burst
        }
        LARGE_INTEGER before, after;
        QueryPerformanceCounter(&before);
        if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
            break;
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        const bool ok = GetThreadContext(thread, &context) != 0;
        if (ok) {
            CaptureStack(context, samples[count]);
        }
        ResumeThread(thread);
        QueryPerformanceCounter(&after);
        suspended_ticks += static_cast<u64>(after.QuadPart - before.QuadPart);
        if (ok && samples[count].depth != 0) {
            ++count;
        }
    }
    if (timer) {
        CloseHandle(timer);
    }
    CloseHandle(thread);

    std::unordered_map<u64, Counts> functions;
    std::unordered_set<u64> seen;
    for (size_t i = 0; i < count; ++i) {
        const auto& sample = samples[i];
        seen.clear();
        for (u32 d = 0; d < sample.depth; ++d) {
            auto& entry = functions[sample.functions[d]];
            if (d == 0) {
                ++entry.self;
            }
            if (seen.insert(sample.functions[d]).second) {
                ++entry.inclusive;
            }
        }
    }

    const u64 exe_base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
    char exe_path[MAX_PATH]{};
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    std::string exe_name = exe_path;
    exe_name = exe_name.substr(exe_name.find_last_of("\\/") + 1);

    // Time spent outside the executable (waits in ntdll, the graphics driver, the C++ runtime),
    // attributed to the innermost function of the executable on the stack: who is waiting or
    // calling into the driver.
    std::unordered_map<u64, bool> is_exe_function;
    const auto in_exe = [&](u64 function) {
        const auto [it, inserted] = is_exe_function.try_emplace(function, false);
        if (inserted) {
            u64 offset = 0;
            it->second = DescribeAddress(function, exe_name.c_str(), exe_base, offset) == exe_name;
        }
        return it->second;
    };
    std::unordered_map<std::string, std::unordered_map<u64, u32>> outside; // leaf module -> caller -> count
    std::unordered_map<std::string, u32> outside_total;
    for (size_t i = 0; i < count; ++i) {
        const auto& sample = samples[i];
        if (in_exe(sample.functions[0])) {
            continue;
        }
        u64 offset = 0;
        const std::string leaf_module =
            DescribeAddress(sample.functions[0], exe_name.c_str(), exe_base, offset);
        for (u32 d = 1; d < sample.depth; ++d) {
            if (in_exe(sample.functions[d])) {
                ++outside[leaf_module][sample.functions[d]];
                break;
            }
        }
        ++outside_total[leaf_module];
    }

    std::vector<std::pair<u64, Counts>> list(functions.begin(), functions.end());
    const auto render = [&](auto&& key, const char* name) {
        std::ranges::sort(list, [&](const auto& a, const auto& b) { return key(a.second) > key(b.second); });
        nlohmann::json array = nlohmann::json::array();
        for (size_t i = 0; i < list.size() && i < top; ++i) {
            u64 offset = 0;
            const std::string module = DescribeAddress(list[i].first, exe_name.c_str(), exe_base, offset);
            array.push_back({{"module", module},
                             {"rva", fmt::format("{:#x}", offset)},
                             {"self_pct", 100.0 * list[i].second.self / std::max<size_t>(count, 1)},
                             {"inclusive_pct",
                              100.0 * list[i].second.inclusive / std::max<size_t>(count, 1)}});
        }
        return array;
    };

    nlohmann::json j;
    j["thread"] = thread_name;
    j["samples"] = count;
    j["seconds"] = seconds;
    j["hz"] = hz;
    j["suspended_pct"] = 100.0 * static_cast<double>(suspended_ticks) /
                         static_cast<double>(frequency.QuadPart) / seconds;
    nlohmann::json outside_json = nlohmann::json::object();
    for (const auto& [leaf_module, total] : outside_total) {
        if (total * 100 < count * 2) {
            continue; // under 2% of the samples
        }
        std::vector<std::pair<u32, u64>> callers;
        for (const auto& [caller, n] : outside[leaf_module]) {
            callers.emplace_back(n, caller);
        }
        std::ranges::sort(callers, std::greater<>{});
        nlohmann::json array = nlohmann::json::array();
        for (size_t i = 0; i < callers.size() && i < 8; ++i) {
            u64 offset = 0;
            DescribeAddress(callers[i].second, exe_name.c_str(), exe_base, offset);
            array.push_back({{"module", exe_name},
                             {"rva", fmt::format("{:#x}", offset)},
                             {"self_pct", 0.0},
                             {"inclusive_pct", 100.0 * callers[i].first / std::max<size_t>(count, 1)}});
        }
        outside_json[leaf_module] = {{"pct_of_samples", 100.0 * total / std::max<size_t>(count, 1)},
                                     {"callers", array}};
    }
    j["time_outside_exe"] = outside_json;
    j["top_self"] = render([](const Counts& c) { return c.self; }, "self");
    j["top_inclusive"] = render([](const Counts& c) { return c.inclusive; }, "inclusive");
    return j;
}

#else

bool IsProfilerUnwinding() noexcept {
    return false;
}

nlohmann::json ProfileThread(const std::string&, double, u32, u32) {
    return {{"error", "the profiler is only available on Windows x64"}};
}

#endif

} // namespace VideoCore
