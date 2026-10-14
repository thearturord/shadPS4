// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/thread_usage.h"

#ifdef _WIN32
#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace VideoCore {

#ifdef _WIN32

namespace {

using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);

std::string ThreadName(HANDLE thread, DWORD tid) {
    static const auto get_description = reinterpret_cast<GetThreadDescriptionFn>(
        GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "GetThreadDescription"));
    if (get_description) {
        PWSTR wide = nullptr;
        if (SUCCEEDED(get_description(thread, &wide)) && wide) {
            std::string name;
            const int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
            if (size > 1) {
                name.resize(static_cast<size_t>(size) - 1);
                WideCharToMultiByte(CP_UTF8, 0, wide, -1, name.data(), size, nullptr, nullptr);
            }
            LocalFree(wide);
            if (!name.empty()) {
                return name;
            }
        }
    }
    return fmt::format("tid{}", tid);
}

ULONGLONG FileTimeValue(const FILETIME& time) {
    return (static_cast<ULONGLONG>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

/// Calls fn(handle, thread id) for every thread of this process. NtGetNextThread walks only this
/// process (about 1 ms); a Toolhelp snapshot lists every thread of the system (about 90 ms here,
/// which showed as frame hitches when done every few seconds).
template <typename Fn>
void ForEachProcessThread(ACCESS_MASK access, Fn&& fn) {
    using NtGetNextThreadFn = LONG(NTAPI*)(HANDLE, HANDLE, ACCESS_MASK, ULONG, ULONG, PHANDLE);
    static const auto next_thread = reinterpret_cast<NtGetNextThreadFn>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtGetNextThread"));
    if (next_thread) {
        HANDLE current = nullptr;
        while (true) {
            HANDLE following = nullptr;
            const LONG status = next_thread(GetCurrentProcess(), current, access, 0, 0, &following);
            if (current) {
                CloseHandle(current);
            }
            if (status < 0 || !following) {
                return;
            }
            current = following;
            fn(current, GetThreadId(current));
        }
    }
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    const DWORD pid = GetCurrentProcessId();
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (Thread32First(snapshot, &entry)) {
        do {
            if (entry.th32OwnerProcessID != pid) {
                continue;
            }
            if (const HANDLE thread = OpenThread(access, FALSE, entry.th32ThreadID)) {
                fn(thread, entry.th32ThreadID);
                CloseHandle(thread);
            }
        } while (Thread32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);
}

} // namespace

std::vector<ThreadUsage> ThreadUsageSampler::Sample(double* total_out, double min_percent,
                                                    size_t max_threads) {
    FILETIME now_ft;
    GetSystemTimeAsFileTime(&now_ft);
    const ULONGLONG now = FileTimeValue(now_ft);
    const ULONGLONG interval = last_wall != 0 ? now - last_wall : 0;

    std::vector<ThreadUsage> entries;
    double total = 0.0;

    std::unordered_map<u32, u64> current;
    ForEachProcessThread(THREAD_QUERY_LIMITED_INFORMATION, [&](HANDLE thread, DWORD tid) {
        FILETIME creation, exit_time, kernel, user;
        if (!GetThreadTimes(thread, &creation, &exit_time, &kernel, &user)) {
            return;
        }
        const ULONGLONG cpu = FileTimeValue(kernel) + FileTimeValue(user);
        current[tid] = cpu;
        const auto it = last_cpu.find(tid);
        if (interval != 0 && it != last_cpu.end() && cpu >= it->second) {
            const double percent =
                100.0 * static_cast<double>(cpu - it->second) / static_cast<double>(interval);
            total += percent;
            if (percent >= min_percent) {
                entries.push_back({ThreadName(thread, tid), percent});
            }
        }
    });

    last_cpu = std::move(current);
    last_wall = now;
    if (total_out) {
        *total_out = total;
    }
    if (interval == 0) {
        return {};
    }
    std::ranges::sort(entries,
                      [](const ThreadUsage& a, const ThreadUsage& b) { return a.percent > b.percent; });
    if (entries.size() > max_threads) {
        entries.resize(max_threads);
    }
    return entries;
}

namespace {
std::mutex isolation_mutex;
std::vector<std::pair<DWORD, u32>> pinned_threads; // thread id, logical processor
std::unordered_set<DWORD> isolated_done;            // threads already restricted to `others`
size_t isolated_for_pins = 0;                       // pinned_threads.size() when isolated_done was built
} // namespace

void ReapplyCoreIsolation() {
    std::scoped_lock lk{isolation_mutex};
    if (pinned_threads.empty()) {
        return;
    }
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return;
    }
    DWORD_PTR excluded = 0;
    for (const auto& [tid, core] : pinned_threads) {
        excluded |= (DWORD_PTR{1} << core) | (DWORD_PTR{1} << (core ^ 1));
    }
    const DWORD_PTR others = process_mask & ~excluded;
    if (others == 0) {
        return;
    }
    if (isolated_for_pins != pinned_threads.size()) {
        isolated_done.clear(); // the excluded cores changed: every thread needs the new mask
        isolated_for_pins = pinned_threads.size();
    }
    ForEachProcessThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
                         [&](HANDLE thread, DWORD tid) {
                             const bool pinned = std::ranges::any_of(
                                 pinned_threads, [&](const auto& p) { return p.first == tid; });
                             if (pinned || !isolated_done.insert(tid).second) {
                                 return; // setting the same mask again only costs time
                             }
                             SetThreadAffinityMask(thread, others);
                         });
}

void ClearCoreIsolation() {
    std::scoped_lock lk{isolation_mutex};
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return;
    }
    ForEachProcessThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
                         [&](HANDLE thread, DWORD) { SetThreadAffinityMask(thread, process_mask); });
    pinned_threads.clear();
    isolated_done.clear();
    isolated_for_pins = 0;
}

void PinThreadAndIsolateCore(u32 core) {
    if (core >= 63) {
        return;
    }
    {
        std::scoped_lock lk{isolation_mutex};
        pinned_threads.emplace_back(GetCurrentThreadId(), core);
    }
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << core);
    ReapplyCoreIsolation();
}

u32 FindThreadId(const std::string& name_part) {
    u32 found = 0;
    ULONGLONG best_cpu = 0;
    ForEachProcessThread(THREAD_QUERY_LIMITED_INFORMATION, [&](HANDLE thread, DWORD tid) {
        // Several threads can share a name (guest threads): take the one that used the most CPU.
        FILETIME creation, exit_time, kernel, user;
        if (ThreadName(thread, tid).find(name_part) != std::string::npos &&
            GetThreadTimes(thread, &creation, &exit_time, &kernel, &user)) {
            const ULONGLONG cpu = FileTimeValue(kernel) + FileTimeValue(user);
            if (found == 0 || cpu > best_cpu) {
                found = tid;
                best_cpu = cpu;
            }
        }
    });
    return found;
}

std::string ThreadUsageReport() {
    static ThreadUsageSampler sampler;
    double total = 0.0;
    const auto entries = sampler.Sample(&total);
    if (entries.empty()) {
        return {};
    }
    std::string report;
    for (size_t i = 0; i < entries.size(); ++i) {
        report += fmt::format("{}{} {:.0f}%", i ? ", " : "", entries[i].name, entries[i].percent);
    }
    return fmt::format("total {:.0f}% ({:.1f} cores): {}", total, total / 100.0, report);
}

#else

void PinThreadAndIsolateCore(u32) {}
void ReapplyCoreIsolation() {}
void ClearCoreIsolation() {}

u32 FindThreadId(const std::string&) {
    return 0;
}

std::vector<ThreadUsage> ThreadUsageSampler::Sample(double* total, double, size_t) {
    if (total) {
        *total = 0.0;
    }
    return {};
}

std::string ThreadUsageReport() {
    return {};
}

#endif

} // namespace VideoCore
