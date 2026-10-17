// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "core/native_readbacks.h"
#include "video_core/readback_policy.h"

namespace Core::NativeReadbacks {

namespace {

struct Range {
    VAddr begin{};
    VAddr end{};
    u32 usage{~0u};
    u32 kind{};
    u64 caller{};
    u64 notes{};          ///< records that write it
    u64 bytes{};          ///< bytes those records write
    u64 maps{};           ///< AsyncMap calls that returned (part of) it
    u64 faults{};         ///< CPU read faults inside it
    u64 shared_faults{};  ///< CPU read faults on a page it shares but outside it
    u64 values{};         ///< 4 byte ranges: the first dword at each map (counters)
    u64 value_sum{};
    u32 value_min{~0u};
    u32 value_max{};
    u64 value_zero{};
    u64 value_repeat{};   ///< maps that saw the same value as the previous one
    u32 value_last{~0u};
    std::map<u64, u64> map_callers;
};

constexpr size_t MaxRanges = 128;
constexpr u32 MaxRangesPerCaller = 6;
std::mutex mutex;
std::array<Range, MaxRanges> ranges;
size_t range_count{};
std::atomic<u64> faults_other{}, maps_other{}, faults_total{};
std::map<u64, u64> other_map_callers;
std::map<std::pair<u64, u32>, std::pair<u64, u64>> per_caller; ///< (caller, record kind) -> records, bytes

std::atomic<u64> explicit_copies{}, explicit_bytes{};
std::array<std::atomic<u64>, 2> explicit_refused{};

LandingGate landing_gate;
std::atomic<u64> gate_checks{}, gate_waits{}, gate_timeouts{}, gate_wait_ns{};
std::atomic<u64> landing_begins{}, landing_ends{}, landings_active{};
std::atomic<u64> readers_active{}, reader_windows{}, reader_ns_total{}, reader_ns_max{};
std::atomic<u64> landing_started_during_read{}, read_started_during_landing{}, windows_with_landing{}, landings_in_windows{};
thread_local u64 reader_start_ns;
thread_local u64 reader_landings_at_start;

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

constexpr u64 PageSize = 4096;

Range* Find(VAddr address, u64 size) {
    for (size_t i = 0; i < range_count; ++i) {
        if (address < ranges[i].end && address + size > ranges[i].begin) {
            return &ranges[i];
        }
    }
    return nullptr;
}

} // namespace

void NoteDestination(VAddr address, u64 size, u32 usage, u32 kind, u64 caller) {
    std::scoped_lock lock{mutex};
    for (size_t i = 0; i < range_count; ++i) {
        if (ranges[i].begin == address && ranges[i].end == address + size) {
            ++ranges[i].notes;
            ranges[i].bytes += size;
            auto& seen = per_caller[{caller, kind}];
            ++seen.first;
            seen.second += size;
            return;
        }
    }
    u32 same_caller = 0;
    for (size_t i = 0; i < range_count; ++i) {
        same_caller += ranges[i].caller == caller && ranges[i].kind == kind;
    }
    auto& total = per_caller[{caller, kind}];
    ++total.first;
    total.second += size;
    if (range_count == MaxRanges || same_caller >= MaxRangesPerCaller) {
        return;
    }
    auto& range = ranges[range_count++];
    range.begin = address;
    range.end = address + size;
    range.usage = usage;
    range.kind = kind;
    range.caller = caller;
    range.notes = 1;
    range.bytes = size;
}

void NoteMap(VAddr address, u64 size, u64 caller, u32 first_dword) {
    std::scoped_lock lock{mutex};
    if (Range* range = Find(address, size ? size : 1)) {
        ++range->maps;
        ++range->map_callers[caller];
        if (size == sizeof(u32)) {
            ++range->values;
            range->value_sum += first_dword;
            range->value_min = std::min(range->value_min, first_dword);
            range->value_max = std::max(range->value_max, first_dword);
            range->value_zero += first_dword == 0;
            range->value_repeat += first_dword == range->value_last;
            range->value_last = first_dword;
        }
    } else {
        maps_other.fetch_add(1, std::memory_order_relaxed);
        ++other_map_callers[caller];
    }
}

void NoteReadFault(VAddr address, u64 size) {
    faults_total.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (Range* range = Find(address, size ? size : 1)) {
        ++range->faults;
        return;
    }
    const VAddr page_begin = address & ~(PageSize - 1);
    const VAddr page_end = (address + (size ? size : 1) + PageSize - 1) & ~(PageSize - 1);
    for (size_t i = 0; i < range_count; ++i) {
        if (page_begin < ranges[i].end && page_end > ranges[i].begin) {
            ++ranges[i].shared_faults;
            return;
        }
    }
    faults_other.fetch_add(1, std::memory_order_relaxed);
}

bool ExplicitEnabled() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && policy.dma_async.load(std::memory_order_relaxed) &&
           (policy.native_sdk_strip.load(std::memory_order_relaxed) & 33554432) != 0;
}

void NoteExplicit(bool done, u64 bytes, u32 reason) {
    if (done) {
        explicit_copies.fetch_add(1, std::memory_order_relaxed);
        explicit_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else if (reason < explicit_refused.size()) {
        explicit_refused[reason].fetch_add(1, std::memory_order_relaxed);
    }
}

void SetLandingGate(LandingGate gate) {
    std::scoped_lock lock{mutex};
    landing_gate = std::move(gate);
}

void GateMap(VAddr address, u64 size) {
    LandingGate gate;
    {
        std::scoped_lock lock{mutex};
        const Range* range = Find(address, size ? size : 1);
        if (range == nullptr || range->usage != 3 || !landing_gate) {
            return;
        }
        gate = landing_gate;
    }
    gate_checks.fetch_add(1, std::memory_order_relaxed);
    const u64 start = NowNs();
    const u32 result = gate(address, size, 20);
    if (result != 0) {
        gate_waits.fetch_add(1, std::memory_order_relaxed);
        gate_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
        if (result == 2) {
            gate_timeouts.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void LandingBegin(VAddr, u64) {
    landing_begins.fetch_add(1, std::memory_order_relaxed);
    landings_active.fetch_add(1, std::memory_order_relaxed);
    if (readers_active.load(std::memory_order_relaxed) != 0) {
        landing_started_during_read.fetch_add(1, std::memory_order_relaxed);
    }
}

void LandingEnd(VAddr, u64) {
    landing_ends.fetch_add(1, std::memory_order_relaxed);
    landings_active.fetch_sub(1, std::memory_order_relaxed);
}

void ReaderEnter() {
    readers_active.fetch_add(1, std::memory_order_relaxed);
    if (landings_active.load(std::memory_order_relaxed) != 0) {
        read_started_during_landing.fetch_add(1, std::memory_order_relaxed);
    }
    reader_start_ns = NowNs();
    reader_landings_at_start = landing_begins.load(std::memory_order_relaxed);
}

void ReaderExit() {
    const u64 duration = NowNs() - reader_start_ns;
    readers_active.fetch_sub(1, std::memory_order_relaxed);
    reader_windows.fetch_add(1, std::memory_order_relaxed);
    reader_ns_total.fetch_add(duration, std::memory_order_relaxed);
    u64 previous = reader_ns_max.load(std::memory_order_relaxed);
    while (duration > previous && !reader_ns_max.compare_exchange_weak(previous, duration, std::memory_order_relaxed)) {
    }
    if (const u64 landings = landing_begins.load(std::memory_order_relaxed) - reader_landings_at_start; landings != 0) {
        windows_with_landing.fetch_add(1, std::memory_order_relaxed);
        landings_in_windows.fetch_add(landings, std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    using nlohmann::json;
    std::scoped_lock lock{mutex};
    json out;
    json list = json::array();
    for (size_t i = 0; i < range_count; ++i) {
        const Range& range = ranges[i];
        json callers = json::object();
        for (const auto& [caller, count] : range.map_callers) {
            callers[fmt::format("{:#x}", caller)] = count;
        }
        list.push_back({{"begin", fmt::format("{:#x}", range.begin)},
                        {"bytes", range.end - range.begin},
                        {"usage", range.usage == ~0u ? json("unknown") : json(range.usage)},
                        {"record_kind", range.kind},
                        {"record_caller", fmt::format("{:#x}", range.caller)},
                        {"records", range.notes},
                        {"bytes_written", range.bytes},
                        {"maps", range.maps},
                        {"map_callers", callers},
                        {"read_faults_inside", range.faults},
                        {"read_faults_same_page", range.shared_faults},
                        {"counter_values", range.values == 0 ? json(nullptr)
                                                              : json{{"maps", range.values},
                                                                     {"min", range.value_min},
                                                                     {"max", range.value_max},
                                                                     {"mean", static_cast<double>(range.value_sum) / static_cast<double>(range.values)},
                                                                     {"zero", range.value_zero},
                                                                     {"same_as_previous", range.value_repeat}}}});
    }
    out["destinations"] = list;
    json callers_json = json::array();
    for (const auto& [key, value] : per_caller) {
        callers_json.push_back({{"caller", fmt::format("{:#x}", key.first)}, {"record_kind", key.second}, {"records", value.first}, {"bytes", value.second}});
    }
    out["record_callers"] = callers_json;
    json other = json::object();
    for (const auto& [caller, count] : other_map_callers) {
        other[fmt::format("{:#x}", caller)] = count;
    }
    out["maps_outside_destinations"] = maps_other.load();
    out["map_callers_outside"] = other;
    out["explicit"] = {{"enabled", ExplicitEnabled()},
                       {"copies", explicit_copies.load()},
                       {"bytes", explicit_bytes.load()},
                       {"refused_other_path", explicit_refused[0].load()},
                       {"refused_too_large", explicit_refused[1].load()}};
    out["gate"] = {{"checks", gate_checks.load()}, {"waits", gate_waits.load()}, {"timeouts", gate_timeouts.load()},
                   {"wait_ms_total", static_cast<double>(gate_wait_ns.load()) / 1e6}};
    {
        const u64 windows = reader_windows.load();
        out["reader"] = {{"windows", windows},
                         {"mean_us", windows ? static_cast<double>(reader_ns_total.load()) / 1e3 / static_cast<double>(windows) : 0.0},
                         {"max_us", static_cast<double>(reader_ns_max.load()) / 1e3},
                         {"landings", landing_ends.load()},
                         {"landing_started_during_a_read", landing_started_during_read.load()},
                         {"read_started_during_a_landing", read_started_during_landing.load()},
                         {"windows_with_a_landing_begin", windows_with_landing.load()},
                         {"landing_begins_in_windows", landings_in_windows.load()}};
    }
    out["read_faults_total"] = faults_total.load();
    out["read_faults_elsewhere"] = faults_other.load();
    return out;
}

} // namespace Core::NativeReadbacks
