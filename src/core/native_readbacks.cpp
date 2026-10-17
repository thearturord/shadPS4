// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "core/native_readbacks.h"

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

void NoteMap(VAddr address, u64 size, u64 caller) {
    std::scoped_lock lock{mutex};
    if (Range* range = Find(address, size ? size : 1)) {
        ++range->maps;
        ++range->map_callers[caller];
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
                        {"read_faults_same_page", range.shared_faults}});
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
    out["read_faults_total"] = faults_total.load();
    out["read_faults_elsewhere"] = faults_other.load();
    return out;
}

} // namespace Core::NativeReadbacks
