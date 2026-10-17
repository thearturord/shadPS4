// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "core/memory.h"
#include "core/native_audit.h"
#include "core/native_resources.h"
#include "video_core/readback_policy.h"

namespace Core::NativeAudit {

namespace {

constexpr u64 PageSize = 4096;
constexpr u32 SlotBits = 20;
constexpr u64 SlotMask = (1ull << SlotBits) - 1;
constexpr u32 TagBits = 28;
/// A slot holds (page << 28) | 28 bits of the hash of the page as it was uploaded; 0 is empty.
std::array<std::atomic<u64>, 1u << SlotBits> slots;

std::atomic<u64> uploads_hashed{}, checks{}, checks_skipped_rate{}, no_record{}, stale{}, racing{}, forgotten{};
std::array<std::atomic<u64>, 20> stale_by_class{};
std::array<std::atomic<u64>, 20> checks_by_class{};
std::atomic<u64> budget_second{};
std::atomic<u32> budget_left{};

std::mutex events_mutex;
struct Event {
    VAddr page;
    std::string where;
    double seconds;
};
std::vector<Event> events;

const auto start_time = std::chrono::steady_clock::now();

u64 PageHash(const u8* data) {
    u64 h = 0xcbf29ce484222325ull;
    for (u32 i = 0; i < PageSize; i += 8) {
        u64 word;
        std::memcpy(&word, data + i, sizeof(word));
        h = (h ^ word) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    return h;
}

bool Sampled(u64 page_index) {
    return ((page_index * 0x9E3779B97F4A7C15ull) >> 59) < 4; // 4 of 32 pages
}

u64 Pack(u64 page_index, u64 hash) {
    return (page_index << TagBits) | (hash & ((1ull << TagBits) - 1)) | (1ull << 63);
}

std::atomic<u64>& Slot(u64 page_index) {
    return slots[(page_index * 0x9E3779B97F4A7C15ull) >> (64 - SlotBits)];
}

/// At most 20000 page checks per second: the audit must not slow the raster thread down.
bool TakeBudget() {
    const u64 second = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count());
    u64 seen = budget_second.load(std::memory_order_relaxed);
    if (seen != second && budget_second.compare_exchange_strong(seen, second)) {
        budget_left.store(20000, std::memory_order_relaxed);
    }
    u32 left = budget_left.load(std::memory_order_relaxed);
    while (left != 0) {
        if (budget_left.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

std::atomic<u64> stale_draws{}, stale_differs{};
std::array<std::atomic<u64>, 8> stale_group_draws{}, stale_group_differs{};
std::mutex stale_mutex;
std::map<std::pair<u64, std::string>, u64> stale_by_caller; ///< (caller, fields that differ) -> draws
struct StaleExample {
    u64 present;
    u64 caller;
    std::string diff;
};
std::vector<StaleExample> stale_examples;
std::map<u64, u64> stale_differs_by_frame_phase; ///< present % 3 -> draws that differ
thread_local u64 plan_caller;

} // namespace

bool StaleEnabled() {
    return (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 134217728) != 0;
}

void SetPlanCaller(u64 caller) {
    plan_caller = caller;
}

u64 PlanCaller() {
    return plan_caller;
}

void NoteStale(u32 groups, bool differs, const std::string& diff, u64 caller, u64 present) {
    stale_draws.fetch_add(1, std::memory_order_relaxed);
    // which group's fields differ (the field names are the ones of DiffFixedState)
    const auto has = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
            if (diff.find(name) != std::string::npos) {
                return true;
            }
        }
        return false;
    };
    const bool group_differs[3] = {has({"cull_mode", "front_face", "polygon_mode", "provoking_first"}),
                                   has({"depth_clip", "negative_one_to_one", "rasterizer_discard", "user_clip_plane_mask", "clip_disabled"}),
                                   has({"depth_clear", "stencil_clear"})};
    for (u32 bit = 0; bit < 3; ++bit) {
        if ((groups >> bit) & 1) {
            stale_group_draws[bit].fetch_add(1, std::memory_order_relaxed);
            if (group_differs[bit]) {
                stale_group_differs[bit].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (!differs) {
        return;
    }
    stale_differs.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{stale_mutex};
    ++stale_by_caller[{caller, diff}];
    ++stale_differs_by_frame_phase[present % 3];
    if (stale_examples.size() < 30) {
        stale_examples.push_back({present, caller, diff});
    }
}

bool Enabled() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return (policy.native_sdk_strip.load(std::memory_order_relaxed) & 67108864) != 0;
}

void NoteUpload(VAddr address, const u8* data, u64 size) {
    if ((address & (PageSize - 1)) != 0) {
        return;
    }
    for (u64 offset = 0; offset + PageSize <= size; offset += PageSize) {
        const u64 page_index = (address + offset) / PageSize;
        if (!Sampled(page_index)) {
            continue;
        }
        Slot(page_index).store(Pack(page_index, PageHash(data + offset)), std::memory_order_relaxed);
        uploads_hashed.fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteGpuWrite(VAddr address, u64 size) {
    const u64 first = address / PageSize;
    const u64 last = (address + size - 1) / PageSize;
    for (u64 page_index = first; page_index <= last; ++page_index) {
        if (!Sampled(page_index)) {
            continue;
        }
        auto& slot = Slot(page_index);
        const u64 value = slot.load(std::memory_order_relaxed);
        if (value != 0 && (value & ~(1ull << 63)) >> TagBits == page_index) {
            slot.store(0, std::memory_order_relaxed);
            forgotten.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void CheckClean(VAddr address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address / PageSize;
    const u64 last = (address + size - 1) / PageSize;
    std::array<u8, PageSize> buffer;
    for (u64 page_index = first; page_index <= last; ++page_index) {
        if (!Sampled(page_index)) {
            continue;
        }
        const u64 recorded = Slot(page_index).load(std::memory_order_relaxed);
        if (recorded == 0 || ((recorded & ~(1ull << 63)) >> TagBits) != page_index) {
            no_record.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!TakeBudget()) {
            checks_skipped_rate.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        auto* memory = Core::Memory::Instance();
        memory->CopySparseMemory(page_index * PageSize, buffer.data(), PageSize);
        const u64 hash = PageHash(buffer.data());
        checks.fetch_add(1, std::memory_order_relaxed);
        const u32 usage_class = Core::NativeResources::UsageClassOf(page_index * PageSize);
        checks_by_class[usage_class % checks_by_class.size()].fetch_add(1, std::memory_order_relaxed);
        if (Pack(page_index, hash) == recorded) {
            continue;
        }
        // differs: look again, the CPU may be writing the page right now
        memory->CopySparseMemory(page_index * PageSize, buffer.data(), PageSize);
        const u64 again = PageHash(buffer.data());
        if (again != hash) {
            racing.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        stale.fetch_add(1, std::memory_order_relaxed);
        stale_by_class[usage_class % stale_by_class.size()].fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lock{events_mutex};
        if (events.size() < 40) {
            events.push_back({page_index * PageSize, Core::NativeResources::DescribeAddress(page_index * PageSize),
                              std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count()});
        }
        // report the page once: take the new hash as the reference
        Slot(page_index).store(Pack(page_index, hash), std::memory_order_relaxed);
    }
}

nlohmann::json Report() {
    using nlohmann::json;
    json out;
    out["enabled"] = Enabled();
    out["pages_hashed_at_upload"] = uploads_hashed.load();
    out["pages_forgotten_by_gpu_writes"] = forgotten.load();
    out["checks"] = checks.load();
    out["checks_skipped_by_rate_limit"] = checks_skipped_rate.load();
    out["checks_without_upload_record"] = no_record.load();
    out["racing_with_a_cpu_write"] = racing.load();
    out["stale"] = stale.load();
    json by_class = json::array();
    for (u32 c = 0; c < checks_by_class.size(); ++c) {
        const u64 n = checks_by_class[c].load();
        if (n != 0) {
            by_class.push_back({{"class", Core::NativeResources::UsageClassNameOf(c)}, {"checks", n}, {"stale", stale_by_class[c].load()}});
        }
    }
    out["by_resource_class"] = by_class;
    json list = json::array();
    {
        std::scoped_lock lock{events_mutex};
        for (const auto& event : events) {
            list.push_back({{"page", fmt::format("{:#x}", event.page)}, {"where", event.where}, {"at_s", event.seconds}});
        }
    }
    out["first_stale_pages"] = list;
    {
        json stale;
        stale["draws"] = stale_draws.load();
        stale["draws_where_the_register_value_differs_from_the_engine_object"] = stale_differs.load();
        stale["by_group"] = {{"surface", {{"draws", stale_group_draws[0].load()}, {"differs", stale_group_differs[0].load()}}},
                             {"clip", {{"draws", stale_group_draws[1].load()}, {"differs", stale_group_differs[1].load()}}},
                             {"clear_flags", {{"draws", stale_group_draws[2].load()}, {"differs", stale_group_differs[2].load()}}}};
        std::scoped_lock lock{stale_mutex};
        json callers = json::array();
        std::vector<std::pair<u64, std::pair<u64, std::string>>> sorted;
        for (const auto& [key, count] : stale_by_caller) {
            sorted.push_back({count, key});
        }
        std::sort(sorted.rbegin(), sorted.rend());
        for (size_t i = 0; i < sorted.size() && i < 25; ++i) {
            callers.push_back({{"draws", sorted[i].first}, {"caller", fmt::format("{:#x}", sorted[i].second.first)}, {"fields", sorted[i].second.second}});
        }
        stale["by_caller_and_fields"] = callers;
        json phases = json::object();
        for (const auto& [phase, count] : stale_differs_by_frame_phase) {
            phases[std::to_string(phase)] = count;
        }
        stale["differing_draws_by_present_number_mod_3"] = phases;
        json examples = json::array();
        for (const auto& e : stale_examples) {
            examples.push_back({{"present", e.present}, {"caller", fmt::format("{:#x}", e.caller)}, {"fields", e.diff}});
        }
        stale["first_examples"] = examples;
        out["stale_state"] = stale;
    }
    out["note"] = "stale: the page differs from what was uploaded and the tracker called it clean; classes need the census (/resources?start=1)";
    return out;
}

} // namespace Core::NativeAudit
