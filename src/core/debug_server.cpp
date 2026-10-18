// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <httplib.h>
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/debug_server.h"
#include "core/native_resources.h"
#include "core/memory.h"
#include "core/native_draw.h"
#include "core/native_entry_dump.h"
#include "core/native_audit.h"
#include "video_core/renderdoc.h"
#include "core/native_readbacks.h"
#include "video_core/readback_policy.h"
#include "video_core/thread_profiler.h"
#include "video_core/thread_usage.h"

namespace Core::DebugServer {

namespace {

struct Entry {
    std::string description;
    Handler handler;
};

std::mutex registry_mutex;
std::map<std::string, Entry> registry;
std::unique_ptr<httplib::Server> server;
std::jthread server_thread;
std::mutex lifecycle_mutex;
std::mutex published_mutex;
std::map<std::string, nlohmann::json> published;

const auto start_time = std::chrono::steady_clock::now();
std::atomic<u64> presents{0};
constexpr u64 TimeRing = 1024;
std::array<std::atomic<u64>, TimeRing> present_times{};

constexpr u64 FaultRegions = 1u << 14; // 4 MB regions, aliased above 64 GB
std::array<std::atomic<u32>, FaultRegions> fault_regions{};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start_time)
                                .count());
}

u64 ParseNumber(const Params& params, const std::string& key, u64 fallback) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return fallback;
    }
    return std::stoull(it->second, nullptr, 0);
}

double FpsOver(double seconds) {
    const u64 count = presents.load(std::memory_order_acquire);
    const u64 now = NowNs();
    const u64 window = static_cast<u64>(seconds * 1e9);
    u64 frames = 0;
    u64 oldest = now;
    for (u64 i = 0; i < std::min<u64>(count, TimeRing); ++i) {
        const u64 t = present_times[(count - 1 - i) & (TimeRing - 1)].load(std::memory_order_relaxed);
        if (t == 0 || now - t > window) {
            break;
        }
        ++frames;
        oldest = t;
    }
    if (frames < 2) {
        return 0.0;
    }
    return static_cast<double>(frames - 1) / (static_cast<double>(now - oldest) / 1e9);
}

nlohmann::json Status(const Params&) {
    nlohmann::json j;
    j["uptime_s"] = static_cast<double>(NowNs()) / 1e9;
    j["presents"] = presents.load();
    j["fps_1s"] = FpsOver(1.0);
    j["fps_5s"] = FpsOver(5.0);
    j["game_serial"] = MemoryPatcher::g_game_serial;
    return j;
}

nlohmann::json EntryDumpEndpoint(const Params& params) {
    if (params.find("frames") != params.end()) {
        Core::EntryDump::Request(static_cast<u32>(std::min<u64>(ParseNumber(params, "frames", 1), 16)));
    }
    return Core::EntryDump::Status();
}

nlohmann::json Native(const Params&) {
    auto& live = NativeDraw::Live();
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    nlohmann::json j;
    j["note"] = "cumulative counters of the native renderer since start; take differences for rates";
    j["mode"] = policy.native_draw_mode.load();
    j["exec"] = policy.native_draw_exec.load();
    j["audit_every"] = policy.native_audit_every.load();
    j["sdk_strip"] = policy.native_sdk_strip.load();
    j["gfx_tags"] = live.gfx_tags.load();
    j["gfx_exec"] = live.gfx_exec.load();
    j["compute_tags"] = live.compute_tags.load();
    j["compute_exec"] = live.compute_exec.load();
    j["asc_exec"] = live.asc_exec.load();
    j["asc_packet"] = live.asc_packet.load();
    j["asc_no_record"] = live.asc_no_record.load();
    j["fill_exec"] = live.fill_exec.load();
    j["mode_switch_exec"] = live.mode_switch_exec.load();
    j["asc_barrier_exec"] = live.asc_barrier_exec.load();
    j["clear_exec"] = live.clear_exec.load();
    j["asc_begin_stripped"] = live.asc_begin_stripped.load();
    j["counter_readback"] = live.counter_readback.load();
    j["async_copy_exec"] = live.async_copy_exec.load();
    j["device_begin_stripped"] = live.device_begin_stripped.load();
    j["sync_units_stripped"] = live.sync_units_stripped.load();
    j["sync_acquires_stripped"] = live.sync_acquires_stripped.load();
    j["sync_events_stripped"] = live.sync_events_stripped.load();
    j["barrier_calls_skipped"] = live.barrier_calls_skipped.load();
    {
        nlohmann::json audit;
        audit["dispatches"] = live.cs_audit_dispatches.load();
        audit["differing"] = live.cs_audit_differing.load();
        nlohmann::json dwords = nlohmann::json::object();
        for (size_t i = 0; i < live.cs_audit_dword.size(); ++i) {
            if (const u64 count = live.cs_audit_dword[i].load(); count != 0) {
                dwords[std::to_string(i)] = count;
            }
        }
        audit["differing_dwords"] = dwords;
        audit["classification"] = {{"draws", live.class_audit_draws.load()}, {"class_differs", live.class_audit_differ.load()}, {"extent_differs", live.extent_audit_differ.load()}};
        audit["mirror_reads"] = live.mirror_reads.load();
        j["cs_audit"] = audit;
    }
    j["map_native"] = {{"tried", live.map_tried.load()}, {"patched", live.map_patched.load()}, {"refused", live.map_refused.load()}, {"executed", live.map_exec.load()}};
    j["fetch_rebuild"] = {{"tried", live.fetch_rebuild_tried.load()}, {"rebuilt", live.fetch_rebuilt.load()}, {"refused", live.fetch_rebuild_refused.load()}, {"executed", live.fetch_rebuild_exec.load()}};
    j["sync_converted_hook"] = live.sync_converted_hook.load();
    j["sync_converted_sweep"] = live.sync_converted_sweep.load();
    j["sync_converted_submit"] = live.sync_converted_submit.load();
    j["sync_packets_kept"] = live.fence_packets_kept.load();
    j["sync_converted"] = {{"eop", live.sync_converted[1].load()}, {"release", live.sync_converted[2].load()}, {"wait", live.sync_converted[3].load()}, {"write", live.sync_converted[4].load()}};
    j["sync_exec"] = {{"eop", live.sync_exec[1].load()}, {"release", live.sync_exec[2].load()}, {"wait", live.sync_exec[3].load()}, {"write", live.sync_exec[4].load()}};
    j["direct_built"] = live.direct_built.load();
    j["direct_pure"] = live.direct_pure.load();
    j["direct_mixed"] = live.direct_mixed.load();
    j["direct_entries"] = live.direct_entries.load();
    j["direct_matched"] = live.direct_matched.load();
    j["direct_entry_mismatch"] = live.direct_entry_mismatch.load();
    j["direct_purity_mismatch"] = live.direct_purity_mismatch.load();
    j["direct_incomplete"] = live.direct_incomplete.load();
    j["direct_executed"] = live.direct_executed.load();
    j["direct_executed_asc"] = live.direct_executed_asc.load();
    j["shadow_anchors"] = live.shadow_anchors.load();
    j["shadow_spans"] = live.shadow_spans.load();
    j["shadow_pure"] = live.shadow_pure.load();
    j["shadow_mixed"] = live.shadow_mixed.load();
    j["shadow_matched"] = live.shadow_matched.load();
    j["shadow_extra"] = live.shadow_extra.load();
    j["shadow_missing"] = live.shadow_missing.load();
    j["shadow_nested"] = live.shadow_nested.load();
    j["shadow_stale"] = live.shadow_stale.load();
    {
        nlohmann::json kinds = nlohmann::json::object();
        for (u32 kind = 0; kind < 64; ++kind) {
            if (const u64 n = Core::NativeDraw::ShadowExtraByKind(kind); n != 0) {
                kinds[std::to_string(kind)] = n;
            }
        }
        j["shadow_extra_by_kind"] = std::move(kinds);
    }
    {
        nlohmann::json other = nlohmann::json::object();
        for (u32 opcode = 0; opcode < 256; ++opcode) {
            if (const u64 n = Core::NativeDraw::ShadowNonEntryCount(opcode); n != 0) {
                other[fmt::format("{:#x}", opcode)] = n;
            }
        }
        j["shadow_non_entry"] = std::move(other);
    }
    j["fixed_native"] = live.fixed_native.load();
    j["fixed_checked"] = live.fixed_checked.load();
    j["fixed_diff"] = live.fixed_diff.load();
    j["track_native"] = live.track_native.load();
    j["track_fce"] = live.track_fce.load();
    j["track_game"] = live.track_game_second.load() + live.track_game_depth.load();
    auto& packets = j["packets"] = nlohmann::json::array();
    for (const auto& p : live.packets) {
        packets.push_back(p.load());
    }
    j["asc_packets"] = live.asc_packets.load();
    return j;
}

nlohmann::json Threads(const Params&) {
    static VideoCore::ThreadUsageSampler sampler;
    static std::mutex mutex;
    std::scoped_lock lk{mutex};
    double total = 0.0;
    const auto usage = sampler.Sample(&total);
    nlohmann::json j;
    j["note"] = "percent of one core since the previous call of this endpoint";
    j["total_percent"] = total;
    auto& list = j["threads"] = nlohmann::json::array();
    for (const auto& entry : usage) {
        list.push_back({{"name", entry.name}, {"percent", entry.percent}});
    }
    return j;
}

nlohmann::json Memory(const Params& params) {
    const u64 address = ParseNumber(params, "addr", 0);
    const u64 size = ParseNumber(params, "size", 64);
    if (size == 0 || size > 0x10000) {
        return {{"error", "size must be 1..65536"}};
    }
    auto* memory = Core::Memory::Instance();
    if (address == 0 || !memory->IsValidMapping(address, size)) {
        return {{"error", "address range is not mapped"}, {"addr", address}, {"size", size}};
    }
    std::vector<u8> data(size);
    memory->CopySparseMemory(address, data.data(), size);
    static constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(size * 2);
    for (const u8 byte : data) {
        hex.push_back(digits[byte >> 4]);
        hex.push_back(digits[byte & 15]);
    }
    nlohmann::json j;
    j["addr"] = fmt::format("{:#x}", address);
    j["size"] = size;
    j["hex"] = hex;
    if (size % 4 == 0 && size <= 256) {
        auto& words = j["u32"] = nlohmann::json::array();
        for (u64 i = 0; i < size; i += 4) {
            u32 word;
            std::memcpy(&word, data.data() + i, 4);
            words.push_back(fmt::format("{:#010x}", word));
        }
    }
    return j;
}

nlohmann::json Log(const Params& params) {
    const u64 lines = std::min<u64>(ParseNumber(params, "lines", 50), 2000);
    std::string name = "shad_log.txt";
    if (const auto it = params.find("file"); it != params.end()) {
        name = it->second;
    }
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find("..") != std::string::npos) {
        return {{"error", "bad file name"}};
    }
    std::string filter;
    if (const auto it = params.find("filter"); it != params.end()) {
        filter = it->second;
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {{"error", "cannot open " + path.string()}};
    }
    in.seekg(0, std::ios::end);
    const auto end = static_cast<s64>(in.tellg());
    const s64 chunk = std::min<s64>(end, 1 << 20);
    in.seekg(end - chunk);
    std::string text(static_cast<size_t>(chunk), '\0');
    in.read(text.data(), chunk);
    std::vector<std::string> result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) {
            eol = text.size();
        }
        std::string line = text.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (filter.empty() || line.find(filter) != std::string::npos) {
            result.push_back(std::move(line));
        }
        pos = eol + 1;
    }
    if (result.size() > lines) {
        result.erase(result.begin(), result.end() - static_cast<s64>(lines));
    }
    return {{"file", name}, {"lines", result}};
}

nlohmann::json Faults(const Params& params) {
    const u64 top = std::min<u64>(ParseNumber(params, "top", 15), 100);
    std::vector<std::pair<u32, u32>> counts; // {count, region}
    u64 total = 0;
    for (u32 i = 0; i < FaultRegions; ++i) {
        const u32 count = fault_regions[i].exchange(0, std::memory_order_relaxed);
        if (count != 0) {
            counts.emplace_back(count, i);
            total += count;
        }
    }
    std::ranges::sort(counts, std::greater<>{});
    nlohmann::json j;
    j["note"] = "CPU write faults on tracked pages since the previous call, by 4 MB region";
    j["total"] = total;
    auto& list = j["top_regions"] = nlohmann::json::array();
    for (size_t i = 0; i < counts.size() && i < top; ++i) {
        list.push_back({{"addr", fmt::format("{:#x}", static_cast<u64>(counts[i].second) << 22)},
                        {"faults", counts[i].first}});
    }
    return j;
}

nlohmann::json Profile(const Params& params) {
    std::string thread = "GpuRaster";
    if (const auto it = params.find("thread"); it != params.end() && !it->second.empty()) {
        thread = it->second;
    }
    const double seconds = static_cast<double>(ParseNumber(params, "seconds", 5));
    const u32 hz = static_cast<u32>(ParseNumber(params, "hz", 500));
    const u32 top = static_cast<u32>(std::min<u64>(ParseNumber(params, "top", 25), 200));
    static std::mutex profile_mutex; // one profile at a time
    std::scoped_lock lk{profile_mutex};
    return VideoCore::ProfileThread(thread, seconds, hz, top);
}

nlohmann::json Policy(const Params&) {
    return VideoCore::ReadbackPolicy::Dump();
}

nlohmann::json SetPolicy(const Params& params) {
    nlohmann::json j;
    for (const auto& [name, value] : params) {
        const bool ok = VideoCore::ReadbackPolicy::Set(name, static_cast<s64>(std::stoll(value, nullptr, 0)));
        j[name] = ok ? "ok" : "unknown name";
    }
    j["policy"] = VideoCore::ReadbackPolicy::Dump();
    return j;
}

std::mutex periodic_mutex;
std::map<std::string, std::vector<std::pair<double, double>>> periodic; // name -> {at_ms, cost_ms}

nlohmann::json Frames(const Params& params) {
    const u64 want = std::min<u64>(ParseNumber(params, "n", 600), TimeRing - 1);
    const u64 count = presents.load(std::memory_order_acquire);
    const u64 available = std::min<u64>(count, TimeRing - 1);
    const u64 n = std::min(want, available);
    nlohmann::json j;
    j["now_ms"] = UptimeMs();
    auto& frames = j["frames"] = nlohmann::json::array();
    for (u64 i = n; i >= 1; --i) {
        const u64 t = present_times[(count - i) & (TimeRing - 1)].load(std::memory_order_relaxed);
        const u64 previous =
            present_times[(count - i - 1) & (TimeRing - 1)].load(std::memory_order_relaxed);
        if (t == 0 || previous == 0) {
            continue;
        }
        frames.push_back({t / 1e6, (t - previous) / 1e6});
    }
    return j;
}

nlohmann::json Periodic(const Params&) {
    std::scoped_lock lk{periodic_mutex};
    nlohmann::json j;
    j["now_ms"] = UptimeMs();
    for (const auto& [name, runs] : periodic) {
        auto& array = j[name] = nlohmann::json::array();
        for (const auto& [at, cost] : runs) {
            array.push_back({{"at_ms", at}, {"cost_ms", cost}});
        }
    }
    return j;
}

nlohmann::json Stats(const Params&) {
    std::scoped_lock lk{published_mutex};
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [group, value] : published) {
        j[group] = value;
    }
    return j;
}

nlohmann::json Index(const Params&) {
    nlohmann::json j;
    std::scoped_lock lk{registry_mutex};
    for (const auto& [path, entry] : registry) {
        j[path] = entry.description;
    }
    return j;
}

void RegisterBuiltins() {
    Register("/", "list of endpoints", Index);
    Register("/status", "uptime, presented frames, fps", Status);
    Register("/native", "native renderer counters (cumulative)", Native);
    Register("/capture",
             "?frames=24&x0=0&y0=0&x1=1&y1=1: save the guest output of the next frames (every game flip once), cropped (fractions), to log/frame_sequence/f<index>_p<present>.png; "
             "without frames: how many are left",
             [](const Params& params) {
                 if (params.find("frames") != params.end()) {
                     const auto number = [&](const char* key, float fallback) {
                         const auto it = params.find(key);
                         return it == params.end() || it->second.empty() ? fallback : std::stof(it->second);
                     };
                     VideoCore::RequestFrameSequence(static_cast<u32>(std::min<u64>(ParseNumber(params, "frames", 24), 600)), number("x0", 0.0f),
                                                     number("y0", 0.0f), number("x1", 1.0f), number("y1", 1.0f));
                 }
                 return nlohmann::json{{"remaining", VideoCore::FrameSequenceRemaining()}};
             });
    Register("/audit", "tracker audit: how often the buffer cache called a page clean that the CPU had changed (native_sdk_strip bit 67108864)",
             [](const Params&) { return Core::NativeAudit::Report(); });
    Register("/readbacks", "explicit readbacks, step 1: destinations of the engine's copy / counter read-back records, AsyncMap ranges, CPU read faults",
             [](const Params&) { return Core::NativeReadbacks::Report(); });
    Register("/entrydump",
             "?frames=1: dump the next frames of the command stream as entries to log/entry_dump.txt (without frames: the state of the last request)",
             EntryDumpEndpoint);
    Register("/resources", "?start=1 turns the resource census on; the engine's GPU memory blocks and who uses, uploads and writes them",
             [](const Params& params) {
                 if (ParseNumber(params, "start", 0) != 0) {
                     Core::NativeResources::Start();
                 }
                 return Core::NativeResources::Report();
             });
    Register("/threads", "CPU use of the busiest threads since the previous call", Threads);
    Register("/memory", "?addr=0x..&size=N (max 65536): guest memory as hex", Memory);
    Register("/log", "?lines=N&filter=text&file=shad_log.txt: tail of a log file", Log);
    Register("/profile",
             "?thread=GpuRaster&seconds=5&hz=500&top=25: stack sampling of one thread (functions as "
             "offsets, resolve with the linker map)",
             Profile);
    Register("/policy", "readback policy switches (video_core/readback_policy.h)", Policy);
    Register("/set", "?name=value changes readback policy switches (development only)",
             SetPolicy);
    Register("/frames", "?n=600: the last n frame intervals (time stamp and length in ms)", Frames);
    Register("/periodic", "last runs of periodic jobs (when they ended, how long they took)",
             Periodic);
    Register("/stats", "figures published by the subsystems (latest interval)", Stats);
    Register("/faults", "CPU write faults on tracked pages by 4 MB region since the previous call",
             Faults);
}

} // namespace

void Register(const std::string& path, const std::string& description, Handler handler) {
    std::scoped_lock lk{registry_mutex};
    registry[path] = Entry{description, std::move(handler)};
}

void Unregister(const std::string& path) {
    std::scoped_lock lk{registry_mutex};
    registry.erase(path);
}

u64 SincePresentNs() {
    const u64 count = presents.load(std::memory_order_acquire);
    if (count == 0) {
        return 0;
    }
    const u64 last = present_times[(count - 1) & (TimeRing - 1)].load(std::memory_order_relaxed);
    const u64 now = NowNs();
    return now > last ? now - last : 0;
}

u64 PresentCount() {
    return presents.load(std::memory_order_acquire);
}

double UptimeMs() {
    return static_cast<double>(NowNs()) / 1e6;
}

void RecordPeriodic(const std::string& name, double cost_ms) {
    const double at = UptimeMs();
    std::scoped_lock lk{periodic_mutex};
    auto& runs = periodic[name];
    runs.emplace_back(at, cost_ms);
    if (runs.size() > 24) {
        runs.erase(runs.begin());
    }
}

void Publish(const std::string& group, nlohmann::json value) {
    std::scoped_lock lk{published_mutex};
    published[group] = std::move(value);
}

void NotePresent() {
    const u64 index = presents.fetch_add(1, std::memory_order_acq_rel);
    present_times[index & (TimeRing - 1)].store(NowNs(), std::memory_order_relaxed);
}

void NoteWriteFault(VAddr address) {
    fault_regions[(address >> 22) & (FaultRegions - 1)].fetch_add(1, std::memory_order_relaxed);
}

void Start(u16 port) {
    std::scoped_lock lk{lifecycle_mutex};
    if (port == 0 || server) {
        return;
    }
    RegisterBuiltins();
    server = std::make_unique<httplib::Server>();
    server->Get(R"(/.*)", [](const httplib::Request& req, httplib::Response& res) {
        Handler handler;
        {
            std::scoped_lock lk{registry_mutex};
            auto it = registry.find(req.path);
            if (it != registry.end()) {
                handler = it->second.handler;
            }
        }
        if (!handler) {
            res.status = 404;
            res.set_content(Index({}).dump(2), "application/json");
            return;
        }
        Params params;
        for (const auto& [key, value] : req.params) {
            params[key] = value;
        }
        try {
            res.set_content(handler(params).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    if (!server->bind_to_port("127.0.0.1", port)) {
        LOG_ERROR(Core, "debug server: cannot listen on 127.0.0.1:{}", port);
        server.reset();
        return;
    }
    server_thread = std::jthread{[] { server->listen_after_bind(); }};
    LOG_INFO(Core, "debug server listening on 127.0.0.1:{}", port);
}

void Stop() {
    std::scoped_lock lk{lifecycle_mutex};
    if (!server) {
        return;
    }
    server->stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }
    server.reset();
}

} // namespace Core::DebugServer
