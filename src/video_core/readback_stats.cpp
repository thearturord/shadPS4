// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/string_util.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "video_core/readback_stats.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#endif

namespace VideoCore::ReadbackStats {

namespace Detail {
std::atomic<bool> enabled{false};
std::atomic<u32> cp_packet{};
std::atomic<const char*> cp_phase{};
thread_local bool is_cp_thread{};
thread_local u32 cp_current{};
thread_local u64 cp_since{};
thread_local u64 cp_packets{};
thread_local u32 cp_yield_kind{};
thread_local CpTimeArray cp_local_ns{};
thread_local CpEventCounters cp_events{};
std::atomic<u64> cp_time_ns[static_cast<u32>(CpTime::Count)]{};
} // namespace Detail

namespace {

constexpr u32 WriterUnknown = 0xFF;

struct Counters {
    std::atomic<u64> presents{};
    std::atomic<u64> submits{};
    std::atomic<u64> draws{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> read_faults{};
    std::atomic<u64> write_faults{};
    std::atomic<u64> fault_ns{};
    std::atomic<u64> protects{};
    std::atomic<u64> protect_ns{};
    std::atomic<u64> finishes{};
    std::atomic<u64> finish_ns{};
    std::atomic<u64> gpu_idle_ns{};
    std::atomic<u64> buf_readbacks{};
    std::atomic<u64> buf_readbacks_from_read{};
    std::atomic<u64> buf_readbacks_from_write{};
    std::atomic<u64> buf_readbacks_empty{};
    std::atomic<u64> buf_bytes{};
    std::atomic<u64> buf_finish_ns{};
    std::atomic<u64> buf_blocked_ns{};
    std::atomic<u64> img_readbacks{};
    std::atomic<u64> img_bytes{};
    std::atomic<u64> img_finish_ns{};
    std::atomic<u64> batches{};
    std::atomic<u64> batch_buffers{};
    std::atomic<u64> batch_images{};
    std::atomic<u64> batch_bytes{};
    std::atomic<u64> batch_wait_ns{};
    std::atomic<u64> regmem_waits{};
    std::atomic<u64> regmem_wait_ns{};
    std::atomic<u64> regmem_shortcuts{};
    // Indexed by [is_compute][WaitClass].
    std::array<std::array<std::atomic<u64>, u32(WaitClass::Count)>, 2> wait_count{};
    std::array<std::array<std::atomic<u64>, u32(WaitClass::Count)>, 2> wait_ns{};
    std::atomic<u64> sem_waits{};
    std::atomic<u64> sem_wait_ns{};
    std::atomic<u64> stream_barriers{};
    std::atomic<u64> cp_spin_ns{};
    std::atomic<u64> cp_flip_spin_ns{};
    std::atomic<u64> gpu_busy_ns{};
    std::atomic<u64> present_cpu_ns{};
    std::atomic<u64> presents_timed{};
    std::atomic<u64> stream_copies{};
    std::atomic<u64> upload_bytes{};
    std::atomic<u64> hot_pages_new{};
    std::atomic<u64> hot_decays{};
    std::atomic<u64> buffers_created{};
    std::atomic<u64> buffers_merged{};
    std::atomic<u64> merge_bytes{};
    std::atomic<u64> render_passes{};
    std::atomic<u64> cmds_since_submit{};
    std::array<std::atomic<u64>, 6> submit_hist{};
    std::array<std::atomic<u64>, static_cast<size_t>(Hazard::Count)> hazards{};
    std::array<std::atomic<u64>, static_cast<size_t>(Reuse::Count)> reuse{};
    std::atomic<u64> prefetch_downloads{};
    std::atomic<u64> prefetch_bytes{};
    std::atomic<u64> prefetch_repeat_downloads{};
    std::atomic<u64> prefetch_repeat_bytes{};
    std::atomic<u64> hot_range_faults{};
    std::atomic<u64> replay_commands{};
    std::atomic<u64> replay_ns{};
    std::atomic<u64> vk_submits{};
    std::atomic<u64> async_fences{};
    std::atomic<u64> async_fence_submits{};
    std::array<std::atomic<u64>, NumFenceKinds> fence_kinds{};
    std::array<std::atomic<u64>, NumFenceKinds> fence_kind_submits{};
    std::atomic<u64> stream_copy_bytes{};
    std::atomic<u64> stream_waits{};
    std::atomic<u64> stream_wait_ns{};
};

struct Writer {
    VAddr end;
    u64 hash;
    u32 stage;
    u64 seq;
};

struct BufferStat {
    VAddr base{};
    u64 size{};
    u64 hash{};
    u32 stage{WriterUnknown};
    u64 count{};
    u64 from_read{};
    u64 from_write{};
    u64 bytes{};
    u64 finish_ns{};
    u64 blocked_ns{};
};

struct ShaderStat {
    u64 hash{};
    u32 stage{};
    u64 count{};
    u64 bytes{};
    u64 blocked_ns{};
};

struct WaitStat {
    VAddr address{};
    bool is_compute{};
    std::array<u64, u32(WaitClass::Count)> count{};
    std::array<u64, u32(WaitClass::Count)> ns{};
    u64 total_ns{};
};

struct ShaderCallStat {
    ShaderCall last{};
    u64 count{};
    u64 total_ns{};
    u64 total_work{};
    CpTimeArray category_ns{};
    CpEventCounters events{};
};

constexpr std::array<const char*, static_cast<size_t>(CpTime::Count)> CpTimeNames = {
    "other",         "draw itself",    "dispatch itself", "pipeline",
    "upload",        "fence",          "submit end",      "fault",
    "protect",       "gpu wait",       "guest cmd",       "flip sleep",
    "idle",          "prepare",        "compute checks",  "bind buffers",
    "bind textures", "render targets", "vertex/index",    "descriptors",
    "dynamic state", "begin pass",     "stream copy",     "buffer lookup",
    "gpu mark",      "tex find image", "tex view/layout", "tex sampler",
    "stats",         "vk cmd",         "vk submit",       "vk other",
    "stream memcpy", "fence images",   "fence prefetch",  "fence submit",
    "pipe stages",   "pipe map",       "pipe compile",    "rt prepare",
    "rt views",      "stage runtime",  "stage flatbuf",   "stage spec",
    "stage find",
};
static_assert(std::ranges::none_of(CpTimeNames, [](const char* name) { return name == nullptr; }),
              "every CpTime category needs a name");

struct BufferRegionStat {
    VAddr region{};
    u64 created{};
    u64 merged{};
    u64 merged_bytes{};
    u64 wanted_bytes{};
    VAddr min_wanted{~0ULL};
    VAddr max_wanted{};
    u64 max_size{};
    u64 stream_leaps{};
};

struct BufferCreation {
    u64 time_ms;
    VAddr wanted_begin;
    VAddr wanted_end;
    VAddr begin;
    VAddr end;
    u32 num_merged;
    u64 merged_bytes;
    bool stream_leap;
};

struct ThreadCpuStat {
    std::string name;
    u64 last_cpu_100ns{};
    double sum_pct{};
    double max_pct{};
    u32 samples{};
};

struct PrefetchStat {
    VAddr base{};
    u64 size{};
    u64 downloads{};
    u64 bytes{};
    u64 repeat_downloads{};
    u64 repeat_bytes{};
    u64 frames{};
    u64 last_frame{~0ULL};
    u64 faults{};
};

struct ImageStat {
    VAddr addr{};
    u32 width{};
    u32 height{};
    u32 num_bits{};
    u64 count{};
    u64 bytes{};
    u64 finish_ns{};
};

Counters counters;

std::mutex maps_mutex;
std::map<VAddr, Writer> writers;
u64 writer_seq{};
std::unordered_map<VAddr, BufferStat> buffer_stats;
std::unordered_map<u64, ShaderStat> shader_stats;
std::unordered_map<VAddr, ImageStat> image_stats;
std::unordered_map<VAddr, PrefetchStat> prefetch_stats;
std::atomic<u64> total_presents{};
std::unordered_map<u64, WaitStat> wait_stats;
std::unordered_map<u64, ShaderCallStat> shader_call_stats;
std::unordered_map<VAddr, BufferRegionStat> buffer_region_stats;
std::vector<BufferCreation> last_buffer_creations; // ring of the last 32
size_t next_buffer_creation{};
std::unordered_map<u32, ThreadCpuStat> thread_cpu_stats; // reporter thread only

std::mutex thread_mutex;
std::condition_variable_any stop_cv;
std::jthread reporter;
u64 start_ns{};

const char* ShaderCallKindName(ShaderCallKind kind) {
    switch (kind) {
    case ShaderCallKind::Draw:
        return "draw";
    case ShaderCallKind::DrawIndirect:
        return "draw-ind";
    case ShaderCallKind::Dispatch:
        return "dispatch";
    case ShaderCallKind::DispatchIndirect:
        return "disp-ind";
    case ShaderCallKind::DispatchHle:
        return "disp-hle";
    case ShaderCallKind::DispatchSkipped:
        return "disp-skip";
    }
    return "?";
}

const char* StageName(u32 stage) {
    switch (stage) {
    case 0:
        return "fs";
    case 1:
        return "tcs";
    case 2:
        return "tes";
    case 3:
        return "vs";
    case 4:
        return "gs";
    case 5:
        return "cs";
    case WriterFill:
        return "fill";
    case WriterCopy:
        return "copy";
    default:
        return "unknown";
    }
}

double Ms(u64 ns) {
    return static_cast<double>(ns) / 1'000'000.0;
}

double Mb(u64 bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

/// Finds the most recent writer containing the address. Caller holds maps_mutex.
const std::pair<const VAddr, Writer>* FindWriter(VAddr addr) {
    auto it = writers.upper_bound(addr);
    const std::pair<const VAddr, Writer>* best = nullptr;
    for (int i = 0; i < 64 && it != writers.begin(); ++i) {
        --it;
        if (addr < it->second.end && (!best || it->second.seq > best->second.seq)) {
            best = &*it;
        }
    }
    return best;
}

std::filesystem::path LogPath(const char* name) {
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir) / name;
}

void WriteTopReport(u64 elapsed_ns) {
    std::vector<BufferStat> buffers;
    std::vector<ShaderStat> shaders;
    std::vector<ImageStat> images;
    std::vector<PrefetchStat> prefetches;
    std::vector<WaitStat> waits;
    std::vector<ShaderCallStat> shader_calls;
    {
        std::scoped_lock lk{maps_mutex};
        shader_calls.reserve(shader_call_stats.size());
        for (const auto& [_, stat] : shader_call_stats) {
            shader_calls.push_back(stat);
        }
        for (const auto& [_, stat] : wait_stats) {
            waits.push_back(stat);
        }
        buffers.reserve(buffer_stats.size());
        for (const auto& [_, stat] : buffer_stats) {
            buffers.push_back(stat);
        }
        for (const auto& [_, stat] : shader_stats) {
            shaders.push_back(stat);
        }
        for (const auto& [_, stat] : image_stats) {
            images.push_back(stat);
        }
        for (const auto& [_, stat] : prefetch_stats) {
            prefetches.push_back(stat);
        }
    }
    std::ranges::sort(prefetches, [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
    std::ranges::sort(buffers,
                      [](const auto& a, const auto& b) { return a.blocked_ns > b.blocked_ns; });
    std::ranges::sort(shaders,
                      [](const auto& a, const auto& b) { return a.blocked_ns > b.blocked_ns; });
    std::ranges::sort(images,
                      [](const auto& a, const auto& b) { return a.finish_ns > b.finish_ns; });
    std::ranges::sort(waits, [](const auto& a, const auto& b) {
        if (a.total_ns != b.total_ns) {
            return a.total_ns > b.total_ns;
        }
        return a.count[0] > b.count[0];
    });

    std::string out;
    out += fmt::format("Readback report after {:.1f} s (sorted by time the game was blocked)\n\n",
                       static_cast<double>(elapsed_ns) / 1e9);

    out += "== Buffers ==\n";
    out += fmt::format("{:>18} {:>10} {:>18} {:>7} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}\n",
                       "base", "size", "writer_hash", "stage", "count", "on_read", "on_write", "MB",
                       "blocked_ms", "gpu_wait_ms");
    for (size_t i = 0; i < std::min<size_t>(buffers.size(), 40); ++i) {
        const auto& b = buffers[i];
        out += fmt::format("{:#18x} {:>10} {:#18x} {:>7} {:>8} {:>8} {:>8} {:>10.2f} {:>12.1f} "
                           "{:>12.1f}\n",
                           b.base, b.size, b.hash, StageName(b.stage), b.count, b.from_read,
                           b.from_write, Mb(b.bytes), Ms(b.blocked_ns), Ms(b.finish_ns));
    }

    out += "\n== Writer shaders ==\n";
    out += fmt::format("{:>18} {:>7} {:>8} {:>10} {:>12}\n", "hash", "stage", "count", "MB",
                       "blocked_ms");
    for (size_t i = 0; i < std::min<size_t>(shaders.size(), 40); ++i) {
        const auto& s = shaders[i];
        out += fmt::format("{:#18x} {:>7} {:>8} {:>10.2f} {:>12.1f}\n", s.hash, StageName(s.stage),
                           s.count, Mb(s.bytes), Ms(s.blocked_ns));
    }

    out += "\n== Linear images ==\n";
    out += fmt::format("{:>18} {:>6} {:>6} {:>5} {:>8} {:>10} {:>12}\n", "addr", "width", "height",
                       "bits", "count", "MB", "gpu_wait_ms");
    for (size_t i = 0; i < std::min<size_t>(images.size(), 40); ++i) {
        const auto& img = images[i];
        out += fmt::format("{:#18x} {:>6} {:>6} {:>5} {:>8} {:>10.2f} {:>12.1f}\n", img.addr,
                           img.width, img.height, img.num_bits, img.count, Mb(img.bytes),
                           Ms(img.finish_ns));
    }

    out += "\n== Prefetched buffer ranges (hot ranges downloaded at fences), by MB ==\n";
    out += "per_frame = downloads per frame the range was downloaded in; repeat = downloads of a\n"
           "range already downloaded earlier in the same frame; faults = CPU faults on the range\n";
    out += fmt::format("{:>18} {:>8} {:>10} {:>9} {:>10} {:>10} {:>11} {:>8}\n", "base", "size",
                       "downloads", "per_frame", "MB", "repeat_MB", "repeat_pct", "faults");
    for (size_t i = 0; i < std::min<size_t>(prefetches.size(), 40); ++i) {
        const auto& p = prefetches[i];
        out += fmt::format(
            "{:#18x} {:>8} {:>10} {:>9.1f} {:>10.2f} {:>10.2f} {:>10.1f}% {:>8}\n", p.base, p.size,
            p.downloads,
            p.frames ? static_cast<double>(p.downloads) / static_cast<double>(p.frames) : 0.0,
            Mb(p.bytes), Mb(p.repeat_bytes),
            p.bytes ? 100.0 * static_cast<double>(p.repeat_bytes) / static_cast<double>(p.bytes)
                    : 0.0,
            p.faults);
    }

    out += "\n== Command stream waits (WaitRegMem), by blocked time ==\n";
    out += "pass = passed on a recorded fence, producer = waited for another ring to record it,\n"
           "fence = waited for a held back fence to land, memory = value not written by a fence\n";
    out += fmt::format("{:>18} {:>7} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>11}\n", "address",
                       "queue", "pass", "producer", "fence", "memory", "producer_ms", "fence_ms",
                       "memory_ms");
    for (size_t i = 0; i < std::min<size_t>(waits.size(), 40); ++i) {
        const auto& w = waits[i];
        out += fmt::format("{:#18x} {:>7} {:>9} {:>9} {:>9} {:>9} {:>11.1f} {:>11.1f} {:>11.1f}\n",
                           w.address, w.is_compute ? "compute" : "gfx", w.count[0], w.count[1],
                           w.count[2], w.count[3], Ms(w.ns[1]), Ms(w.ns[2]), Ms(w.ns[3]));
    }

    std::ranges::sort(shader_calls,
                      [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
    const double seconds = std::max(1.0, static_cast<double>(elapsed_ns) / 1e9);
    u64 all_ns = 0;
    u64 all_calls = 0;
    for (const auto& s : shader_calls) {
        all_ns += s.total_ns;
        all_calls += s.count;
    }
    out += fmt::format("\n== Shaders by command processor time (whole run: {} shaders, {} calls, "
                       "{:.1f} ms/s) ==\n",
                       shader_calls.size(), all_calls, Ms(all_ns) / seconds);
    out += "time includes everything done for the call (bindings, uploads, faults); work = "
           "workgroups for dispatches, vertices x instances for draws (0 = indirect); "
           "bufs/imgs = bound (written/storage); rt = color targets (+d = depth)\n";
    out += fmt::format("{:>9} {:>18} {:>18} {:>9} {:>9} {:>8} {:>7} {:>11} {:>7} {:>8} {:>8} {:>5} "
                       "{:>6}\n",
                       "kind", "shader", "pixel_shader", "calls", "calls/s", "ms/s", "cum%",
                       "avg_work", "thr/grp", "bufs", "imgs", "rt", "avg_us");
    u64 cumulative_ns = 0;
    for (size_t i = 0; i < std::min<size_t>(shader_calls.size(), 60); ++i) {
        const auto& s = shader_calls[i];
        const auto& c = s.last;
        cumulative_ns += s.total_ns;
        out += fmt::format(
            "{:>9} {:#18x} {:#18x} {:>9} {:>9.1f} {:>8.2f} {:>6.1f}% {:>11} {:>7} {:>8} {:>8} "
            "{:>5} {:>6.1f}\n",
            ShaderCallKindName(c.kind), c.hash, c.ps_hash, s.count,
            static_cast<double>(s.count) / seconds, Ms(s.total_ns) / seconds,
            all_ns ? 100.0 * static_cast<double>(cumulative_ns) / static_cast<double>(all_ns) : 0.0,
            s.count ? s.total_work / s.count : 0, c.threads_per_group,
            fmt::format("{}({})", c.buffers, c.written_buffers),
            fmt::format("{}({})", c.images, c.storage_images),
            fmt::format("{}{}", c.color_targets, c.depth_target ? "+d" : ""),
            s.count ? static_cast<double>(s.total_ns) / static_cast<double>(s.count) / 1000.0
                    : 0.0);
    }

    std::vector<BufferRegionStat> regions;
    std::vector<BufferCreation> creations;
    {
        std::scoped_lock lk{maps_mutex};
        for (const auto& [_, stat] : buffer_region_stats) {
            regions.push_back(stat);
        }
        for (size_t i = 0; i < last_buffer_creations.size(); ++i) {
            creations.push_back(
                last_buffer_creations[(next_buffer_creation + i) % last_buffer_creations.size()]);
        }
    }
    std::ranges::sort(regions, [](const auto& a, const auto& b) { return a.created > b.created; });
    out += "\n== Buffer creations by 16 MB region (whole run) ==\n";
    out += fmt::format("{:>14} {:>9} {:>9} {:>10} {:>10} {:>14} {:>14} {:>10} {:>6}\n", "region",
                       "created", "merged", "merged_MB", "wanted_MB", "min_wanted", "max_wanted",
                       "max_KB", "leaps");
    for (size_t i = 0; i < std::min<size_t>(regions.size(), 20); ++i) {
        const auto& r = regions[i];
        out +=
            fmt::format("{:#14x} {:>9} {:>9} {:>10.1f} {:>10.1f} {:#14x} {:#14x} {:>10} {:>6}\n",
                        r.region << 24, r.created, r.merged, Mb(r.merged_bytes), Mb(r.wanted_bytes),
                        r.min_wanted, r.max_wanted, r.max_size / 1024, r.stream_leaps);
    }
    out += "\n== Last buffer creations ==\n";
    out += fmt::format("{:>9} {:>14} {:>9} {:>14} {:>10} {:>7} {:>10} {:>5}\n", "time_ms", "wanted",
                       "wanted_KB", "created", "size_KB", "merged", "merged_KB", "leap");
    for (const auto& c : creations) {
        out += fmt::format("{:>9} {:#14x} {:>9} {:#14x} {:>10} {:>7} {:>10} {:>5}\n", c.time_ms,
                           c.wanted_begin, (c.wanted_end - c.wanted_begin) / 1024, c.begin,
                           (c.end - c.begin) / 1024, c.num_merged, c.merged_bytes / 1024,
                           c.stream_leap ? "yes" : "");
    }

    std::vector<ThreadCpuStat> threads;
    for (const auto& [_, stat] : thread_cpu_stats) {
        if (stat.samples > 0) {
            threads.push_back(stat);
        }
    }
    std::ranges::sort(threads, [](const auto& a, const auto& b) {
        return a.sum_pct / a.samples > b.sum_pct / b.samples;
    });
    out += "\n== Threads by CPU use, seconds with more than 20,000 draws (100% = one core) ==\n";
    out += fmt::format("{:>40} {:>8} {:>8} {:>8}\n", "thread", "avg_%", "max_%", "seconds");
    for (size_t i = 0; i < std::min<size_t>(threads.size(), 30); ++i) {
        const auto& t = threads[i];
        out += fmt::format("{:>40} {:>8.1f} {:>8.1f} {:>8}\n", t.name.substr(0, 40),
                           t.sum_pct / t.samples, t.max_pct, t.samples);
    }

    out += "\n== Where the time of the top 25 shaders goes (per call) ==\n";
    for (size_t i = 0; i < std::min<size_t>(shader_calls.size(), 25); ++i) {
        const auto& s = shader_calls[i];
        if (s.count == 0) {
            continue;
        }
        const double calls = static_cast<double>(s.count);
        const auto per_call_us = [&](u64 ns) { return static_cast<double>(ns) / calls / 1000.0; };
        std::array<u32, static_cast<size_t>(CpTime::Count)> order{};
        for (u32 c = 0; c < order.size(); ++c) {
            order[c] = c;
        }
        std::ranges::sort(order, [&](u32 a, u32 b) { return s.category_ns[a] > s.category_ns[b]; });
        std::string top;
        for (u32 k = 0; k < 6; ++k) {
            const u32 c = order[k];
            if (s.category_ns[c] == 0) {
                break;
            }
            top += fmt::format("{}{} {:.1f}", k ? ", " : "", CpTimeNames[c],
                               per_call_us(s.category_ns[c]));
        }
        const auto& e = s.events;
        const auto kb = [&](u64 bytes) { return static_cast<double>(bytes) / calls / 1024.0; };
        out +=
            fmt::format("{:>9} {:#x}/{:#x}: {:.1f} us/call = {}\n", ShaderCallKindName(s.last.kind),
                        s.last.hash, s.last.ps_hash, per_call_us(s.total_ns), top);
        out += fmt::format(
            "          per call: upload {:.1f} KB, image uploads {:.2f}, protects {:.1f}, faults "
            "{:.2f}, buffers created {:.3f}, GPU-marked {:.1f} KB, stream copy {:.1f} KB, "
            "readback {:.1f} KB\n",
            kb(e.upload_bytes), static_cast<double>(e.image_uploads) / calls,
            static_cast<double>(e.protect_calls) / calls, static_cast<double>(e.faults) / calls,
            static_cast<double>(e.buffers_created) / calls, kb(e.gpu_marked_bytes),
            kb(e.stream_copy_bytes), kb(e.readback_bytes));
    }

    std::ofstream file{LogPath("readback_top.txt"), std::ios::trunc};
    file << out;
}

/// CPU use of every thread of the process during the last interval (Windows only).
void SampleThreads(u64 interval_ns, bool heavy_second) {
#ifdef _WIN32
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    const DWORD pid = GetCurrentProcessId();
    THREADENTRY32 entry{.dwSize = sizeof(THREADENTRY32)};
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid) {
            continue;
        }
        const HANDLE thread =
            OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
        if (!thread) {
            continue;
        }
        FILETIME creation, exit, kernel, user;
        if (GetThreadTimes(thread, &creation, &exit, &kernel, &user)) {
            const auto to_u64 = [](const FILETIME& t) {
                return (u64{t.dwHighDateTime} << 32) | t.dwLowDateTime;
            };
            const u64 cpu = to_u64(kernel) + to_u64(user);
            auto [it, inserted] = thread_cpu_stats.try_emplace(entry.th32ThreadID);
            auto& stat = it->second;
            if (inserted || stat.name.empty()) {
                PWSTR name = nullptr;
                if (SUCCEEDED(GetThreadDescription(thread, &name)) && name) {
                    stat.name = Common::UTF16ToUTF8(name);
                    LocalFree(name);
                }
                if (stat.name.empty()) {
                    stat.name = fmt::format("thread {}", entry.th32ThreadID);
                }
            }
            if (!inserted && heavy_second && interval_ns > 0 && cpu >= stat.last_cpu_100ns) {
                const double pct = 100.0 * static_cast<double>(cpu - stat.last_cpu_100ns) * 100.0 /
                                   static_cast<double>(interval_ns);
                stat.sum_pct += pct;
                stat.max_pct = std::max(stat.max_pct, pct);
                ++stat.samples;
            }
            stat.last_cpu_100ns = cpu;
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
#endif
}

/// The report is diagnostics only: a failure is logged instead of taking the emulator down.
void SafeWriteTopReport(u64 elapsed_ns) {
    try {
        WriteTopReport(elapsed_ns);
    } catch (const std::exception& e) {
        LOG_ERROR(Render, "Readback stats: writing readback_top.txt failed: {}", e.what());
    }
}

void ReporterThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:ReadbackStats");

    std::ofstream csv{LogPath("readback_stats.csv"), std::ios::trunc};
    csv << "time_s,presents,submits,draws,dispatches,read_faults,write_faults,fault_ms,"
           "protects,protect_ms,buf_readbacks,buf_rb_on_read,buf_rb_on_write,buf_rb_empty,"
           "buf_MB,buf_gpu_wait_ms,game_blocked_ms,img_readbacks,img_MB,img_gpu_wait_ms,"
           "finish_calls,finish_ms,gpu_thread_busy_pct,batches,batch_buffers,batch_images,"
           "batch_MB,batch_wait_ms,regmem_waits,regmem_wait_ms,regmem_shortcuts,stream_waits,"
           "stream_wait_ms,gfx_pass,gfx_producer,gfx_producer_ms,gfx_fence,gfx_fence_ms,"
           "gfx_memory,gfx_memory_ms,cmp_pass,cmp_producer,cmp_producer_ms,cmp_fence,"
           "cmp_fence_ms,cmp_memory,cmp_memory_ms,sem_waits,sem_wait_ms,stream_barriers,"
           "cp_other_ms,cp_draw_ms,cp_dispatch_ms,cp_pipeline_ms,cp_upload_ms,cp_fence_ms,"
           "cp_submit_end_ms,cp_fault_ms,cp_protect_ms,cp_gpu_wait_ms,cp_guest_cmd_ms,"
           "cp_flip_sleep_ms,cp_idle_ms,rec_prepare_ms,rec_compute_checks_ms,"
           "rec_bind_buffers_ms,rec_bind_textures_ms,rec_render_targets_ms,rec_vertex_index_ms,"
           "rec_descriptors_ms,rec_dynamic_state_ms,rec_begin_pass_ms,buf_stream_copy_ms,"
           "buf_lookup_ms,buf_gpu_mark_ms,tex_find_image_ms,tex_view_layout_ms,tex_sampler_ms,"
           "stats_overhead_ms,vk_cmd_ms,vk_submit_ms,vk_other_ms,stream_memcpy_ms,"
           "fence_image_copies_ms,fence_prefetch_ms,fence_submit_ms,pipeline_stages_ms,"
           "pipeline_map_ms,pipeline_compile_ms,rt_prepare_ms,rt_views_ms,stage_runtime_ms,"
           "stage_flatbuf_ms,stage_spec_ms,stage_find_ms,"
           "cp_wait_spin_ms,cp_flip_spin_ms,gpu_busy_ms,present_cpu_ms,"
           "stream_copies,stream_copy_MB,vk_submits,async_fences,async_fence_submits,"
           "upload_MB,hot_pages_new,hot_decays,buffers_created,buffers_merged,merge_MB,"
           "render_passes,submit_cmds_0,submit_cmds_1_7,submit_cmds_8_31,submit_cmds_32_127,"
           "submit_cmds_128_511,submit_cmds_512p,cp_guest_writes,cp_write_overlaps,"
           "gpu_mark_overlaps,draws_checked,draw_pipeline_same,draw_ctx_regs_same,"
           "draw_rt_regs_same,draw_user_data_same,draw_all_same,dispatches_checked,"
           "dispatch_pipeline_same,dispatch_user_data_same,stream_rep_tick_same,"
           "stream_rep_tick_changed,stream_rep_old_same,stream_rep_old_changed,"
           "stream_rep_same_MB,stream_reused,stream_reuse_resets,rt_reused,stage_lookups,"
           "stage_variant_compares,stage_same_as_last,prefetch_downloads,"
           "prefetch_MB,prefetch_repeat_downloads,prefetch_repeat_MB,hot_range_faults,"
           "cmdlist_commands,cmdlist_replay_ms,fences_eop,fences_eos,fences_release,"
           "fences_held_write,fences_submit_end,fence_submits_eop,fence_submits_eos,"
           "fence_submits_release,fence_submits_held_write,fence_submits_submit_end\n";
    csv.flush();

    u64 last_ns = NowNs();
    u32 ticks = 0;
    u32 stalled_seconds = 0;
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{thread_mutex};
            stop_cv.wait_for(lk, stoken, std::chrono::seconds{1}, [] { return false; });
        }
        const u64 now = NowNs();
        const u64 interval = now - last_ns;
        last_ns = now;

        const auto take = [](std::atomic<u64>& v) { return v.exchange(0); };
        const u64 idle = take(counters.gpu_idle_ns);
        const double busy =
            interval ? 100.0 * (1.0 - std::min(1.0, static_cast<double>(idle) / interval)) : 0.0;

        SampleThreads(interval, counters.draws.load() >= 20000);

        // Stall watchdog: the command processor is busy but records nothing.
        const bool no_work = counters.submits.load() == 0 && counters.draws.load() == 0 &&
                             counters.dispatches.load() == 0 && counters.presents.load() == 0;
        if (no_work && busy > 95.0) {
            if (++stalled_seconds == 3) {
                const u32 packet = Detail::cp_packet.load(std::memory_order_relaxed);
                const char* phase = Detail::cp_phase.load(std::memory_order_relaxed);
                LOG_WARNING(Render,
                            "Hang check: command processor busy for 3s without recording work. "
                            "Last packet: queue {} opcode {:#x}, phase: {}",
                            packet >> 16, packet & 0xFFFF, phase ? phase : "packet processing");
            }
        } else {
            stalled_seconds = 0;
        }
        // Command processor time. Spinning over waiting queues happens in the "other" category,
        // so it is reported separately and taken out of it.
        std::string cp_columns;
        {
            std::array<u64, static_cast<u32>(CpTime::Count)> cp{};
            for (u32 i = 0; i < cp.size(); ++i) {
                cp[i] = Detail::cp_time_ns[i].exchange(0, std::memory_order_relaxed);
            }
            const u64 spin = take(counters.cp_spin_ns);
            const u64 flip_spin = take(counters.cp_flip_spin_ns);
            cp[0] -= std::min(cp[0], spin + flip_spin);
            for (const u64 ns : cp) {
                cp_columns += fmt::format(",{:.2f}", Ms(ns));
            }
            const u64 presents_timed = take(counters.presents_timed);
            cp_columns += fmt::format(",{:.2f},{:.2f},{:.2f},{:.2f}", Ms(spin), Ms(flip_spin),
                                      Ms(take(counters.gpu_busy_ns)),
                                      presents_timed ? Ms(take(counters.present_cpu_ns)) /
                                                           static_cast<double>(presents_timed)
                                                     : 0.0);
            cp_columns +=
                fmt::format(",{},{:.2f},{},{},{}", take(counters.stream_copies),
                            Mb(take(counters.stream_copy_bytes)), take(counters.vk_submits),
                            take(counters.async_fences), take(counters.async_fence_submits));
            cp_columns +=
                fmt::format(",{:.2f},{},{},{},{},{:.2f},{}", Mb(take(counters.upload_bytes)),
                            take(counters.hot_pages_new), take(counters.hot_decays),
                            take(counters.buffers_created), take(counters.buffers_merged),
                            Mb(take(counters.merge_bytes)), take(counters.render_passes));
            for (auto& bucket : counters.submit_hist) {
                cp_columns += fmt::format(",{}", take(bucket));
            }
            for (auto& hazard : counters.hazards) {
                cp_columns += fmt::format(",{}", take(hazard));
            }
            for (u32 i = 0; i < counters.reuse.size(); ++i) {
                const u64 value = take(counters.reuse[i]);
                cp_columns += i == static_cast<u32>(Reuse::StreamRepeatSameBytes)
                                  ? fmt::format(",{:.2f}", Mb(value))
                                  : fmt::format(",{}", value);
            }
            cp_columns += fmt::format(
                ",{},{:.2f},{},{:.2f},{}", take(counters.prefetch_downloads),
                Mb(take(counters.prefetch_bytes)), take(counters.prefetch_repeat_downloads),
                Mb(take(counters.prefetch_repeat_bytes)), take(counters.hot_range_faults));
            cp_columns += fmt::format(",{},{:.2f}", take(counters.replay_commands),
                                      Ms(take(counters.replay_ns)));
            for (auto& count : counters.fence_kinds) {
                cp_columns += fmt::format(",{}", take(count));
            }
            for (auto& count : counters.fence_kind_submits) {
                cp_columns += fmt::format(",{}", take(count));
            }
        }
        std::string wait_columns;
        for (u32 q = 0; q < 2; ++q) {
            for (u32 c = 0; c < u32(WaitClass::Count); ++c) {
                const u64 count = take(counters.wait_count[q][c]);
                const u64 ns = take(counters.wait_ns[q][c]);
                wait_columns +=
                    c == 0 ? fmt::format(",{}", count) : fmt::format(",{},{:.2f}", count, Ms(ns));
            }
        }

        csv << fmt::format(
            "{:.1f},{},{},{},{},{},{},{:.2f},{},{:.2f},{},{},{},{},{:.2f},{:.2f},{:.2f},{},{:.2f},"
            "{:.2f},{},{:.2f},{:.1f},{},{},{},{:.2f},{:.2f},{},{:.2f},{},{},{:.2f}{},{},{:.2f},"
            "{}{}\n",
            static_cast<double>(now - start_ns) / 1e9, take(counters.presents),
            take(counters.submits), take(counters.draws), take(counters.dispatches),
            take(counters.read_faults), take(counters.write_faults), Ms(take(counters.fault_ns)),
            take(counters.protects), Ms(take(counters.protect_ns)), take(counters.buf_readbacks),
            take(counters.buf_readbacks_from_read), take(counters.buf_readbacks_from_write),
            take(counters.buf_readbacks_empty), Mb(take(counters.buf_bytes)),
            Ms(take(counters.buf_finish_ns)), Ms(take(counters.buf_blocked_ns)),
            take(counters.img_readbacks), Mb(take(counters.img_bytes)),
            Ms(take(counters.img_finish_ns)), take(counters.finishes), Ms(take(counters.finish_ns)),
            busy, take(counters.batches), take(counters.batch_buffers), take(counters.batch_images),
            Mb(take(counters.batch_bytes)), Ms(take(counters.batch_wait_ns)),
            take(counters.regmem_waits), Ms(take(counters.regmem_wait_ns)),
            take(counters.regmem_shortcuts), take(counters.stream_waits),
            Ms(take(counters.stream_wait_ns)), wait_columns, take(counters.sem_waits),
            Ms(take(counters.sem_wait_ns)), take(counters.stream_barriers), cp_columns);
        csv.flush();

        if (++ticks % 5 == 0) {
            SafeWriteTopReport(now - start_ns);
        }
    }
    SafeWriteTopReport(NowNs() - start_ns);
}

} // Anonymous namespace

void Start() {
    if (!EmulatorSettings.IsReadbackStatsEnabled() || reporter.joinable()) {
        return;
    }
    start_ns = NowNs();
    Detail::enabled = true;
    reporter = std::jthread{ReporterThread};
    LOG_INFO(Render, "Readback stats enabled, writing readback_stats.csv and readback_top.txt");
}

void Stop() {
    if (!reporter.joinable()) {
        return;
    }
    Detail::enabled = false;
    reporter.request_stop();
    reporter.join();
}

u64 FrameIndex() {
    return total_presents.load(std::memory_order_relaxed);
}

void OnPrefetch(VAddr range_start, u64 range_size, u64 bytes, bool repeat) {
    counters.prefetch_downloads.fetch_add(1, std::memory_order_relaxed);
    counters.prefetch_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (repeat) {
        counters.prefetch_repeat_downloads.fetch_add(1, std::memory_order_relaxed);
        counters.prefetch_repeat_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    const u64 frame = FrameIndex();
    std::scoped_lock lk{maps_mutex};
    auto& stat = prefetch_stats[range_start];
    stat.base = range_start;
    stat.size = range_size;
    ++stat.downloads;
    stat.bytes += bytes;
    if (repeat) {
        ++stat.repeat_downloads;
        stat.repeat_bytes += bytes;
    }
    if (stat.last_frame != frame) {
        stat.last_frame = frame;
        ++stat.frames;
    }
}

void OnHotRangeFault(VAddr range_start, u64 range_size) {
    counters.hot_range_faults.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lk{maps_mutex};
    auto& stat = prefetch_stats[range_start];
    stat.base = range_start;
    stat.size = range_size;
    ++stat.faults;
}

void OnPresent() {
    total_presents.fetch_add(1, std::memory_order_relaxed);
    counters.presents.fetch_add(1, std::memory_order_relaxed);
}

void OnSubmit() {
    counters.submits.fetch_add(1, std::memory_order_relaxed);
}

void OnDraw() {
    counters.draws.fetch_add(1, std::memory_order_relaxed);
    counters.cmds_since_submit.fetch_add(1, std::memory_order_relaxed);
}

void OnDispatch() {
    counters.dispatches.fetch_add(1, std::memory_order_relaxed);
    counters.cmds_since_submit.fetch_add(1, std::memory_order_relaxed);
}

void OnFault(bool is_write, u64 handler_ns) {
    (is_write ? counters.write_faults : counters.read_faults)
        .fetch_add(1, std::memory_order_relaxed);
    counters.fault_ns.fetch_add(handler_ns, std::memory_order_relaxed);
}

void OnProtect(u64 ns) {
    counters.protects.fetch_add(1, std::memory_order_relaxed);
    counters.protect_ns.fetch_add(ns, std::memory_order_relaxed);
}

void OnFinish(u64 wait_ns) {
    counters.finishes.fetch_add(1, std::memory_order_relaxed);
    counters.finish_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

void OnGpuThreadIdle(u64 idle_ns) {
    counters.gpu_idle_ns.fetch_add(idle_ns, std::memory_order_relaxed);
}

void RecordWriter(VAddr base, u64 size, u64 shader_hash, u32 stage) {
    std::scoped_lock lk{maps_mutex};
    writers[base] = Writer{
        .end = base + size,
        .hash = shader_hash,
        .stage = stage,
        .seq = ++writer_seq,
    };
}

void OnBufferReadback(VAddr fault_addr, u64 bytes, u64 finish_ns, u64 blocked_ns,
                      bool write_triggered) {
    if (bytes == 0) {
        counters.buf_readbacks_empty.fetch_add(1, std::memory_order_relaxed);
        counters.buf_blocked_ns.fetch_add(blocked_ns, std::memory_order_relaxed);
        return;
    }
    counters.buf_readbacks.fetch_add(1, std::memory_order_relaxed);
    (write_triggered ? counters.buf_readbacks_from_write : counters.buf_readbacks_from_read)
        .fetch_add(1, std::memory_order_relaxed);
    counters.buf_bytes.fetch_add(bytes, std::memory_order_relaxed);
    counters.buf_finish_ns.fetch_add(finish_ns, std::memory_order_relaxed);
    counters.buf_blocked_ns.fetch_add(blocked_ns, std::memory_order_relaxed);

    std::scoped_lock lk{maps_mutex};
    VAddr base = fault_addr & ~VAddr{0xFFFF};
    u64 size = 0x10000;
    u64 hash = 0;
    u32 stage = WriterUnknown;
    if (const auto* writer = FindWriter(fault_addr)) {
        base = writer->first;
        size = writer->second.end - writer->first;
        hash = writer->second.hash;
        stage = writer->second.stage;
    }
    auto& stat = buffer_stats[base];
    stat.base = base;
    stat.size = std::max(stat.size, size);
    stat.hash = hash;
    stat.stage = stage;
    ++stat.count;
    stat.from_read += write_triggered ? 0 : 1;
    stat.from_write += write_triggered ? 1 : 0;
    stat.bytes += bytes;
    stat.finish_ns += finish_ns;
    stat.blocked_ns += blocked_ns;

    auto& shader = shader_stats[hash ^ (u64{stage} << 56)];
    shader.hash = hash;
    shader.stage = stage;
    ++shader.count;
    shader.bytes += bytes;
    shader.blocked_ns += blocked_ns;
}

void OnBatch(u32 num_buffers, u32 num_images, u64 bytes, u64 wait_ns) {
    counters.batches.fetch_add(1, std::memory_order_relaxed);
    counters.batch_buffers.fetch_add(num_buffers, std::memory_order_relaxed);
    counters.batch_images.fetch_add(num_images, std::memory_order_relaxed);
    counters.batch_bytes.fetch_add(bytes, std::memory_order_relaxed);
    counters.batch_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

void OnCommandWait(bool is_compute, VAddr address, WaitClass cls, u64 wait_ns) {
    const u32 q = is_compute ? 1 : 0;
    const u32 c = static_cast<u32>(cls);
    if (cls == WaitClass::FencePass) {
        counters.regmem_shortcuts.fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.regmem_waits.fetch_add(1, std::memory_order_relaxed);
        counters.regmem_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    }
    counters.wait_count[q][c].fetch_add(1, std::memory_order_relaxed);
    counters.wait_ns[q][c].fetch_add(wait_ns, std::memory_order_relaxed);

    std::scoped_lock lk{maps_mutex};
    auto& stat = wait_stats[address ^ (u64{q} << 63)];
    stat.address = address;
    stat.is_compute = is_compute;
    ++stat.count[c];
    stat.ns[c] += wait_ns;
    stat.total_ns += wait_ns;
}

void OnSemaphoreWait(u64 wait_ns) {
    counters.sem_waits.fetch_add(1, std::memory_order_relaxed);
    counters.sem_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

void OnStreamBarrier() {
    counters.stream_barriers.fetch_add(1, std::memory_order_relaxed);
}

void OnCpSpin(bool flip, u64 ns) {
    (flip ? counters.cp_flip_spin_ns : counters.cp_spin_ns)
        .fetch_add(ns, std::memory_order_relaxed);
}

void OnShaderCall(const ShaderCall& call) {
    const u64 key =
        call.hash ^ (call.ps_hash * 0x9E3779B97F4A7C15ULL) ^ (static_cast<u64>(call.kind) << 59);
    std::scoped_lock lk{maps_mutex};
    auto& stat = shader_call_stats[key];
    stat.last = call;
    ++stat.count;
    stat.total_ns += call.ns;
    stat.total_work += call.work;
    for (size_t i = 0; i < stat.category_ns.size(); ++i) {
        stat.category_ns[i] += call.category_ns[i];
    }
    stat.events += call.events;
}

void OnRasterizerSubmit() {
    const u64 commands = counters.cmds_since_submit.exchange(0, std::memory_order_relaxed);
    const size_t bucket = commands == 0    ? 0
                          : commands < 8   ? 1
                          : commands < 32  ? 2
                          : commands < 128 ? 3
                          : commands < 512 ? 4
                                           : 5;
    counters.submit_hist[bucket].fetch_add(1, std::memory_order_relaxed);
}

void OnRenderPassBegin() {
    counters.render_passes.fetch_add(1, std::memory_order_relaxed);
}

void OnUpload(u64 bytes) {
    counters.upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void OnHotPages(u32 count) {
    if (IsEnabled()) {
        counters.hot_pages_new.fetch_add(count, std::memory_order_relaxed);
    }
}

void OnHotPageDecay() {
    if (IsEnabled()) {
        counters.hot_decays.fetch_add(1, std::memory_order_relaxed);
    }
}

void OnBufferCreated(VAddr wanted_begin, VAddr wanted_end, VAddr begin, VAddr end, u32 num_merged,
                     u64 merged_bytes, bool stream_leap) {
    counters.buffers_created.fetch_add(1, std::memory_order_relaxed);
    counters.buffers_merged.fetch_add(num_merged, std::memory_order_relaxed);
    counters.merge_bytes.fetch_add(merged_bytes, std::memory_order_relaxed);
    std::scoped_lock lk{maps_mutex};
    auto& region = buffer_region_stats[wanted_begin >> 24];
    region.region = wanted_begin >> 24;
    ++region.created;
    region.merged += num_merged;
    region.merged_bytes += merged_bytes;
    region.wanted_bytes += wanted_end - wanted_begin;
    region.min_wanted = std::min(region.min_wanted, wanted_begin);
    region.max_wanted = std::max(region.max_wanted, wanted_begin);
    region.max_size = std::max<u64>(region.max_size, end - begin);
    region.stream_leaps += stream_leap ? 1 : 0;
    const BufferCreation creation{
        .time_ms = (NowNs() - start_ns) / 1'000'000,
        .wanted_begin = wanted_begin,
        .wanted_end = wanted_end,
        .begin = begin,
        .end = end,
        .num_merged = num_merged,
        .merged_bytes = merged_bytes,
        .stream_leap = stream_leap,
    };
    constexpr size_t MaxCreations = 32;
    if (last_buffer_creations.size() < MaxCreations) {
        last_buffer_creations.push_back(creation);
    } else {
        last_buffer_creations[next_buffer_creation] = creation;
        next_buffer_creation = (next_buffer_creation + 1) % MaxCreations;
    }
}

void OnHazard(Hazard hazard) {
    counters.hazards[static_cast<size_t>(hazard)].fetch_add(1, std::memory_order_relaxed);
}

void OnReuse(Reuse reuse, u64 amount) {
    counters.reuse[static_cast<size_t>(reuse)].fetch_add(amount, std::memory_order_relaxed);
}

void OnCommandReplay(u64 commands, u64 ns) {
    counters.replay_commands.fetch_add(commands, std::memory_order_relaxed);
    counters.replay_ns.fetch_add(ns, std::memory_order_relaxed);
}

void OnVkSubmit() {
    counters.vk_submits.fetch_add(1, std::memory_order_relaxed);
}

void OnAsyncFence(FenceKind kind, bool submitted) {
    const u32 index = static_cast<u32>(std::countr_zero(static_cast<u32>(kind)));
    counters.async_fences.fetch_add(1, std::memory_order_relaxed);
    if (index < NumFenceKinds) {
        counters.fence_kinds[index].fetch_add(1, std::memory_order_relaxed);
    }
    if (submitted) {
        counters.async_fence_submits.fetch_add(1, std::memory_order_relaxed);
        if (index < NumFenceKinds) {
            counters.fence_kind_submits[index].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void OnStreamCopy(u64 bytes) {
    counters.stream_copies.fetch_add(1, std::memory_order_relaxed);
    counters.stream_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void OnGpuBusy(u64 ns) {
    counters.gpu_busy_ns.fetch_add(ns, std::memory_order_relaxed);
}

void OnPresentCpu(u64 ns) {
    counters.present_cpu_ns.fetch_add(ns, std::memory_order_relaxed);
    counters.presents_timed.fetch_add(1, std::memory_order_relaxed);
}

void OnStreamWait(u64 wait_ns) {
    counters.stream_waits.fetch_add(1, std::memory_order_relaxed);
    counters.stream_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

void OnImageReadback(VAddr addr, u64 bytes, u32 width, u32 height, u32 num_bits, u64 finish_ns) {
    counters.img_readbacks.fetch_add(1, std::memory_order_relaxed);
    counters.img_bytes.fetch_add(bytes, std::memory_order_relaxed);
    counters.img_finish_ns.fetch_add(finish_ns, std::memory_order_relaxed);

    std::scoped_lock lk{maps_mutex};
    auto& stat = image_stats[addr];
    stat.addr = addr;
    stat.width = width;
    stat.height = height;
    stat.num_bits = num_bits;
    ++stat.count;
    stat.bytes += bytes;
    stat.finish_ns += finish_ns;
}

} // namespace VideoCore::ReadbackStats
