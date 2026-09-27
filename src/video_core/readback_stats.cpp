// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
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
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "video_core/readback_stats.h"

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
    std::atomic<u64> vk_submits{};
    std::atomic<u64> async_fences{};
    std::atomic<u64> async_fence_submits{};
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
std::unordered_map<u64, WaitStat> wait_stats;

std::mutex thread_mutex;
std::condition_variable_any stop_cv;
std::jthread reporter;
u64 start_ns{};

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
    std::vector<WaitStat> waits;
    {
        std::scoped_lock lk{maps_mutex};
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
    }
    std::ranges::sort(buffers, [](const auto& a, const auto& b) {
        return a.blocked_ns > b.blocked_ns;
    });
    std::ranges::sort(shaders, [](const auto& a, const auto& b) {
        return a.blocked_ns > b.blocked_ns;
    });
    std::ranges::sort(images, [](const auto& a, const auto& b) {
        return a.finish_ns > b.finish_ns;
    });
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
                       "base", "size", "writer_hash", "stage", "count", "on_read", "on_write",
                       "MB", "blocked_ms", "gpu_wait_ms");
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
        out += fmt::format("{:#18x} {:>7} {:>8} {:>10.2f} {:>12.1f}\n", s.hash,
                           StageName(s.stage), s.count, Mb(s.bytes), Ms(s.blocked_ns));
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

    out += "\n== Command stream waits (WaitRegMem), by blocked time ==\n";
    out += "pass = passed on a recorded fence, producer = waited for another ring to record it,\n"
           "fence = waited for a held back fence to land, memory = value not written by a fence\n";
    out += fmt::format("{:>18} {:>7} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>11}\n", "address",
                       "queue", "pass", "producer", "fence", "memory", "producer_ms",
                       "fence_ms", "memory_ms");
    for (size_t i = 0; i < std::min<size_t>(waits.size(), 40); ++i) {
        const auto& w = waits[i];
        out += fmt::format("{:#18x} {:>7} {:>9} {:>9} {:>9} {:>9} {:>11.1f} {:>11.1f} {:>11.1f}\n",
                           w.address, w.is_compute ? "compute" : "gfx", w.count[0], w.count[1],
                           w.count[2], w.count[3], Ms(w.ns[1]), Ms(w.ns[2]), Ms(w.ns[3]));
    }

    std::ofstream file{LogPath("readback_top.txt"), std::ios::trunc};
    file << out;
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
           "stats_overhead_ms,vk_cmd_ms,vk_submit_ms,vk_other_ms,cp_wait_spin_ms,cp_flip_spin_ms,gpu_busy_ms,present_cpu_ms,"
           "stream_copies,stream_copy_MB,vk_submits,async_fences,async_fence_submits\n";
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
            cp_columns += fmt::format(",{},{:.2f},{},{},{}", take(counters.stream_copies),
                                      Mb(take(counters.stream_copy_bytes)),
                                      take(counters.vk_submits), take(counters.async_fences),
                                      take(counters.async_fence_submits));
        }
        std::string wait_columns;
        for (u32 q = 0; q < 2; ++q) {
            for (u32 c = 0; c < u32(WaitClass::Count); ++c) {
                const u64 count = take(counters.wait_count[q][c]);
                const u64 ns = take(counters.wait_ns[q][c]);
                wait_columns += c == 0 ? fmt::format(",{}", count)
                                       : fmt::format(",{},{:.2f}", count, Ms(ns));
            }
        }

        csv << fmt::format(
            "{:.1f},{},{},{},{},{},{},{:.2f},{},{:.2f},{},{},{},{},{:.2f},{:.2f},{:.2f},{},{:.2f},"
            "{:.2f},{},{:.2f},{:.1f},{},{},{},{:.2f},{:.2f},{},{:.2f},{},{},{:.2f}{},{},{:.2f},"
            "{}{}\n",
            static_cast<double>(now - start_ns) / 1e9, take(counters.presents),
            take(counters.submits), take(counters.draws), take(counters.dispatches),
            take(counters.read_faults), take(counters.write_faults), Ms(take(counters.fault_ns)),
            take(counters.protects), Ms(take(counters.protect_ns)),
            take(counters.buf_readbacks), take(counters.buf_readbacks_from_read),
            take(counters.buf_readbacks_from_write), take(counters.buf_readbacks_empty),
            Mb(take(counters.buf_bytes)), Ms(take(counters.buf_finish_ns)),
            Ms(take(counters.buf_blocked_ns)), take(counters.img_readbacks),
            Mb(take(counters.img_bytes)), Ms(take(counters.img_finish_ns)),
            take(counters.finishes), Ms(take(counters.finish_ns)), busy,
            take(counters.batches), take(counters.batch_buffers), take(counters.batch_images),
            Mb(take(counters.batch_bytes)), Ms(take(counters.batch_wait_ns)),
            take(counters.regmem_waits), Ms(take(counters.regmem_wait_ns)),
            take(counters.regmem_shortcuts), take(counters.stream_waits),
            Ms(take(counters.stream_wait_ns)), wait_columns, take(counters.sem_waits),
            Ms(take(counters.sem_wait_ns)), take(counters.stream_barriers), cp_columns);
        csv.flush();

        if (++ticks % 5 == 0) {
            WriteTopReport(now - start_ns);
        }
    }
    WriteTopReport(NowNs() - start_ns);
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

void OnPresent() {
    counters.presents.fetch_add(1, std::memory_order_relaxed);
}

void OnSubmit() {
    counters.submits.fetch_add(1, std::memory_order_relaxed);
}

void OnDraw() {
    counters.draws.fetch_add(1, std::memory_order_relaxed);
}

void OnDispatch() {
    counters.dispatches.fetch_add(1, std::memory_order_relaxed);
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
    (flip ? counters.cp_flip_spin_ns : counters.cp_spin_ns).fetch_add(ns,
                                                                      std::memory_order_relaxed);
}

void OnVkSubmit() {
    counters.vk_submits.fetch_add(1, std::memory_order_relaxed);
}

void OnAsyncFence(bool submitted) {
    counters.async_fences.fetch_add(1, std::memory_order_relaxed);
    if (submitted) {
        counters.async_fence_submits.fetch_add(1, std::memory_order_relaxed);
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
