// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
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
}

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
    {
        std::scoped_lock lk{maps_mutex};
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
           "stream_wait_ms\n";
    csv.flush();

    u64 last_ns = NowNs();
    u32 ticks = 0;
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

        csv << fmt::format(
            "{:.1f},{},{},{},{},{},{},{:.2f},{},{:.2f},{},{},{},{},{:.2f},{:.2f},{:.2f},{},{:.2f},"
            "{:.2f},{},{:.2f},{:.1f},{},{},{},{:.2f},{:.2f},{},{:.2f},{},{},{:.2f}\n",
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
            Ms(take(counters.stream_wait_ns)));
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

void OnRegMemWait(u64 wait_ns, bool shortcut) {
    if (shortcut) {
        counters.regmem_shortcuts.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    counters.regmem_waits.fetch_add(1, std::memory_order_relaxed);
    counters.regmem_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
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
