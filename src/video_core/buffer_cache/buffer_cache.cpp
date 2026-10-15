// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/readback_batch.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/readback_policy.h"
#include "video_core/untracked_ring.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      stream_reuse_enabled{EmulatorSettings.IsReadbackStreamReuseEnabled()},
      prefetch_mode{EmulatorSettings.GetReadbackPrefetchMode()},
      async_guest_faults{EmulatorSettings.IsReadbackAsyncGuestFaultsEnabled()},
      prefetch_lifetime{EmulatorSettings.GetReadbackPrefetchLifetime()},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
    LoadHotFile();
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::LoadHotFile() {
    hot_file = EmulatorSettings.GetReadbackHotFile();
    if (hot_file.empty()) {
        return;
    }
    std::ifstream in{hot_file};
    unsigned long long start{}, end{};
    while (in >> std::hex >> start >> end) {
        if (end <= start) {
            continue;
        }
        hot_file_windows[start] = end;
        seeded_windows.insert(start);
        hot_ranges[start].end = end;
        hot_ranges[start].last_fault_epoch = 0;
    }
    LOG_INFO(Render_Vulkan, "readback_hot_file: {} windows loaded from {}", hot_file_windows.size(),
             hot_file);
}

static u64 SteadyNowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void BufferCache::SaveHotFile() {
    hot_file_dirty = false;
    hot_file_last_write_ns = SteadyNowNs();
    std::ofstream out{hot_file, std::ios::trunc};
    for (const auto& [start, end] : hot_file_windows) {
        out << std::hex << start << ' ' << end << "\n";
    }
}

std::pair<VAddr, VAddr> BufferCache::ReadbackWindow(const Buffer* arena, VAddr device_addr,
                                                    u64 size) const {
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_start = arena->cpu_addr;
    const VAddr arena_end = arena_start + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena_start);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    return {window_start, window_end};
}

void BufferCache::NoteHotRange(VAddr window_start, VAddr window_end, bool is_write) {
    const s32 mode_override = ReadbackPolicy::Get().prefetch_mode_override.load();
    const u32 mode = mode_override >= 0 ? static_cast<u32>(mode_override) : prefetch_mode;
    if (mode == 1 || (mode == 2 && !is_write)) {
        // Remember ranges the CPU reads back so they can be prefetched at the next fence.
        auto& hot_range = hot_ranges[window_start];
        hot_range.end = window_end;
        hot_range.last_fault_epoch = hot_epoch;
        if (!is_write && !hot_file.empty()) {
            auto& known_end = hot_file_windows[window_start];
            if (known_end < window_end) {
                known_end = window_end;
                hot_file_dirty = true;
            }
        }
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    if (async_guest_faults && !assume_locks && !Vulkan::Scheduler::OnPriorityThread()) {
        ReadMemoryForGuest(device_addr, size, is_write);
        return;
    }
    const auto flush_request = [this, device_addr, size, is_write] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        const auto [window_start, window_end] = ReadbackWindow(arena, device_addr, size);
        const u64 window_size = window_end - window_start;
        ApplyPendingUnmarks();
        auto result = DownloadMemory(arena, window_start, window_size);
        const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
        const u64 page_size = page_end - page_start;
        if (result.bytes == 0 && memory_tracker->IsRegionGpuModified(page_start, page_size) &&
            IsDownloadInFlight(page_start, page_size)) {
            // The page has nothing left to download: an asynchronous readback of it was already
            // submitted. Wait for that one to land instead of finishing all GPU work.
            WaitForDownloads(page_start, page_size);
            ApplyPendingUnmarks();
        }
        FinishFaultReadback(device_addr, size, is_write, window_start, window_size, result.bytes);
        if (result.bytes != 0) {
            NoteHotRange(window_start, window_end, is_write);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::FinishFaultReadback(VAddr device_addr, u64 size, bool is_write,
                                      VAddr window_start, u64 window_size, u64& bytes) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    const u64 page_size = page_end - page_start;
    // The faulting page must end up accessible, or the access faults again forever. Download the
    // page, then drop a GPU mark with nothing left behind it.
    if (memory_tracker->IsRegionGpuModified(page_start, page_size)) {
        const auto* arena = GetArena(page_start >> block_shift, (page_end - 1) >> block_shift);
        bytes += DownloadMemory(arena, page_start, page_size).bytes;
        // Downloads of this page have landed by now, so a mark with no GPU write behind it is
        // stale.
        const bool still_marked = memory_tracker->IsRegionGpuModified(page_start, page_size);
        UnmarkSettledPages(page_start, page_size);
        if (still_marked && !memory_tracker->IsRegionGpuModified(page_start, page_size)) {
            static u32 num_logged = 0;
            if (num_logged < 20) {
                ++num_logged;
                LOG_INFO(Render_Vulkan,
                         "Readback: cleared a stale GPU mark on {:#x}-{:#x} (fault at {:#x})",
                         page_start, page_end, device_addr);
            }
        }
    }
    const bool window_has_gpu_bytes = [&] {
        std::shared_lock lock{gpu_ranges_mutex};
        return gpu_modified_ranges.Intersects(window_start, window_size);
    }();
    if (is_write && !window_has_gpu_bytes &&
        memory_tracker->IsRegionGpuModified(window_start, window_size)) {
        // Everything in the window is in guest memory, drop any stale GPU mark before marking
        // the CPU write: a page can't be both, it would need write-only protection.
        memory_tracker->UnmarkRegionAsGpuModified(window_start, window_size, false);
    }
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::ReadMemoryForGuest(VAddr device_addr, u64 size, bool is_write) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    const u64 page_size = page_end - page_start;
    auto landed = std::make_shared<std::atomic<bool>>(false);
    bool recorded = false;
    bool in_flight = false;
    VAddr window_start{};
    u64 window_size{};
    u64 bytes = 0;

    // 1. Command processor: record the downloads of the window and of the faulting page, and
    // submit them. The write backs run on the priority thread once the GPU is done, then this
    // thread is released. Guest memory and protections are not touched yet.
    liverpool->SendCommand<true>([&] {
        ApplyPendingUnmarks();
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        const auto [win_start, win_end] = ReadbackWindow(arena, device_addr, size);
        window_start = win_start;
        window_size = win_end - win_start;
        std::vector<Common::UniqueFunction<void>> callbacks;
        const auto record = [&](VAddr range_start, u64 range_size) {
            DownloadCopies copies;
            const u64 total = CollectDownloadCopies(arena, range_start, range_size, copies);
            if (total == 0) {
                return;
            }
            bytes += total;
            callbacks.push_back(RecordDownloadCopies(arena, range_start, range_size,
                                                     std::move(copies), total, true));
        };
        record(window_start, window_size);
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr range_start = std::max<VAddr>(page_start, arena->cpu_addr);
        const VAddr range_end = std::min<VAddr>(page_end, arena_end);
        if (range_end > range_start) {
            record(range_start, range_end - range_start);
        }
        if (!callbacks.empty()) {
            recorded = true;
            scheduler.DeferPriorityOperation(
                [callbacks = std::move(callbacks), landed]() mutable {
                    for (auto& callback : callbacks) {
                        callback();
                    }
                    landed->store(true, std::memory_order_release);
                    landed->notify_all();
                },
                scheduler.CurrentTick());
            scheduler.Flush();
        } else if (memory_tracker->IsRegionGpuModified(page_start, page_size) &&
                   IsDownloadInFlight(page_start, page_size)) {
            // Nothing left to download: an asynchronous readback of the page is on its way.
            in_flight = true;
            if (scheduler.PriorityOperationsNeedSubmit()) {
                scheduler.Flush();
            }
        }
        if (bytes != 0) {
            NoteHotRange(window_start, window_start + window_size, is_write);
        }
    });

    // 2. This (guest) thread waits for the data. The command processor goes on meanwhile.
    if (recorded) {
        landed->wait(false, std::memory_order_acquire);
    }
    if (in_flight) {
        while (IsDownloadInFlight(page_start, page_size)) {
            std::this_thread::yield();
        }
    }

    // 3. Command processor: apply the protection changes of the landed downloads. Anything the
    // GPU wrote to the page after step 1 is downloaded there, waiting as before (rare).
    liverpool->SendCommand<true>([&] {
        ApplyPendingUnmarks();
        FinishFaultReadback(device_addr, size, is_write, window_start, window_size, bytes);
    });
}

u64 BufferCache::CollectDownloadCopies(const Buffer* arena, VAddr device_addr, u64 size,
                                       DownloadCopies& copies) {
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 range_size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        std::unique_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.ForEachInRange(address, range_size, add_download);
        gpu_modified_ranges.Subtract(address, range_size);
    });
    return total_size_bytes;
}

Common::UniqueFunction<void> BufferCache::RecordDownloadCopies(const Buffer* arena,
                                                               VAddr device_addr, u64 size,
                                                               DownloadCopies&& copies,
                                                               u64 total_size_bytes,
                                                               bool deferred_unmark) {
    const auto download = staging_pool.Request(total_size_bytes, MemoryType::HostCached, 0,
                                               deferred_unmark);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    u64 inflight_id = 0;
    if (deferred_unmark) {
        std::scoped_lock lk{pending_unmarks_mutex};
        inflight_id = ++next_inflight_id;
        inflight_downloads.push_back({device_addr, size, inflight_id});
    }
    return [this, copies = std::move(copies), download, arena_base = arena->cpu_addr, device_addr,
            size, deferred_unmark, inflight_id] {
        download.buffer->Invalidate(download.offset, download.size);
        for (const auto& copy : copies) {
            auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
            memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                    copy.size);
        }
        if (!deferred_unmark) {
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
            return;
        }
        // Running outside the GPU thread. The GPU may have written the range again after the
        // copy was recorded, so the unprotect is applied on the GPU thread, where writes are
        // tracked (see ApplyPendingUnmarks). Until then the range stays protected.
        staging_pool.FreeDeferred(download);
        std::scoped_lock lk{pending_unmarks_mutex};
        std::erase_if(inflight_downloads,
                      [inflight_id](const auto& item) { return item.id == inflight_id; });
        pending_unmarks.emplace_back(device_addr, size);
    };
}

BufferCache::DownloadResult BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr,
                                                        u64 size) {
    DownloadCopies copies;
    const u64 total_size_bytes = CollectDownloadCopies(arena, device_addr, size, copies);
    if (total_size_bytes == 0) {
        return {};
    }
    auto write_data = RecordDownloadCopies(arena, device_addr, size, std::move(copies),
                                           total_size_bytes, false);
    scheduler.Finish();
    write_data();
    return {.bytes = total_size_bytes};
}

void BufferCache::ApplyPendingUnmarks() {
    std::vector<std::pair<VAddr, u64>> unmarks;
    {
        std::scoped_lock lk{pending_unmarks_mutex};
        if (pending_unmarks.empty()) {
            return;
        }
        unmarks.swap(pending_unmarks);
    }
    for (const auto& [device_addr, size] : unmarks) {
        // Parts the GPU wrote again stay protected so the CPU reads the new data. The rest is
        // unprotected: skipping the whole range left pages protected with nothing left to
        // download, which made the CPU fault on them forever.
        UnmarkSettledPages(device_addr, size);
    }
}

bool BufferCache::IsDownloadInFlight(VAddr device_addr, u64 size) {
    std::scoped_lock lk{pending_unmarks_mutex};
    return std::ranges::any_of(inflight_downloads, [&](const InflightDownload& download) {
        return download.addr < device_addr + size && device_addr < download.addr + download.size;
    });
}

void BufferCache::WaitForDownloads(VAddr device_addr, u64 size) {
    // Asynchronous downloads are submitted when recorded. Submit anyway if an operation waits on
    // the command buffer being recorded, since the downloads land in order after it.
    if (scheduler.PriorityOperationsNeedSubmit()) {
        scheduler.Flush();
    }
    const u64 start = SteadyNowNs();
    bool reported = false;
    while (IsDownloadInFlight(device_addr, size)) {
        if (!reported && SteadyNowNs() - start > 2'000'000'000ULL) {
            reported = true;
            LOG_WARNING(Render_Vulkan,
                        "Hang check: waiting >2s for an asynchronous readback of {:#x}:{:#x}",
                        device_addr, size);
        }
        std::this_thread::yield();
    }
}

void BufferCache::UnmarkSettledPages(VAddr device_addr, u64 size) {
    const VAddr start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    RangeSet settled;
    settled.Add(start, end - start);
    // Only data the GPU wrote and that wasn't downloaded yet keeps a page protected. A page with
    // a newer download still in flight is unprotected too: its guest memory holds everything that
    // landed so far, and the game only sees the fence of the newer write once it has landed
    // (keeping it protected made the CPU wait for the hot ranges that are prefetched at nearly
    // every fence).
    {
        std::shared_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.ForEachInRange(start, end - start,
                                           [&](VAddr range_start, VAddr range_end) {
                                               settled.Subtract(range_start,
                                                                range_end - range_start);
                                           });
    }
    // Protection is per page: a page partly outside the settled ranges stays protected.
    settled.ForEach([&](VAddr range_start, VAddr range_end) {
        const VAddr page_start = Common::AlignUp(range_start, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignDown(range_end, BYTES_PER_PAGE);
        if (page_end > page_start) {
            memory_tracker->UnmarkRegionAsGpuModified(page_start, page_end - page_start, false);
        }
    });
}

void BufferCache::PrefetchHotRanges(ReadbackBatch& batch, bool deferred_unmark, u32 fence_kind) {
    // Ranges that stop faulting are prefetched until they expire, then re-learned on the next
    // fault. Expiry keeps ranges the game no longer reads from costing downloads forever.
    ApplyPendingUnmarks();
    if (ReadbackPolicy::ReadbacksMode() == 0) {
        return;
    }
    const auto& policy = ReadbackPolicy::Get();
    if (!(policy.prefetch_kinds_mask.load(std::memory_order_relaxed) & fence_kind) ||
        policy.prefetch_mode_override.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const u32 lifetime = policy.prefetch_lifetime.load(std::memory_order_relaxed);
    if (hot_file_dirty && SteadyNowNs() - hot_file_last_write_ns > 5'000'000'000ULL) {
        SaveHotFile();
    }
    for (auto it = hot_ranges.begin(); it != hot_ranges.end();) {
        if (seeded_windows.contains(it->first)) {
            it->second.last_fault_epoch = hot_epoch;
        } else if (hot_epoch - it->second.last_fault_epoch > lifetime) {
            it = hot_ranges.erase(it);
            continue;
        }
        const VAddr range_start = it->first;
        const VAddr range_end = it->second.end;
        ++it;
        // Prefetched at every fence where the range is GPU modified again. Limiting this to once
        // per frame made later writes in the frame fault, each costing a synchronous GPU wait.
        if (!IsRegionGpuModified(range_start, range_end - range_start)) {
            continue;
        }
        const Buffer* arena = address_space[range_start >> ARENA_PAGE_BITS];
        if (!arena) {
            continue;
        }
        const VAddr start = std::max(arena->cpu_addr, range_start);
        const VAddr end = std::min(arena->cpu_addr + arena->size_bytes, range_end);
        if (start >= end) {
            continue;
        }
        DownloadCopies copies;
        const u64 total_size_bytes = CollectDownloadCopies(arena, start, end - start, copies);
        if (total_size_bytes == 0) {
            continue;
        }
        batch.Reserve(total_size_bytes);
        batch.Add(total_size_bytes, false,
                  RecordDownloadCopies(arena, start, end - start, std::move(copies),
                                       total_size_bytes, deferred_unmark));
    }
}

void BufferCache::InvalidateStreamReuse(VAddr device_addr, u64 size) {
    if (stream_reuse.empty() || device_addr >= stream_reuse_max ||
        device_addr + size <= stream_reuse_min) {
        return;
    }
    stream_reuse.clear();
    stream_reuse_min = ~0ULL;
    stream_reuse_max = 0;
}

void BufferCache::RecordDmaDownload(const Buffer* arena, VAddr dst, u32 size) {
    // The game's own readbacks are small lists and counters; a bigger copy is not one of them.
    constexpr u32 MaxDmaDownload = 4_MB;
    if (size == 0 || size > MaxDmaDownload) {
        return;
    }
    const auto staging = staging_pool.Request(size, MemoryType::HostCached, 0, true);
    const vk::BufferCopy region = {
        .srcOffset = arena->Offset(dst),
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, staging.buffer, std::span{&region, 1});
    auto& policy = ReadbackPolicy::Get();
    policy.dma_async_copies.fetch_add(1, std::memory_order_relaxed);
    policy.dma_async_bytes.fetch_add(size, std::memory_order_relaxed);
    async_recorded.store(1, std::memory_order_release);
    // The priority thread runs its operations in the order they were deferred, so a fence deferred
    // after this one is signaled only once the bytes are in guest memory.
    scheduler.DeferPriorityOperation(
        [this, staging, dst, size] {
            staging.buffer->Invalidate(staging.offset, size);
            memory->TryWriteBacking(std::bit_cast<u8*>(dst), staging.mapped, size);
            staging_pool.FreeDeferred(staging);
        },
        scheduler.CurrentTick());
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        // Stream copies are at most STREAM_THRESHOLD bytes, so the size fits in the low 16 bits.
        const u64 reuse_key = (device_addr << 16) | size;
        if (stream_reuse_enabled) {
            const u64 tick = scheduler.CurrentTick();
            const u64 wraps = stream_buffer.WrapCount();
            if (tick != stream_reuse_tick || wraps != stream_reuse_wraps) {
                // A new command buffer: the game may have changed the data since. A wrap of the
                // stream buffer may overwrite earlier copies.
                stream_reuse_tick = tick;
                stream_reuse_wraps = wraps;
                stream_reuse.clear();
                stream_reuse_min = ~0ULL;
                stream_reuse_max = 0;
            } else if (const auto it = stream_reuse.find(reuse_key); it != stream_reuse.end()) {
                return {&stream_buffer, it->second};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        if (stream_reuse_enabled && stream_buffer.WrapCount() == stream_reuse_wraps) {
            stream_reuse.emplace(reuse_key, offset);
            stream_reuse_min = std::min<VAddr>(stream_reuse_min, device_addr);
            stream_reuse_max = std::max<VAddr>(stream_reuse_max, device_addr + size);
        }
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        std::unique_lock lock{gpu_ranges_mutex};
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    for (const auto& [lo, hi] : resident_range_cache) {
        if (first_block >= lo && last_block < hi) {
            return;
        }
    }
    const bool cacheable = last_block - first_block < 4;
    if (cacheable) {
        bool all_known = true;
        for (u64 block = first_block; block <= last_block; ++block) {
            if (resident_block_cache[block & 4095] != block + 1) {
                all_known = false;
                break;
            }
        }
        if (all_known) {
            return;
        }
    }
    const auto remember = [&] {
        if (cacheable) {
            for (u64 block = first_block; block <= last_block; ++block) {
                resident_block_cache[block & 4095] = block + 1;
            }
        } else {
            resident_range_cache[resident_range_next++ & 7] = {first_block, last_block + 1};
        }
    };

    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        remember();
        return;
    }

    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info = {
        .pNext = &alloc_flags,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    remember();
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const auto add_copy = [&](u64 addr, u64 copy_size) {
        copies.emplace_back(total_size_bytes, addr, copy_size);
        total_size_bytes += copy_size;
    };
    const auto& ring = UntrackedRing::Get();
    if (ring.Active() && !is_written) {
        const auto [lo, hi] = ring.Clip(device_addr, size);
        if (lo < hi && !memory_tracker->IsRegionGpuModified(lo, hi - lo)) {
            // Untracked upload ring: nobody told us what the CPU wrote, so the requested part is uploaded.
            if (device_addr < lo) {
                memory_tracker->ForEachUploadRange(device_addr, lo - device_addr, is_written, add_copy);
            }
            // Read again in the same command buffer: the data cannot have changed since the first upload.
            auto& done = ring_uploads[(lo >> 6) & 255];
            const u64 tick = scheduler.CurrentTick();
            if (done.tick != tick || done.lo > lo || done.hi < hi) {
                add_copy(lo, hi - lo);
                done = {lo, hi, tick};
            }
            if (hi < device_addr + size) {
                memory_tracker->ForEachUploadRange(hi, device_addr + size - hi, is_written, add_copy);
            }
        } else {
            memory_tracker->ForEachUploadRange(device_addr, size, is_written, add_copy);
        }
    } else {
        memory_tracker->ForEachUploadRange(device_addr, size, is_written, add_copy);
    }
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    std::unique_lock queue_lock{Vulkan::Scheduler::submit_mutex, std::defer_lock};
    if (!Vulkan::Scheduler::submit_lock_held) {
        queue_lock.lock();
    }
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
