// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "common/hash_u64.h"
#include "common/interval_set.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;
class ReadbackBatch;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

    /// True once if downloads for the priority thread were recorded since the last call (the
    /// command buffer must reach the GPU soon).
    [[nodiscard]] bool TakeAsyncRecorded() {
        return async_recorded.exchange(0, std::memory_order_acq_rel) != 0;
    }

    /// Records downloads of recently read back ranges that are GPU modified again.
    /// With deferred_unmark the write backs may run outside the GPU thread.
    void PrefetchHotRanges(ReadbackBatch& batch, bool deferred_unmark = false,
                           u32 fence_kind = ~0u);

    /// Advances the fence counter used to expire hot ranges.
    void AdvanceHotEpoch() {
        ++hot_epoch;
    }

    /// A guest memory write by the command processor (WriteData, occlusion results, DMA...).
    /// Drops reusable stream copies it overlaps.
    void NoteCpGuestWrite(VAddr device_addr, u64 size) {
        if (stream_reuse_enabled) {
            InvalidateStreamReuse(device_addr, size);
        }
    }

    /// readback_dma_async: copies [dst, dst + size) of the arena into a private staging buffer; the
    /// priority thread puts the bytes into guest memory once the GPU executed the copy.
    void RecordDmaDownload(const Buffer* arena, VAddr dst, u32 size);

    /// Unprotects ranges whose asynchronous readback has landed. Must run on the GPU thread.
    void ApplyPendingUnmarks();

    /// Returns true if an asynchronous download of part of the range has not landed yet.
    bool IsDownloadInFlight(VAddr device_addr, u64 size);

    /// Waits until asynchronous downloads of the range have landed in guest memory.
    void WaitForDownloads(VAddr device_addr, u64 size);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    struct DownloadResult {
        u64 bytes{};
    };

    struct HotRange {
        VAddr end;
        u64 last_fault_epoch;
    };

    using DownloadCopies = boost::container::small_vector<vk::BufferCopy, 1>;

    DownloadResult DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    /// Gathers GPU modified ranges to download and clears them from the modified range set.
    u64 CollectDownloadCopies(const Buffer* arena, VAddr device_addr, u64 size,
                              DownloadCopies& copies);

    /// Records the copy into download memory. The returned function writes the data back to
    /// guest memory and must only run once the GPU has executed the copy. With deferred_unmark
    /// the download memory is held until then and the protection of the range is lifted on the
    /// GPU thread (see ApplyPendingUnmarks).
    Common::UniqueFunction<void> RecordDownloadCopies(const Buffer* arena, VAddr device_addr,
                                                      u64 size, DownloadCopies&& copies,
                                                      u64 total_size_bytes, bool deferred_unmark);

    /// Window around a readback request: GPU-modified ranges come as many small scattered
    /// islands, so a download is widened to the 512 KB block around the request.
    std::pair<VAddr, VAddr> ReadbackWindow(const Buffer* arena, VAddr device_addr, u64 size) const;

    /// Readback for a guest thread: the command processor records and submits the downloads and
    /// goes on while the faulting thread waits for them to land.
    void ReadMemoryForGuest(VAddr device_addr, u64 size, bool is_write);

    /// End of a fault readback, on the GPU thread: drops GPU marks with nothing left behind them,
    /// downloads what the GPU wrote to the page in the meantime (waiting), and marks the CPU write.
    void FinishFaultReadback(VAddr device_addr, u64 size, bool is_write, VAddr window_start,
                             u64 window_size, u64& bytes);

    void NoteHotRange(VAddr window_start, VAddr window_end, bool is_write);

    /// Unprotects the pages in the range that the GPU didn't write again since their data was
    /// downloaded. Must run on the GPU thread.
    void UnmarkSettledPages(VAddr device_addr, u64 size);

    void LoadHotFile();
    void SaveHotFile();

    /// Drops the reusable stream copies if a guest memory write overlaps them.
    void InvalidateStreamReuse(VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    RangeSet gpu_modified_ranges;
    /// Guards gpu_modified_ranges.
    mutable std::shared_mutex gpu_ranges_mutex;

    /// readback_stream_reuse_enabled: stream copies of the current command buffer by range. A copy
    /// is only reused while the command buffer, and the stream buffer pass, are the same.
    const bool stream_reuse_enabled;
    /// readback_prefetch_mode: 0 = no prefetch, 1 = read and write faults, 2 = read faults only.
    const u32 prefetch_mode;
    /// readback_async_guest_faults: guest threads wait for their readbacks themselves.
    const bool async_guest_faults;
    /// readback_prefetch_lifetime: fences a hot range stays prefetched after its last fault.
    const u32 prefetch_lifetime;
    tsl::robin_map<u64, u64, Common::U64Hash> stream_reuse;
    u64 stream_reuse_tick{};
    u64 stream_reuse_wraps{};
    VAddr stream_reuse_min{~0ULL};
    VAddr stream_reuse_max{};

    /// readback_hot_file: windows the game read (learned from read faults), loaded at start as
    /// permanent prefetch ranges and written back when new ones are learned.
    std::string hot_file;
    std::map<VAddr, VAddr> hot_file_windows; // start -> end
    std::unordered_set<VAddr> seeded_windows;
    bool hot_file_dirty{};
    u64 hot_file_last_write_ns{};
    std::unordered_map<VAddr, HotRange> hot_ranges;
    u64 hot_epoch{};

    /// Asynchronous downloads recorded but not landed in guest memory yet.
    struct InflightDownload {
        VAddr addr;
        u64 size;
        u64 id;
    };
    std::mutex pending_unmarks_mutex;
    std::vector<std::pair<VAddr, u64>> pending_unmarks;
    std::vector<InflightDownload> inflight_downloads; // guarded by pending_unmarks_mutex
    u64 next_inflight_id{};

    std::mutex dma_mutex;
    std::vector<std::unique_ptr<Buffer>> dma_pool;
    std::atomic<u32> async_recorded{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
