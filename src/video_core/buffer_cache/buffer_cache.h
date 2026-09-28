// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
}

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class MemoryTracker;
class PageManager;
class ReadbackBatch;

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    /// Window around a readback request: GPU-modified ranges come as many small scattered
    /// islands, so a download is widened to the 512 KB block around the request.
    std::pair<VAddr, VAddr> ReadbackWindow(const Buffer& buffer, VAddr device_addr, u64 size) const;

    /// Binds host vertex buffers for the current draw.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                           boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Bind host index buffer for the current draw.
    void BindIndexBuffer(u32 index_offset,
                         boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {});

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector();

    /// Records downloads of recently read back ranges that are GPU modified again.
    /// With deferred_unmark the write backs may run outside the GPU thread.
    void PrefetchHotRanges(ReadbackBatch& batch, bool deferred_unmark = false);

    /// Advances the frame counter used to expire hot ranges.
    void AdvanceHotEpoch();

    /// Unprotects ranges whose asynchronous readback has landed. Must run on the GPU thread.
    void ApplyPendingUnmarks();

    /// Unprotects the pages in the range that the GPU didn't write again since their data was
    /// downloaded. Must run on the GPU thread.
    void UnmarkSettledPages(VAddr device_addr, u64 size);

    /// Measurement for the multi-core plan: a guest memory write by the command processor
    /// (WriteData, occlusion results, DMA...). Drops reusable stream copies it overlaps.
    void NoteCpGuestWrite(VAddr device_addr, u64 size);

    /// Returns true if an asynchronous download of part of the range has not landed yet.
    bool IsDownloadInFlight(VAddr device_addr, u64 size);

    /// True if guest memory holds the current contents of the range: no GPU write that isn't
    /// downloaded, and no download of it on its way. Command processor thread only.
    bool IsCpuCopyCurrent(VAddr device_addr, u64 size);

    /// Waits until asynchronous downloads of the range have landed in guest memory.
    void WaitForDownloads(VAddr device_addr, u64 size);

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    struct DownloadResult {
        u64 bytes{};
        u64 finish_ns{};
    };

    struct HotRange {
        VAddr end;
        u64 last_fault_epoch;
        u64 last_prefetch_frame{~0ULL}; ///< Readback stats: frame of the last prefetch.
    };

    using DownloadCopies = boost::container::small_vector<vk::BufferCopy, 1>;

    template <bool async>
    DownloadResult DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size);

    /// Gathers GPU modified ranges to download and clears them from the modified range set.
    u64 CollectDownloadCopies(Buffer& buffer, VAddr device_addr, u64 size, DownloadCopies& copies);

    /// Records the copy into download memory. The returned function writes the data back to
    /// guest memory and must only run once the GPU has executed the copy.
    Common::UniqueFunction<void> RecordDownloadCopies(Buffer& buffer, VAddr device_addr, u64 size,
                                                      DownloadCopies&& copies, u64 total_size_bytes,
                                                      bool deferred_unmark);

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    RangeSet gpu_modified_ranges;
    std::unordered_map<VAddr, HotRange> hot_ranges;
    std::mutex pending_unmarks_mutex;
    std::vector<std::pair<VAddr, u64>> pending_unmarks;
    /// Asynchronous downloads recorded but not landed in guest memory yet.
    struct InflightDownload {
        VAddr addr;
        u64 size;
        u64 id;
    };
    std::vector<InflightDownload> inflight_downloads; // guarded by pending_unmarks_mutex
    // Stream copy sources of the current command buffer (readback stats only).
    void NotePendingStreamSource(VAddr device_addr, u64 size);
    bool OverlapsPendingStreamSource(VAddr device_addr, u64 size);
    std::vector<std::pair<VAddr, VAddr>> pending_stream_sources;
    /// readback_stream_reuse_enabled: stream copies of the current command buffer by range.
    /// A copy is only reused while the command buffer, and the stream buffer pass, are the same.
    const bool stream_reuse_enabled;
    /// readback_prefetch_mode: 0 = no prefetch, 1 = read and write faults, 2 = read faults only.
    const u32 prefetch_mode;
    /// readback_async_guest_faults: guest threads wait for their readbacks themselves.
    const bool async_guest_faults;
    void ReadMemoryForGuest(VAddr device_addr, u64 size, bool is_write);
    /// End of a fault readback, on the command processor thread: drops GPU marks with nothing
    /// left behind them, downloads what the GPU wrote to the page in the meantime (waiting), and
    /// marks the CPU write.
    void FinishFaultReadback(VAddr device_addr, u64 size, bool is_write, VAddr window_start,
                             u64 window_size, u64& bytes);
    void NoteHotRange(VAddr window_start, VAddr window_end, bool is_write);
    /// readback_prefetch_lifetime: fences a hot range stays prefetched after its last fault.
    const u32 prefetch_lifetime;
    tsl::robin_map<u64, u64> stream_reuse;
    u64 stream_reuse_tick{};
    u64 stream_reuse_wraps{};
    VAddr stream_reuse_min{~0ULL};
    VAddr stream_reuse_max{};
    /// Drops the reusable copies if a guest memory write overlaps them.
    void InvalidateStreamReuse(VAddr device_addr, u64 size);
    /// Readback stats: how often a stream copy repeats an earlier one (same range, same data).
    void NoteStreamRepeat(VAddr device_addr, u64 size);
    struct StreamSeen {
        u64 tick;
        u64 hash;
    };
    std::unordered_map<u64, StreamSeen> stream_seen;
    u64 pending_stream_tick{};
    VAddr pending_stream_min{~0ULL};
    VAddr pending_stream_max{};
    u64 next_inflight_id{};
    u64 hot_epoch{};
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
};

} // namespace VideoCore
