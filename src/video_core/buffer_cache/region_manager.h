// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/types.h"
#include <array>

#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    /// Hot pages: pages the CPU rewrites right after nearly every upload. Write-protecting them
    /// again after each upload only buys a write fault per rewrite, so they stay CPU-modified and
    /// writable and are uploaded at each use instead. A page used as a GPU write target stops
    /// being hot. Must be called under the lock.
    void SetHotPagesEnabled(bool enabled) noexcept {
        hot_pages_enabled = enabled;
    }

    /// Counts CPU writes to pages an upload just write-protected again. Returns the number of
    /// pages that became hot. Must be called under the lock.
    u32 NoteCpuWrite(u64 dirty_addr, u64 size) noexcept {
        if (!hot_pages_enabled) {
            return 0;
        }
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page = std::min<size_t>(
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE),
            NUM_PAGES_PER_REGION);
        u32 new_hot = 0;
        for (size_t page = start_page; page < end_page; ++page) {
            if (!reprotected.Get(page)) {
                continue;
            }
            reprotected.Unset(page);
            if (refaults[page] < 255) {
                ++refaults[page];
            }
            if (refaults[page] >= HotRefaults && !hot.Get(page)) {
                hot.Set(page);
                ++new_hot;
            }
        }
        return new_hot;
    }

    /// Forgets hot pages and halves the rewrite counts, so pages the CPU stopped rewriting get
    /// write-protected again at their next upload. Must be called under the lock.
    void DecayHotPages() noexcept {
        hot.Clear();
        for (auto& count : refaults) {
            count >>= 1;
        }
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
            UpdateProtection<enable, true>();
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func,
                              bool keep_hot = false) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            if constexpr (type == Type::CPU) {
                if (hot_pages_enabled) {
                    RegionBits range;
                    range.SetRange(start_page, end_page);
                    if (keep_hot) {
                        // Uploaded for reading: hot pages stay modified (and writable).
                        const RegionBits cleared = range & ~hot;
                        bits &= ~cleared;
                        reprotected |= mask & cleared;
                    } else {
                        // Uploaded before a GPU write: every page goes back to normal tracking.
                        bits.UnsetRange(start_page, end_page);
                        hot &= ~range;
                        reprotected &= ~range;
                    }
                } else {
                    bits.UnsetRange(start_page, end_page);
                }
                UpdateProtection<true, false>();
            } else {
                bits.UnsetRange(start_page, end_page);
                if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
                    UpdateProtection<false, true>();
                }
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        const RegionBits& bits = GetRegionBits<type>();
        return bits.AnyInRange(start_page, end_page);
    }

    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
    }

    /// CPU rewrites after an upload before a page becomes hot.
    static constexpr u8 HotRefaults = 3;

    PageManager* tracker;
    VAddr cpu_addr = 0;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
    bool hot_pages_enabled{};
    RegionBits hot{};         ///< Kept CPU-modified and writable (see SetHotPagesEnabled).
    RegionBits reprotected{}; ///< Write-protected again by an upload, not rewritten since.
    std::array<u8, NUM_PAGES_PER_REGION> refaults{}; ///< CPU rewrites after an upload.
};

} // namespace VideoCore
