// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <shared_mutex>
#include <xxhash.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/logging/log.h"
#include "core/native_draw.h"
#include "core/native_resources.h"
#include "video_core/readback_policy.h"

namespace Core::NativeResources {

bool census_enabled = false;

namespace {

constexpr u32 MaxBlocks = 262144;
constexpr std::array<u64, 4> SizeLimits{4096, 65536, 1048576, 16777216};
constexpr u32 NumSizeClasses = 5;
constexpr const char* SizeClassNames[NumSizeClasses] = {"<=4K", "<=64K", "<=1M", "<=16M", ">16M"};
constexpr const char* HeapNames[3] = {"garlic", "onion", "outside"};

/// What the engine says a block is (D3D11-style description of a JsBuffer / JsTexture).
struct Resource {
    u64 object{};
    u64 id{};
    u64 address{};
    u32 kind{};  ///< 1 buffer, 2 texture 1D, 3 texture 2D, 4 texture 3D
    u32 usage{}; ///< 0 default, 1 immutable, 2 dynamic, 3 staging
    u32 bind{};
    u32 misc{};
    u32 format{};
    u32 width{}, height{}, mips{}, layers{};
};

struct Block {
    u64 address{};
    u64 size{};
    u8 heap{};
    u8 usage_class{}; ///< 0 unknown, else 1 + (kind - 1) * 4 + usage
};

struct Counters {
    std::atomic<u64> requests{}, request_bytes{}, written{};
    std::atomic<u64> stream_copy{}, stream_reuse{}, memo{}, arena{};
    std::atomic<u64> upload_bytes{}, stream_bytes{};
    std::atomic<u64> image_uploads{}, image_upload_bytes{}, image_upload_from_gpu_memory{};
    std::atomic<u64> cpu_writes{}, cpu_write_bytes{};
};

constexpr u32 NumUsageClasses = 17;
constexpr const char* KindNames[5] = {"unknown", "buffer", "tex1d", "tex2d", "tex3d"};
constexpr const char* UsageNames[4] = {"default", "immutable", "dynamic", "staging"};

std::string UsageClassName(u32 c) {
    if (c == 0) {
        return "unknown";
    }
    return fmt::format("{}/{}", KindNames[(c - 1) / 4 + 1], UsageNames[(c - 1) % 4]);
}

struct Snapshot {
    std::vector<Block> blocks;       ///< sorted by address
    std::vector<Resource> resources; ///< parallel to blocks (object 0: the block is not a known resource)
    std::unique_ptr<Counters[]> counters;
    std::array<u64, 2> live_bytes{};
};

/// Totals of one block (identified by address and size) over the snapshots it was alive in.
struct History {
    Block block;
    Resource resource;
    u64 requests{}, written{}, upload_bytes{}, image_uploads{}, image_upload_bytes{}, cpu_writes{}, cpu_write_bytes{};
    u64 first_scan{}, last_scan{};
};

std::atomic<Snapshot*> current{nullptr};
std::vector<std::unique_ptr<Snapshot>> retired;
std::array<std::array<Counters, NumSizeClasses>, 3> classes;
std::array<Counters, NumUsageClasses> usage_classes;
std::mutex resource_mutex;
std::unordered_map<u64, Resource> resources_by_address;
std::unordered_map<u64, u64> address_by_object;
std::vector<u64> pending_textures;
std::mutex history_mutex;
std::map<std::pair<u64, u64>, History> history;
std::atomic<u64> scans{};
std::atomic<u64> allocated_blocks{}, freed_blocks{};
std::atomic<bool> started{false};
u64 start_ns{};
std::array<Block, MaxBlocks> scan_buffer;

// ---- immutable buffers --------------------------------------------------------------------------------------------------
std::shared_mutex immutable_mutex;
std::map<u64, std::shared_ptr<ImmutableBuffer>> immutables;                  ///< by begin address
std::unordered_map<u64, std::shared_ptr<ImmutableBuffer>> immutable_by_object;
std::vector<std::pair<u64, std::shared_ptr<ImmutableBuffer>>> immutable_graveyard; ///< destroyed, kept alive for readers
std::atomic<u64> immutable_epoch{1};
// ---- frame ring upload redundancy ---------------------------------------------------------------------------------------
// 64 byte granules of the frame ring already copied to the GPU in the current lap of the ring (a lap ends when the engine's
// write offset wraps; the scan thread notices that once per second, so the figure is approximate).
constexpr u64 RingGranule = 64;
std::atomic<u64> ring_base{0}, ring_size{0};
std::unique_ptr<std::atomic<u8>[]> ring_marks;
std::atomic<u64> ring_laps{}, ring_bytes_new{}, ring_bytes_repeated{}, ring_requests_with_upload{};
std::atomic<u64> ring_last_offset{0};

std::atomic<u64> image_clears_done{};
std::array<std::atomic<u64>, 3> image_clears_refused;
std::atomic<u64> image_copies_done{};
std::array<std::atomic<u64>, 3> image_copies_refused;
std::atomic<u64> immutable_hits{}, immutable_syncs{}, immutable_drops{}, immutable_checks{}, immutable_stale{};

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

u32 SizeClass(u64 size) {
    u32 c = 0;
    while (c < SizeLimits.size() && size > SizeLimits[c]) {
        ++c;
    }
    return c;
}

/// Reads the allocated blocks of one of the engine's Gomalloc heaps: 512 hash buckets from heap + 0xf0, each a list of
/// 32 byte nodes (address, size, next, previous). ~0 when the heap changed while it was read.
#ifdef _WIN32
__declspec(noinline) u32 ScanRaw(u64 heap, u8 id, Block* out, u32 capacity) {
    u32 n = 0;
    __try {
        for (u32 bucket = 0; bucket < 512; ++bucket) {
            u64 node = *reinterpret_cast<const u64*>(heap + 0xf0 + bucket * 8);
            for (u32 guard = 0; node != 0 && guard < 100000; ++guard) {
                if (n < capacity) {
                    out[n++] = {*reinterpret_cast<const u64*>(node), *reinterpret_cast<const u64*>(node + 8), id};
                }
                node = *reinterpret_cast<const u64*>(node + 0x10);
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return ~0u;
    }
    return n;
}
#else
u32 ScanRaw(u64, u8, Block*, u32) {
    return ~0u;
}
#endif

#ifdef _WIN32
__declspec(noinline) bool SafeRead64(u64 address, u64* out) {
    __try {
        *out = *reinterpret_cast<const u64*>(address);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}
#else
bool SafeRead64(u64, u64*) {
    return false;
}
#endif

void MergeInto(std::map<std::pair<u64, u64>, History>& into, const Snapshot& old, u64 scan) {
    for (size_t i = 0; i < old.blocks.size(); ++i) {
        const auto& c = old.counters[i];
        auto& h = into[{old.blocks[i].address, old.blocks[i].size}];
        if (h.block.size == 0) {
            h.block = old.blocks[i];
            h.resource = old.resources[i];
            h.first_scan = scan;
        }
        h.last_scan = scan;
        h.requests += c.requests.load(std::memory_order_relaxed);
        h.written += c.written.load(std::memory_order_relaxed);
        h.upload_bytes += c.upload_bytes.load(std::memory_order_relaxed);
        h.image_uploads += c.image_uploads.load(std::memory_order_relaxed);
        h.image_upload_bytes += c.image_upload_bytes.load(std::memory_order_relaxed);
        h.cpu_writes += c.cpu_writes.load(std::memory_order_relaxed);
        h.cpu_write_bytes += c.cpu_write_bytes.load(std::memory_order_relaxed);
    }
}

/// Reads the description of a JsBuffer / JsTexture (JsResource layout: id +0x10, address +0x30, kind +0x124, description +0x180).
bool ReadResource(u64 object, Resource& out) {
    std::array<u64, 5> desc{};
    u64 id = 0, address = 0, kind = 0;
    if (!SafeRead64(object + 0x10, &id) || !SafeRead64(object + 0x30, &address) || !SafeRead64(object + 0x120, &kind)) {
        return false;
    }
    for (u32 i = 0; i < desc.size(); ++i) {
        if (!SafeRead64(object + 0x180 + i * 8, &desc[i])) {
            return false;
        }
    }
    const auto dword = [&](u32 index) { return static_cast<u32>(desc[index / 2] >> ((index & 1) * 32)); };
    out = {};
    out.object = object;
    out.id = id;
    out.address = address;
    out.kind = static_cast<u32>(kind >> 32);
    switch (out.kind) {
    case 1: // ByteWidth, Usage, BindFlags, CPUAccessFlags, MiscFlags, StructureByteStride
        out.width = dword(0);
        out.usage = dword(1);
        out.bind = dword(2);
        out.misc = dword(4);
        break;
    case 2: // Width, MipLevels, ArraySize, Format, Usage, BindFlags
        out.width = dword(0);
        out.mips = dword(1);
        out.layers = dword(2);
        out.format = dword(3);
        out.usage = dword(4);
        out.bind = dword(5);
        break;
    case 3: // Width, Height, MipLevels, ArraySize, Format, SampleCount, SampleQuality, Usage, BindFlags
        out.width = dword(0);
        out.height = dword(1);
        out.mips = dword(2);
        out.layers = dword(3);
        out.format = dword(4);
        out.usage = dword(7);
        out.bind = dword(8);
        break;
    case 4: // Width, Height, Depth, MipLevels, Format, Usage, BindFlags
        out.width = dword(0);
        out.height = dword(1);
        out.layers = dword(2);
        out.mips = dword(3);
        out.format = dword(4);
        out.usage = dword(5);
        out.bind = dword(6);
        break;
    default:
        return false;
    }
    return out.address != 0 && out.usage < 4;
}

void AddImmutable(const Resource& resource) {
    auto entry = std::make_shared<ImmutableBuffer>();
    entry->begin = resource.address;
    entry->end = resource.address + ((u64{resource.width} + 3) & ~u64{3});
    entry->object = resource.object;
    std::unique_lock lock{immutable_mutex};
    if (const auto it = immutables.find(entry->begin); it != immutables.end()) {
        it->second->alive.store(false, std::memory_order_release);
        immutable_graveyard.emplace_back(NowNs(), it->second);
        immutable_epoch.fetch_add(1, std::memory_order_release);
    }
    immutables[entry->begin] = entry;
    immutable_by_object[entry->object] = entry;
}

void AddResource(const Resource& resource) {
    std::lock_guard lock{resource_mutex};
    resources_by_address[resource.address] = resource;
    address_by_object[resource.object] = resource.address;
}

void UpdateRing() {
    u64 allocator = 0, first = 0, base = 0;
    const u64 device = Core::NativeDraw::EngineDevice();
    if (device == 0 || !SafeRead64(device + 0x10238, &allocator) || allocator == 0 || !SafeRead64(allocator, &first) ||
        !SafeRead64(allocator + 0x18, &base)) {
        return;
    }
    const u64 size = first & 0xffffffffull;
    const u64 offset = first >> 32;
    if (size == 0 || size > (1ull << 30)) {
        return;
    }
    if (ring_size.load() != size || ring_base.load() != base) {
        auto marks = std::make_unique<std::atomic<u8>[]>(size / RingGranule + 1);
        ring_marks = std::move(marks); // not safe against a concurrent event, once at the start only
        ring_base.store(base);
        ring_size.store(size);
    }
    ring_last_offset.store(offset);
}

void NoteRingUpload(VAddr address, u64 bytes) {
    const u64 base = ring_base.load(std::memory_order_relaxed);
    const u64 size = ring_size.load(std::memory_order_relaxed);
    if (size == 0 || address < base || address >= base + size || !ring_marks) {
        return;
    }
    ring_requests_with_upload.fetch_add(1, std::memory_order_relaxed);
    {
        // the engine's frame counter (JsDevice + 0x10240): the marks are per frame
        static std::atomic<u32> last_frame{~0u};
        const u64 device = Core::NativeDraw::EngineDevice();
        u64 frame_word = 0;
        if (device != 0 && SafeRead64(device + 0x10240, &frame_word)) {
            const u32 frame = static_cast<u32>(frame_word);
            if (last_frame.exchange(frame, std::memory_order_relaxed) != frame) {
                ring_laps.fetch_add(1, std::memory_order_relaxed); // counts frames from here on
                for (u64 i = 0; i < size / RingGranule + 1; ++i) {
                    ring_marks[i].store(0, std::memory_order_relaxed);
                }
            }
        }
    }
    const u64 first = (address - base) / RingGranule;
    const u64 last = std::min<u64>((address - base + bytes - 1) / RingGranule, size / RingGranule);
    u64 fresh = 0;
    for (u64 g = first; g <= last; ++g) {
        if (ring_marks[g].exchange(1, std::memory_order_relaxed) == 0) {
            ++fresh;
        }
    }
    ring_bytes_new.fetch_add(fresh * RingGranule, std::memory_order_relaxed);
    ring_bytes_repeated.fetch_add((last - first + 1 - fresh) * RingGranule, std::memory_order_relaxed);
}

void ScanOnce() {
    UpdateRing();
    {
        std::vector<u64> textures;
        {
            std::lock_guard lock{resource_mutex};
            textures.swap(pending_textures);
        }
        for (const u64 object : textures) {
            Resource resource;
            if (ReadResource(object, resource)) {
                AddResource(resource);
            }
        }
    }
    const u64 device = Core::NativeDraw::EngineDevice();
    if (device == 0) {
        return;
    }
    // JsDevice: garlic heap at +0x100b8, onion heap at +0x100e8
    const std::array<std::pair<u64, u8>, 2> heaps{{{device + 0x100b8, 0}, {device + 0x100e8, 1}}};
    auto next = std::make_unique<Snapshot>();
    for (const auto& [slot, id] : heaps) {
        u64 heap = 0;
        if (!SafeRead64(slot, &heap)) {
            return;
        }
        if (heap == 0) {
            continue;
        }
        const u32 n = ScanRaw(heap, id, scan_buffer.data(), MaxBlocks);
        if (n == ~0u) {
            return;
        }
        for (u32 i = 0; i < n; ++i) {
            next->blocks.push_back(scan_buffer[i]);
            next->live_bytes[id] += scan_buffer[i].size;
        }
    }
    std::sort(next->blocks.begin(), next->blocks.end(), [](const Block& a, const Block& b) { return a.address < b.address; });
    next->resources.resize(next->blocks.size());
    {
        std::lock_guard lock{resource_mutex};
        for (size_t i = 0; i < next->blocks.size(); ++i) {
            if (const auto it = resources_by_address.find(next->blocks[i].address); it != resources_by_address.end()) {
                next->resources[i] = it->second;
                next->blocks[i].usage_class = static_cast<u8>(1 + (it->second.kind - 1) * 4 + it->second.usage);
            }
        }
    }
    next->counters = std::make_unique<Counters[]>(next->blocks.size() + 1);

    Snapshot* old = current.load(std::memory_order_acquire);
    const u64 scan = scans.fetch_add(1, std::memory_order_relaxed) + 1;
    if (old) {
        // churn: blocks (address, size) that appeared or vanished since the last scan
        size_t i = 0, j = 0;
        while (i < old->blocks.size() || j < next->blocks.size()) {
            if (j == next->blocks.size() || (i < old->blocks.size() && old->blocks[i].address < next->blocks[j].address)) {
                freed_blocks.fetch_add(1, std::memory_order_relaxed);
                ++i;
            } else if (i == old->blocks.size() || next->blocks[j].address < old->blocks[i].address) {
                allocated_blocks.fetch_add(1, std::memory_order_relaxed);
                ++j;
            } else {
                if (old->blocks[i].size != next->blocks[j].size) {
                    freed_blocks.fetch_add(1, std::memory_order_relaxed);
                    allocated_blocks.fetch_add(1, std::memory_order_relaxed);
                }
                ++i;
                ++j;
            }
        }
        std::lock_guard lock{history_mutex};
        MergeInto(history, *old, scan);
    }
    current.store(next.get(), std::memory_order_release);
    // events in flight may still hold the previous snapshot: free it two scans later
    retired.push_back(std::move(next));
    if (retired.size() > 3) {
        retired.erase(retired.begin());
    }
}

const Block* Find(const Snapshot* snapshot, VAddr address, size_t* index) {
    if (snapshot == nullptr) {
        return nullptr;
    }
    const auto& blocks = snapshot->blocks;
    auto it = std::upper_bound(blocks.begin(), blocks.end(), address, [](u64 a, const Block& b) { return a < b.address; });
    if (it == blocks.begin()) {
        return nullptr;
    }
    --it;
    if (address >= it->address + it->size) {
        return nullptr;
    }
    *index = static_cast<size_t>(it - blocks.begin());
    return &*it;
}

struct Located {
    Counters* block{};
    Counters* cls{};
    Counters* usage{};
};

Located Locate(VAddr address, u64 size_hint) {
    const Snapshot* snapshot = current.load(std::memory_order_acquire);
    size_t index = 0;
    const Block* block = Find(snapshot, address, &index);
    Located out;
    if (block) {
        out.block = &snapshot->counters[index];
        out.cls = &classes[block->heap][SizeClass(block->size)];
        out.usage = &usage_classes[block->usage_class];
    } else {
        out.cls = &classes[2][SizeClass(size_hint)];
        out.usage = &usage_classes[0];
    }
    return out;
}

} // namespace

void NoteBufferObject(u64 object) {
    Resource resource;
    if (ReadResource(object, resource)) {
        AddResource(resource);
        // kind 1 = buffer, usage 1 = immutable; buffers on memory the caller provided (misc bit 19) are owned elsewhere
        if (resource.kind == 1 && resource.usage == 1 && (resource.misc & 0x80000) == 0 && resource.width != 0) {
            AddImmutable(resource);
        }
    }
}

void NoteTextureObject(u64 object) {
    std::lock_guard lock{resource_mutex};
    pending_textures.push_back(object);
}

void NoteBufferDestroyed(u64 object) {
    {
        std::unique_lock lock{immutable_mutex};
        if (const auto it = immutable_by_object.find(object); it != immutable_by_object.end()) {
            it->second->alive.store(false, std::memory_order_release);
            if (const auto at = immutables.find(it->second->begin); at != immutables.end() && at->second == it->second) {
                immutables.erase(at);
            }
            const u64 now = NowNs();
            immutable_graveyard.emplace_back(now, it->second);
            immutable_by_object.erase(it);
            immutable_epoch.fetch_add(1, std::memory_order_release);
            std::erase_if(immutable_graveyard, [now](const auto& g) { return now - g.first > 5'000'000'000ull; });
        }
    }
    std::lock_guard lock{resource_mutex};
    if (const auto it = address_by_object.find(object); it != address_by_object.end()) {
        resources_by_address.erase(it->second);
        address_by_object.erase(it);
    }
}

void NoteBufferRequest(VAddr address, u32 size, bool written, BufferPath path, u64 uploaded) {
    if (!census_enabled) {
        return;
    }
    const Located at = Locate(address, size);
    if (uploaded != 0) {
        NoteRingUpload(address, size);
    }
    for (Counters* c : {at.block, at.cls, at.usage}) {
        if (c == nullptr) {
            continue;
        }
        c->requests.fetch_add(1, std::memory_order_relaxed);
        c->request_bytes.fetch_add(size, std::memory_order_relaxed);
        if (written) {
            c->written.fetch_add(1, std::memory_order_relaxed);
        }
        switch (path) {
        case BufferPath::StreamCopy:
            c->stream_copy.fetch_add(1, std::memory_order_relaxed);
            break;
        case BufferPath::StreamReuse:
            c->stream_reuse.fetch_add(1, std::memory_order_relaxed);
            break;
        case BufferPath::Memo:
            c->memo.fetch_add(1, std::memory_order_relaxed);
            break;
        case BufferPath::Arena:
            c->arena.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        (path == BufferPath::StreamCopy ? c->stream_bytes : c->upload_bytes).fetch_add(uploaded, std::memory_order_relaxed);
    }
}

void NoteImageUpload(VAddr address, u64 size, bool from_gpu_memory) {
    if (!census_enabled) {
        return;
    }
    const Located at = Locate(address, size);
    for (Counters* c : {at.block, at.cls, at.usage}) {
        if (c == nullptr) {
            continue;
        }
        c->image_uploads.fetch_add(1, std::memory_order_relaxed);
        c->image_upload_bytes.fetch_add(size, std::memory_order_relaxed);
        if (from_gpu_memory) {
            c->image_upload_from_gpu_memory.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void NoteCpuWrite(VAddr address, u64 size) {
    {
        std::shared_lock lock{immutable_mutex};
        if (!immutables.empty()) {
            auto it = immutables.upper_bound(address);
            if (it != immutables.begin()) {
                --it;
            }
            for (; it != immutables.end() && it->first < address + size; ++it) {
                ImmutableBuffer& buffer = *it->second;
                if (buffer.end > address) {
                    buffer.generation.fetch_add(1, std::memory_order_acq_rel);
                    if (buffer.synced.exchange(false, std::memory_order_acq_rel)) {
                        immutable_drops.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
    if (!census_enabled) {
        return;
    }
    const Located at = Locate(address, size);
    for (Counters* c : {at.block, at.cls, at.usage}) {
        if (c == nullptr) {
            continue;
        }
        c->cpu_writes.fetch_add(1, std::memory_order_relaxed);
        c->cpu_write_bytes.fetch_add(size, std::memory_order_relaxed);
    }
}

void NoteImageCopy(bool done, u32 reason) {
    if (done) {
        image_copies_done.fetch_add(1, std::memory_order_relaxed);
    } else if (reason < image_copies_refused.size()) {
        image_copies_refused[reason].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteImageClear(bool done, u32 reason) {
    if (done) {
        image_clears_done.fetch_add(1, std::memory_order_relaxed);
    } else if (reason < image_clears_refused.size()) {
        image_clears_refused[reason].fetch_add(1, std::memory_order_relaxed);
    }
}

bool ImageCopyEnabled() {
    return (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 131072) != 0;
}

bool ImmutableResidencyEnabled() {
    return (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 65536) != 0;
}

ImmutableBuffer* FindImmutable(VAddr address, u32 size) {
    struct Cache {
        u64 epoch{};
        std::array<ImmutableBuffer*, 512> entries{};
    };
    static thread_local Cache cache;
    const u64 epoch = immutable_epoch.load(std::memory_order_acquire);
    if (cache.epoch != epoch) {
        cache.entries.fill(nullptr);
        cache.epoch = epoch;
    }
    const auto contains = [&](const ImmutableBuffer* b) {
        return b != nullptr && address >= b->begin && address + size <= b->end && b->alive.load(std::memory_order_acquire);
    };
    // the cache is indexed by the 1 KB unit of the address: the request's own unit, which lies inside its buffer
    ImmutableBuffer*& slot = cache.entries[(address >> 10) & 511];
    if (contains(slot)) {
        return slot;
    }
    std::shared_lock lock{immutable_mutex};
    auto it = immutables.upper_bound(address);
    if (it == immutables.begin()) {
        return nullptr;
    }
    --it;
    ImmutableBuffer* buffer = it->second.get();
    if (!contains(buffer)) {
        return nullptr;
    }
    slot = buffer;
    return buffer;
}

void ImmutableSynced(ImmutableBuffer& buffer, u32 generation) {
    buffer.hash.store(0, std::memory_order_relaxed);
    if (buffer.generation.load(std::memory_order_acquire) == generation) {
        buffer.synced.store(true, std::memory_order_release);
    }
}

bool ImmutableCheck(ImmutableBuffer& buffer) {
    if ((buffer.requests.fetch_add(1, std::memory_order_relaxed) & 4095) != 0) {
        return false;
    }
    immutable_checks.fetch_add(1, std::memory_order_relaxed);
    const u64 now = XXH3_64bits(reinterpret_cast<const void*>(buffer.begin), buffer.end - buffer.begin) | 1;
    const u64 before = buffer.hash.load(std::memory_order_relaxed);
    if (before == 0) {
        buffer.hash.store(now, std::memory_order_relaxed);
        return false;
    }
    if (now == before) {
        return false;
    }
    if (immutable_stale.fetch_add(1, std::memory_order_relaxed) < 10) {
        LOG_ERROR(Core, "immutable engine buffer {:#x} + {:#x} changed without a write fault", buffer.begin,
                  buffer.end - buffer.begin);
    }
    buffer.synced.store(false, std::memory_order_release);
    return true;
}

void NoteImmutableHit() {
    immutable_hits.fetch_add(1, std::memory_order_relaxed);
}

void NoteImmutableSync() {
    immutable_syncs.fetch_add(1, std::memory_order_relaxed);
}

u32 UsageClassOf(VAddr address) {
    const Snapshot* snapshot = current.load(std::memory_order_acquire);
    size_t index = 0;
    const Block* block = Find(snapshot, address, &index);
    return block ? block->usage_class : 0;
}

std::string UsageClassNameOf(u32 usage_class) {
    return UsageClassName(usage_class);
}

std::string DescribeAddress(VAddr address) {
    const Snapshot* snapshot = current.load(std::memory_order_acquire);
    size_t index = 0;
    const Block* block = Find(snapshot, address, &index);
    if (block == nullptr) {
        return "unknown";
    }
    return fmt::format("{} block {:#x}+{:#x} (heap {})", UsageClassName(block->usage_class), block->address, block->size, block->heap);
}

u32 ResourceUsage(u64 object) {
    Resource resource;
    return ReadResource(object, resource) ? resource.usage : ~0u;
}

void Start() {
    if (started.exchange(true)) {
        return;
    }
    start_ns = NowNs();
    census_enabled = true;
    std::thread([] {
        while (true) {
            ScanOnce();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
    LOG_INFO(Core, "resource census started");
}

nlohmann::json Report() {
    using nlohmann::json;
    json out;
    out["enabled"] = census_enabled;
    out["scans"] = scans.load();
    out["seconds"] = start_ns ? (NowNs() - start_ns) / 1e9 : 0.0;
    out["blocks_allocated_since_start"] = allocated_blocks.load();
    out["blocks_freed_since_start"] = freed_blocks.load();
    const Snapshot* snapshot = current.load(std::memory_order_acquire);
    if (snapshot) {
        std::array<u64, 2> count{};
        for (const auto& b : snapshot->blocks) {
            ++count[b.heap];
        }
        out["heaps"] = {{"garlic", {{"blocks", count[0]}, {"bytes", snapshot->live_bytes[0]}}},
                        {"onion", {{"blocks", count[1]}, {"bytes", snapshot->live_bytes[1]}}}};
    }
    json table = json::array();
    for (u32 h = 0; h < 3; ++h) {
        for (u32 s = 0; s < NumSizeClasses; ++s) {
            const Counters& c = classes[h][s];
            if (c.requests == 0 && c.image_uploads == 0 && c.cpu_writes == 0) {
                continue;
            }
            table.push_back({{"heap", HeapNames[h]},
                             {"block_size", SizeClassNames[s]},
                             {"requests", c.requests.load()},
                             {"request_bytes", c.request_bytes.load()},
                             {"written", c.written.load()},
                             {"stream_copy", c.stream_copy.load()},
                             {"stream_reuse", c.stream_reuse.load()},
                             {"memo", c.memo.load()},
                             {"arena", c.arena.load()},
                             {"buffer_upload_bytes", c.upload_bytes.load()},
                             {"stream_copy_bytes", c.stream_bytes.load()},
                             {"image_uploads", c.image_uploads.load()},
                             {"image_upload_bytes", c.image_upload_bytes.load()},
                             {"image_uploads_from_gpu_memory", c.image_upload_from_gpu_memory.load()},
                             {"cpu_writes", c.cpu_writes.load()},
                             {"cpu_write_bytes", c.cpu_write_bytes.load()}});
        }
    }
    out["classes"] = table;
    json by_usage = json::array();
    for (u32 u = 0; u < NumUsageClasses; ++u) {
        const Counters& c = usage_classes[u];
        if (c.requests == 0 && c.image_uploads == 0 && c.cpu_writes == 0) {
            continue;
        }
        by_usage.push_back({{"resource", UsageClassName(u)},
                            {"requests", c.requests.load()},
                            {"request_bytes", c.request_bytes.load()},
                            {"written", c.written.load()},
                            {"stream_copy", c.stream_copy.load()},
                            {"stream_reuse", c.stream_reuse.load()},
                            {"memo", c.memo.load()},
                            {"arena", c.arena.load()},
                            {"buffer_upload_bytes", c.upload_bytes.load()},
                             {"stream_copy_bytes", c.stream_bytes.load()},
                            {"image_uploads", c.image_uploads.load()},
                            {"image_upload_bytes", c.image_upload_bytes.load()},
                            {"image_uploads_from_gpu_memory", c.image_upload_from_gpu_memory.load()},
                            {"cpu_writes", c.cpu_writes.load()},
                            {"cpu_write_bytes", c.cpu_write_bytes.load()}});
    }
    out["by_resource_kind"] = by_usage;
    {
        std::lock_guard lock{resource_mutex};
        out["registered_resources"] = resources_by_address.size();
    }
    // the allocations that cost most: copies from guest memory (buffers and images) and CPU writes
    std::vector<History> all;
    {
        std::lock_guard lock{history_mutex};
        auto merged = history;
        if (snapshot) {
            MergeInto(merged, *snapshot, scans.load());
        }
        for (const auto& [key, h] : merged) {
            all.push_back(h);
        }
    }
    const auto top = [&](auto&& key, const char* name, size_t n) {
        std::vector<History> sorted = all;
        std::sort(sorted.begin(), sorted.end(), [&](const History& a, const History& b) { return key(a) > key(b); });
        json list = json::array();
        for (size_t i = 0; i < sorted.size() && i < n; ++i) {
            const History& h = sorted[i];
            if (key(h) == 0) {
                break;
            }
            list.push_back({{"address", fmt::format("{:#x}", h.block.address)},
                            {"size", h.block.size},
                            {"heap", HeapNames[h.block.heap]},
                            {"resource", UsageClassName(h.block.usage_class)},
                            {"bind_flags", fmt::format("{:#x}", h.resource.bind)},
                            {"misc_flags", fmt::format("{:#x}", h.resource.misc)},
                            {"format", h.resource.format},
                            {"dims", fmt::format("{}x{} mips {} layers {}", h.resource.width, h.resource.height, h.resource.mips,
                                                 h.resource.layers)},
                            {"scans_alive", h.last_scan - h.first_scan + 1},
                            {"requests", h.requests},
                            {"buffer_upload_bytes", h.upload_bytes},
                            {"image_uploads", h.image_uploads},
                            {"image_upload_bytes", h.image_upload_bytes},
                            {"cpu_writes", h.cpu_writes},
                            {"cpu_write_bytes", h.cpu_write_bytes}});
        }
        out[name] = list;
    };
    top([](const History& h) { return h.upload_bytes + h.image_upload_bytes; }, "top_uploaded_from_guest_memory", 20);
    top([](const History& h) { return h.cpu_writes; }, "top_written_by_cpu", 20);
    top([](const History& h) { return h.requests; }, "top_requested", 20);
    out["blocks_with_history"] = all.size();
    {
        // the frame allocator (JsVolatileMemory): size and write offset in its first qword, base address at +0x18
        u64 allocator = 0, first = 0, base = 0;
        const u64 device = Core::NativeDraw::EngineDevice();
        if (device != 0 && SafeRead64(device + 0x10238, &allocator) && allocator != 0 && SafeRead64(allocator, &first) &&
            SafeRead64(allocator + 0x18, &base)) {
            out["frame_ring"] = {{"base", fmt::format("{:#x}", base)},
                                 {"size", first & 0xffffffffull},
                                 {"offset", first >> 32},
                                 {"frames_seen", ring_laps.load()},
                                 {"requests_that_uploaded", ring_requests_with_upload.load()},
                                 {"requested_bytes_not_uploaded_before_this_lap", ring_bytes_new.load()},
                                 {"requested_bytes_already_uploaded_this_lap", ring_bytes_repeated.load()}};
        }
    }
    out["image_clears"] = {{"native", image_clears_done.load()},
                           {"refused_no_image", image_clears_refused[0].load()},
                           {"refused_not_a_plain_colour_image", image_clears_refused[1].load()},
                           {"refused_pattern_or_format", image_clears_refused[2].load()}};
    out["image_copies"] = {{"enabled", ImageCopyEnabled()},
                           {"native", image_copies_done.load()},
                           {"refused_no_source_image", image_copies_refused[0].load()},
                           {"refused_no_destination_image", image_copies_refused[1].load()},
                           {"refused_layout_differs", image_copies_refused[2].load()}};
    {
        std::shared_lock lock{immutable_mutex};
        out["immutable_residency"] = {{"enabled", ImmutableResidencyEnabled()},
                                      {"buffers", immutables.size()},
                                      {"hits", immutable_hits.load()},
                                      {"whole_buffer_syncs", immutable_syncs.load()},
                                      {"dropped_by_cpu_writes", immutable_drops.load()},
                                      {"content_checks", immutable_checks.load()},
                                      {"changed_without_fault", immutable_stale.load()}};
    }
    return out;
}

} // namespace Core::NativeResources
