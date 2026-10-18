// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/guest_hooks.h"
#include "core/native_direct.h"
#include "core/native_draw.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/native_draw_tag.h"
#include "video_core/readback_policy.h"

namespace Core::NativeDirect {

namespace {

using AmdGpu::PM4CmdNop;

enum class Class : u8 { Inert, Tag, Sync, Flip, Indirect, Packet, Other };

struct Classified {
    Class cls;
    u32 opcode; ///< Sync: the packet the tag stands for; Other: the index in `other`
    u32 dwords;
};

constexpr u32 OtherGameTag = 0;      ///< a tag whose game packets follow (the draw is not executed from its record)
constexpr u32 OtherMarker = 1;       ///< a debug marker while the guest markers are on
constexpr u32 OtherBadPacket = 2;    ///< not a type 3 packet, or a packet that runs past the buffer
constexpr u32 OtherIndirect = 3;     ///< an indirect buffer that cannot be followed
constexpr u32 OtherConstantBuffer = 4; ///< the submission has a constant engine buffer
constexpr u32 BuildOnlyLimit = 5;    ///< indices below this (and above 1) are only found by the build

bool RegisterPacketsOn() {
    return VideoCore::ReadbackPolicy::Get().register_packets.load(std::memory_order_relaxed) != 0;
}

Classified Classify(const u32* packet, bool markers, u32 queue) {
    const u32 header = packet[0];
    const u32 count = (header >> 16) & 0x3fff;
    const u32 opcode = (header >> 8) & 0xff;
    const u32 dwords = count + 2;
    if (opcode == 0x3f) {
        return {Class::Indirect, opcode, dwords};
    }
    if (opcode == 0x50 && queue == 0 && dwords == 7) {
        return {Class::Packet, opcode, dwords}; // a CP DMA of the graphics queue: self-contained, no register state
    }
    if (queue == 0 && RegisterPacketsOn()) {
        // SET_SH_REG and DISPATCH_DIRECT (the game's own compute helpers, such as the texture map) and a CPU style write to memory: the
        // walk's cases, run from the list
        if (opcode == 0x76 || (opcode == 0x15 && dwords == 5)) {
            return {Class::Packet, opcode, dwords};
        }
        // the graphics side of the constant engine's synchronization (increment the draw engine counter, wait on the constant engine counter)
        // context register writes (the extent hints of the targets follow them in the buffer) and event writes: the walk's cases
        if (opcode == 0x69 || opcode == 0x46) {
            return {Class::Packet, opcode, dwords};
        }
        if ((opcode == 0x85 && dwords == 2) || (opcode == 0x86 && dwords == 2)) {
            return {Class::Packet, opcode, dwords};
        }
        if (opcode == 0x37 && dwords >= 4 && (packet[1] & 0x10000) == 0 && (((packet[1] >> 8) & 0xf) == 2 || ((packet[1] >> 8) & 0xf) == 5)) {
            return {Class::Packet, opcode, dwords};
        }
    }
    if (opcode != 0x10) {
        return {Class::Other, opcode, dwords};
    }
    if (count == 0) {
        return {Class::Inert, 0, dwords};
    }
    const u32 magic = packet[1];
    if (magic == VideoCore::NativeTagMagic && count + 1 >= VideoCore::NativeTagPayloadDwords) {
        const u32 meta = packet[2];
        if ((meta & VideoCore::NativeTagExecFlag) != 0 && (meta & VideoCore::NativeTagFollowFlag) == 0) {
            return {Class::Tag, 0, dwords};
        }
        return {Class::Other, OtherGameTag, dwords};
    }
    std::array<u32, 8> restored;
    if (const u32 sync = Core::NativeDraw::RestoreSyncTag(packet, restored, false)) {
        if ((queue == 0 && sync == 0x49) || (queue != 0 && sync == 0x47)) {
            return {Class::Other, sync, dwords}; // release memory belongs to the async queues, end-of-pipe to the graphics queue
        }
        return {Class::Sync, sync, dwords};
    }
    if (magic == PM4CmdNop::PayloadType::PatchedFlip) {
        return {queue == 0 ? Class::Flip : Class::Inert, 0, dwords};
    }
    if (markers && (magic == PM4CmdNop::PayloadType::DebugMarkerPush || magic == PM4CmdNop::PayloadType::DebugColorMarkerPush ||
                    magic == PM4CmdNop::PayloadType::DebugMarkerPop)) {
        return {Class::Other, OtherMarker, dwords};
    }
    return {Class::Inert, 0, dwords};
}

struct IndirectTarget {
    u64 address;
    u32 dwords;
};

IndirectTarget IndirectOf(const u32* packet) {
    return {packet[1] | static_cast<u64>(packet[2] & 0xffff) << 32, packet[3] & 0xfffff};
}

bool Plausible(const IndirectTarget& target) {
    return target.address >= 0x200000000ull && target.address < 0x1000000000ull && (target.address & 3) == 0 && target.dwords != 0 &&
           target.dwords <= (1u << 20);
}

bool StateTemplatesOn() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
           (policy.native_sdk_strip.load(std::memory_order_relaxed) & 8192) != 0;
}

void AddEntry(Buffer& buffer, EntryKind kind, const u32* packet, u32 dwords, u64 address, u32 opcode) {
    Entry entry;
    entry.kind = kind;
    entry.opcode = opcode;
    entry.address = address;
    if (kind == EntryKind::Packet && dwords > entry.words.size()) {
        entry.dwords = dwords;
        entry.seq = buffer.pool.size();
        buffer.pool.insert(buffer.pool.end(), packet, packet + dwords);
    } else {
        entry.dwords = std::min<u32>(dwords, entry.words.size());
        for (u32 i = 0; i < entry.dwords; ++i) {
            entry.words[i] = packet[i];
        }
    }
    if (kind == EntryKind::Tag) {
        entry.seq = packet[9] | static_cast<u64>(packet[10]) << 32;
    }
    buffer.entries.push_back(entry);
}

u32 carry_dwords[64]; ///< per async queue: dwords of a packet that began at the end of the previous window

void AddSpan(Buffer& buffer, const u32* begin, u32 dwords, u32 depth, bool markers, bool templates, u32* carry_out = nullptr) {
    const u32 queue = buffer.queue;
    const u32* at = begin;
    const u32* const end = begin + dwords;
    while (at < end) {
        const u32 type = *at >> 30;
        if (type == 2) {
            ++at;
            continue;
        }
        if (type != 3) {
            ++buffer.other[OtherBadPacket];
            return;
        }
        const Classified packet = Classify(at, markers, queue);
        if (at + packet.dwords > end) {
            ++buffer.other[OtherBadPacket];
            if (carry_out != nullptr) {
                *carry_out = static_cast<u32>(at + packet.dwords - end);
            }
            return;
        }
        const u64 address = reinterpret_cast<u64>(at);
        switch (packet.cls) {
        case Class::Inert:
            break;
        case Class::Tag:
            AddEntry(buffer, EntryKind::Tag, at, packet.dwords, address, 0);
            break;
        case Class::Sync:
            AddEntry(buffer, EntryKind::Sync, at, packet.dwords, address, packet.opcode);
            break;
        case Class::Flip:
            AddEntry(buffer, EntryKind::Flip, at, packet.dwords, address, 0);
            break;
        case Class::Packet:
            if (packet.opcode == 0x69) {
                // the walk reads the NOP packet with the extent hint of a target that follows the register write (up to 22 dwords after it)
                constexpr u32 HintDwords = 24;
                if (at + packet.dwords + HintDwords > end) {
                    ++buffer.other[0x69];
                    break;
                }
                AddEntry(buffer, EntryKind::Packet, at, packet.dwords + HintDwords, address, packet.opcode);
                break;
            }
            AddEntry(buffer, EntryKind::Packet, at, packet.dwords, address, packet.opcode);
            break;
        case Class::Other:
            ++buffer.other[packet.opcode & 255];
            break;
        case Class::Indirect: {
            const IndirectTarget target = IndirectOf(at);
            if (!Plausible(target)) {
                ++buffer.other[OtherIndirect];
                break;
            }
            const auto* target_packets = reinterpret_cast<const u32*>(target.address);
            const Core::NativeDraw::StateTemplate* state =
                target.dwords <= 256 && templates ? Core::NativeDraw::IndirectStateTemplate(target_packets, target.dwords) : nullptr;
            if (state != nullptr) {
                AddEntry(buffer, EntryKind::StateBuffer, at, std::min<u32>(packet.dwords, 4), target.address, 0);
                buffer.entries.back().seq = reinterpret_cast<u64>(state);
            } else if (depth < 3) {
                AddSpan(buffer, target_packets, target.dwords, depth + 1, markers, templates);
            } else {
                ++buffer.other[OtherIndirect];
            }
            break;
        }
        }
        at += packet.dwords;
    }
}

} // namespace

bool PacketPathOff() {
    // Without the game's hooks no draw is a record: the packets are all there is.
    return VideoCore::ReadbackPolicy::Get().packet_path.load(std::memory_order_relaxed) == 0 && GuestHooks::IsActive();
}

bool ExecuteEnabled() {
    return Enabled() && (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 8388608) != 0;
}

bool Enabled() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const u32 bits = policy.native_sdk_strip.load(std::memory_order_relaxed);
    return policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && (bits & 2097152) != 0 && (bits & 4194304) != 0;
}

std::shared_ptr<Buffer> Build(u32 queue, const u32* dcb, u32 dwords, bool has_constant_buffer, bool markers) {
    if (!Enabled() || dcb == nullptr || dwords == 0) {
        if (queue != 0 && dwords != 0) {
            carry_dwords[(queue - 1) & 63] = 0;
        }
        return nullptr;
    }
    auto buffer = std::make_shared<Buffer>();
    buffer->queue = queue;
    buffer->begin = reinterpret_cast<u64>(dcb);
    buffer->dwords = dwords;
    buffer->markers = markers;
    const bool templates = StateTemplatesOn();
    // A submission with a constant engine buffer is scanned like any other: the engine's counters are entries and the buffer runs next to its own
    // constant engine task (ProcessDirect); only a state template (a small buffer of register writes) cannot have one.
    if (const auto* state = !has_constant_buffer && dwords <= 256 && templates ? Core::NativeDraw::IndirectStateTemplate(dcb, dwords) : nullptr) {
        Entry entry;
        entry.kind = EntryKind::StateBuffer;
        entry.address = reinterpret_cast<u64>(dcb);
        entry.seq = reinterpret_cast<u64>(state);
        buffer->entries.push_back(entry);
    } else if (queue != 0) {
        // an async window: the ring may split a packet between two windows (the walk keeps the first part); such windows stay on the walk
        u32& carry = carry_dwords[(queue - 1) & 63];
        const u32 skip = carry;
        carry = 0;
        if (skip >= dwords) {
            carry = skip - dwords;
            ++buffer->other[OtherBadPacket];
        } else {
            buffer->entries.reserve(dwords / 8);
            AddSpan(*buffer, dcb + skip, dwords - skip, 0, markers, false, &carry);
            if (skip != 0) {
                ++buffer->other[OtherBadPacket];
            }
        }
    } else {
        buffer->entries.reserve(dwords / 8);
        AddSpan(*buffer, dcb, dwords, 0, markers, templates);
    }
    u32 other_total = 0;
    std::string signature;
    std::string key = fmt::format("q{}", queue);
    for (u32 index = 0; index < 256; ++index) {
        if (buffer->other[index] != 0) {
            other_total += buffer->other[index];
            signature += fmt::format(" {:#x}:{}", index, buffer->other[index]);
            key += fmt::format(" {:#x}", index);
        }
    }
    buffer->pure = other_total == 0;
    if (!buffer->pure && PacketPathOff()) {
        auto& live_off = Core::NativeDraw::Live();
        live_off.packet_path_skipped_buffers.fetch_add(1, std::memory_order_relaxed);
        live_off.packet_path_skipped_packets.fetch_add(other_total, std::memory_order_relaxed);
        static std::mutex off_mutex;
        static std::set<std::string> off_signatures;
        std::scoped_lock off_lock{off_mutex};
        if (off_signatures.size() < 64 && off_signatures.insert(key).second) {
            LOG_ERROR(Core, "packet path is off: {} packets of a command buffer (queue {}, {} dwords) are not executed (opcode:count):{}", other_total, queue, dwords,
                      signature);
        }
    }
    buffer->complete = buffer->other[OtherBadPacket] == 0 && buffer->other[OtherIndirect] == 0 && buffer->other[OtherConstantBuffer] == 0;
    auto& live = Core::NativeDraw::Live();
    live.direct_built.fetch_add(1, std::memory_order_relaxed);
    live.direct_entries.fetch_add(buffer->entries.size(), std::memory_order_relaxed);
    if (buffer->pure) {
        live.direct_pure.fetch_add(1, std::memory_order_relaxed);
    } else {
        live.direct_mixed.fetch_add(1, std::memory_order_relaxed);
        static std::mutex mutex;
        static std::set<std::string> signatures;
        std::scoped_lock lock{mutex};
        if (signatures.size() < 64 && signatures.insert(key).second) {
            LOG_INFO(Core, "direct submission: mixed buffer, queue {}, {} dwords, {} entries, packets that are not entries (opcode:count):{}", queue,
                     dwords, buffer->entries.size(), signature);
        }
    }
    return buffer;
}

void Buffer::Expect(EntryKind kind, u64 address, u32 opcode, u64 seq) {
    if (cursor < entries.size()) {
        const Entry& entry = entries[cursor];
        if (entry.kind == kind && entry.address == address && entry.opcode == opcode && (kind == EntryKind::StateBuffer || kind == EntryKind::Packet || entry.seq == seq)) {
            ++cursor;
            ++matched;
            return;
        }
    }
    ++mismatched;
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 20) {
        if (cursor < entries.size()) {
            const Entry& entry = entries[cursor];
            LOG_INFO(Core,
                     "direct submission: the walk found entry kind {} at {:#x} (opcode {:#x}, seq {}), the list has kind {} at {:#x} (opcode {:#x}, seq "
                     "{}) as number {} of {}",
                     static_cast<u32>(kind), address, opcode, seq, static_cast<u32>(entry.kind), entry.address, entry.opcode, entry.seq, cursor,
                     entries.size());
        } else {
            LOG_INFO(Core, "direct submission: the walk found entry kind {} at {:#x} after the end of the list ({} entries)", static_cast<u32>(kind),
                     address, entries.size());
        }
    }
    // resynchronize on the list: the entry may be further on
    for (size_t i = cursor; i < entries.size(); ++i) {
        if (entries[i].address == address && entries[i].kind == kind) {
            cursor = i + 1;
            return;
        }
    }
}

void Buffer::Walk(const u32* packet) {
    if (!complete) {
        return;
    }
    const Classified found = Classify(packet, markers, queue);
    const u64 address = reinterpret_cast<u64>(packet);
    switch (found.cls) {
    case Class::Inert:
    case Class::Indirect:
        break;
    case Class::Tag:
        Expect(EntryKind::Tag, address, 0, packet[9] | static_cast<u64>(packet[10]) << 32);
        break;
    case Class::Sync:
        Expect(EntryKind::Sync, address, found.opcode, 0);
        break;
    case Class::Flip:
        Expect(EntryKind::Flip, address, 0, 0);
        break;
    case Class::Packet:
        Expect(EntryKind::Packet, address, found.opcode, 0);
        break;
    case Class::Other:
        ++walk_other[found.opcode & 255];
        break;
    }
}

void Buffer::WalkStateBuffer(u64 address) {
    if (!complete) {
        return;
    }
    Expect(EntryKind::StateBuffer, address, 0, 0);
}

void Buffer::Finish() {
    auto& live = Core::NativeDraw::Live();
    if (!complete) {
        live.direct_incomplete.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (cursor < entries.size()) {
        mismatched += static_cast<u32>(entries.size() - cursor);
    }
    bool purity_differs = false;
    for (u32 index = 0; index < 256; ++index) {
        const bool build_only = index >= 2 && index < BuildOnlyLimit;
        if (!build_only && other[index] != walk_other[index]) {
            purity_differs = true;
        }
    }
    live.direct_matched.fetch_add(matched, std::memory_order_relaxed);
    live.direct_entry_mismatch.fetch_add(mismatched, std::memory_order_relaxed);
    if (purity_differs) {
        live.direct_purity_mismatch.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 10) {
            std::string built, walked;
            for (u32 index = 0; index < 256; ++index) {
                if (other[index] != 0) {
                    built += fmt::format(" {:#x}:{}", index, other[index]);
                }
                if (walk_other[index] != 0) {
                    walked += fmt::format(" {:#x}:{}", index, walk_other[index]);
                }
            }
            LOG_INFO(Core, "direct submission: queue {}, {} dwords: packets that are not entries differ, built:{} walked:{}", queue, dwords, built, walked);
        }
    }
}

} // namespace Core::NativeDirect
