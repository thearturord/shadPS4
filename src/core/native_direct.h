// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <vector>
#include "common/types.h"

// Direct submission, step 3 (documents/tlg/native-direct-submission.md). When a graphics command buffer is submitted, the entries in
// it (native tags, synchronization tags, the flip, small state buffers) are copied into a list, in order, with indirect buffers
// flattened. A buffer whose packets are all entries or inert is "pure": the list is all the command processor needs. Step 3a only
// builds the list and compares it with what the packet walk finds; nothing executes from it yet.
namespace Core::NativeDirect {

enum class EntryKind : u8 { Tag, Sync, Flip, StateBuffer, Packet };

struct Entry {
    EntryKind kind{};
    u32 opcode{};  ///< Sync: the packet the tag stands for (0x47, 0x49, 0x3c, 0x37); Packet: its opcode
    u32 dwords{};  ///< dwords copied into `words`
    u64 address{}; ///< where it sits in the game's buffer (StateBuffer: the buffer)
    u64 seq{};     ///< Tag: the record
    std::array<u32, 13> words{};
};

class Buffer {
public:
    u32 queue{};
    u64 begin{};
    u32 dwords{};
    bool pure{true};
    bool markers{};
    bool complete{true}; ///< the list holds every entry (false: the build gave up: constant engine buffer, bad packet, indirect buffer it cannot follow)
    u32 depth{}; ///< ProcessGraphics frames running over this buffer (the top one and the indirect buffers it calls)
    std::vector<Entry> entries;
    std::vector<u32> pool; ///< the dwords of Packet entries longer than `Entry::words` (the entry's `seq` is the offset)

    /// The dwords of a Packet entry.
    const u32* PacketWords(const Entry& entry) const {
        return entry.dwords > entry.words.size() ? pool.data() + entry.seq : entry.words.data();
    }
    std::array<u32, 256> other{}; ///< packets by opcode that are not entries (index 0: a tag whose game packets follow, 1: a debug marker)

    /// The walk is at the type 3 packet `packet`.
    void Walk(const u32* packet);
    /// The walk replaced the (indirect) buffer at `address` by its state template.
    void WalkStateBuffer(u64 address);
    /// The last frame of the walk is done: counts what matched.
    void Finish();

private:
    void Expect(EntryKind kind, u64 address, u32 opcode, u64 seq);

    size_t cursor{};
    u32 matched{};
    u32 mismatched{};
    std::array<u32, 256> walk_other{};
};

/// Pure graphics buffers run from the list (bit 8388608 of `native_sdk_strip`, needs the bit above).
bool ExecuteEnabled();

/// Setting `native_packet_path` 0 (live `/set?packet_path=0`): the packet walk is off. A buffer that is not pure runs from its entries; its other packets are not executed.
bool PacketPathOff();

/// Direct submission is on: execution, the sync tags and the setting for it (bit 4194304 of `native_sdk_strip`).
bool Enabled();

/// The entries of the graphics command buffer `dcb` (null when the setting is off). `markers`: the guest debug markers are on.
std::shared_ptr<Buffer> Build(u32 queue, const u32* dcb, u32 dwords, bool has_constant_buffer, bool markers);

} // namespace Core::NativeDirect
