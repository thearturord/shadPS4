// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <nlohmann/json.hpp>
#include "common/types.h"

// Direct submission, step 1b (documents/tlg/native-direct-submission.md): a text dump of one frame of the command stream,
// listed as entries (the native tags with their records) and the packets between them that are not entries
// (fences, waits, flips, indirect buffers). It replaces the PM4 frame dump for buffers that are submitted directly.
namespace Core::EntryDump {

/// Dump the next `frames` frames: the capture starts at the next flip and ends at the flip that closes the last frame.
void Request(u32 frames);

/// State, and the file and line count of the last dump.
nlohmann::json Status();

/// One walked command buffer span (a command buffer or an indirect buffer). `queue`: 0 graphics, 1 + n async compute queue n.
class Span {
public:
    Span(u32 queue, const u32* begin, u32 dwords);
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;

    /// The walk is at the type 3 packet `header` of this span.
    void Packet(const void* header);

private:
    bool tracked{};
    u32 queue{};
    u32 dwords{};
    u32 id{};
    u32 generation{};
    u64 begin{};
};

} // namespace Core::EntryDump
