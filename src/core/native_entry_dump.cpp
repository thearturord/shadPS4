// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/native_draw.h"
#include "core/native_entry_dump.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/native_draw_tag.h"

namespace Core::EntryDump {

namespace {

using namespace AmdGpu;

enum State : u32 { Idle, Armed, Capturing };

constexpr size_t MaxLines = 400000;

std::atomic<u32> state{Idle};
std::atomic<u32> next_span{1};
std::mutex mutex;
std::vector<std::string> lines;
std::map<std::string, u32> by_label;
u32 capture_generation{};
u32 frames_wanted{};
u32 frames_done{};
u64 order{};
bool truncated{};
std::string last_file;
size_t last_lines{};
u64 last_dumps{};

const char* DataSelectName(DataSelect select) {
    switch (select) {
    case DataSelect::None: return "none";
    case DataSelect::Data32Low: return "data32";
    case DataSelect::Data64: return "data64";
    case DataSelect::GpuClock64: return "clock";
    case DataSelect::PerfCounter: return "perf";
    case DataSelect::GdsMemStore: return "gds";
    }
    return "?";
}

const char* FunctionName(PM4CmdWaitRegMem::Function function) {
    switch (function) {
    case PM4CmdWaitRegMem::Function::Always: return "always";
    case PM4CmdWaitRegMem::Function::LessThan: return "<";
    case PM4CmdWaitRegMem::Function::LessThanEqual: return "<=";
    case PM4CmdWaitRegMem::Function::Equal: return "==";
    case PM4CmdWaitRegMem::Function::NotEqual: return "!=";
    case PM4CmdWaitRegMem::Function::GreaterThanEqual: return ">=";
    case PM4CmdWaitRegMem::Function::GreaterThan: return ">";
    default: return "?";
    }
}

std::string DescribePacket(const PM4Header* header, PM4ItOpcode opcode, u32 dwords) {
    switch (opcode) {
    case PM4ItOpcode::EventWriteEop: {
        const auto* eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
        return fmt::format("SIGNAL end-of-pipe label={:#x} value={:#x} data={} irq={}", reinterpret_cast<u64>(eop->Address<u8>()), eop->DataQWord(),
                           DataSelectName(eop->data_sel.Value()), static_cast<u32>(eop->int_sel.Value()));
    }
    case PM4ItOpcode::ReleaseMem: {
        const auto* release = reinterpret_cast<const PM4CmdReleaseMem*>(header);
        return fmt::format("SIGNAL release-mem label={:#x} value={:#x} data={} irq={}", release->Address<u64>(), release->DataQWord(),
                           DataSelectName(release->data_sel.Value()), static_cast<u32>(release->int_sel.Value()));
    }
    case PM4ItOpcode::WriteData: {
        const auto* write = reinterpret_cast<const PM4CmdWriteData*>(header);
        const u32 count = dwords >= 4 ? dwords - 3 : 0;
        std::string text = fmt::format("WRITE label={:#x} dst_sel={} dwords={}", write->Address<u64>(), write->dst_sel.Value(), count);
        for (u32 i = 0; i < count && i < 2; ++i) {
            text += fmt::format(" {:#x}", write->data[i]);
        }
        return text;
    }
    case PM4ItOpcode::WaitRegMem: {
        const auto* wait = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
        if (wait->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
            return fmt::format("WAIT label={:#x} {} {:#x} mask={:#x}", reinterpret_cast<u64>(wait->Address()), FunctionName(wait->function.Value()),
                               wait->ref, wait->mask);
        }
        return fmt::format("WAIT register={:#x} {} {:#x} mask={:#x}", wait->Reg(), FunctionName(wait->function.Value()), wait->ref, wait->mask);
    }
    case PM4ItOpcode::IndirectBuffer: {
        const auto* indirect = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
        return fmt::format("INDIRECT address={:#x} dwords={}", reinterpret_cast<u64>(indirect->Address<u8>()), indirect->ib_size.Value());
    }
    default:
        return fmt::format("PACKET opcode={:#x} dwords={}", static_cast<u32>(opcode), dwords);
    }
}

void CountLabel(const std::string& text) {
    const size_t first = text.find(' ');
    if (text.starts_with("ENTRY SIGNAL")) {
        ++by_label[text.substr(0, text.find(' ', text.find(' ', first + 1) + 1))];
    } else if (first != std::string::npos && (text.starts_with("ENTRY") || text.starts_with("SIGNAL"))) {
        ++by_label[text.substr(0, text.find(' ', first + 1))];
    } else {
        ++by_label[text.substr(0, first)];
    }
}

void Finish() {
    state.store(Idle, std::memory_order_release);
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "entry_dump.txt";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        LOG_ERROR(Core, "entry dump: cannot write {}", path.string());
        lines.clear();
        by_label.clear();
        return;
    }
    out << fmt::format("entry dump: {} frame(s), {} lines{}\n", frames_done, lines.size(), truncated ? " (truncated)" : "");
    out << "columns: walk order, queue (0 graphics, 1+n async compute n), span number, offset in the span (bytes), what\n";
    out << "walk order is the order the command processor walks the buffers: the queues interleave\n";
    out << "ENTRY mode: record = executed from the record, no game packets; at-draw-packet = executed when the draw packet arrives; "
           "packets-follow = the game writes the packets too\n";
    out << "summary:\n";
    for (const auto& [label, count] : by_label) {
        out << fmt::format("  {:8} {}\n", count, label);
    }
    out << "----\n";
    for (const auto& line : lines) {
        out << line << '\n';
    }
    last_file = path.string();
    last_lines = lines.size();
    ++last_dumps;
    LOG_INFO(Core, "entry dump: {} lines written to {}", lines.size(), last_file);
    lines.clear();
    lines.shrink_to_fit();
    by_label.clear();
}

} // namespace

void Request(u32 frames) {
    std::scoped_lock lock{mutex};
    lines.clear();
    by_label.clear();
    ++capture_generation;
    frames_wanted = frames == 0 ? 1 : frames;
    frames_done = 0;
    order = 0;
    truncated = false;
    state.store(Armed, std::memory_order_release);
}

nlohmann::json Status() {
    std::scoped_lock lock{mutex};
    nlohmann::json j;
    const u32 current = state.load();
    j["state"] = current == Idle ? "idle" : current == Armed ? "armed, waiting for a flip" : "capturing";
    j["lines_so_far"] = lines.size();
    j["dumps"] = last_dumps;
    j["last_file"] = last_file;
    j["last_lines"] = last_lines;
    return j;
}

Span::Span(u32 queue_id, const u32* span_begin, u32 span_dwords)
    : tracked{state.load(std::memory_order_relaxed) != Idle}, queue{queue_id}, dwords{span_dwords}, id{next_span.fetch_add(1, std::memory_order_relaxed)},
      begin{reinterpret_cast<u64>(span_begin)} {}

void Span::Packet(const void* packet, u64 address, bool show) {
    if (!tracked || !show) {
        return;
    }
    const u64 packet_address = address != 0 ? address : reinterpret_cast<u64>(packet);
    if (state.load(std::memory_order_acquire) == Idle) {
        tracked = false;
        return;
    }
    const auto* header = static_cast<const PM4Header*>(packet);
    const PM4ItOpcode opcode = header->type3.opcode;
    const u32 packet_dwords = header->type3.NumWords();
    std::string text;
    bool flip = false;
    if (opcode == PM4ItOpcode::Nop) {
        const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
        if (nop->header.count.Value() == 0) {
            return;
        }
        if (nop->data_block[0] == PM4CmdNop::PayloadType::NativeDrawTag && nop->header.count.Value() + 1 >= VideoCore::NativeTagPayloadDwords) {
            if (state.load(std::memory_order_acquire) == Armed) {
                return;
            }
            text = Core::NativeDraw::DescribeEntry(header);
        } else if (std::array<u32, 8> restored; Core::NativeDraw::RestoreSyncTag(reinterpret_cast<const u32*>(header), restored, false) != 0) {
            if (state.load(std::memory_order_acquire) == Armed) {
                return;
            }
            text = "ENTRY " + DescribePacket(reinterpret_cast<const PM4Header*>(restored.data()), static_cast<PM4ItOpcode>((restored[0] >> 8) & 0xff),
                                             (restored[0] >> 16 & 0x3fff) + 2);
        } else if (nop->data_block[0] == PM4CmdNop::PayloadType::PatchedFlip) {
            flip = true;
            text = fmt::format("FLIP patched flip request");
        } else {
            return;
        }
    } else {
        if (state.load(std::memory_order_acquire) == Armed) {
            return;
        }
        text = DescribePacket(header, opcode, packet_dwords);
    }
    std::scoped_lock lock{mutex};
    const u32 current = state.load(std::memory_order_relaxed);
    if (current == Idle) {
        return;
    }
    if (flip && queue == 0 && current == Armed) {
        state.store(Capturing, std::memory_order_release);
        lines.emplace_back("---- frame start (flip seen) ----");
        return;
    }
    if (current != Capturing) {
        return;
    }
    if (generation != capture_generation) {
        generation = capture_generation;
        lines.push_back(fmt::format("        q{} s{:<5} span begins: address {:#x}, {} dwords", queue, id, begin, dwords));
    }
    if (lines.size() >= MaxLines) {
        truncated = true;
        Finish();
        return;
    }
    CountLabel(text);
    lines.push_back(fmt::format("{:7} q{} s{:<5} +{:06x}  {}", order++, queue, id, packet_address - begin, text));
    if (flip && queue == 0) {
        lines.emplace_back("---- frame end (flip) ----");
        if (++frames_done >= frames_wanted) {
            Finish();
        }
    }
}

} // namespace Core::EntryDump
