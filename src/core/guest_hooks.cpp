// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <nlohmann/json.hpp>
#include <Zydis/Zydis.h>
#include <xbyak/xbyak.h>
#include "common/decoder.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/cpu_patches.h"
#include "core/debug_server.h"
#include "core/guest_hooks.h"
#include "core/memory.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Xbyak::util;

namespace Core::GuestHooks {

namespace {

constexpr u64 RingHeaderSize = 64;
constexpr u64 RingEntrySize = 64;
constexpr size_t MinDisplacedBytes = 5; // size of a near jump
constexpr size_t MaxDisplacedBytes = 15;

enum class Mode { Count, Args };

struct Deref {
    u32 reg{};  // index into rdi, rsi, rdx, rcx, r8, r9
    u64 offset{};
    u32 size{}; // 1, 2, 4 or 8
};

struct Hook {
    std::string name;
    std::vector<Deref> derefs;
    u64 offset{};
    Mode mode{Mode::Count};
    u8* entry{};

    // header (64 bytes): u64 index, then for "args" hooks the ring of 64-byte entries:
    //   rdi rsi rdx rcx r8 r9 ret seq
    std::unique_ptr<u8[]> storage;
    u8* header{};
    u64 ring_entries{};
    u64 last_count{};

    std::atomic<u64>* counter() const {
        return reinterpret_cast<std::atomic<u64>*>(header);
    }
};

std::vector<std::unique_ptr<Hook>> hooks;
std::ofstream counts_file;
std::ofstream args_file;
std::ofstream deref_file;
std::ofstream fault_file;
std::atomic<u64> frame_index{0};

// Write faults seen by the fault handler: {frame, address}, drained once per frame.
constexpr u64 FaultRingSize = 1u << 20;
struct FaultRecord {
    std::atomic<u64> frame{~0ull};
    std::atomic<u64> address{};
};
std::unique_ptr<FaultRecord[]> fault_ring;
std::atomic<u64> fault_head{0};
u64 fault_tail = 0;

std::filesystem::path ConfigPath() {
    const auto& serial = MemoryPatcher::g_game_serial;
    return Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (serial + ".hooks.json");
}

u64 ParseHex(const nlohmann::json& value) {
    if (value.is_number_unsigned()) {
        return value.get<u64>();
    }
    return std::stoull(value.get<std::string>(), nullptr, 16);
}

/// Decodes whole instructions from `code` until at least MinDisplacedBytes are covered. Returns 0 if
/// an instruction cannot be copied to another address unchanged.
size_t DisplacedLength(const u8* code, std::string& why) {
    size_t length = 0;
    while (length < MinDisplacedBytes) {
        ZydisDecodedInstruction inst;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        const auto status = Common::Decoder::Instance()->decodeInstruction(
            inst, operands, const_cast<u8*>(code) + length, MaxDisplacedBytes);
        if (!ZYAN_SUCCESS(status)) {
            why = "cannot decode instruction at +" + std::to_string(length);
            return 0;
        }
        if ((inst.attributes & ZYDIS_ATTRIB_IS_RELATIVE) != 0) {
            why = "relative operand at +" + std::to_string(length);
            return 0;
        }
        switch (inst.meta.category) {
        case ZYDIS_CATEGORY_CALL:
        case ZYDIS_CATEGORY_COND_BR:
        case ZYDIS_CATEGORY_UNCOND_BR:
        case ZYDIS_CATEGORY_RET:
        case ZYDIS_CATEGORY_INTERRUPT:
        case ZYDIS_CATEGORY_SYSCALL:
            why = "branch or call at +" + std::to_string(length);
            return 0;
        default:
            break;
        }
        length += inst.length;
        if (length > MaxDisplacedBytes) {
            why = "displaced bytes exceed the limit";
            return 0;
        }
    }
    return length;
}

/// Stub for a counter-only hook: lock inc [header].
void EmitCountStub(Xbyak::CodeGenerator& c, const Hook& hook) {
    c.pushfq();
    c.push(r10);
    c.mov(r10, reinterpret_cast<u64>(hook.header));
    c.lock();
    c.inc(qword[r10]);
    c.pop(r10);
    c.popfq();
}

/// Stub for an argument-recording hook. Takes a slot with lock xadd, stores rdi..r9 and the return
/// address, and publishes the sequence number last.
void EmitArgsStub(Xbyak::CodeGenerator& c, const Hook& hook) {
    c.pushfq();
    c.push(rax);
    c.push(r10);
    c.push(r11);
    c.mov(r10, reinterpret_cast<u64>(hook.header));
    c.mov(r11d, 1);
    c.lock();
    c.xadd(qword[r10], r11); // r11 = sequence number of this call
    c.mov(rax, r11);
    c.and_(rax, static_cast<uint32_t>(hook.ring_entries - 1));
    c.shl(rax, 6);
    c.lea(r10, ptr[r10 + rax + RingHeaderSize]);
    c.mov(ptr[r10 + 0], rdi);
    c.mov(ptr[r10 + 8], rsi);
    c.mov(ptr[r10 + 16], rdx);
    c.mov(ptr[r10 + 24], rcx);
    c.mov(ptr[r10 + 32], r8);
    c.mov(ptr[r10 + 40], r9);
    c.mov(rax, ptr[rsp + 32]); // return address: [r11][r10][rax][flags][ret]
    c.mov(ptr[r10 + 48], rax);
    c.mov(ptr[r10 + 56], r11); // publish
    c.pop(r11);
    c.pop(r10);
    c.pop(rax);
    c.popfq();
}

bool InstallHook(Hook& hook, u8* image_base, u64 image_size) {
    if (hook.offset + 16 > image_size) {
        LOG_ERROR(Core, "guest hook '{}': offset {:#x} is outside the image", hook.name, hook.offset);
        return false;
    }
    u8* entry = image_base + hook.offset;
    std::string why;
    const size_t displaced = DisplacedLength(entry, why);
    if (displaced == 0) {
        LOG_ERROR(Core, "guest hook '{}' at {:#x} refused: {}", hook.name, hook.offset, why);
        return false;
    }
    u8 original[MaxDisplacedBytes];
    std::memcpy(original, entry, displaced);

    const u64 storage_size =
        RingHeaderSize + (hook.mode == Mode::Args ? hook.ring_entries * RingEntrySize : 0) + 64;
    hook.storage = std::make_unique<u8[]>(storage_size);
    std::memset(hook.storage.get(), 0, storage_size);
    const auto raw = reinterpret_cast<u64>(hook.storage.get());
    hook.header = reinterpret_cast<u8*>((raw + 63) & ~u64{63});

    const u8* stub = nullptr;
    const bool emitted = EmitInTrampolineArea(entry, [&](Xbyak::CodeGenerator& c) {
        stub = c.getCurr<const u8*>();
        if (hook.mode == Mode::Args) {
            EmitArgsStub(c, hook);
        } else {
            EmitCountStub(c, hook);
        }
        c.db(original, displaced);
        c.jmp(entry + displaced, Xbyak::CodeGenerator::T_NEAR);
        return true;
    });
    if (!emitted) {
        LOG_ERROR(Core, "guest hook '{}': no room in the trampoline area", hook.name);
        return false;
    }

    const s64 rel = reinterpret_cast<s64>(stub) - reinterpret_cast<s64>(entry + 5);
    if (rel < INT32_MIN || rel > INT32_MAX) {
        LOG_ERROR(Core, "guest hook '{}': trampoline is out of near-jump range", hook.name);
        return false;
    }
    u8 patch[MaxDisplacedBytes];
    patch[0] = 0xE9;
    const auto rel32 = static_cast<s32>(rel);
    std::memcpy(&patch[1], &rel32, sizeof(rel32));
    std::memset(&patch[5], 0x90, displaced - 5);
    std::memcpy(entry, patch, displaced);
#ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), entry, displaced);
#endif
    hook.entry = entry;
    LOG_INFO(Core, "guest hook '{}' installed at {:#x} ({} bytes displaced, mode {})", hook.name,
             reinterpret_cast<u64>(entry), displaced, hook.mode == Mode::Args ? "args" : "count");
    return true;
}

} // namespace

void InstallForEboot(VAddr base, u64 image_size) {
    const auto path = ConfigPath();
    if (!std::filesystem::exists(path)) {
        return;
    }
    nlohmann::json config;
    try {
        std::ifstream in(path);
        in >> config;
    } catch (const std::exception& e) {
        LOG_ERROR(Core, "guest hooks: cannot read {}: {}", path.string(), e.what());
        return;
    }

    try {
        const u64 ghidra_base = config.contains("ghidra_image_base")
                                    ? ParseHex(config.at("ghidra_image_base"))
                                    : 0x1000000;
        u64 ring_entries = config.value("ring_entries", u64{16384});
        if (ring_entries < 16 || (ring_entries & (ring_entries - 1)) != 0 ||
            ring_entries > (1u << 24)) {
            LOG_WARNING(Core, "guest hooks: ring_entries must be a power of two in 16..16M, using 16384");
            ring_entries = 16384;
        }

        u32 installed = 0;
        u32 failed = 0;
        for (const auto& item : config.at("hooks")) {
            auto hook = std::make_unique<Hook>();
            hook->name = item.at("name").get<std::string>();
            if (item.contains("ghidra")) {
                const u64 address = ParseHex(item.at("ghidra"));
                if (address < ghidra_base) {
                    LOG_ERROR(Core, "guest hook '{}': address below the image base", hook->name);
                    ++failed;
                    continue;
                }
                hook->offset = address - ghidra_base;
            } else {
                hook->offset = ParseHex(item.at("offset"));
            }
            hook->mode = item.value("mode", std::string("count")) == "args" ? Mode::Args : Mode::Count;
            if (item.contains("deref")) {
                static const char* const reg_names[] = {"rdi", "rsi", "rdx", "rcx", "r8", "r9"};
                for (const auto& d : item.at("deref")) {
                    Deref deref;
                    const auto name = d.at("reg").get<std::string>();
                    deref.reg = 99;
                    for (u32 i = 0; i < 6; ++i) {
                        if (name == reg_names[i]) {
                            deref.reg = i;
                        }
                    }
                    deref.offset = ParseHex(d.at("off"));
                    deref.size = d.value("size", 8u);
                    if (deref.reg == 99 ||
                        (deref.size != 1 && deref.size != 2 && deref.size != 4 && deref.size != 8) ||
                        hook->derefs.size() >= 6) {
                        LOG_ERROR(Core, "guest hook '{}': bad deref entry", hook->name);
                        continue;
                    }
                    hook->derefs.push_back(deref);
                }
            }
            hook->ring_entries = ring_entries;
            if (InstallHook(*hook, reinterpret_cast<u8*>(base), image_size)) {
                hooks.push_back(std::move(hook));
                ++installed;
            } else {
                ++failed;
            }
        }
        LOG_INFO(Core, "guest hooks: {} installed, {} failed ({})", installed, failed, path.string());
    } catch (const std::exception& e) {
        LOG_ERROR(Core, "guest hooks: bad configuration {}: {}", path.string(), e.what());
    }

    if (hooks.empty()) {
        return;
    }
    const auto log_dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    counts_file.open(log_dir / "guest_hooks.csv", std::ios::trunc);
    counts_file << "frame,name,calls,total\n";
    const bool any_args = std::any_of(hooks.begin(), hooks.end(),
                                      [](const auto& h) { return h->mode == Mode::Args; });
    if (any_args) {
        args_file.open(log_dir / "guest_hooks_args.csv", std::ios::trunc);
        args_file << "frame,name,seq,rdi,rsi,rdx,rcx,r8,r9,ret\n";
    }
    if (std::any_of(hooks.begin(), hooks.end(),
                    [](const auto& h) { return !h->derefs.empty(); })) {
        deref_file.open(log_dir / "guest_hooks_deref.csv", std::ios::trunc);
        deref_file << "frame,name,seq,d0,d1,d2,d3,d4,d5\n";
    }
    fault_ring = std::make_unique<FaultRecord[]>(FaultRingSize);
    fault_file.open(log_dir / "write_faults.csv", std::ios::trunc);
    fault_file << "frame,page\n";
}

void NoteWriteFault(VAddr address) {
    if (!fault_ring) {
        return;
    }
    const u64 slot = fault_head.fetch_add(1, std::memory_order_relaxed) & (FaultRingSize - 1);
    fault_ring[slot].address.store(address, std::memory_order_relaxed);
    fault_ring[slot].frame.store(frame_index.load(std::memory_order_relaxed),
                                 std::memory_order_release);
}

bool IsActive() {
    return !hooks.empty();
}

void OnFrame() {
    if (hooks.empty()) {
        return;
    }
    const u64 frame = frame_index.fetch_add(1, std::memory_order_relaxed) + 1;
    // Write faults of the frame that just ended, one line per page.
    {
        const u64 head = fault_head.load(std::memory_order_acquire);
        if (head - fault_tail > FaultRingSize) {
            fault_tail = head - FaultRingSize;
        }
        std::unordered_set<u64> seen;
        for (u64 i = fault_tail; i < head; ++i) {
            const auto& record = fault_ring[i & (FaultRingSize - 1)];
            const u64 fault_frame = record.frame.load(std::memory_order_acquire);
            if (fault_frame == ~0ull) {
                continue;
            }
            const u64 page = record.address.load(std::memory_order_relaxed) & ~u64{0xFFF};
            if (seen.insert(page ^ (fault_frame << 52)).second) {
                fault_file << fault_frame << ',' << std::hex << page << std::dec << '\n';
            }
        }
        fault_tail = head;
    }
    nlohmann::json published_calls = nlohmann::json::object();
    for (auto& hook : hooks) {
        const u64 total = hook->counter()->load(std::memory_order_relaxed);
        const u64 delta = total - hook->last_count;
        published_calls[hook->name] = {{"calls_last_frame", delta}, {"total", total}};
        if (delta != 0) {
            counts_file << frame << ',' << hook->name << ',' << delta << ',' << total << '\n';
        }
        if (hook->mode == Mode::Args && args_file.is_open() && delta != 0) {
            const u64 first =
                total - hook->last_count > hook->ring_entries ? total - hook->ring_entries
                                                              : hook->last_count;
            for (u64 seq = first; seq < total; ++seq) {
                const u64 slot = seq & (hook->ring_entries - 1);
                const auto* entry = reinterpret_cast<const volatile u64*>(
                    hook->header + RingHeaderSize + slot * RingEntrySize);
                u64 values[8];
                for (int i = 0; i < 8; ++i) {
                    values[i] = entry[i];
                }
                if (values[7] != seq) {
                    continue; // not published yet, or overwritten
                }
                args_file << frame << ',' << hook->name << ',' << seq;
                args_file << std::hex;
                for (int i = 0; i < 7; ++i) {
                    args_file << ',' << values[i];
                }
                args_file << std::dec << '\n';
                if (!hook->derefs.empty() && deref_file.is_open()) {
                    deref_file << frame << ',' << hook->name << ',' << seq << std::hex;
                    for (size_t d = 0; d < 6; ++d) {
                        deref_file << ',';
                        if (d >= hook->derefs.size()) {
                            continue;
                        }
                        const auto& deref = hook->derefs[d];
                        const u64 address = values[deref.reg] + deref.offset;
                        u64 value = ~0ull;
                        if (values[deref.reg] != 0 &&
                            Core::Memory::Instance()->IsValidMapping(address, deref.size)) {
                            value = 0;
                            std::memcpy(&value, reinterpret_cast<const void*>(address), deref.size);
                        }
                        deref_file << value;
                    }
                    deref_file << std::dec << '\n';
                }
            }
        }
        hook->last_count = total;
    }
    DebugServer::Publish("hooks", std::move(published_calls));
    if (frame % 60 == 0) {
        counts_file.flush();
        if (args_file.is_open()) {
            args_file.flush();
        }
        if (deref_file.is_open()) {
            deref_file.flush();
        }
        fault_file.flush();
    }
}

} // namespace Core::GuestHooks
