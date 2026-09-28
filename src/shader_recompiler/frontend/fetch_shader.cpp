// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "core/memory.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {

/**
 * s_load_dwordx4 s[8:11], s[2:3], 0x00
 * s_load_dwordx4 s[12:15], s[2:3], 0x04
 * s_load_dwordx4 s[16:19], s[2:3], 0x08
 * s_waitcnt     lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
 * buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen
 * buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen
 * s_waitcnt     0
 * s_setpc_b64   s[0:1]

 * s_load_dwordx4  s[4:7], s[2:3], 0x0
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
 * s_load_dwordx4  s[4:7], s[2:3], 0x8
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[8:11], v0, s[4:7], 0 idxen
 * s_waitcnt       vmcnt(0) & expcnt(0) & lgkmcnt(0)
 * s_setpc_b64     s[0:1]

 * A normal fetch shader looks like the above, the instructions are generated
 * using input semantics on cpu side. Load instructions can either be separate or interleaved
 * We take the reverse way, extract the original input semantics from these instructions.
 **/

static bool IsTypedBufferLoad(const Gcn::GcnInst& inst) {
    return inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_X ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XY ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZ ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZW;
}

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base) {
    const u32* code;
    std::memcpy(&code, &info.user_data[sgpr_base], sizeof(code));
    return code;
}

static FetchShaderData ParseFetchShaderCode(const u32* code);

namespace {
bool backing_reads = false;

/// Where to read the fetch shader code from: its physical backing if enabled (no readback fault
/// when the page also holds GPU-written data, the GPU never writes shader code), else the guest
/// mapping. `size` must cover everything that is read.
const u32* ReadableCode(const u32* code, u64 size) {
    if (backing_reads) {
        if (const u8* backing =
                Core::Memory::Instance()->BackingPointer(reinterpret_cast<VAddr>(code), size)) {
            return reinterpret_cast<const u32*>(backing);
        }
    }
    return code;
}
} // Anonymous namespace

void SetFetchShaderBackingReads(bool enabled) {
    backing_reads = enabled;
}

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info) {
    if (!info.has_fetch_shader) {
        return std::nullopt;
    }

    // This runs for every draw (pipeline lookup), but fetch shaders are tiny and rarely change.
    // Keep the result per code address and decode again only if the code bytes differ.
    struct CachedFetchShader {
        std::vector<u32> code;
        FetchShaderData data;
    };
    thread_local std::unordered_map<const u32*, CachedFetchShader> cache;

    const auto* code = GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    if (const auto it = cache.find(code); it != cache.end()) {
        const auto& cached = it->second;
        const u64 size = cached.code.size() * sizeof(u32);
        if (std::memcmp(ReadableCode(code, size), cached.code.data(), size) == 0) {
            return cached.data;
        }
    }
    // Fetch shaders are a few dozen instructions; a longer one is read through the guest mapping.
    constexpr u64 MaxBackingBytes = 1024;
    const u32* readable = ReadableCode(code, MaxBackingBytes);
    FetchShaderData data = ParseFetchShaderCode(readable);
    if (readable != code && data.size > MaxBackingBytes) {
        readable = code;
        data = ParseFetchShaderCode(code);
    }
    if (cache.size() >= 4096) {
        cache.clear();
    }
    cache[code] = CachedFetchShader{
        .code = std::vector<u32>(readable, readable + data.size / sizeof(u32)),
        .data = data,
    };
    return data;
}

static FetchShaderData ParseFetchShaderCode(const u32* code) {
    FetchShaderData data{};
    GcnCodeSlice code_slice(code, code + std::numeric_limits<u32>::max());
    GcnDecodeContext decoder;

    struct VsharpLoad {
        u32 dword_offset{};
        u32 base_sgpr{};
    };
    std::array<VsharpLoad, 104> loads{};

    u32 semantic_index = 0;
    while (!code_slice.atEnd()) {
        const auto inst = decoder.decodeInstruction(code_slice);
        data.size += inst.length;

        if (inst.opcode == Opcode::S_SETPC_B64) {
            break;
        }

        if (inst.inst_class == InstClass::ScalarMemRd) {
            loads[inst.dst[0].code] = VsharpLoad{inst.control.smrd.offset, inst.src[0].code * 2};
            continue;
        }

        if (inst.opcode == Opcode::V_ADD_I32) {
            const auto vgpr = inst.dst[0].code;
            const auto sgpr = s8(inst.src[0].code);
            switch (vgpr) {
            case 0: // V0 is always the vertex offset
                data.vertex_offset_sgpr = sgpr;
                break;
            case 3: // V3 is always the instance offset
                data.instance_offset_sgpr = sgpr;
                break;
            default:
                UNREACHABLE();
            }
        }

        if (inst.inst_class == InstClass::VectorMemBufFmt) {
            // SRSRC is in units of 4 SPGRs while SBASE is in pairs of SGPRs
            const u32 base_sgpr = inst.src[2].code * 4;
            auto& attrib = data.attributes.emplace_back();
            attrib.semantic = semantic_index++;
            attrib.dest_vgpr = inst.src[1].code;
            attrib.num_elements = inst.control.mubuf.count;
            attrib.sgpr_base = loads[base_sgpr].base_sgpr;
            attrib.dword_offset = loads[base_sgpr].dword_offset;
            attrib.inst_offset = inst.control.mtbuf.offset;
            attrib.instance_data = inst.src[0].code;
            if (IsTypedBufferLoad(inst)) {
                attrib.data_format = inst.control.mtbuf.dfmt;
                attrib.num_format = inst.control.mtbuf.nfmt;
            }
        }
    }

    return data;
}

} // namespace Shader::Gcn
