#ifndef SYNAPTIK_TASK0065_RNG_DROPOUT_KERNELS_H
#define SYNAPTIK_TASK0065_RNG_DROPOUT_KERNELS_H

static NSString *const SynaptikTask0065RngDropoutKernelSource =
@"\n"
"struct RngDropoutMeta0065 { ulong first; ulong second; uint complementBits; uint count; uint valueType; uint reserved; };\n"
"inline ulong mix64_0065(ulong z) { z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ul; z = (z ^ (z >> 27u)) * 0x94d049bb133111ebul; return z ^ (z >> 31u); }\n"
"kernel void initial_state_0065(device ulong *output [[buffer(0)]], constant RngDropoutMeta0065 &m [[buffer(1)]], uint gid [[thread_position_in_grid]]) { if (gid != 0u) return; output[0] = m.first; output[1] = m.second; }\n"
"kernel void dropout_numeric_0065(device const uchar *input [[buffer(0)]], device const ulong *state [[buffer(1)]], device uchar *output [[buffer(2)]], device uchar *mask [[buffer(3)]], device ulong *nextState [[buffer(4)]], constant RngDropoutMeta0065 &m [[buffer(5)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= m.count) return;\n"
"    ulong key = state[0], counter = state[1];\n"
"    ulong keyOffset = mix64_0065(key + 0x9e3779b97f4a7c15ul);\n"
"    ulong word = mix64_0065(counter + ulong(gid) + keyOffset);\n"
"    bool keep = (word >> 11u) >= m.first;\n"
"    mask[gid] = keep ? uchar(1u) : uchar(0u);\n"
"    if (keep) {\n"
"        float scale = 1.0f / as_type<float>(m.complementBits);\n"
"        float value = synaptik_load_numeric(input, ulong(gid), m.valueType) * scale;\n"
"        synaptik_store_numeric(output, ulong(gid), m.valueType, value);\n"
"    } else {\n"
"        store_word(output, ulong(gid), lp_width(m.valueType), 0ul);\n"
"    }\n"
"    if (gid == 0u) { nextState[0] = key; nextState[1] = counter + ulong(m.count); }\n"
"}\n"
;

#endif
