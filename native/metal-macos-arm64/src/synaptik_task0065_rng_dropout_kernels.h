#ifndef SYNAPTIK_TASK0065_RNG_DROPOUT_KERNELS_H
#define SYNAPTIK_TASK0065_RNG_DROPOUT_KERNELS_H

static NSString *const SynaptikTask0065RngDropoutKernelSource =
@"\n"
"struct RngDropoutMeta0065 { ulong first; ulong second; uint complementBits; uint count; };\n"
"inline ulong mix64_0065(ulong z) { z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ul; z = (z ^ (z >> 27u)) * 0x94d049bb133111ebul; return z ^ (z >> 31u); }\n"
"kernel void initial_state_0065(device ulong *output [[buffer(0)]], constant RngDropoutMeta0065 &m [[buffer(1)]], uint gid [[thread_position_in_grid]]) { if (gid != 0u) return; output[0] = m.first; output[1] = m.second; }\n"
"kernel void dropout_f32_0065(device const float *input [[buffer(0)]], device const ulong *state [[buffer(1)]], device float *output [[buffer(2)]], device uchar *mask [[buffer(3)]], device ulong *nextState [[buffer(4)]], constant RngDropoutMeta0065 &m [[buffer(5)]], uint gid [[thread_position_in_grid]]) { if (gid >= m.count) return; ulong key = state[0]; ulong counter = state[1]; ulong keyOffset = mix64_0065(key + 0x9e3779b97f4a7c15ul); ulong word = mix64_0065(counter + ulong(gid) + keyOffset); bool keep = (word >> 11u) >= m.first; mask[gid] = keep ? uchar(1u) : uchar(0u); if (keep) { float complement = as_type<float>(m.complementBits); float scale = 1.0f / complement; output[gid] = input[gid] * scale; } else { output[gid] = as_type<float>(0u); } if (gid == 0u) { nextState[0] = key; nextState[1] = counter + ulong(m.count); } }\n"
;

#endif
