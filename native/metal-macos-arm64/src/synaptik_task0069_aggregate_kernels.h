#ifndef SYNAPTIK_TASK0069_AGGREGATE_KERNELS_H
#define SYNAPTIK_TASK0069_AGGREGATE_KERNELS_H

static NSString *const SynaptikTask0069AggregateKernelSource =
@"\n"
"inline uint l1_abs_word_0069(uint word) { return word & 0x7fffffffu; }\n"
"inline float l1_add_0069(float accumulator, float contributor) { return accumulator + contributor; }\n"
"kernel void l1_norm_f32_0069(device const uchar *input [[buffer(0)]], device uchar *output [[buffer(1)]], constant DataMeta &m [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear != 0ul || m.elementCount != 1ul || m.inputExtents[0] == 0ul) return; uint firstWord = l1_abs_word_0069(uint(load_word(input, 0ul, 4u))); float accumulator = as_type<float>(firstWord); for (ulong contributor = 1ul; contributor < m.inputExtents[0]; ++contributor) { uint contributorWord = l1_abs_word_0069(uint(load_word(input, contributor, 4u))); accumulator = l1_add_0069(accumulator, as_type<float>(contributorWord)); } store_word(output, 0ul, 4u, ulong(as_type<uint>(accumulator))); }\n";

#endif
