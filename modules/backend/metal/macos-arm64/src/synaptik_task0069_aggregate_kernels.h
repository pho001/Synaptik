#ifndef SYNAPTIK_TASK0069_AGGREGATE_KERNELS_H
#define SYNAPTIK_TASK0069_AGGREGATE_KERNELS_H

static NSString *const SynaptikTask0069AggregateKernelSource =
@"\n"
@"inline float scatter_add_site_0069(float accumulator, float contributor) { return accumulator + contributor; }\n"
@"kernel void scatter_add_f32_0069(device const uchar *data [[buffer(0)]], device const uchar *indices [[buffer(1)]], device const uchar *updates [[buffer(2)]], device uchar *output [[buffer(3)]], constant DataMeta &m [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) { ulong target = data_linear_id(gid, m); if (target >= m.elementCount) return; uint baseWord = uint(load_word(data, target, 4u)); float accumulator = as_type<float>(baseWord); bool addressed = false; for (ulong ordinal = 0ul; ordinal < m.auxDims[0]; ++ordinal) { ulong indexStorage = m.roleOffsets[1] + ordinal * m.roleStrides[1][0]; long selected = read_index(indices, indexStorage, m.indexType); if (selected == long(target)) { accumulator = scatter_add_site_0069(accumulator, as_type<float>(uint(load_word(updates, ordinal, 4u)))); addressed = true; } } uint resultWord = addressed ? as_type<uint>(accumulator) : baseWord; store_word(output, target, 4u, ulong(resultWord)); }\n";

#endif
