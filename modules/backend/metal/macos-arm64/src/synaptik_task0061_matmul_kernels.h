#ifndef SYNAPTIK_TASK0061_MATMUL_KERNELS_H
#define SYNAPTIK_TASK0061_MATMUL_KERNELS_H

static NSString *const SynaptikTask0061MatmulKernelSource =
@"\n"
"inline void matmul_coordinates_0061(ulong linear, ulong contraction, constant DataMeta &m,\n"
"        thread ulong *left, thread ulong *right) {\n"
"    ulong out[16]; decode_coords(linear, m.outDims, m.outputRank, out);\n"
"    uint leftBatch = m.inputRank > 1u ? m.inputRank - 2u : 0u;\n"
"    uint rightBatch = m.auxRank > 1u ? m.auxRank - 2u : 0u;\n"
"    uint batchRank = max(leftBatch, rightBatch);\n"
"    uint leftPad = batchRank - leftBatch, rightPad = batchRank - rightBatch;\n"
"    for (uint axis = 0u; axis < leftBatch; ++axis)\n"
"        left[axis] = m.inDims[axis] == 1ul ? 0ul : out[leftPad + axis];\n"
"    for (uint axis = 0u; axis < rightBatch; ++axis)\n"
"        right[axis] = m.auxDims[axis] == 1ul ? 0ul : out[rightPad + axis];\n"
"    if (m.inputRank == 1u) left[0] = contraction;\n"
"    else { left[m.inputRank - 2u] = out[batchRank]; left[m.inputRank - 1u] = contraction; }\n"
"    if (m.auxRank == 1u) right[0] = contraction;\n"
"    else { right[m.auxRank - 2u] = contraction;\n"
"        right[m.auxRank - 1u] = out[batchRank + (m.inputRank > 1u ? 1u : 0u)]; }\n"
"}\n"
"inline void matmul_offsets_0061(ulong linear, ulong contraction, constant DataMeta &m,\n"
"        thread ulong &leftOffset, thread ulong &rightOffset) {\n"
"    ulong left[16], right[16]; matmul_coordinates_0061(linear, contraction, m, left, right);\n"
"    leftOffset = coords_storage(left, m.inputStrides, m.inputRank, m.inputOffset);\n"
"    rightOffset = coords_storage(right, m.auxiliaryStrides, m.auxRank, m.auxiliaryOffset);\n"
"}\n"
"inline float matmul_fma_0061(float left, float right, float accumulator) { return fma(left, right, accumulator); }\n"
"inline float matmul_f32_0061(device const uchar *left, device const uchar *right, ulong linear, constant DataMeta &m, bool leftBfloat, bool rightBfloat) { float accumulator = 0.0f; ulong contractions = m.inputExtents[0]; for (ulong contraction = 0ul; contraction < contractions; ++contraction) { ulong leftOffset, rightOffset; matmul_offsets_0061(linear, contraction, m, leftOffset, rightOffset); uint leftWord = uint(load_word(left, leftOffset, leftBfloat ? 2u : 4u)); uint rightWord = uint(load_word(right, rightOffset, rightBfloat ? 2u : 4u)); float leftValue = as_type<float>(leftBfloat ? leftWord << 16u : leftWord); float rightValue = as_type<float>(rightBfloat ? rightWord << 16u : rightWord); accumulator = matmul_fma_0061(leftValue, rightValue, accumulator); } return accumulator; }\n"
"kernel void matmul_i32_i32_i32_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; uint accumulator = 0u; for (ulong contraction = 0ul; contraction < m.inputExtents[0]; ++contraction) { ulong leftOffset, rightOffset; matmul_offsets_0061(linear, contraction, m, leftOffset, rightOffset); accumulator += uint(load_word(left, leftOffset, 4u)) * uint(load_word(right, rightOffset, 4u)); } store_word(output, linear, 4u, ulong(accumulator)); }\n"
"kernel void matmul_i32_i64_i64_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; ulong accumulator = 0ul; for (ulong contraction = 0ul; contraction < m.inputExtents[0]; ++contraction) { ulong leftOffset, rightOffset; matmul_offsets_0061(linear, contraction, m, leftOffset, rightOffset); ulong leftValue = ulong(long(int(uint(load_word(left, leftOffset, 4u))))); accumulator += leftValue * load_word(right, rightOffset, 8u); } store_word(output, linear, 8u, accumulator); }\n"
"kernel void matmul_i64_i32_i64_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; ulong accumulator = 0ul; for (ulong contraction = 0ul; contraction < m.inputExtents[0]; ++contraction) { ulong leftOffset, rightOffset; matmul_offsets_0061(linear, contraction, m, leftOffset, rightOffset); ulong rightValue = ulong(long(int(uint(load_word(right, rightOffset, 4u))))); accumulator += load_word(left, leftOffset, 8u) * rightValue; } store_word(output, linear, 8u, accumulator); }\n"
"kernel void matmul_i64_i64_i64_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; ulong accumulator = 0ul; for (ulong contraction = 0ul; contraction < m.inputExtents[0]; ++contraction) { ulong leftOffset, rightOffset; matmul_offsets_0061(linear, contraction, m, leftOffset, rightOffset); accumulator += load_word(left, leftOffset, 8u) * load_word(right, rightOffset, 8u); } store_word(output, linear, 8u, accumulator); }\n"
"kernel void matmul_f32_f32_f32_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; store_word(output, linear, 4u, ulong(as_type<uint>(matmul_f32_0061(left, right, linear, m, false, false)))); }\n"
"kernel void matmul_bf16_f32_f32_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; store_word(output, linear, 4u, ulong(as_type<uint>(matmul_f32_0061(left, right, linear, m, true, false)))); }\n"
"kernel void matmul_f32_bf16_f32_0061(device const uchar *left [[buffer(0)]], device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]], constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return; store_word(output, linear, 4u, ulong(as_type<uint>(matmul_f32_0061(left, right, linear, m, false, true)))); }\n";

/*
 * Controlled FLOAT32 terminal contractions reuse the ordinary coordinate proof and
 * DataMeta without changing its layout. Assemble this source after both ordinary
 * matmul source and SynaptikGeneralScanKernelSource. Bindings 0/1/2/3 are left,
 * right, canonical output, DataMeta; 28/29 are scanControl/ScanStepControl. The
 * control POD is owned and statically size-checked by the general-Scan component.
 * The binder authenticates each operand's original logical row axis/inner, scope
 * and complete physical spans. No B=1 projection or arithmetic activity masking:
 * both scope checks dominate BOTH operand loads, even when one operand is SHARED.
 * Separate multiply/add sites follow the represented contraction; no implicit FMA.
 * One output writer, no workspace, ordinary custom_grid geometry, FLOAT32 only.
 */
static NSString *const SynaptikControlledMatmulKernelSource =
@"\n"
"inline float matmul_f32_controlled_0061(device const uchar *left, device const uchar *right,\n"
"        ulong linear, constant DataMeta &m, device const ulong *scanControl,\n"
"        constant ScanStepControl &scanStep) {\n"
"    float accumulator = 0.0f;\n"
"    for (ulong contraction = 0ul; contraction < m.inputExtents[0]; ++contraction) {\n"
"        ulong lc[16], rc[16]; matmul_coordinates_0061(linear, contraction, m, lc, rc);\n"
"        ulong leftLogical = coords_linear(lc, m.inDims, m.inputRank);\n"
"        ulong rightLogical = coords_linear(rc, m.auxDims, m.auxRank);\n"
"        if (!scan_control_operand(scanControl, scanStep, 0u, leftLogical)\n"
"                || !scan_control_operand(scanControl, scanStep, 1u, rightLogical)) continue;\n"
"        ulong lo = coords_storage(lc, m.inputStrides, m.inputRank, m.inputOffset);\n"
"        ulong ro = coords_storage(rc, m.auxiliaryStrides, m.auxRank, m.auxiliaryOffset);\n"
"        float lv = as_type<float>(uint(load_word(left, lo, 4u)));\n"
"        float rv = as_type<float>(uint(load_word(right, ro, 4u)));\n"
"        float product = lv * rv; accumulator = accumulator + product;\n"
"    }\n"
"    return accumulator;\n"
"}\n"
"kernel void matmul_f32_f32_f32_controlled_0061(device const uchar *left [[buffer(0)]],\n"
"        device const uchar *right [[buffer(1)]], device uchar *output [[buffer(2)]],\n"
"        constant DataMeta &m [[buffer(3)]], device const ulong *scanControl [[buffer(28)]],\n"
"        constant ScanStepControl &scanStep [[buffer(29)]], uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return;\n"
"    if (!scan_control_output(scanControl, scanStep, linear)) return;\n"
"    float result = matmul_f32_controlled_0061(left, right, linear, m, scanControl, scanStep);\n"
"    store_word(output, linear, 4u, ulong(as_type<uint>(result)));\n"
"}\n";

#endif
