#ifndef SYNAPTIK_NORMALIZATION_LOSS_KERNELS_H
#define SYNAPTIK_NORMALIZATION_LOSS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/** FLOAT32-only physical read descriptor. Offsets and strides are in elements. */
typedef struct {
    uint64_t offset;
    uint64_t strides[16];
} SynaptikNormalizationLossView;

/**
 * Prepared, checked metadata for ordinary normalization and categorical losses.
 *
 * mode is the unchanged ordinary operation wire (86..93). reduction is NONE=0,
 * SUM=1, MEAN=2. flags are scale=1, bias=2, ignore-index-present=4; epsilonBits
 * and momentumBits preserve the exact FLOAT32 scalar bits. indexType is the
 * existing carrier wire (INT32=2 or INT64=6), never a floating conversion.
 *
 * dims[0..rank) describes input 0 (rank <= 16). For softmax/CE, axis is the
 * class axis, featureCount=dims[axis], rowCount=product(other dimensions).
 * For LN/RMS, axis starts the normalized suffix, featureCount=product(suffix),
 * rowCount=product(prefix). Their scale/bias descriptors index the suffix,
 * starting at strides[0]. For BN, axis is the channel axis, featureCount=C,
 * rowCount=N=product(nonchannel dimensions), and roles 1..4 are rank-one.
 * Dense CE target strides match logits rank; index CE target strides describe
 * the rank-minus-one nonclass shape. All outputs are canonical and offset zero.
 * elementCount is the checked product of dims, equal to rowCount*featureCount.
 *
 * All kernels bind roles input0..4 at 0..4, output0..4 at 5..9, metadata at 10.
 * Unused bindings can use any valid buffer; they are never read or written.
 * BN training writes all five canonical siblings in one dispatch: normalized,
 * next running mean, next running variance, saved mean, saved inverse stddev.
 * Saved slots 3/4 are not independently differentiable public output roots.
 * Other operations write only output0. Grids: rowCount for normalization_rows,
 * featureCount for batch_norm; CE NONE rowCount, CE SUM/MEAN exactly one thread.
 * Empty norm/BN outputs need no dispatch; reduced CE must dispatch even at zero
 * rows, publishing +0 for SUM and NaN for MEAN. No scratch/workspace is used.
 *
 * The owner must validate shapes, typed attrs, supported flags, checked address
 * ranges, nonaliasing outputs, FLOAT32 roles/results, exact canonical sibling
 * identity, and BN training C==0 || N>=2 before encoding. Index CE must validate
 * all nonignored targets against [0,C) BEFORE ANY output mutation, comparing
 * the ignore value in the original INT32/INT64 domain before bounds checking.
 * Dense target validity remains the Model caller obligation. Each bound
 * buffer must cover its actual affine read range; this header grants no low,
 * FLOAT64 arithmetic, gradient, or public-symbolic admission by itself.
 *
 * Assemble after SynaptikFloat32MathKernelSource and SynaptikGeneralScanKernelSource.
 * Calls synaptik_f32_exp/log/sqrt and scan_control_output/output_at/any_output;
 * BN siblings use each output's authenticated row geometry and demand, not
 * output zero as a surrogate. Nonempty mean sums seed the first contributor
 * so intermediate means preserve their contributor-derived zero sign. The ordinary and
 * _scan entry points share numeric bodies, with optional control at 28/29.
 * Scan admission must prove B outside normalized/class axes, BN training B at
 * channel, reduced CE SHARED; NONE CE uses its nonclass output row axis.
 * Disable fast-math contraction/reassociation
 * as for existing named-site kernels. Statistics are computed once per slice,
 * then reused for every feature; reduced CE visits every row once, not once per
 * output element. The serial reduced CE avoids an undeclared intermediate.
 */
typedef struct {
    uint32_t mode, rank, axis, reduction, flags, indexType, epsilonBits, momentumBits;
    uint64_t rowCount, featureCount, elementCount;
    int64_t ignoreIndex;
    uint64_t dims[16];
    SynaptikNormalizationLossView views[5];
} SynaptikNormalizationLossMeta;

_Static_assert(sizeof(SynaptikNormalizationLossView) == 136, "normalization view ABI");
_Static_assert(sizeof(SynaptikNormalizationLossMeta) == 872, "normalization metadata ABI");
_Static_assert(offsetof(SynaptikNormalizationLossMeta, dims) == 64, "normalization dimensions ABI");
_Static_assert(offsetof(SynaptikNormalizationLossMeta, views) == 192, "normalization views ABI");

static NSString *const SynaptikNormalizationLossKernelSource =
@"\n"
"struct NormalizationLossView { ulong offset; ulong strides[16]; };\n"
"struct NormalizationLossMeta { uint mode, rank, axis, reduction, flags, indexType, epsilonBits, momentumBits; ulong rowCount, featureCount, elementCount; long ignoreIndex; ulong dims[16]; NormalizationLossView views[5]; };\n"
"inline float nl_nan() { return as_type<float>(0x7fc00000u); }\n"
"inline ulong nl_flat_offset(ulong linear, constant NormalizationLossMeta &m, uint role, uint first) { ulong offset = m.views[role].offset; for (uint a = m.rank; a-- > first;) { ulong c = linear % m.dims[a]; linear /= m.dims[a]; offset += c * m.views[role].strides[a-first]; } return offset; }\n"
"inline ulong nl_axis_offset(ulong row, ulong feature, constant NormalizationLossMeta &m, uint role, bool omitAxis) { ulong offset = m.views[role].offset; for (uint a = m.rank; a-- > 0u;) { if (a == m.axis) { if (!omitAxis) offset += feature * m.views[role].strides[a]; } else { ulong c = row % m.dims[a]; row /= m.dims[a]; uint s = omitAxis && a > m.axis ? a-1u : a; offset += c * m.views[role].strides[s]; } } return offset; }\n"
"inline ulong nl_axis_output(ulong row, ulong feature, constant NormalizationLossMeta &m) { ulong result = 0ul, stride = 1ul; for (uint a = m.rank; a-- > 0u;) { ulong c; if (a == m.axis) c = feature; else { c = row % m.dims[a]; row /= m.dims[a]; } result += c * stride; stride *= m.dims[a]; } return result; }\n"
"inline float nl_max(float a, float b) { return b > a || (b == 0.0f && a == 0.0f && !signbit(b)) ? b : a; }\n"
"inline void nl_normalization_rows(device const float *input, device const float *scale, device const float *bias, device float *output, constant NormalizationLossMeta &m, ulong row, device const ulong *scanControl, constant ScanStepControl *scanStep) {\n"
"  if (row >= m.rowCount || m.featureCount == 0ul) return;\n"
"  bool soft = m.mode == 92u || m.mode == 93u;\n"
"  if (scanControl != nullptr && !scan_control_output(scanControl,*scanStep,soft ? nl_axis_output(row,0ul,m) : row*m.featureCount)) return;\n"
"  float sum = 0.0f, maximum = -INFINITY, first = 0.0f; bool nan = false, infinity = false, positiveInfinity = false, constantSlice = true;\n"
"  for (ulong f = 0ul; f < m.featureCount; ++f) {\n"
"    ulong p = soft ? nl_axis_offset(row, f, m, 0u, false) : nl_flat_offset(row*m.featureCount+f, m, 0u, 0u);\n"
"    float x = input[p]; if (f == 0ul) first = x; else constantSlice = constantSlice && x == first;\n"
"    nan = nan || isnan(x); infinity = infinity || isinf(x); positiveInfinity = positiveInfinity || x == INFINITY;\n"
"    if (soft) maximum = nl_max(maximum, x); else { float contribution = m.mode == 91u ? x*x : x; sum = f == 0ul ? contribution : sum+contribution; }\n"
"  }\n"
"  float mean = 0.0f, denominator = 0.0f;\n"
"  if (soft) {\n"
"    // Negative-infinity-containing slices deliberately keep the unshifted rule.\n"
"    if (!nan && !positiveInfinity) for (ulong f = 0ul; f < m.featureCount; ++f) { float x = input[nl_axis_offset(row,f,m,0u,false)]; sum = sum + synaptik_f32_exp(infinity ? x : x-maximum); }\n"
"    if (m.mode == 93u) denominator = synaptik_f32_log(sum);\n"
"  } else {\n"
"    if (m.mode == 90u) {\n"
"      mean = sum / float(m.featureCount); sum = 0.0f;\n"
"      for (ulong f = 0ul; f < m.featureCount; ++f) { float delta = input[nl_flat_offset(row*m.featureCount+f,m,0u,0u)] - mean; sum = sum + delta*delta; }\n"
"    }\n"
"    denominator = synaptik_f32_sqrt(sum / float(m.featureCount) + as_type<float>(m.epsilonBits));\n"
"  }\n"
"  for (ulong f = 0ul; f < m.featureCount; ++f) {\n"
"    ulong p = soft ? nl_axis_offset(row,f,m,0u,false) : nl_flat_offset(row*m.featureCount+f,m,0u,0u); float x = input[p], y;\n"
"    if (soft) {\n"
"      if (nan) y = nl_nan();\n"
"      else if (positiveInfinity) y = x == INFINITY ? nl_nan() : (m.mode == 92u ? 0.0f : -INFINITY);\n"
"      else { float shift = infinity ? x : x-maximum; y = m.mode == 92u ? synaptik_f32_exp(shift)/sum : shift-denominator; }\n"
"    } else {\n"
"      if (m.mode == 90u) y = nan || infinity ? nl_nan() : (constantSlice ? 0.0f : (x-mean)/denominator);\n"
"      else y = x/denominator;\n"
"      if ((m.flags & 1u) != 0u) y = y * scale[nl_flat_offset(f,m,1u,m.axis)];\n"
"      if ((m.flags & 2u) != 0u) y = y + bias[nl_flat_offset(f,m,2u,m.axis)];\n"
"    }\n"
"    output[soft ? nl_axis_output(row,f,m) : row*m.featureCount+f] = y;\n"
"  }\n"
"}\n"
"inline void nl_batch_norm(device const float *input, device const float *scale, device const float *bias, device const float *runningMean, device const float *runningVariance, device float *output, device float *nextMean, device float *nextVariance, device float *savedMean, device float *savedInvStd, constant NormalizationLossMeta &m, ulong channel, device const ulong *scanControl, constant ScanStepControl *scanStep) {\n"
"  if (channel >= m.featureCount) return;\n"
"  bool training = m.mode == 89u;\n"
"  bool wantValue = true, wantNextMean = training, wantNextVariance = training, wantSavedMean = training, wantSavedInv = training;\n"
"  if (scanControl != nullptr) {\n"
"    if (!scan_control_any_output(scanControl,*scanStep)) return;\n"
"    if (training || scanStep->rowAxis == int(m.axis)) wantValue = scan_control_output_at(scanControl,*scanStep,0u,nl_axis_output(0ul,channel,m));\n"
"    if (training) {\n"
"      wantNextMean = scan_control_output_at(scanControl,*scanStep,1u,channel);\n"
"      wantNextVariance = scan_control_output_at(scanControl,*scanStep,2u,channel);\n"
"      wantSavedMean = scan_control_output_at(scanControl,*scanStep,3u,channel);\n"
"      wantSavedInv = scan_control_output_at(scanControl,*scanStep,4u,channel);\n"
"    }\n"
"  }\n"
"  if (!(wantValue || wantNextMean || wantNextVariance || wantSavedMean || wantSavedInv)) return;\n"
"  float mean = 0.0f, variance = 0.0f, squared = 0.0f;\n"
"  if (training) {\n"
"    // Nonempty means start with the first actual contributor, preserving -0.\n"
"    float sum = input[nl_axis_offset(0ul,channel,m,0u,false)];\n"
"    for (ulong r = 1ul; r < m.rowCount; ++r) sum = sum + input[nl_axis_offset(r,channel,m,0u,false)];\n"
"    mean = sum / float(m.rowCount);\n"
"    if (wantValue || wantNextVariance || wantSavedInv) {\n"
"      for (ulong r = 0ul; r < m.rowCount; ++r) { float delta = input[nl_axis_offset(r,channel,m,0u,false)] - mean; squared = squared + delta*delta; }\n"
"      variance = squared / float(m.rowCount);\n"
"    }\n"
"  } else {\n"
"    mean = runningMean[m.views[3].offset+channel*m.views[3].strides[0]];\n"
"    variance = runningVariance[m.views[4].offset+channel*m.views[4].strides[0]];\n"
"  }\n"
"  float root = wantValue || wantSavedInv ? synaptik_f32_sqrt(variance + as_type<float>(m.epsilonBits)) : 0.0f;\n"
"  if (wantValue) {\n"
"    float gamma = scale[m.views[1].offset+channel*m.views[1].strides[0]], beta = bias[m.views[2].offset+channel*m.views[2].strides[0]];\n"
"    for (ulong r = 0ul; r < m.rowCount; ++r) {\n"
"      ulong linear = nl_axis_output(r,channel,m);\n"
"      if (scanControl != nullptr && !scan_control_output_at(scanControl,*scanStep,0u,linear)) continue;\n"
"      float centered = input[nl_axis_offset(r,channel,m,0u,false)]-mean; float normalized = centered/root; float affine = normalized*gamma; output[linear] = affine+beta;\n"
"    }\n"
"  }\n"
"  if (wantNextMean || wantNextVariance) {\n"
"    float momentum = as_type<float>(m.momentumBits), retain = 1.0f-momentum;\n"
"    if (wantNextMean) {\n"
"      float oldMean = runningMean[m.views[3].offset+channel*m.views[3].strides[0]];\n"
"      float retainedMean = retain*oldMean, observedMean = momentum*mean;\n"
"      nextMean[channel] = retainedMean+observedMean;\n"
"    }\n"
"    if (wantNextVariance) {\n"
"      float oldVariance = runningVariance[m.views[4].offset+channel*m.views[4].strides[0]];\n"
"      float unbiased = squared/float(m.rowCount-1ul);\n"
"      float retainedVariance = retain*oldVariance, observedVariance = momentum*unbiased;\n"
"      nextVariance[channel] = retainedVariance+observedVariance;\n"
"    }\n"
"  }\n"
"  if (wantSavedMean) savedMean[channel] = mean;\n"
"  if (wantSavedInv) savedInvStd[channel] = 1.0f/root;\n"
"}\n"
"inline long nl_index(device const uchar *target, ulong address, uint type) { return type == 2u ? long(((device const int *)target)[address]) : ((device const long *)target)[address]; }\n"
"inline void nl_categorical_loss(device const float *logits, device const uchar *target, device float *output, constant NormalizationLossMeta &m, uint gid, device const ulong *scanControl, constant ScanStepControl *scanStep) {\n"
"  ulong firstRow = m.reduction == 0u ? ulong(gid) : 0ul; if (m.reduction == 0u ? firstRow >= m.rowCount : gid != 0u) return;\n"
"  if (scanControl != nullptr && !scan_control_output(scanControl,*scanStep,firstRow)) return;\n"
"  ulong lastRow = m.reduction == 0u ? firstRow+1ul : m.rowCount, included = 0ul; float reduced = 0.0f;\n"
"  for (ulong row = firstRow; row < lastRow; ++row) {\n"
"    long selected = 0l;\n"
"    if (m.mode == 87u) {\n"
"      selected = nl_index(target,nl_axis_offset(row,0ul,m,1u,true),m.indexType);\n"
"      if ((m.flags & 4u) != 0u && selected == m.ignoreIndex) { if (m.reduction == 0u) output[row] = 0.0f; continue; }\n"
"    }\n"
"    ++included; float maximum = -INFINITY; bool nan = false;\n"
"    for (ulong c = 0ul; c < m.featureCount; ++c) { float z = logits[nl_axis_offset(row,c,m,0u,false)]; nan = nan || isnan(z); maximum = nl_max(maximum,z); }\n"
"    float sum = 0.0f;\n"
"    for (ulong c = 0ul; c < m.featureCount; ++c) { float shift = logits[nl_axis_offset(row,c,m,0u,false)]-maximum; sum = sum+synaptik_f32_exp(shift); }\n"
"    float lse = nan ? nl_nan() : maximum+synaptik_f32_log(sum), loss = 0.0f;\n"
"    if (m.mode == 86u) {\n"
"      for (ulong c = 0ul; c < m.featureCount; ++c) {\n"
"        float weight = ((device const float *)target)[nl_axis_offset(row,c,m,1u,false)];\n"
"        // The exact zero-target guard precedes subtraction and multiplication.\n"
"        if (weight != 0.0f) { float difference = lse-logits[nl_axis_offset(row,c,m,0u,false)]; float contribution = weight*difference; loss = loss+contribution; }\n"
"      }\n"
"    } else loss = lse-logits[nl_axis_offset(row,ulong(selected),m,0u,false)];\n"
"    if (m.reduction == 0u) output[row] = loss; else reduced = reduced+loss;\n"
"  }\n"
"  if (m.reduction != 0u) output[0] = m.reduction == 2u ? reduced/float(included) : reduced;\n"
"}\n"
"#define SYNAPTIK_NL_BINDINGS device const float *input0 [[buffer(0)]], device const float *input1 [[buffer(1)]], device const float *input2 [[buffer(2)]], device const float *input3 [[buffer(3)]], device const float *input4 [[buffer(4)]], device float *output0 [[buffer(5)]], device float *output1 [[buffer(6)]], device float *output2 [[buffer(7)]], device float *output3 [[buffer(8)]], device float *output4 [[buffer(9)]], constant NormalizationLossMeta &m [[buffer(10)]], uint gid [[thread_position_in_grid]]\n"
"#define SYNAPTIK_NL_SCAN_BINDINGS SYNAPTIK_NL_BINDINGS, device const ulong *scanControl [[buffer(28)]], constant ScanStepControl &scanStep [[buffer(29)]]\n"
"kernel void normalization_rows_f32(SYNAPTIK_NL_BINDINGS) { nl_normalization_rows(input0,input1,input2,output0,m,ulong(gid),nullptr,nullptr); }\n"
"kernel void normalization_rows_f32_scan(SYNAPTIK_NL_SCAN_BINDINGS) { nl_normalization_rows(input0,input1,input2,output0,m,ulong(gid),scanControl,&scanStep); }\n"
"kernel void batch_norm_f32(SYNAPTIK_NL_BINDINGS) { nl_batch_norm(input0,input1,input2,input3,input4,output0,output1,output2,output3,output4,m,ulong(gid),nullptr,nullptr); }\n"
"kernel void batch_norm_f32_scan(SYNAPTIK_NL_SCAN_BINDINGS) { nl_batch_norm(input0,input1,input2,input3,input4,output0,output1,output2,output3,output4,m,ulong(gid),scanControl,&scanStep); }\n"
"kernel void categorical_loss_f32(SYNAPTIK_NL_BINDINGS) { nl_categorical_loss(input0,(device const uchar *)input1,output0,m,gid,nullptr,nullptr); }\n"
"kernel void categorical_loss_f32_scan(SYNAPTIK_NL_SCAN_BINDINGS) { nl_categorical_loss(input0,(device const uchar *)input1,output0,m,gid,scanControl,&scanStep); }\n"
"#undef SYNAPTIK_NL_SCAN_BINDINGS\n"
"#undef SYNAPTIK_NL_BINDINGS\n";

#endif
