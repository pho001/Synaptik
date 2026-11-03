#ifndef SYNAPTIK_GENERAL_SCAN_PLAN_H
#define SYNAPTIK_GENERAL_SCAN_PLAN_H

/* Included in the existing executable implementation after its ordinary metadata/build/encode
 * owners. No exported entry point, alternate ordinary decoder, or Runtime body executor is added.
 * This decoder consumes exactly MetalScanEncoding v1 attribute words in the authenticated
 * enclosing program schema. All mutable local/control storage is a declared invocation workspace region.
 */
#include "synaptik_general_scan_binding_kernels.h"

typedef struct {
    uint64_t elements, bytes, offset, rowInner;
    uint64_t dimensions[16], strides[16];
    uint32_t rank, type, width, gradient, terminal;
    int32_t rowAxis;
} SynaptikMetalScanValueBinding;

typedef struct {
    SynaptikMetalDecodedNode node;
    SynaptikMetalScanStepControl control;
    int32_t accumulator;
    uint32_t contraction;
    uint32_t firstStage, stageCount, semanticScatterReduction;
    uint64_t scratchOffset, scratchBytes;
} SynaptikMetalScanUnitBinding;

@interface SynaptikMetalScanPlan : NSObject
@property(nonatomic, strong) NSData *words;
@property(nonatomic, strong) NSData *values;
@property(nonatomic, strong) NSData *units;
@property(nonatomic, strong) NSData *inputReferences;
@property(nonatomic, strong) NSData *valueScopes;
@property(nonatomic, strong) NSData *carryOffsets;
@property(nonatomic, strong) NSData *carryPromoted;
@property(nonatomic, strong) NSData *scopeMemberships;
@property(nonatomic, strong) NSArray<SynaptikMetalProgramStep *> *steps;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *stepControls;
@property(nonatomic, strong) id<MTLComputePipelineState> bindingPipeline;
@end
@implementation SynaptikMetalScanPlan
@end

static BOOL synaptik_scan_add(uint64_t a, uint64_t b, uint64_t *result) {
    if (a > UINT64_MAX - b) return NO;
    *result = a + b; return YES;
}
static BOOL synaptik_scan_multiply(uint64_t a, uint64_t b, uint64_t *result) {
    if (a != 0U && b > UINT64_MAX / a) return NO;
    *result = a * b; return YES;
}
static uint32_t synaptik_scan_width(uint64_t type) {
    switch (type) { case 1U: case 2U: return 4U; case 5U: case 7U: return 2U; case 6U: return 8U; case 3U: return 1U; default: return 0U; }
}
static BOOL synaptik_scan_differentiable_type(uint64_t type) { return type == 1U || type == 5U || type == 7U; }
static BOOL synaptik_scan_scope(uint64_t scope, uint64_t count) {
    return scope == UINT64_MAX || scope < count;
}

/* The plan validates the ordinary contract before appending its complete controlled stage
 * sequence. These private hooks never re-enter Scan lowering and are not callbacks retained
 * in the plan or public/native ABI exports.
 */
static BOOL synaptik_append_scan_ordinary_steps(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        id<MTLDevice> device, id<MTLLibrary> library, id<MTLLibrary> lowLibrary,
        const SynaptikMetalScanStepControl *control, const int32_t *allValueRowAxes,
        const uint64_t *allValueRowInner, uint32_t semanticScatterReduction,
        NSMutableArray<SynaptikMetalProgramStep *> *sink);
static int32_t synaptik_encode_scan_step(SynaptikMetalProgramStep *step,
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets);
static int32_t synaptik_validate_scan_step(SynaptikMetalProgramStep *step,
        SynaptikMetalDecodedNode node, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets,
        const uint64_t *control, const SynaptikMetalScanStepControl *stepControl);
static BOOL synaptik_decode_scan_ordinary_attributes(SynaptikMetalDecodedNode *node,
        const uint8_t *rawWords, uint32_t rawWordCount, const uint8_t *types,
        uint32_t *semanticScatterReduction);
static BOOL synaptik_scan_validate_gradient_flags(SynaptikMetalDecodedNode node,
        const uint8_t *allTypes, const uint8_t *allValueGradFlags);
static BOOL synaptik_scan_validate_ordinary(SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const uint8_t *types, const uint64_t *strides, const uint64_t *offsets, const uint8_t *states);
static BOOL synaptik_scan_prove_coordinates(SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const int32_t *rows, const uint8_t *terminals, uint64_t batch);

static SynaptikMetalScanPlan *synaptik_decode_scan_plan(const uint8_t *bytes, uint32_t wordCount,
        id<MTLDevice> device, id<MTLLibrary> library, id<MTLLibrary> lowLibrary) {
    if (bytes == NULL || wordCount < 32U || wordCount > (1U << 20U) || device == nil || library == nil) return nil;
    NSMutableData *wordData = [NSMutableData dataWithLength:(NSUInteger)wordCount * 8U];
    if (wordData == nil) return nil;
    uint64_t *w = wordData.mutableBytes;
    for (uint32_t i = 0U; i < wordCount; i++) w[i] = synaptik_read_le64(bytes + (size_t)i * 8U);
    if (w[0] != UINT64_C(0x5343414e504c414e) || w[1] != 1U || w[2] != wordCount || w[29] != wordCount
            || w[3] > INT64_MAX || (w[4] & ~7U) != 0U || w[5] > 4096U || w[6] == 0U || w[6] > 4096U
            || w[7] > 4096U || w[8] > 4096U || w[9] > 4096U || w[10] > w[5] || w[11] > INT64_MAX
            || w[12] == 0U || w[12] > 4096U || w[13] > 512U || w[14] != w[5] + w[6] + w[7]
            || w[15] != w[5] + w[8] + w[9] || w[14] > 4096U || w[15] > 4096U || w[5] + w[8] == 0U || w[16] > w[15]
            || w[17] != w[19] + w[20] || w[17] > w[14] || w[18] > 4608U
            || w[19] + w[21] > w[5] || w[23] != 32U || (w[22] & 7U) != 0U
            || (w[30] & 7U) != 0U || w[30] > w[22] || w[31] < 10U
            || w[22] > NSUIntegerMax || w[31] > (NSUIntegerMax - w[30]) / 8U
            || w[22] != w[30] + w[31] * 8U) return nil;
    for (uint32_t i = 23U; i < 29U; i++) if (w[i] > w[i + 1U] || w[i + 1U] > wordCount) return nil;
    BOOL lengthAware = (w[4] & 2U) != 0U, activity = (w[4] & 4U) != 0U;
    if ((!lengthAware && (w[10] != 0U || w[11] != 0U))
            || (!activity && (w[17] != 0U || w[18] != 0U || w[19] != 0U || w[20] != 0U || w[21] != 0U))) return nil;
    uint32_t valueCount = (uint32_t)w[12], unitCount = (uint32_t)w[13];
    NSMutableData *valueData = [NSMutableData dataWithLength:(NSUInteger)valueCount * sizeof(SynaptikMetalScanValueBinding)];
    NSMutableData *unitData = [NSMutableData dataWithLength:(NSUInteger)unitCount * sizeof(SynaptikMetalScanUnitBinding)];
    NSMutableData *scopeData = [NSMutableData dataWithLength:(NSUInteger)valueCount * sizeof(uint32_t)];
    NSMutableData *availabilityData = [NSMutableData dataWithLength:valueCount];
    NSMutableData *typeData = [NSMutableData dataWithLength:valueCount];
    NSMutableData *stateData = [NSMutableData dataWithLength:valueCount];
    NSMutableData *gradientData = [NSMutableData dataWithLength:valueCount];
    NSMutableData *terminalData = [NSMutableData dataWithLength:valueCount];
    NSMutableData *strideData = [NSMutableData dataWithLength:(NSUInteger)valueCount * 16U * sizeof(uint64_t)];
    NSMutableData *offsetData = [NSMutableData dataWithLength:(NSUInteger)valueCount * sizeof(uint64_t)];
    NSMutableData *rowAxesData = [NSMutableData dataWithLength:(NSUInteger)valueCount * sizeof(int32_t)];
    NSMutableData *rowInnerData = [NSMutableData dataWithLength:(NSUInteger)valueCount * sizeof(uint64_t)];
    if (valueData == nil || unitData == nil || scopeData == nil || availabilityData == nil || typeData == nil
            || stateData == nil || gradientData == nil || terminalData == nil || strideData == nil || offsetData == nil
            || rowAxesData == nil || rowInnerData == nil) return nil;
    SynaptikMetalScanValueBinding *values = valueData.mutableBytes;
    SynaptikMetalScanUnitBinding *units = unitData.mutableBytes;
    uint32_t *valueScopes = scopeData.mutableBytes;
    uint8_t *available = availabilityData.mutableBytes, *types = typeData.mutableBytes, *states = stateData.mutableBytes;
    uint8_t *gradients = gradientData.mutableBytes;
    uint8_t *allValueTerminals = terminalData.mutableBytes;
    uint64_t *strides = strideData.mutableBytes, *offsets = offsetData.mutableBytes;
    int32_t *allValueRowAxes = rowAxesData.mutableBytes;
    uint64_t *allValueRowInner = rowInnerData.mutableBytes;
    memset(valueScopes, 0xff, (size_t)valueCount * sizeof(uint32_t));
    NSMutableArray<MPSShape *> *shapes = [NSMutableArray arrayWithCapacity:valueCount];
    uint64_t cursor = w[23], allocatedEnd = 0U;
    for (uint32_t v = 0U; v < valueCount; v++) {
        if (cursor > w[24] || w[24] - cursor < 13U) return nil;
        uint64_t *r = w + cursor, rank = r[2];
        if (rank > 16U || r[0] != 13U + 2U * rank || r[0] > w[24] - cursor
                || synaptik_scan_width(r[1]) == 0U || r[3] > 1U || (r[3] != 0U && !synaptik_scan_differentiable_type(r[1]))
                || r[4] != 1U || r[5] != 0U || r[10] > 1U || r[12] != 0U) return nil;
        int64_t row = (int64_t)r[9];
        if (row < -1 || row >= (int64_t)rank || (!lengthAware && (row != -1 || r[10] != 0U))
                || (row >= 0 && r[10] != 0U)) return nil;
        SynaptikMetalScanValueBinding value = { .rank = (uint32_t)rank, .type = (uint32_t)r[1],
                .width = synaptik_scan_width(r[1]), .gradient = (uint32_t)r[3], .terminal = (uint32_t)r[10],
                .rowAxis = (int32_t)row, .offset = r[7], .bytes = r[8], .rowInner = r[11] };
        NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        uint64_t product = 1U, inner = row < 0 ? 0U : 1U, stride = 1U;
        for (uint32_t a = 0U; a < rank; a++) {
            uint64_t dim = r[13U + a];
            if (dim > INT64_MAX || !synaptik_scan_multiply(product, dim, &product)) return nil;
            value.dimensions[a] = dim; [shape addObject:@(dim)];
            if ((int64_t)a > row && row >= 0 && !synaptik_scan_multiply(inner, dim, &inner)) return nil;
        }
        for (uint32_t reverse = (uint32_t)rank; reverse > 0U; reverse--) {
            uint32_t a = reverse - 1U;
            if (r[13U + rank + a] != stride) return nil;
            value.strides[a] = stride; strides[(size_t)v * 16U + a] = stride;
            if (!synaptik_scan_multiply(stride, value.dimensions[a], &stride)) return nil;
        }
        uint64_t byteCount;
        if (!synaptik_scan_multiply(product, value.width, &byteCount) || r[6] != product || byteCount != value.bytes
                || inner != value.rowInner || (row >= 0 && value.dimensions[row] != w[11])) return nil;
        value.elements = product;
        if (value.offset != UINT64_MAX) {
            if ((value.offset & 7U) != 0U || allocatedEnd > UINT64_MAX - 7U
                    || value.offset != ((allocatedEnd + 7U) & ~UINT64_C(7)) || value.offset > w[30]
                    || value.bytes > w[30] - value.offset) return nil;
            allocatedEnd = value.offset + value.bytes;
        }
        values[v] = value; types[v] = (uint8_t)value.type; states[v] = 1U; offsets[v] = 0U;
        gradients[v] = (uint8_t)value.gradient;
        allValueTerminals[v] = (uint8_t)value.terminal;
        allValueRowAxes[v] = value.rowAxis; allValueRowInner[v] = value.rowInner;
        [shapes addObject:[shape copy]]; cursor += r[0];
    }
    if (cursor != w[24] || w[25] + w[14] != w[26] || w[26] + w[15] != w[27]
            || w[27] + w[16] != w[28] || ((allocatedEnd + 7U) & ~UINT64_C(7)) > w[30]) return nil;
    for (uint32_t f = 0U; f < w[14]; f++) {
        uint64_t v = w[w[25] + f];
        if (v >= valueCount || available[v] || values[v].terminal
                || (f < w[5] ? values[v].offset == UINT64_MAX : values[v].offset != UINT64_MAX)) return nil;
        if (lengthAware && ((f < w[10] && values[v].rowAxis != 0)
                || (f >= w[10] && f < w[5] && values[v].rowAxis != -1)
                || (f >= w[5] + w[6] && values[v].rowAxis != -1))) return nil;
        available[v] = 1U;
    }
    NSMutableData *carryOffsetsData = [NSMutableData dataWithLength:(NSUInteger)w[5] * sizeof(uint64_t)];
    NSMutableData *carryPromotedData = [NSMutableData dataWithLength:(NSUInteger)w[5]];
    if (carryOffsetsData == nil || carryPromotedData == nil) return nil;
    uint64_t *carryOffsets = carryOffsetsData.mutableBytes;
    uint8_t *carryPromoted = carryPromotedData.mutableBytes;
    for (uint32_t c = 0U; c < w[5]; c++) {
        uint64_t next = w[w[26] + c]; if (next >= valueCount) return nil;
        BOOL unique = YES;
        for (uint32_t other = 0U; other < w[5]; other++) if (other != c && w[w[26] + other] == next) unique = NO;
        if (!available[next] && unique && values[next].offset != UINT64_MAX) {
            carryOffsets[c] = values[next].offset; carryPromoted[c] = 1U;
        } else {
            if (allocatedEnd > UINT64_MAX - 7U) return nil;
            allocatedEnd = (allocatedEnd + 7U) & ~UINT64_C(7); carryOffsets[c] = allocatedEnd;
            if (!synaptik_scan_add(allocatedEnd, values[w[w[25] + c]].bytes, &allocatedEnd) || allocatedEnd > w[30]) return nil;
        }
    }
    uint64_t previousDemand = UINT64_MAX;
    for (uint32_t d = 0U; d < w[16]; d++) {
        uint64_t port = w[w[27] + d];
        if (port >= w[15] || (d != 0U && port <= previousDemand)) return nil;
        previousDemand = port;
    }
    /* Decode variable input vectors once into aligned, plan-owned storage. Nodes borrow
     * these immutable references during preparation and execution, never a stack/inline copy. */
    uint64_t inputReferenceCount = 0U, referenceCursor = w[24];
    for (uint32_t u = 0U; u < unitCount; u++) {
        if (referenceCursor > w[25] || w[25] - referenceCursor < 14U) return nil;
        const uint64_t *record = w + referenceCursor;
        if (record[0] < 14U || record[0] > w[25] - referenceCursor
                || record[3] > record[0] - 14U
                || !synaptik_scan_add(inputReferenceCount, record[3], &inputReferenceCount)) return nil;
        referenceCursor += record[0];
    }
    if (referenceCursor != w[25] || inputReferenceCount > NSUIntegerMax / sizeof(uint32_t)) return nil;
    NSMutableData *inputReferenceData = [NSMutableData dataWithLength:(NSUInteger)inputReferenceCount * sizeof(uint32_t)];
    if (inputReferenceData == nil) return nil;
    uint32_t *inputReferences = inputReferenceData.mutableBytes;
    uint64_t inputReferenceOffset = 0U;
    NSMutableArray<SynaptikMetalProgramStep *> *steps = [NSMutableArray arrayWithCapacity:(NSUInteger)unitCount * 2U];
    NSMutableArray<id<MTLBuffer>> *controls = [NSMutableArray arrayWithCapacity:unitCount];
    cursor = w[24];
    for (uint32_t u = 0U; u < unitCount; u++) {
        if (cursor > w[25] || w[25] - cursor < 14U) return nil;
        uint64_t *r = w + cursor;
        if (r[0] < 14U || r[0] > w[25] - cursor || r[0] != 14U + r[3] + r[4] + r[5]
                || r[1] < 1U || r[1] > 115U
                || ((r[1] != 77U && r[1] != 78U) && r[3] > SYNAPTIK_MAX_NODE_INPUTS) || r[4] == 0U
                || r[4] > SYNAPTIK_MAX_NODE_OUTPUTS || r[5] > SYNAPTIK_MAX_ATTRIBUTE_WORDS
                || r[3] < SYNAPTIK_MIN_INPUTS[r[1]]
                || ((r[1] != 77U && r[1] != 78U) && r[3] > SYNAPTIK_MAX_INPUTS[r[1]])
                || r[4] < SYNAPTIK_MIN_OUTPUTS[r[1]] || r[4] > SYNAPTIK_MAX_OUTPUTS[r[1]]
                || !synaptik_scan_scope(r[6], w[18]) || !synaptik_scan_scope(r[7], w[18])
                || !synaptik_scan_scope(r[8], w[18]) || ((r[7] == UINT64_MAX) != (r[8] == UINT64_MAX))
                || r[10] > 1U || (r[11] != UINT64_MAX && r[11] >= w[5])) return nil;
        if ((activity && r[6] == UINT64_MAX) || (!activity
                && (r[6] != UINT64_MAX || r[7] != UINT64_MAX || r[8] != UINT64_MAX || r[11] != UINT64_MAX))) return nil;
        SynaptikMetalScanUnitBinding unit = {0};
        unit.node.operation = (uint32_t)r[1]; unit.node.attribute_kind = (uint32_t)r[2];
        unit.node.input_count = (uint32_t)r[3]; unit.node.output_count = (uint32_t)r[4]; unit.node.attribute_count = (uint32_t)r[5];
        unit.control.scope = (uint32_t)r[6]; unit.control.leftScope = (uint32_t)r[7]; unit.control.rightScope = (uint32_t)r[8];
        unit.control.rowAxis = (int32_t)r[9]; unit.contraction = (uint32_t)r[10]; unit.accumulator = (int32_t)r[11];
        unit.scratchOffset = r[12]; unit.scratchBytes = r[13];
        if (unit.scratchBytes == 0U) {
            if (unit.scratchOffset != UINT64_MAX) return nil;
        } else {
            if ((unit.scratchOffset & 7U) != 0U || allocatedEnd > UINT64_MAX - 7U
                    || unit.scratchOffset != ((allocatedEnd + 7U) & ~UINT64_C(7))
                    || unit.scratchOffset > w[30] || unit.scratchBytes > w[30] - unit.scratchOffset) return nil;
            allocatedEnd = unit.scratchOffset + unit.scratchBytes;
        }
        for (uint32_t role = 0U; role < 4U; role++) unit.control.inputRowAxes[role] = -1;
        unit.node.inputs = inputReferences + inputReferenceOffset;
        for (uint32_t i = 0U; i < unit.node.input_count; i++) {
            uint64_t v = r[14U + i]; if (v >= valueCount || !available[v]) return nil;
            inputReferences[inputReferenceOffset + i] = (uint32_t)v;
            if (i < 4U) { unit.control.inputRowAxes[i] = values[v].rowAxis; unit.control.inputRowInner[i] = values[v].rowInner; }
        }
        inputReferenceOffset += unit.node.input_count;
        for (uint32_t o = 0U; o < unit.node.output_count; o++) {
            uint64_t v = r[14U + unit.node.input_count + o];
            if (v >= valueCount || available[v] || values[v].offset == UINT64_MAX) return nil;
            unit.node.outputs[o] = (uint32_t)v; available[v] = 1U; valueScopes[v] = unit.control.scope;
            unit.control.outputRowAxes[o] = values[v].rowAxis;
            unit.control.outputScopes[o] = unit.control.scope;
            unit.control.outputRowInner[o] = values[v].rowInner;
            unit.control.outputElements[o] = values[v].elements;
        }
        unit.control.outputCount = unit.node.output_count;
        unit.node.first_input = unit.node.input_count > 0U ? unit.node.inputs[0] : UINT32_MAX;
        unit.node.second_input = unit.node.input_count > 1U ? unit.node.inputs[1] : UINT32_MAX;
        unit.node.auxiliary = unit.node.input_count > 2U ? unit.node.inputs[2] : 0U;
        unit.node.output = unit.node.outputs[0];
        if (unit.control.rowAxis != values[unit.node.output].rowAxis) return nil;
        BOOL anyInputRow = NO;
        for (uint32_t i = 0U; i < unit.node.input_count; i++) anyInputRow |= allValueRowAxes[unit.node.inputs[i]] >= 0;
        if (unit.contraction != (uint32_t)(unit.control.rowAxis < 0 && anyInputRow)) return nil;
        unit.control.rowInner = values[unit.node.output].rowInner;
        const uint8_t *attributeBytes = bytes + (cursor + 14U + r[3] + r[4]) * 8U;
        if (!synaptik_decode_scan_ordinary_attributes(&unit.node, attributeBytes, (uint32_t)r[5],
                types, &unit.semanticScatterReduction)) return nil;
        /* Operand claims are proved declarations or earlier validated SSA results. Ordinary
         * equations precede the independent coordinate proof; no output claim feeds inference.
         * Both checks run before any pipeline is built, including genuine zero-output markers.
         */
        if (!synaptik_scan_validate_ordinary(unit.node, shapes, types, strides, offsets, states)
                || !synaptik_scan_validate_gradient_flags(unit.node, types, gradients)
                || (lengthAware && !synaptik_scan_prove_coordinates(
                        unit.node, shapes, allValueRowAxes, allValueTerminals, w[11]))) return nil;
        if (unit.control.leftScope != UINT32_MAX && (unit.node.operation != 2U || unit.node.input_count != 2U || unit.node.output_count != 1U)) return nil;
        uint64_t expectedScratch = 0U;
        if (unit.node.operation == 35U && (unit.node.output_count == 1U
                || values[unit.node.outputs[1]].type != 1U)
                && (unit.node.output_count > 1U || values[unit.node.output].elements != 0U)) {
            const SynaptikMetalScanValueBinding *output = values + unit.node.output;
            const SynaptikMetalScanValueBinding *key = values + unit.node.second_input;
            if (output->rank < 2U || key->rank < 2U) return nil;
            expectedScratch = 1U;
            for (uint32_t a = 0U; a + 1U < output->rank; a++)
                if (!synaptik_scan_multiply(expectedScratch, output->dimensions[a], &expectedScratch)) return nil;
            if (!synaptik_scan_multiply(expectedScratch, key->dimensions[key->rank - 2U], &expectedScratch)
                    || !synaptik_scan_multiply(expectedScratch, 4U, &expectedScratch)) return nil;
        }
        if (unit.scratchBytes != expectedScratch) return nil;
        unit.firstStage = (uint32_t)steps.count;
        if (!synaptik_append_scan_ordinary_steps(unit.node, shapes, types, strides, offsets,
                states, device, library, lowLibrary, &unit.control, allValueRowAxes, allValueRowInner,
                unit.semanticScatterReduction, steps)) return nil;
        unit.stageCount = (uint32_t)steps.count - unit.firstStage;
        for (uint32_t stage = 0U; stage < unit.stageCount; stage++) {
            SynaptikMetalProgramStep *step = steps[unit.firstStage + stage];
            step.scratchOffset = unit.scratchBytes == 0U ? 0U : (NSUInteger)unit.scratchOffset;
            step.scratchInput = unit.scratchBytes == 0U ? UINT32_MAX : valueCount;
        }
        id<MTLBuffer> controlBuffer = [device newBufferWithBytes:&unit.control length:sizeof(unit.control) options:MTLResourceStorageModeShared];
        if (controlBuffer == nil) return nil;
        [controls addObject:controlBuffer]; units[u] = unit; cursor += r[0];
    }
    if (cursor != w[25] || ((allocatedEnd + 7U) & ~UINT64_C(7)) != w[30]) return nil;
    for (uint32_t r = 0U; r < w[15]; r++) {
        uint64_t v = w[w[26] + r]; if (v >= valueCount || !available[v]) return nil;
        if (r < w[5]) {
            uint64_t f = w[w[25] + r];
            if (values[v].type != values[f].type || values[v].rank != values[f].rank
                    || memcmp(values[v].dimensions, values[f].dimensions, sizeof(values[v].dimensions)) != 0) return nil;
        }
        if (lengthAware && ((r < w[10] && (values[v].rowAxis != 0 || values[v].terminal))
                || (r >= w[10] && r < w[5] && values[v].rowAxis != -1)
                || (r >= w[5] && r < w[5] + w[8] && (values[v].rowAxis != 0 || values[v].terminal)))) return nil;
    }
    if (lengthAware) {
        NSMutableIndexSet *dependent = [NSMutableIndexSet indexSet], *seen = [NSMutableIndexSet indexSet];
        for (uint32_t c = (uint32_t)w[10]; c < w[5]; c++) [dependent addIndex:(NSUInteger)w[w[25] + c]];
        for (uint32_t u = 0U; u < unitCount; u++) for (uint32_t i = 0U; i < units[u].node.input_count; i++) {
            uint32_t input = units[u].node.inputs[i]; if (![dependent containsIndex:input]) continue;
            uint32_t position = UINT32_MAX;
            for (uint32_t c = (uint32_t)w[10]; c < w[5]; c++) if (w[w[25] + c] == input) position = c;
            if (position == UINT32_MAX || i != 0U || units[u].node.operation != 2U || units[u].node.input_count != 2U
                    || units[u].node.output_count != 1U || units[u].node.output != w[w[26] + position]) return nil;
            [seen addIndex:position]; [dependent addIndex:units[u].node.output];
        }
        if (seen.count != w[5] - w[10]) return nil;
    }
    cursor = w[28];
    uint64_t rootSetWords = (w[17] + 63U) / 64U;
    uint64_t membershipWords;
    if (!synaptik_scan_multiply(w[18] + w[17], rootSetWords, &membershipWords)
            || membershipWords > (1U << 20U)) return nil;
    NSMutableData *membershipData = [NSMutableData dataWithLength:(NSUInteger)membershipWords * 8U];
    if (membershipData == nil) return nil;
    uint64_t *memberships = membershipData.mutableBytes;
    NSMutableIndexSet *numericCarries = [NSMutableIndexSet indexSet], *seedPositions = [NSMutableIndexSet indexSet];
    for (uint32_t c = 0U; c < w[19]; c++) {
        if (cursor + 3U > w[29] || w[cursor] >= w[5] || w[cursor + 1U] > 1U || w[cursor + 2U] >= w[18]
                || [numericCarries containsIndex:(NSUInteger)w[cursor]] || (lengthAware && w[cursor] >= w[10])
                || !synaptik_scan_differentiable_type(values[w[w[25] + w[cursor]]].type)) return nil;
        [numericCarries addIndex:(NSUInteger)w[cursor]];
        valueScopes[w[w[25] + w[cursor]]] = (uint32_t)(w[18] + c); cursor += 3U;
    }
    for (uint32_t s = 0U; s < w[20]; s++) {
        if (cursor >= w[29] || w[cursor] < w[5] || w[cursor] >= w[5] + w[6]
                || [seedPositions containsIndex:(NSUInteger)w[cursor]]
                || !synaptik_scan_differentiable_type(values[w[w[25] + w[cursor]]].type)) return nil;
        [seedPositions addIndex:(NSUInteger)w[cursor]];
        valueScopes[w[w[25] + w[cursor]]] = (uint32_t)(w[18] + w[19] + s); cursor++;
    }
    for (uint32_t s = 0U; s < w[18]; s++) {
        if (cursor >= w[29]) return nil;
        uint64_t count = w[cursor++], previous = UINT64_MAX;
        if (count > w[17] || count > w[29] - cursor) return nil;
        for (uint64_t r = 0U; r < count; r++) {
            uint64_t root = w[cursor++]; if (root >= w[17] || (r != 0U && root <= previous)) return nil; previous = root;
            memberships[(uint64_t)s * rootSetWords + root / 64U] |= UINT64_C(1) << (root % 64U);
        }
    }
    for (uint64_t root = 0U; root < w[17]; root++)
        memberships[(w[18] + root) * rootSetWords + root / 64U] |= UINT64_C(1) << (root % 64U);
    for (uint32_t u = 0U; u < unitCount; u++) {
        const SynaptikMetalScanUnitBinding *unit = units + u;
        for (uint32_t i = 0U; i < unit->node.input_count; i++) {
            uint32_t availableScope = valueScopes[unit->node.inputs[i]];
            uint32_t requiredScope = unit->control.leftScope == UINT32_MAX ? unit->control.scope
                    : i == 0U ? unit->control.leftScope : unit->control.rightScope;
            if (availableScope == UINT32_MAX) continue;
            if (requiredScope == UINT32_MAX) return nil;
            for (uint64_t word = 0U; word < rootSetWords; word++)
                if ((memberships[(uint64_t)requiredScope * rootSetWords + word]
                        & ~memberships[(uint64_t)availableScope * rootSetWords + word]) != 0U) return nil;
        }
        if (unit->control.leftScope != UINT32_MAX) {
            for (uint64_t word = 0U; word < rootSetWords; word++)
                if ((memberships[(uint64_t)unit->control.leftScope * rootSetWords + word]
                        | memberships[(uint64_t)unit->control.rightScope * rootSetWords + word])
                        != memberships[(uint64_t)unit->control.scope * rootSetWords + word]) return nil;
            const SynaptikMetalScanValueBinding *output = values + unit->node.output;
            for (uint32_t i = 0U; i < 2U; i++) {
                const SynaptikMetalScanValueBinding *input = values + unit->node.inputs[i];
                if (input->type != output->type || input->rank != output->rank
                        || memcmp(input->dimensions, output->dimensions, sizeof(input->dimensions)) != 0) return nil;
            }
        }
    }
    for (uint32_t c = 0U; c < w[19]; c++) {
        uint64_t position = w[w[28] + c * 3U], expectedScope = w[w[28] + c * 3U + 2U];
        uint32_t actualScope = valueScopes[w[w[26] + position]];
        for (uint64_t word = 0U; word < rootSetWords; word++) {
            uint64_t actualBits = actualScope == UINT32_MAX ? 0U : memberships[(uint64_t)actualScope * rootSetWords + word];
            if (memberships[expectedScope * rootSetWords + word] != actualBits) return nil;
        }
    }
    NSMutableIndexSet *accumulatorUnits = [NSMutableIndexSet indexSet];
    for (uint32_t a = 0U; a < w[21]; a++) {
        if (cursor + 2U > w[29] || w[cursor] >= w[5] || w[cursor + 1U] >= w[18]
                || [numericCarries containsIndex:(NSUInteger)w[cursor]] || (lengthAware && w[cursor] < w[10])
                || !synaptik_scan_differentiable_type(values[w[w[25] + w[cursor]]].type)) return nil;
        uint64_t position = w[cursor], next = w[w[26] + position], formal = w[w[25] + position];
        uint32_t producer = UINT32_MAX;
        for (uint32_t u = 0U; u < unitCount; u++)
            for (uint32_t o = 0U; o < units[u].node.output_count; o++) if (units[u].node.outputs[o] == next) producer = u;
        if (producer == UINT32_MAX || [accumulatorUnits containsIndex:producer]) return nil;
        const SynaptikMetalScanUnitBinding *unit = units + producer;
        if (unit->node.operation != 2U || unit->node.input_count != 2U || unit->node.output_count != 1U
                || unit->node.inputs[0] != formal || unit->control.leftScope != UINT32_MAX || unit->accumulator != (int32_t)position) return nil;
        for (uint64_t word = 0U; word < rootSetWords; word++)
            if (memberships[(uint64_t)unit->control.scope * rootSetWords + word]
                    != memberships[w[cursor + 1U] * rootSetWords + word]) return nil;
        [accumulatorUnits addIndex:producer]; [numericCarries addIndex:(NSUInteger)position]; cursor += 2U;
    }
    for (uint32_t u = 0U; u < unitCount; u++)
        if ((units[u].accumulator >= 0) != [accumulatorUnits containsIndex:u]) return nil;
    uint64_t rootWords = (w[17] + 63U) / 64U, rows = lengthAware ? w[11] : 1U, scopeWords, presenceWords, expectedWords;
    if (cursor != w[29] || (activity && numericCarries.count != w[5])
            || !synaptik_scan_multiply(w[18] + w[17], rootWords, &scopeWords)
            || !synaptik_scan_multiply(rows, rootWords, &presenceWords)
            || !synaptik_scan_add(10U, scopeWords, &expectedWords)
            || !synaptik_scan_add(expectedWords, presenceWords, &expectedWords)
            || !synaptik_scan_add(expectedWords, lengthAware ? w[11] : 0U, &expectedWords)
            || !synaptik_scan_add(expectedWords, presenceWords, &expectedWords) || expectedWords != w[31]) return nil;
    id<MTLFunction> bindingFunction = [library newFunctionWithName:@"scan_bind_raw"];
    NSError *error = nil;
    id<MTLComputePipelineState> bindingPipeline = bindingFunction == nil ? nil
            : [device newComputePipelineStateWithFunction:bindingFunction error:&error];
    if (bindingPipeline == nil || error != nil) return nil;
    SynaptikMetalScanPlan *plan = [SynaptikMetalScanPlan new];
    plan.inputReferences = inputReferenceData;
    plan.words = wordData; plan.values = valueData; plan.units = unitData; plan.valueScopes = scopeData;
    plan.carryOffsets = carryOffsetsData;
    plan.scopeMemberships = membershipData;
    plan.carryPromoted = carryPromotedData;
    plan.steps = [steps copy]; plan.stepControls = [controls copy]; plan.bindingPipeline = bindingPipeline;
    return plan;
}

/* Complete independent outer binding check. Reference spans are borrowed little-endian bytes
 * from the enclosing authenticated image; no fixed ordinary16/5 arrays or extra copies apply.
 * Result references enumerate demanded roles only; all latent role identities remain local.
 * Ordinary complete-program validation has independently authenticated canonical represented
 * storage for both canonical and materialized affine-view inputs before this binding check. */
static BOOL synaptik_scan_validate_outer(SynaptikMetalScanPlan *plan,
        uint32_t inputCount, const uint8_t *inputRefs, uint32_t outputCount, const uint8_t *outputRefs,
        uint32_t valueCount, const uint32_t *ranks, const uint64_t *dimensions,
        const uint8_t *types, const uint8_t *gradientFlags, const uint8_t *states) {
    if (plan == nil || inputRefs == NULL || (outputCount != 0U && outputRefs == NULL)
            || ranks == NULL || dimensions == NULL || types == NULL || gradientFlags == NULL || states == NULL) return NO;
    const uint64_t *w = plan.words.bytes;
    const SynaptikMetalScanValueBinding *locals = plan.values.bytes;
    BOOL lengths = (w[4] & 2U) != 0U, gradient = NO;
    if (inputCount != w[14] + (lengths ? 1U : 0U) || outputCount != w[16]) return NO;
    for (uint32_t f = 0U; f < w[14]; f++) {
        uint32_t actual = synaptik_read_le32(inputRefs + (size_t)f * 4U);
        const SynaptikMetalScanValueBinding *formal = locals + w[w[25] + f];
        BOOL sequence = f >= w[5] && f < w[5] + w[6];
        if (actual >= valueCount || types[actual] != formal->type || gradientFlags[actual] != formal->gradient
                || (states[actual] != 1U && states[actual] != 2U)
                || ranks[actual] != formal->rank + (sequence ? 1U : 0U)) return NO;
        const uint64_t *shape = dimensions + (size_t)actual * SYNAPTIK_MAX_RANK;
        if (sequence && shape[0] != w[3]) return NO;
        for (uint32_t a = 0U; a < formal->rank; a++) if (shape[a + (sequence ? 1U : 0U)] != formal->dimensions[a]) return NO;
        gradient |= gradientFlags[actual] != 0U;
        if (lengths) for (uint32_t earlier = 0U; earlier < f; earlier++)
            if (synaptik_read_le32(inputRefs + (size_t)earlier * 4U) == actual) {
                int32_t row = formal->rowAxis;
                const SynaptikMetalScanValueBinding *previous = locals + w[w[25] + earlier];
                int32_t previousRow = previous->rowAxis;
                if (row >= 0 && sequence) row++;
                if (previousRow >= 0 && earlier >= w[5] && earlier < w[5] + w[6]) previousRow++;
                if (row != previousRow) return NO;
            }
    }
    if (lengths) {
        uint32_t actual = synaptik_read_le32(inputRefs + (size_t)(inputCount - 1U) * 4U);
        if (actual >= valueCount || types[actual] != 6U || gradientFlags[actual] != 0U
                || (states[actual] != 1U && states[actual] != 2U)
                || ranks[actual] != 1U || dimensions[(size_t)actual * SYNAPTIK_MAX_RANK] != w[11]) return NO;
    }
    for (uint32_t d = 0U; d < outputCount; d++) {
        uint32_t actual = synaptik_read_le32(outputRefs + (size_t)d * 4U);
        uint64_t port = w[w[27] + d];
        const SynaptikMetalScanValueBinding *role = locals + w[w[26] + port];
        BOOL history = port >= w[5];
        if (actual >= valueCount || types[actual] != role->type || states[actual] != 1U
                || gradientFlags[actual] != (uint8_t)(gradient && synaptik_scan_differentiable_type(role->type))
                || ranks[actual] != role->rank + (history ? 1U : 0U)) return NO;
        const uint64_t *shape = dimensions + (size_t)actual * SYNAPTIK_MAX_RANK;
        if (history && shape[0] != w[3]) return NO;
        for (uint32_t a = 0U; a < role->rank; a++) if (shape[a + (history ? 1U : 0U)] != role->dimensions[a]) return NO;
    }
    return YES;
}

/* Allocation-free direct-feed and producer-ready length preflight. The optional maximum is
 * a host scalar used only to skip globally inactive time coordinates, never a specialization. */
static int32_t synaptik_scan_preflight_lengths(SynaptikMetalScanPlan *plan,
        id<MTLBuffer> lengthBuffer, uint64_t *maximum) {
    if (plan == nil) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    const uint64_t *w = plan.words.bytes;
    if ((w[4] & 2U) == 0U) { if (maximum != NULL) *maximum = w[3]; return SYNAPTIK_METAL_STATUS_OK; }
    if (lengthBuffer == nil || w[11] > lengthBuffer.length / 8U
            || (w[11] != 0U && (lengthBuffer.contents == NULL || ((uintptr_t)lengthBuffer.contents & 7U) != 0U)))
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    const int64_t *lengths = lengthBuffer.contents;
    uint64_t activeTime = 0U;
    for (uint64_t b = 0U; b < w[11]; b++) {
        if (lengths[b] < 0 || (uint64_t)lengths[b] > w[3]) return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        activeTime = MAX(activeTime, (uint64_t)lengths[b]);
    }
    if (maximum != NULL) *maximum = activeTime;
    return SYNAPTIK_METAL_STATUS_OK;
}

static int32_t synaptik_scan_flush(id<MTLCommandBuffer> __strong *command,
        id<MTLComputeCommandEncoder> __strong *encoder) {
    if (*command == nil) return SYNAPTIK_METAL_STATUS_OK;
    if (*encoder != nil) { [*encoder endEncoding]; *encoder = nil; }
    [*command commit]; [*command waitUntilCompleted];
    BOOL success = (*command).status == MTLCommandBufferStatusCompleted && (*command).error == nil;
    *command = nil;
    return success ? SYNAPTIK_METAL_STATUS_OK : SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
}

static BOOL synaptik_scan_encoder(id<MTLCommandQueue> queue, id<MTLCommandBuffer> __strong *command,
        id<MTLComputeCommandEncoder> __strong *encoder) {
    if (*command == nil) *command = [queue commandBuffer];
    if (*command == nil) return NO;
    if (*encoder == nil) *encoder = [*command computeCommandEncoder];
    return *encoder != nil;
}

static BOOL synaptik_scan_bind(SynaptikMetalScanPlan *plan, id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> source, NSUInteger sourceOffset, id<MTLBuffer> destination,
        NSUInteger destinationOffset, uint64_t elements, uint32_t width, uint32_t mode,
        id<MTLBuffer> workspace, NSUInteger controlOffset, SynaptikMetalScanStepControl step,
        id<MTLBuffer> oldCarry, NSUInteger oldCarryOffset) {
    if (elements == 0U) return YES;
    SynaptikMetalScanBindMeta meta = { .elementCount = elements, .width = width, .mode = mode };
    MTLSize grid;
    uint64_t gridWidth, gridHeight;
    if (!custom_grid(elements, &grid, &gridWidth, &gridHeight)) return NO;
    // custom_grid bounds each coordinate to UINT32_MAX before publication.
    meta.gridWidth = (uint32_t)gridWidth;
    meta.gridHeight = (uint32_t)gridHeight;
    [encoder setComputePipelineState:plan.bindingPipeline];
    [encoder setBuffer:source offset:sourceOffset atIndex:0U];
    [encoder setBuffer:destination offset:destinationOffset atIndex:1U];
    [encoder setBytes:&meta length:sizeof(meta) atIndex:2U];
    [encoder setBuffer:oldCarry offset:oldCarryOffset atIndex:3U];
    [encoder setBuffer:workspace offset:controlOffset atIndex:28U];
    [encoder setBytes:&step length:sizeof(step) atIndex:29U];
    NSUInteger threads = MAX((NSUInteger)1U, MIN(plan.bindingPipeline.threadExecutionWidth,
            plan.bindingPipeline.maxTotalThreadsPerThreadgroup));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return YES;
}

/* actuals are ordered declared inputs, results are ONLY demanded roles in increasing demand
 * order. workspace is the existing Run-owned prepared resource with exact header22 bytes.
 * The caller validates context/access/alias facts at the ordinary executable boundary; this
 * function independently checks byte sizes and refuses an overlapping workspace.
 */
static int32_t synaptik_run_scan_plan(SynaptikMetalScanPlan *plan, id<MTLCommandQueue> queue,
        id<MTLBuffer> __unsafe_unretained const *actuals, uint32_t actualCount,
        id<MTLBuffer> __unsafe_unretained const *results, uint32_t resultCount, id<MTLBuffer> workspace) {
    if (plan == nil || queue == nil || workspace == nil || actuals == NULL || (resultCount != 0U && results == NULL))
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    const uint64_t *w = plan.words.bytes;
    const SynaptikMetalScanValueBinding *values = plan.values.bytes;
    const SynaptikMetalScanUnitBinding *units = plan.units.bytes;
    const uint32_t *valueScopes = plan.valueScopes.bytes;
    const uint64_t *carryOffsets = plan.carryOffsets.bytes;
    const uint8_t *carryPromoted = plan.carryPromoted.bytes;
    BOOL lengthAware = (w[4] & 2U) != 0U, reverse = (w[4] & 1U) != 0U;
    if (actualCount != w[14] + (lengthAware ? 1U : 0U) || resultCount != w[16]
            || workspace.length < w[22] || workspace.contents == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    for (uint32_t f = 0U; f < w[14]; f++) {
        const SynaptikMetalScanValueBinding *v = values + w[w[25] + f];
        uint64_t bytes = v->bytes;
        if (f >= w[5] && f < w[5] + w[6] && !synaptik_scan_multiply(bytes, w[3], &bytes))
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        if (actuals[f] == workspace || actuals[f].length < bytes) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t d = 0U; d < w[16]; d++) {
        uint64_t port = w[w[27] + d], bytes = values[w[w[26] + port]].bytes;
        if (port >= w[5] && !synaptik_scan_multiply(bytes, w[3], &bytes)) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        if (results[d] == workspace || results[d].length < bytes) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    }
    const int64_t *lengths = NULL;
    uint64_t activeTime = w[3];
    if (lengthAware) {
        id<MTLBuffer> lengthBuffer = actuals[actualCount - 1U];
        if (lengthBuffer == workspace) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        int32_t status = synaptik_scan_preflight_lengths(plan, lengthBuffer, &activeTime);
        if (status != SYNAPTIK_METAL_STATUS_OK) return status;
        lengths = lengthBuffer.contents;
    }
    uint64_t *control = (uint64_t *)((uint8_t *)workspace.contents + w[30]);
    memset(control, 0, (size_t)w[31] * 8U);
    uint64_t rootWords = (w[17] + 63U) / 64U, rows = lengthAware ? w[11] : 1U;
    control[1] = w[11]; control[2] = w[17]; control[3] = w[18] + w[17]; control[4] = rootWords;
    control[5] = 10U; control[6] = control[5] + control[3] * rootWords;
    control[7] = control[6] + rows * rootWords; control[8] = lengthAware ? 1U : 0U; control[9] = rows;
    if (plan.scopeMemberships.length != 0U) memcpy(control + control[5], plan.scopeMemberships.bytes, plan.scopeMemberships.length);
    if (lengthAware && rows != 0U) memcpy(control + control[7], lengths, (size_t)rows * 8U);
    uint64_t nextPresenceOffset = control[7] + (lengthAware ? rows : 0U);
    uint64_t activityCursor = w[28], carryTable = activityCursor;
    for (uint32_t c = 0U; c < w[19]; c++) {
        if (w[activityCursor + 1U] != 0U) for (uint64_t b = 0U; b < rows; b++)
            control[control[6] + b * rootWords + c / 64U] |= UINT64_C(1) << (c % 64U);
        activityCursor += 3U;
    }
    for (uint32_t s = 0U; s < w[20]; s++) {
        uint64_t root = w[19] + s;
        for (uint64_t b = 0U; b < rows; b++)
            control[control[6] + b * rootWords + root / 64U] |= UINT64_C(1) << (root % 64U);
        activityCursor++;
    }
    for (uint32_t scope = 0U; scope < w[18]; scope++) {
        uint64_t count = w[activityCursor++]; activityCursor += count;
    }
    uint64_t accumulatorTable = activityCursor;
    id<MTLBuffer> __unsafe_unretained buffers[4097];
    NSUInteger bindingOffsets[4097];
    for (uint32_t v = 0U; v < w[12]; v++) {
        buffers[v] = workspace; bindingOffsets[v] = values[v].offset == UINT64_MAX ? 0U : (NSUInteger)values[v].offset;
    }
    buffers[w[12]] = workspace; bindingOffsets[w[12]] = 0U;
    for (uint32_t f = (uint32_t)w[5]; f < w[14]; f++) buffers[(NSUInteger)w[w[25] + f]] = actuals[f];
    id<MTLCommandBuffer> initial = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    for (uint32_t c = 0U; c < w[5]; c++) {
        const SynaptikMetalScanValueBinding *v = values + w[w[25] + c];
        if (v->bytes == 0U) continue;
        if (blit == nil) {
            initial = [queue commandBuffer]; blit = [initial blitCommandEncoder];
            if (initial == nil || blit == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        }
        BOOL present = YES;
        for (uint32_t root = 0U; root < w[19]; root++)
            if (w[carryTable + root * 3U] == c) present = w[carryTable + root * 3U + 1U] != 0U;
        if (present) [blit copyFromBuffer:actuals[c] sourceOffset:0U toBuffer:workspace destinationOffset:(NSUInteger)v->offset size:(NSUInteger)v->bytes];
        else [blit fillBuffer:workspace range:NSMakeRange((NSUInteger)v->offset, (NSUInteger)v->bytes) value:0U];
    }
    if (initial != nil) {
        [blit endEncoding]; [initial commit]; [initial waitUntilCompleted];
        if (initial.status != MTLCommandBufferStatusCompleted || initial.error != nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
    }
    for (uint64_t iteration = 0U; iteration < activeTime; iteration++) {
        uint64_t time = reverse ? activeTime - 1U - iteration : iteration; control[0] = time;
        if (!synaptik_scan_control_any(control, UINT32_MAX)) continue;
        @autoreleasepool {
        for (uint32_t s = 0U; s < w[6]; s++) {
            uint64_t v = w[w[25] + w[5] + s]; bindingOffsets[v] = (NSUInteger)(time * values[v].bytes);
        }
        for (uint32_t c = 0U; c < w[5]; c++) if (carryPromoted[c]) {
            uint64_t formal = w[w[25] + c], next = w[w[26] + c];
            bindingOffsets[next] = bindingOffsets[formal] == values[formal].offset
                    ? (NSUInteger)carryOffsets[c] : (NSUInteger)values[formal].offset;
        }
        id<MTLCommandBuffer> command = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        for (uint32_t u = 0U; u < w[13]; u++) {
            const SynaptikMetalScanUnitBinding *unit = units + u;
            if (!synaptik_scan_control_any(control, unit->control.scope)) continue;
            BOOL requiresValidation = NO, hasOutput = NO;
            for (uint32_t i = 0U; i < unit->node.input_count; i++)
                requiresValidation |= values[unit->node.inputs[i]].type == 3U;
            uint32_t op = unit->node.operation;
            requiresValidation |= op == 16U || op == 17U || op == 18U || op == 69U || op == 70U || op == 71U || op == 72U || op == 87U;
            if (requiresValidation) {
                int32_t status = synaptik_scan_flush(&command, &encoder);
                if (status != SYNAPTIK_METAL_STATUS_OK) return status;
                status = synaptik_validate_scan_step(plan.steps[unit->firstStage], unit->node, buffers, (uint32_t)w[12] + 1U,
                        bindingOffsets, control, &unit->control);
                if (status != SYNAPTIK_METAL_STATUS_OK) return status;
            }
            for (uint32_t o = 0U; o < unit->node.output_count; o++) hasOutput |= values[unit->node.outputs[o]].bytes != 0U;
            if (!hasOutput) continue;
            if (!synaptik_scan_encoder(queue, &command, &encoder)) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            [encoder setBuffer:workspace offset:(NSUInteger)w[30] atIndex:28U];
            [encoder setBuffer:plan.stepControls[u] offset:0U atIndex:29U];
            for (uint32_t stage = 0U; stage < unit->stageCount; stage++) {
                int32_t status = synaptik_encode_scan_step(plan.steps[unit->firstStage + stage], encoder,
                        buffers, (uint32_t)w[12] + 1U, bindingOffsets);
                if (status != SYNAPTIK_METAL_STATUS_OK) { [encoder endEncoding]; return status; }
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
        for (uint32_t d = 0U; d < w[16]; d++) {
            uint64_t port = w[w[27] + d]; if (port < w[5]) continue;
            uint64_t v = w[w[26] + port]; if (values[v].bytes == 0U) continue;
            SynaptikMetalScanStepControl step = { .rowInner = values[v].rowInner, .rowAxis = values[v].rowAxis,
                    .scope = valueScopes[v], .leftScope = UINT32_MAX, .rightScope = UINT32_MAX };
            if (!synaptik_scan_encoder(queue, &command, &encoder)
                    || !synaptik_scan_bind(plan, encoder, buffers[v], bindingOffsets[v], results[d], (NSUInteger)(time * values[v].bytes),
                            values[v].elements, values[v].width, 0U, workspace, (NSUInteger)w[30], step, buffers[v], bindingOffsets[v]))
                return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        }
        for (uint32_t c = 0U; c < w[5]; c++) {
            uint64_t old = w[w[25] + c], next = w[w[26] + c]; if (values[old].bytes == 0U) continue;
            uint32_t scope = UINT32_MAX, mode = 2U;
            for (uint32_t root = 0U; root < w[19]; root++) if (w[carryTable + root * 3U] == c) {
                scope = (uint32_t)w[carryTable + root * 3U + 2U]; mode = 1U;
            }
            for (uint32_t a = 0U; a < w[21]; a++) if (w[accumulatorTable + a * 2U] == c)
                scope = (uint32_t)w[accumulatorTable + a * 2U + 1U];
            if (carryPromoted[c]) {
                if (!lengthAware && (w[4] & 4U) == 0U) continue;
                mode += 2U;
            }
            SynaptikMetalScanStepControl step = { .rowInner = values[old].rowInner, .rowAxis = values[old].rowAxis,
                    .scope = scope, .leftScope = UINT32_MAX, .rightScope = UINT32_MAX };
            if (!synaptik_scan_encoder(queue, &command, &encoder)
                    || !synaptik_scan_bind(plan, encoder, buffers[next], bindingOffsets[next], workspace,
                            bindingOffsets[old] == values[old].offset ? (NSUInteger)carryOffsets[c] : (NSUInteger)values[old].offset,
                            values[old].elements, values[old].width, mode, workspace, (NSUInteger)w[30], step,
                            buffers[old], bindingOffsets[old]))
                return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        }
        int32_t status = synaptik_scan_flush(&command, &encoder); if (status != SYNAPTIK_METAL_STATUS_OK) return status;
        for (uint32_t c = 0U; c < w[5]; c++) {
            uint64_t v = w[w[25] + c];
            bindingOffsets[v] = bindingOffsets[v] == values[v].offset ? (NSUInteger)carryOffsets[c] : (NSUInteger)values[v].offset;
        }
        /* Simultaneous next-presence: never overwrite an old carried root while another next
         * scope still reads it. Inactive rows retain both represented carry and presence bits. */
        uint64_t presenceWords = rows * rootWords;
        if (presenceWords != 0U) memcpy(control + nextPresenceOffset, control + control[6], (size_t)presenceWords * 8U);
        for (uint64_t b = 0U; b < rows; b++) {
            if (!synaptik_scan_control_row(control, UINT32_MAX, b)) continue;
            for (uint32_t root = 0U; root < w[19]; root++) {
                uint64_t *word = control + nextPresenceOffset + b * rootWords + root / 64U;
                uint64_t bit = UINT64_C(1) << (root % 64U);
                if (synaptik_scan_control_row(control, (uint32_t)w[carryTable + root * 3U + 2U], b)) *word |= bit;
                else *word &= ~bit;
            }
        }
        if (presenceWords != 0U) memcpy(control + control[6], control + nextPresenceOffset, (size_t)presenceWords * 8U);
        }
    }
    id<MTLCommandBuffer> final = nil;
    blit = nil;
    for (uint32_t d = 0U; d < w[16]; d++) {
        uint64_t port = w[w[27] + d];
        const SynaptikMetalScanValueBinding *v = values + w[w[26] + port];
        if (v->bytes == 0U || (port >= w[5] && activeTime == w[3])) continue;
        if (blit == nil) {
            final = [queue commandBuffer]; blit = [final blitCommandEncoder];
            if (final == nil || blit == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        }
        if (port < w[5]) {
            [blit copyFromBuffer:workspace sourceOffset:bindingOffsets[w[w[25] + port]]
                    toBuffer:results[d] destinationOffset:0U size:(NSUInteger)v->bytes];
        } else {
            [blit fillBuffer:results[d] range:NSMakeRange((NSUInteger)(activeTime * v->bytes),
                    (NSUInteger)((w[3] - activeTime) * v->bytes)) value:0U];
        }
    }
    if (final == nil) return SYNAPTIK_METAL_STATUS_OK;
    [blit endEncoding]; [final commit]; [final waitUntilCompleted];
    return final.status == MTLCommandBufferStatusCompleted && final.error == nil
            ? SYNAPTIK_METAL_STATUS_OK : SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
}

#endif
