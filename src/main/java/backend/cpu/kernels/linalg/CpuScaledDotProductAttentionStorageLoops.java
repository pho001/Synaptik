package backend.cpu.kernels.linalg;

import backend.cpu.storage.CpuStorageView;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class CpuScaledDotProductAttentionStorageLoops {
    private CpuScaledDotProductAttentionStorageLoops() {
    }

    static void executeF64(
            CpuStorageView query,
            CpuStorageView key,
            CpuStorageView value,
            CpuStorageView mask,
            CpuStorageView output,
            double scale,
            boolean fastExp,
            CpuStorageView cachedWeights
    ) {
        double[] queryArray = query.isArray() ? query.requireF64Array() : null;
        double[] keyArray = key.isArray() ? key.requireF64Array() : null;
        double[] valueArray = value.isArray() ? value.requireF64Array() : null;
        double[] outputArray = output.isArray() ? output.requireF64Array() : null;
        double[] cacheArray = cachedWeights == null ? null : cachedWeights.requireF64Array();
        byte[] maskArray = mask == null || !mask.isArray() ? null : mask.requireBoolArray();
        MemorySegment querySegment = query.isMemorySegment() ? query.requireSegment() : null;
        MemorySegment keySegment = key.isMemorySegment() ? key.requireSegment() : null;
        MemorySegment valueSegment = value.isMemorySegment() ? value.requireSegment() : null;
        MemorySegment outputSegment = output.isMemorySegment() ? output.requireSegment() : null;
        MemorySegment maskSegment = mask != null && mask.isMemorySegment() ? mask.requireSegment() : null;

        AttentionStoragePlan plan = AttentionStoragePlan.create(query, key, value, mask, output);
        double[] rowScores = new double[plan.keyLen];
        for (int row = 0; row < plan.totalRows; row++) {
            computeRowF64(
                    plan,
                    row,
                    queryArray,
                    querySegment,
                    keyArray,
                    keySegment,
                    valueArray,
                    valueSegment,
                    maskArray,
                    maskSegment,
                    outputArray,
                    outputSegment,
                    cacheArray,
                    scale,
                    fastExp,
                    rowScores
            );
        }
    }

    static void executeF32(
            CpuStorageView query,
            CpuStorageView key,
            CpuStorageView value,
            CpuStorageView mask,
            CpuStorageView output,
            float scale,
            boolean fastExp,
            CpuStorageView cachedWeights
    ) {
        float[] queryArray = query.isArray() ? query.requireF32Array() : null;
        float[] keyArray = key.isArray() ? key.requireF32Array() : null;
        float[] valueArray = value.isArray() ? value.requireF32Array() : null;
        float[] outputArray = output.isArray() ? output.requireF32Array() : null;
        float[] cacheArray = cachedWeights == null ? null : cachedWeights.requireF32Array();
        byte[] maskArray = mask == null || !mask.isArray() ? null : mask.requireBoolArray();
        MemorySegment querySegment = query.isMemorySegment() ? query.requireSegment() : null;
        MemorySegment keySegment = key.isMemorySegment() ? key.requireSegment() : null;
        MemorySegment valueSegment = value.isMemorySegment() ? value.requireSegment() : null;
        MemorySegment outputSegment = output.isMemorySegment() ? output.requireSegment() : null;
        MemorySegment maskSegment = mask != null && mask.isMemorySegment() ? mask.requireSegment() : null;

        AttentionStoragePlan plan = AttentionStoragePlan.create(query, key, value, mask, output);
        float[] rowScores = new float[plan.keyLen];
        for (int row = 0; row < plan.totalRows; row++) {
            computeRowF32(
                    plan,
                    row,
                    queryArray,
                    querySegment,
                    keyArray,
                    keySegment,
                    valueArray,
                    valueSegment,
                    maskArray,
                    maskSegment,
                    outputArray,
                    outputSegment,
                    cacheArray,
                    scale,
                    fastExp,
                    rowScores
            );
        }
    }

    static void executeBF16(
            CpuStorageView query,
            CpuStorageView key,
            CpuStorageView value,
            CpuStorageView mask,
            CpuStorageView output,
            float scale,
            boolean fastExp,
            CpuStorageView cachedWeights
    ) {
        short[] queryArray = query.isArray() ? query.requireBF16Array() : null;
        short[] keyArray = key.isArray() ? key.requireBF16Array() : null;
        short[] valueArray = value.isArray() ? value.requireBF16Array() : null;
        short[] outputArray = output.isArray() ? output.requireBF16Array() : null;
        float[] cacheArray = cachedWeights == null ? null : cachedWeights.requireF32Array();
        byte[] maskArray = mask == null || !mask.isArray() ? null : mask.requireBoolArray();
        MemorySegment querySegment = query.isMemorySegment() ? query.requireSegment() : null;
        MemorySegment keySegment = key.isMemorySegment() ? key.requireSegment() : null;
        MemorySegment valueSegment = value.isMemorySegment() ? value.requireSegment() : null;
        MemorySegment outputSegment = output.isMemorySegment() ? output.requireSegment() : null;
        MemorySegment maskSegment = mask != null && mask.isMemorySegment() ? mask.requireSegment() : null;

        AttentionStoragePlan plan = AttentionStoragePlan.create(query, key, value, mask, output);
        float[] rowScores = new float[plan.keyLen];
        for (int row = 0; row < plan.totalRows; row++) {
            computeRowBF16(
                    plan,
                    row,
                    queryArray,
                    querySegment,
                    keyArray,
                    keySegment,
                    valueArray,
                    valueSegment,
                    maskArray,
                    maskSegment,
                    outputArray,
                    outputSegment,
                    cacheArray,
                    scale,
                    fastExp,
                    rowScores
            );
        }
    }

    private static void computeRowF64(
            AttentionStoragePlan plan,
            int row,
            double[] queryArray,
            MemorySegment querySegment,
            double[] keyArray,
            MemorySegment keySegment,
            double[] valueArray,
            MemorySegment valueSegment,
            byte[] maskArray,
            MemorySegment maskSegment,
            double[] outputArray,
            MemorySegment outputSegment,
            double[] cacheArray,
            double scale,
            boolean fastExp,
            double[] rowScores
    ) {
        int batch = row / plan.queryLen;
        int queryIndex = row % plan.queryLen;
        int queryBase = plan.queryStorageOffset + plan.queryBatchOffsets[batch] + queryIndex * plan.queryRowStride;
        int keyBatchBase = plan.keyStorageOffset + plan.keyBatchOffsets[batch];
        int valueBatchBase = plan.valueStorageOffset + plan.valueBatchOffsets[batch];
        int outputBase = plan.outputStorageOffset + plan.outputBatchOffsets[batch] + queryIndex * plan.outputRowStride;
        int maskBase = plan.maskBatchOffsets == null
                ? -1
                : plan.maskStorageOffset + plan.maskBatchOffsets[batch] + queryIndex * plan.maskRowStride;

        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0d;
                continue;
            }
            int keyBase = keyBatchBase + keyIndex * plan.keyRowStride;
            double score = 0.0d;
            for (int depth = 0; depth < plan.depth; depth++) {
                score += readF64(queryArray, querySegment, queryBase + depth * plan.queryColStride)
                        * readF64(keyArray, keySegment, keyBase + depth * plan.keyColStride);
            }
            score *= scale;
            rowScores[keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }

        if (!anyValid) {
            double uniform = 1.0d / plan.keyLen;
            writeCacheF64(cacheArray, row, plan.keyLen, rowScores, uniform);
            writeWeightedValuesF64(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, uniform);
            return;
        }

        double sum = 0.0d;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0d;
                continue;
            }
            double shifted = rowScores[keyIndex] - max;
            double exp = fastExp ? FastTranscendentals.fastExpF64(shifted) : Math.exp(shifted);
            rowScores[keyIndex] = exp;
            sum += exp;
        }

        double inv = 1.0d / sum;
        if (cacheArray != null) {
            int weightsBase = row * plan.keyLen;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                double weight = rowScores[keyIndex] * inv;
                rowScores[keyIndex] = weight;
                cacheArray[weightsBase + keyIndex] = weight;
            }
        } else {
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                rowScores[keyIndex] *= inv;
            }
        }
        writeWeightedValuesF64(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, 0.0d);
    }

    private static void computeRowF32(
            AttentionStoragePlan plan,
            int row,
            float[] queryArray,
            MemorySegment querySegment,
            float[] keyArray,
            MemorySegment keySegment,
            float[] valueArray,
            MemorySegment valueSegment,
            byte[] maskArray,
            MemorySegment maskSegment,
            float[] outputArray,
            MemorySegment outputSegment,
            float[] cacheArray,
            float scale,
            boolean fastExp,
            float[] rowScores
    ) {
        int batch = row / plan.queryLen;
        int queryIndex = row % plan.queryLen;
        int queryBase = plan.queryStorageOffset + plan.queryBatchOffsets[batch] + queryIndex * plan.queryRowStride;
        int keyBatchBase = plan.keyStorageOffset + plan.keyBatchOffsets[batch];
        int valueBatchBase = plan.valueStorageOffset + plan.valueBatchOffsets[batch];
        int outputBase = plan.outputStorageOffset + plan.outputBatchOffsets[batch] + queryIndex * plan.outputRowStride;
        int maskBase = plan.maskBatchOffsets == null
                ? -1
                : plan.maskStorageOffset + plan.maskBatchOffsets[batch] + queryIndex * plan.maskRowStride;

        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            int keyBase = keyBatchBase + keyIndex * plan.keyRowStride;
            float score = 0.0f;
            for (int depth = 0; depth < plan.depth; depth++) {
                score += readF32(queryArray, querySegment, queryBase + depth * plan.queryColStride)
                        * readF32(keyArray, keySegment, keyBase + depth * plan.keyColStride);
            }
            score *= scale;
            rowScores[keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }

        if (!anyValid) {
            float uniform = 1.0f / plan.keyLen;
            writeCacheF32(cacheArray, row, plan.keyLen, rowScores, uniform);
            writeWeightedValuesF32(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, uniform);
            return;
        }

        float sum = 0.0f;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            float shifted = rowScores[keyIndex] - max;
            float exp = fastExp ? FastTranscendentals.fastExpF32(shifted) : (float) Math.exp(shifted);
            rowScores[keyIndex] = exp;
            sum += exp;
        }

        float inv = 1.0f / sum;
        if (cacheArray != null) {
            int weightsBase = row * plan.keyLen;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                float weight = rowScores[keyIndex] * inv;
                rowScores[keyIndex] = weight;
                cacheArray[weightsBase + keyIndex] = weight;
            }
        } else {
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                rowScores[keyIndex] *= inv;
            }
        }
        writeWeightedValuesF32(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, 0.0f);
    }

    private static void computeRowBF16(
            AttentionStoragePlan plan,
            int row,
            short[] queryArray,
            MemorySegment querySegment,
            short[] keyArray,
            MemorySegment keySegment,
            short[] valueArray,
            MemorySegment valueSegment,
            byte[] maskArray,
            MemorySegment maskSegment,
            short[] outputArray,
            MemorySegment outputSegment,
            float[] cacheArray,
            float scale,
            boolean fastExp,
            float[] rowScores
    ) {
        int batch = row / plan.queryLen;
        int queryIndex = row % plan.queryLen;
        int queryBase = plan.queryStorageOffset + plan.queryBatchOffsets[batch] + queryIndex * plan.queryRowStride;
        int keyBatchBase = plan.keyStorageOffset + plan.keyBatchOffsets[batch];
        int valueBatchBase = plan.valueStorageOffset + plan.valueBatchOffsets[batch];
        int outputBase = plan.outputStorageOffset + plan.outputBatchOffsets[batch] + queryIndex * plan.outputRowStride;
        int maskBase = plan.maskBatchOffsets == null
                ? -1
                : plan.maskStorageOffset + plan.maskBatchOffsets[batch] + queryIndex * plan.maskRowStride;

        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            int keyBase = keyBatchBase + keyIndex * plan.keyRowStride;
            float score = 0.0f;
            for (int depth = 0; depth < plan.depth; depth++) {
                score += readBF16(queryArray, querySegment, queryBase + depth * plan.queryColStride)
                        * readBF16(keyArray, keySegment, keyBase + depth * plan.keyColStride);
            }
            score *= scale;
            rowScores[keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }

        if (!anyValid) {
            float uniform = 1.0f / plan.keyLen;
            writeCacheF32(cacheArray, row, plan.keyLen, rowScores, uniform);
            writeWeightedValuesBF16(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, uniform);
            return;
        }

        float sum = 0.0f;
        for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
            if (isMasked(maskArray, maskSegment, maskBase, keyIndex, plan.maskColStride)) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            float shifted = rowScores[keyIndex] - max;
            float exp = fastExp ? FastTranscendentals.fastExpF32(shifted) : (float) Math.exp(shifted);
            rowScores[keyIndex] = exp;
            sum += exp;
        }

        float inv = 1.0f / sum;
        if (cacheArray != null) {
            int weightsBase = row * plan.keyLen;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                float weight = rowScores[keyIndex] * inv;
                rowScores[keyIndex] = weight;
                cacheArray[weightsBase + keyIndex] = weight;
            }
        } else {
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                rowScores[keyIndex] *= inv;
            }
        }
        writeWeightedValuesBF16(plan, valueArray, valueSegment, outputArray, outputSegment, valueBatchBase, outputBase, rowScores, 0.0f);
    }

    private static void writeWeightedValuesF64(
            AttentionStoragePlan plan,
            double[] valueArray,
            MemorySegment valueSegment,
            double[] outputArray,
            MemorySegment outputSegment,
            int valueBatchBase,
            int outputBase,
            double[] rowWeights,
            double fallbackWeight
    ) {
        for (int valueColumn = 0; valueColumn < plan.valueDim; valueColumn++) {
            double sum = 0.0d;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                double weight = fallbackWeight == 0.0d ? rowWeights[keyIndex] : fallbackWeight;
                if (weight != 0.0d) {
                    int valueOffset = valueBatchBase + keyIndex * plan.valueRowStride + valueColumn * plan.valueColStride;
                    sum += readF64(valueArray, valueSegment, valueOffset) * weight;
                }
            }
            writeF64(outputArray, outputSegment, outputBase + valueColumn * plan.outputColStride, sum);
        }
    }

    private static void writeWeightedValuesF32(
            AttentionStoragePlan plan,
            float[] valueArray,
            MemorySegment valueSegment,
            float[] outputArray,
            MemorySegment outputSegment,
            int valueBatchBase,
            int outputBase,
            float[] rowWeights,
            float fallbackWeight
    ) {
        for (int valueColumn = 0; valueColumn < plan.valueDim; valueColumn++) {
            float sum = 0.0f;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                float weight = fallbackWeight == 0.0f ? rowWeights[keyIndex] : fallbackWeight;
                if (weight != 0.0f) {
                    int valueOffset = valueBatchBase + keyIndex * plan.valueRowStride + valueColumn * plan.valueColStride;
                    sum += readF32(valueArray, valueSegment, valueOffset) * weight;
                }
            }
            writeF32(outputArray, outputSegment, outputBase + valueColumn * plan.outputColStride, sum);
        }
    }

    private static void writeWeightedValuesBF16(
            AttentionStoragePlan plan,
            short[] valueArray,
            MemorySegment valueSegment,
            short[] outputArray,
            MemorySegment outputSegment,
            int valueBatchBase,
            int outputBase,
            float[] rowWeights,
            float fallbackWeight
    ) {
        for (int valueColumn = 0; valueColumn < plan.valueDim; valueColumn++) {
            float sum = 0.0f;
            for (int keyIndex = 0; keyIndex < plan.keyLen; keyIndex++) {
                float weight = fallbackWeight == 0.0f ? rowWeights[keyIndex] : fallbackWeight;
                if (weight != 0.0f) {
                    int valueOffset = valueBatchBase + keyIndex * plan.valueRowStride + valueColumn * plan.valueColStride;
                    sum += readBF16(valueArray, valueSegment, valueOffset) * weight;
                }
            }
            writeBF16(outputArray, outputSegment, outputBase + valueColumn * plan.outputColStride, sum);
        }
    }

    private static void writeCacheF64(double[] cacheArray, int row, int keyLen, double[] rowScores, double value) {
        Arrays.fill(rowScores, 0, keyLen, value);
        if (cacheArray == null) {
            return;
        }
        int weightsBase = row * keyLen;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            cacheArray[weightsBase + keyIndex] = value;
        }
    }

    private static void writeCacheF32(float[] cacheArray, int row, int keyLen, float[] rowScores, float value) {
        Arrays.fill(rowScores, 0, keyLen, value);
        if (cacheArray == null) {
            return;
        }
        int weightsBase = row * keyLen;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            cacheArray[weightsBase + keyIndex] = value;
        }
    }

    private static boolean isMasked(byte[] maskArray, MemorySegment maskSegment, int maskBase, int keyIndex, int maskColStride) {
        return maskBase >= 0 && readBool(maskArray, maskSegment, maskBase + keyIndex * maskColStride) == 0;
    }

    private static double readF64(double[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(double[] array, MemorySegment segment, int offset, double value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static float readF32(float[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(float[] array, MemorySegment segment, int offset, float value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static float readBF16(short[] array, MemorySegment segment, int offset) {
        short bits = array != null ? array[offset] : segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    private static void writeBF16(short[] array, MemorySegment segment, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (array != null) {
            array[offset] = bits;
        } else {
            segment.set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    private static byte readBool(byte[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_BYTE, offset);
    }

    private static int[] computeBatchOffsets(int[] inputShape, int[] inputStrides, int[] outputShape) {
        int inputBatchRank = inputShape.length - 2;
        int outputBatchRank = outputShape.length - 2;
        int batchCount = batchCount(outputShape);
        int[] offsets = new int[batchCount];
        if (outputBatchRank == 0) {
            return offsets;
        }
        int[] outputBatchShape = Arrays.copyOf(outputShape, outputBatchRank);
        int[] outputBatchDenseStrides = denseStrides(outputBatchShape);
        int shapeOffset = outputBatchRank - inputBatchRank;
        for (int batch = 0; batch < batchCount; batch++) {
            int remaining = batch;
            int offset = 0;
            for (int dim = 0; dim < outputBatchRank; dim++) {
                int coordinate = remaining / outputBatchDenseStrides[dim];
                remaining %= outputBatchDenseStrides[dim];
                int inputDimIndex = dim - shapeOffset;
                if (inputDimIndex >= 0 && inputShape[inputDimIndex] != 1) {
                    offset += coordinate * inputStrides[inputDimIndex];
                }
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int batchCount(int[] shape) {
        int count = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            count *= shape[i];
        }
        return count;
    }

    private record AttentionStoragePlan(
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            int totalRows,
            int queryStorageOffset,
            int keyStorageOffset,
            int valueStorageOffset,
            int outputStorageOffset,
            int maskStorageOffset,
            int queryRowStride,
            int queryColStride,
            int keyRowStride,
            int keyColStride,
            int valueRowStride,
            int valueColStride,
            int outputRowStride,
            int outputColStride,
            int maskRowStride,
            int maskColStride,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] outputBatchOffsets,
            int[] maskBatchOffsets
    ) {
        static AttentionStoragePlan create(
                CpuStorageView query,
                CpuStorageView key,
                CpuStorageView value,
                CpuStorageView mask,
                CpuStorageView output
        ) {
            int[] queryShape = query.shape();
            int[] keyShape = key.shape();
            int[] valueShape = value.shape();
            int[] outputShape = output.shape();
            int[] queryStrides = query.strides();
            int[] keyStrides = key.strides();
            int[] valueStrides = value.strides();
            int[] outputStrides = output.strides();
            int[] maskShape = mask == null ? null : mask.shape();
            int[] maskStrides = mask == null ? null : mask.strides();
            int batchCount = batchCount(outputShape);
            int queryLen = outputShape[outputShape.length - 2];
            int keyLen = keyShape[keyShape.length - 2];
            return new AttentionStoragePlan(
                    queryLen,
                    keyLen,
                    queryShape[queryShape.length - 1],
                    outputShape[outputShape.length - 1],
                    batchCount * queryLen,
                    query.storageOffset(),
                    key.storageOffset(),
                    value.storageOffset(),
                    output.storageOffset(),
                    mask == null ? 0 : mask.storageOffset(),
                    queryStrides[queryStrides.length - 2],
                    queryStrides[queryStrides.length - 1],
                    keyStrides[keyStrides.length - 2],
                    keyStrides[keyStrides.length - 1],
                    valueStrides[valueStrides.length - 2],
                    valueStrides[valueStrides.length - 1],
                    outputStrides[outputStrides.length - 2],
                    outputStrides[outputStrides.length - 1],
                    mask == null ? 0 : maskStrides[maskStrides.length - 2],
                    mask == null ? 0 : maskStrides[maskStrides.length - 1],
                    computeBatchOffsets(queryShape, queryStrides, outputShape),
                    computeBatchOffsets(keyShape, keyStrides, outputShape),
                    computeBatchOffsets(valueShape, valueStrides, outputShape),
                    computeBatchOffsets(outputShape, outputStrides, outputShape),
                    mask == null ? null : computeBatchOffsets(maskShape, maskStrides, outputShape)
            );
        }
    }
}
