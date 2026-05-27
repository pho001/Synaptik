package backend.cpu.kernels.linalg;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.plan.linalg.attention.ResolvedAttentionHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.kernels.linalg.matmul.common.MatMulBatchingSupport;
import backend.cpu.storage.CpuStorageResolver;
import backend.cpu.storage.CpuStorageView;
import operations.linalg.scaledDotProductAttention;
import tensor.DataType;
import tensor.Tensor;
import utils.FastTranscendentals;

import java.util.Arrays;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

final class ScaledDotProductAttentionExecutor {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final ThreadLocal<double[]> F64_ATTENTION_SCORES = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_ATTENTION_SCORES = ThreadLocal.withInitial(() -> new float[0]);

    private ScaledDotProductAttentionExecutor() {}

    static void executeF64(
            scaledDotProductAttention attention,
            Tensor[] inputTensors,
            CpuStorageView[] inputViews,
            Tensor node,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        Tensor maskTensor = inputTensors.length == 4 ? inputTensors[3] : null;
        CpuStorageView query = inputViews[0];
        CpuStorageView key = inputViews[1];
        CpuStorageView value = inputViews[2];
        CpuStorageView mask = inputViews.length == 4 ? inputViews[3] : null;
        validate(attention, inputTensors[0], inputTensors[1], inputTensors[2], maskTensor, node);
        validateViews(inputTensors, inputViews, node, output);
        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] outputShape = output.shape();
        int[] scoresShape = scoreShape(queryShape, keyShape);
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        CpuStorageView weights = cacheView(runtimeCache);
        if (canUseDenseArrayForward(inputViews, output)) {
            executeDirectF64(
                    query.requireF64Array(),
                    queryShape,
                    key.requireF64Array(),
                    keyShape,
                    value.requireF64Array(),
                    valueShape,
                    mask == null ? null : mask.requireBoolArray(),
                    mask == null ? null : mask.shape(),
                    output.requireF64Array(),
                    outputShape,
                    attention.getScale(),
                    plan.forwardDirectHints(),
                    context.useFastExpApprox(),
                    weights == null ? null : weights.requireF64Array()
            );
            return;
        }
        CpuScaledDotProductAttentionStorageLoops.executeF64(
                query, key, value, mask, output, attention.getScale(), context.useFastExpApprox(), weights);
    }

    static void executeF32(
            scaledDotProductAttention attention,
            Tensor[] inputTensors,
            CpuStorageView[] inputViews,
            Tensor node,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        Tensor maskTensor = inputTensors.length == 4 ? inputTensors[3] : null;
        CpuStorageView query = inputViews[0];
        CpuStorageView key = inputViews[1];
        CpuStorageView value = inputViews[2];
        CpuStorageView mask = inputViews.length == 4 ? inputViews[3] : null;
        validate(attention, inputTensors[0], inputTensors[1], inputTensors[2], maskTensor, node);
        validateViews(inputTensors, inputViews, node, output);
        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] outputShape = output.shape();
        int[] scoresShape = scoreShape(queryShape, keyShape);
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        CpuStorageView weights = cacheView(runtimeCache);
        if (canUseDenseArrayForward(inputViews, output)) {
            executeDirectF32(
                    query.requireF32Array(),
                    queryShape,
                    key.requireF32Array(),
                    keyShape,
                    value.requireF32Array(),
                    valueShape,
                    mask == null ? null : mask.requireBoolArray(),
                    mask == null ? null : mask.shape(),
                    output.requireF32Array(),
                    outputShape,
                    (float) attention.getScale(),
                    plan.forwardDirectHints(),
                    context.useFastExpApprox(),
                    weights == null ? null : weights.requireF32Array(),
                    null
            );
            return;
        }
        CpuScaledDotProductAttentionStorageLoops.executeF32(
                query, key, value, mask, output, (float) attention.getScale(), context.useFastExpApprox(), weights);
    }

    static void executeBF16(
            scaledDotProductAttention attention,
            Tensor[] inputTensors,
            CpuStorageView[] inputViews,
            Tensor node,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        Tensor maskTensor = inputTensors.length == 4 ? inputTensors[3] : null;
        CpuStorageView query = inputViews[0];
        CpuStorageView key = inputViews[1];
        CpuStorageView value = inputViews[2];
        CpuStorageView mask = inputViews.length == 4 ? inputViews[3] : null;
        validate(attention, inputTensors[0], inputTensors[1], inputTensors[2], maskTensor, node);
        validateViews(inputTensors, inputViews, node, output);

        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] outputShape = output.shape();
        int[] scoresShape = scoreShape(queryShape, keyShape);
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        CpuStorageView weights = cacheView(runtimeCache);
        if (canUseDenseArrayForward(inputViews, output)) {
            float[] queryF32 = resolveBF16InputF32(context, 0, query);
            float[] keyF32 = resolveBF16InputF32(context, 1, key);
            float[] valueF32 = resolveBF16InputF32(context, 2, value);
            float[] outF32 = context.publishFloatContinuation() && context.cpuWorkspace() != null
                    ? context.cpuWorkspace().requireFloatWorkspace()
                    : new float[output.logicalSize()];
            executeDirectF32(
                    queryF32,
                    queryShape,
                    keyF32,
                    keyShape,
                    valueF32,
                    valueShape,
                    mask == null ? null : mask.requireBoolArray(),
                    mask == null ? null : mask.shape(),
                    outF32,
                    outputShape,
                    (float) attention.getScale(),
                    plan.forwardDirectHints(),
                    context.useFastExpApprox(),
                    weights == null ? null : weights.requireF32Array(),
                    null
            );
            if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
                context.cpuWorkspace().publishFloatContinuation(output.logicalSize());
                return;
            }
            writeBF16(outF32, output.requireBF16Array(), output.storageOffset(), output.logicalSize());
            return;
        }
        CpuScaledDotProductAttentionStorageLoops.executeBF16(
                query, key, value, mask, output, (float) attention.getScale(), context.useFastExpApprox(), weights);
    }

    private static void executeDirectF32(
            float[] query, int[] queryShape,
            float[] key, int[] keyShape,
            float[] value, int[] valueShape,
            byte[] mask, int[] maskShape,
            float[] out, int[] outShape,
            float scale,
            ResolvedAttentionHints hints,
            boolean fastExp,
            float[] cachedWeightsF32,
            short[] cachedWeightsBF16
    ) {
        int batchCount = batchCount(outShape);
        int queryLen = outShape[outShape.length - 2];
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        int valueDim = outShape[outShape.length - 1];
        int[] queryBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(queryShape, outShape);
        int[] keyBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(keyShape, outShape);
        int[] valueBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(valueShape, outShape);
        int[] maskBatchOffsets = mask == null ? null : MatMulBatchingSupport.computeBatchOffsets(maskShape, outShape);
        int totalRows = batchCount * queryLen;
        if (hints == null) {
            throw new IllegalStateException("Missing prepared forward attention hints.");
        }
        if (hints.parallel() && totalRows > 1) {
            int rowsPerChunk = hints.taskChunkSize();
            int chunks = (totalRows + rowsPerChunk - 1) / rowsPerChunk;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * rowsPerChunk;
                int end = Math.min(start + rowsPerChunk, totalRows);
                float[] rowScores = ensureF32ScoreCapacity(keyLen);
                for (int row = start; row < end; row++) {
                    computeAttentionRowF32(
                            query, key, value, mask, out,
                            queryBatchOffsets, keyBatchOffsets, valueBatchOffsets, maskBatchOffsets,
                            row, queryLen, keyLen, depth, valueDim, scale, hints.vectorized(), fastExp, rowScores,
                            cachedWeightsF32, cachedWeightsBF16
                    );
                }
            });
            return;
        }
        float[] rowScores = ensureF32ScoreCapacity(keyLen);
        for (int row = 0; row < totalRows; row++) {
            computeAttentionRowF32(
                    query, key, value, mask, out,
                    queryBatchOffsets, keyBatchOffsets, valueBatchOffsets, maskBatchOffsets,
                    row, queryLen, keyLen, depth, valueDim, scale, hints.vectorized(), fastExp, rowScores,
                    cachedWeightsF32, cachedWeightsBF16
            );
        }
    }

    private static void executeDirectF64(
            double[] query, int[] queryShape,
            double[] key, int[] keyShape,
            double[] value, int[] valueShape,
            byte[] mask, int[] maskShape,
            double[] out, int[] outShape,
            double scale,
            ResolvedAttentionHints hints,
            boolean fastExp,
            double[] cachedWeights
    ) {
        int batchCount = batchCount(outShape);
        int queryLen = outShape[outShape.length - 2];
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        int valueDim = outShape[outShape.length - 1];
        int[] queryBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(queryShape, outShape);
        int[] keyBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(keyShape, outShape);
        int[] valueBatchOffsets = MatMulBatchingSupport.computeBatchOffsets(valueShape, outShape);
        int[] maskBatchOffsets = mask == null ? null : MatMulBatchingSupport.computeBatchOffsets(maskShape, outShape);
        int totalRows = batchCount * queryLen;
        if (hints == null) {
            throw new IllegalStateException("Missing prepared forward attention hints.");
        }
        if (hints.parallel() && totalRows > 1) {
            int rowsPerChunk = hints.taskChunkSize();
            int chunks = (totalRows + rowsPerChunk - 1) / rowsPerChunk;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * rowsPerChunk;
                int end = Math.min(start + rowsPerChunk, totalRows);
                double[] rowScores = ensureF64ScoreCapacity(keyLen);
                for (int row = start; row < end; row++) {
                    computeAttentionRowF64(
                            query, key, value, mask, out,
                            queryBatchOffsets, keyBatchOffsets, valueBatchOffsets, maskBatchOffsets,
                            row, queryLen, keyLen, depth, valueDim, scale, hints.vectorized(), fastExp, rowScores, cachedWeights
                    );
                }
            });
            return;
        }
        double[] rowScores = ensureF64ScoreCapacity(keyLen);
        for (int row = 0; row < totalRows; row++) {
            computeAttentionRowF64(
                    query, key, value, mask, out,
                    queryBatchOffsets, keyBatchOffsets, valueBatchOffsets, maskBatchOffsets,
                    row, queryLen, keyLen, depth, valueDim, scale, hints.vectorized(), fastExp, rowScores, cachedWeights
            );
        }
    }

    private static void computeAttentionRowF32(
            float[] query,
            float[] key,
            float[] value,
            byte[] mask,
            float[] out,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] maskBatchOffsets,
            int row,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            float scale,
            boolean vectorized,
            boolean fastExp,
            float[] rowScores,
            float[] cachedWeightsF32,
            short[] cachedWeightsBF16
    ) {
        int batch = row / queryLen;
        int queryIndex = row % queryLen;
        int queryBase = queryBatchOffsets[batch] + queryIndex * depth;
        int keyBase = keyBatchOffsets[batch];
        int valueBase = valueBatchOffsets[batch];
        int outBase = row * valueDim;
        int maskBase = mask == null ? -1 : maskBatchOffsets[batch] + queryIndex * keyLen;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            float score = (vectorized
                    ? dotF32(query, queryBase, key, keyBase + keyIndex * depth, depth, true)
                    : dotF32(query, queryBase, key, keyBase + keyIndex * depth, depth, false)) * scale;
            rowScores[keyIndex] = score;
            if (score > max) {
                max = score;
            }
            anyValid = true;
        }

        if (!anyValid) {
            float uniform = 1.0f / keyLen;
            Arrays.fill(out, outBase, outBase + valueDim, 0.0f);
            int weightsBase = row * keyLen;
            for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                if (cachedWeightsF32 != null) {
                    cachedWeightsF32[weightsBase + keyIndex] = uniform;
                } else if (cachedWeightsBF16 != null) {
                    cachedWeightsBF16[weightsBase + keyIndex] = TensorDTypeOps.toBFloat16Bits(uniform);
                }
                if (vectorized) {
                    accumulateWeightedF32(out, outBase, value, valueBase + keyIndex * valueDim, uniform, valueDim, true);
                } else {
                    accumulateWeightedF32(out, outBase, value, valueBase + keyIndex * valueDim, uniform, valueDim, false);
                }
            }
            return;
        }

        float sum = 0.0f;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                rowScores[keyIndex] = 0.0f;
                continue;
            }
            float shifted = rowScores[keyIndex] - max;
            float exp = fastExp ? FastTranscendentals.fastExpF32(shifted) : (float) Math.exp(shifted);
            rowScores[keyIndex] = exp;
            sum += exp;
        }

        float inv = 1.0f / sum;
        Arrays.fill(out, outBase, outBase + valueDim, 0.0f);
        int weightsBase = row * keyLen;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            float weight = rowScores[keyIndex] * inv;
            if (cachedWeightsF32 != null) {
                cachedWeightsF32[weightsBase + keyIndex] = weight;
            } else if (cachedWeightsBF16 != null) {
                cachedWeightsBF16[weightsBase + keyIndex] = TensorDTypeOps.toBFloat16Bits(weight);
            }
            if (weight == 0.0f) {
                continue;
            }
            if (vectorized) {
                accumulateWeightedF32(out, outBase, value, valueBase + keyIndex * valueDim, weight, valueDim, true);
            } else {
                accumulateWeightedF32(out, outBase, value, valueBase + keyIndex * valueDim, weight, valueDim, false);
            }
        }
    }

    private static void computeAttentionRowF64(
            double[] query,
            double[] key,
            double[] value,
            byte[] mask,
            double[] out,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] maskBatchOffsets,
            int row,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            double scale,
            boolean vectorized,
            boolean fastExp,
            double[] rowScores,
            double[] cachedWeights
    ) {
        int batch = row / queryLen;
        int queryIndex = row % queryLen;
        int queryBase = queryBatchOffsets[batch] + queryIndex * depth;
        int keyBase = keyBatchOffsets[batch];
        int valueBase = valueBatchOffsets[batch];
        int outBase = row * valueDim;
        int maskBase = mask == null ? -1 : maskBatchOffsets[batch] + queryIndex * keyLen;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                rowScores[keyIndex] = 0.0d;
                continue;
            }
            double score = (vectorized
                    ? dotF64(query, queryBase, key, keyBase + keyIndex * depth, depth, true)
                    : dotF64(query, queryBase, key, keyBase + keyIndex * depth, depth, false)) * scale;
            rowScores[keyIndex] = score;
            if (score > max) {
                max = score;
            }
            anyValid = true;
        }

        if (!anyValid) {
            double uniform = 1.0d / keyLen;
            Arrays.fill(out, outBase, outBase + valueDim, 0.0d);
            int weightsBase = row * keyLen;
            for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                if (cachedWeights != null) {
                    cachedWeights[weightsBase + keyIndex] = uniform;
                }
                if (vectorized) {
                    accumulateWeightedF64(out, outBase, value, valueBase + keyIndex * valueDim, uniform, valueDim, true);
                } else {
                    accumulateWeightedF64(out, outBase, value, valueBase + keyIndex * valueDim, uniform, valueDim, false);
                }
            }
            return;
        }

        double sum = 0.0d;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                rowScores[keyIndex] = 0.0d;
                continue;
            }
            double shifted = rowScores[keyIndex] - max;
            double exp = fastExp ? FastTranscendentals.fastExpF64(shifted) : Math.exp(shifted);
            rowScores[keyIndex] = exp;
            sum += exp;
        }

        double inv = 1.0d / sum;
        Arrays.fill(out, outBase, outBase + valueDim, 0.0d);
        int weightsBase = row * keyLen;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            double weight = rowScores[keyIndex] * inv;
            if (cachedWeights != null) {
                cachedWeights[weightsBase + keyIndex] = weight;
            }
            if (weight == 0.0d) {
                continue;
            }
            if (vectorized) {
                accumulateWeightedF64(out, outBase, value, valueBase + keyIndex * valueDim, weight, valueDim, true);
            } else {
                accumulateWeightedF64(out, outBase, value, valueBase + keyIndex * valueDim, weight, valueDim, false);
            }
        }
    }

    private static double[] ensureF64ScoreCapacity(int length) {
        double[] buffer = F64_ATTENTION_SCORES.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        double[] resized = new double[newLength];
        F64_ATTENTION_SCORES.set(resized);
        return resized;
    }

    private static float[] ensureF32ScoreCapacity(int length) {
        float[] buffer = F32_ATTENTION_SCORES.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        float[] resized = new float[newLength];
        F32_ATTENTION_SCORES.set(resized);
        return resized;
    }

    private static ScaledDotProductAttentionRuntimeCache prepareRuntimeCache(CpuKernelContext context, Tensor node, int[] scoresShape) {
        if (context == null) {
            throw new IllegalStateException("attention execution requires CpuKernelContext");
        }
        if (!node.getRequiresGrad()) {
            context.clearRuntimeState(node);
            return null;
        }
        DataType cacheType = node.getDataType() == DataType.BFLOAT16 ? DataType.FLOAT32 : node.getDataType();
        ScaledDotProductAttentionRuntimeCache runtimeCache = context.runtimeStateFor(node, ScaledDotProductAttentionRuntimeCache.class);
        if (runtimeCache != null
                && Arrays.equals(runtimeCache.weights().getShapeUnsafe(), scoresShape)
                && runtimeCache.weights().getDataType() == cacheType) {
            runtimeCache.resetForNextExecution();
            return runtimeCache;
        }
        Tensor weights = new Tensor(scoresShape.clone(), List.of(), "attention_weights_cache", cacheType);
        runtimeCache = new ScaledDotProductAttentionRuntimeCache(weights);
        context.putRuntimeState(node, runtimeCache);
        return runtimeCache;
    }

    private static float dotF32(float[] left, int leftBase, float[] right, int rightBase, int length, boolean vectorized) {
        if (!vectorized) {
            float sum = 0.0f;
            for (int i = 0; i < length; i++) {
                sum += left[leftBase + i] * right[rightBase + i];
            }
            return sum;
        }
        int i = 0;
        FloatVector acc = FloatVector.zero(F32);
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            acc = acc.add(
                    FloatVector.fromArray(F32, left, leftBase + i)
                            .mul(FloatVector.fromArray(F32, right, rightBase + i))
            );
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static double dotF64(double[] left, int leftBase, double[] right, int rightBase, int length, boolean vectorized) {
        if (!vectorized) {
            double sum = 0.0d;
            for (int i = 0; i < length; i++) {
                sum += left[leftBase + i] * right[rightBase + i];
            }
            return sum;
        }
        int i = 0;
        DoubleVector acc = DoubleVector.zero(F64);
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            acc = acc.add(
                    DoubleVector.fromArray(F64, left, leftBase + i)
                            .mul(DoubleVector.fromArray(F64, right, rightBase + i))
            );
        }
        double sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static void accumulateWeightedF32(float[] out, int outBase, float[] value, int valueBase, float weight, int length, boolean vectorized) {
        if (!vectorized) {
            for (int i = 0; i < length; i++) {
                out[outBase + i] += value[valueBase + i] * weight;
            }
            return;
        }
        int i = 0;
        FloatVector scaleVector = FloatVector.broadcast(F32, weight);
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector outVector = FloatVector.fromArray(F32, out, outBase + i);
            FloatVector valueVector = FloatVector.fromArray(F32, value, valueBase + i);
            outVector.add(valueVector.mul(scaleVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] += value[valueBase + i] * weight;
        }
    }

    private static void accumulateWeightedF64(double[] out, int outBase, double[] value, int valueBase, double weight, int length, boolean vectorized) {
        if (!vectorized) {
            for (int i = 0; i < length; i++) {
                out[outBase + i] += value[valueBase + i] * weight;
            }
            return;
        }
        int i = 0;
        DoubleVector scaleVector = DoubleVector.broadcast(F64, weight);
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector outVector = DoubleVector.fromArray(F64, out, outBase + i);
            DoubleVector valueVector = DoubleVector.fromArray(F64, value, valueBase + i);
            outVector.add(valueVector.mul(scaleVector)).intoArray(out, outBase + i);
        }
        for (; i < length; i++) {
            out[outBase + i] += value[valueBase + i] * weight;
        }
    }

    private static float[] resolveBF16InputF32(CpuKernelContext context, int inputIndex, CpuStorageView input) {
        float[] continuation = context == null ? null : context.inputFloatContinuation(inputIndex, input.logicalSize());
        if (continuation != null) {
            return continuation;
        }
        return toF32(input.requireBF16Array(), input.storageOffset(), input.logicalSize());
    }

    private static float[] toF32(short[] src, int offset, int length) {
        float[] out = new float[length];
        for (int i = 0; i < length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(src[offset + i]);
        }
        return out;
    }

    private static void writeBF16(float[] src, short[] dst, int offset, int length) {
        for (int i = 0; i < length; i++) {
            dst[offset + i] = TensorDTypeOps.toBFloat16Bits(src[i]);
        }
    }

    private static int[] scoreShape(int[] queryShape, int[] keyShape) {
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] outBatch = broadcastLeadingShape(qBatch, kBatch);
        int[] out = Arrays.copyOf(outBatch, outBatch.length + 2);
        out[outBatch.length] = queryShape[queryShape.length - 2];
        out[outBatch.length + 1] = keyShape[keyShape.length - 2];
        return out;
    }

    private static int[] outputShape(int[] queryShape, int[] keyShape, int[] valueShape) {
        int[] scoresShape = scoreShape(queryShape, keyShape);
        int[] scoresBatch = Arrays.copyOf(scoresShape, scoresShape.length - 2);
        int[] valueBatch = Arrays.copyOf(valueShape, valueShape.length - 2);
        int[] outBatch = broadcastLeadingShape(scoresBatch, valueBatch);
        int[] out = Arrays.copyOf(outBatch, outBatch.length + 2);
        out[outBatch.length] = queryShape[queryShape.length - 2];
        out[outBatch.length + 1] = valueShape[valueShape.length - 1];
        return out;
    }

    private static int[] broadcastLeadingShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int a = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int b = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (a != b && a != 1 && b != 1) {
                throw new IllegalArgumentException("attention batch dimensions are not broadcast-compatible.");
            }
            out[i] = Math.max(a, b);
        }
        return out;
    }

    private static int batchCount(int[] shape) {
        int count = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            count *= shape[i];
        }
        return count;
    }

    private static void validate(
            scaledDotProductAttention attention,
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            Tensor node
    ) {
        if (attention == null || query == null || key == null || value == null || node == null) {
            throw new IllegalArgumentException("attention execution arguments cannot be null");
        }
        if (query.getDataType() != key.getDataType()
                || query.getDataType() != value.getDataType()
                || query.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("attention requires matching floating dtypes for q, k, v and output");
        }
        if (query.getDataType() == DataType.BOOL || query.getDataType() == DataType.INT32 || query.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("attention requires floating inputs");
        }
        if (mask != null && mask.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("attention mask must have BOOL dtype");
        }
        if (query.getShapeUnsafe().length < 2 || key.getShapeUnsafe().length < 2 || value.getShapeUnsafe().length < 2) {
            throw new IllegalArgumentException("attention requires rank >= 2 tensors");
        }
        if (query.getShapeUnsafe()[query.getShapeUnsafe().length - 1] != key.getShapeUnsafe()[key.getShapeUnsafe().length - 1]) {
            throw new IllegalArgumentException("attention query/key head dimension mismatch");
        }
        if (key.getShapeUnsafe()[key.getShapeUnsafe().length - 2] != value.getShapeUnsafe()[value.getShapeUnsafe().length - 2]) {
            throw new IllegalArgumentException("attention key/value sequence dimension mismatch");
        }
        int[] expectedScoresShape = scoreShape(query.getShapeUnsafe(), key.getShapeUnsafe());
        int[] expectedOutShape = outputShape(query.getShapeUnsafe(), key.getShapeUnsafe(), value.getShapeUnsafe());
        if (!Arrays.equals(expectedOutShape, node.getShapeUnsafe())) {
            throw new IllegalArgumentException("attention output shape mismatch");
        }
        if (mask != null && !Arrays.equals(mask.getShapeUnsafe(), expectedScoresShape)) {
            throw new IllegalArgumentException("attention mask shape must equal broadcasted scores shape");
        }
    }

    private static void validateViews(Tensor[] inputTensors, CpuStorageView[] inputViews, Tensor node, CpuStorageView output) {
        if (output.dtype() != node.getDataType() || !Arrays.equals(output.shape(), node.getShapeUnsafe())) {
            throw new IllegalArgumentException("attention output storage view does not match output tensor metadata");
        }
        for (int i = 0; i < inputTensors.length; i++) {
            if (inputViews[i].dtype() != inputTensors[i].getDataType()
                    || !Arrays.equals(inputViews[i].shape(), inputTensors[i].getShapeUnsafe())) {
                throw new IllegalArgumentException("attention input storage view " + i
                        + " does not match input tensor metadata");
            }
        }
    }

    private static CpuStorageView cacheView(ScaledDotProductAttentionRuntimeCache runtimeCache) {
        if (runtimeCache == null) {
            return null;
        }
        return new CpuStorageResolver().bindArrayOnly(runtimeCache.weights());
    }

    private static boolean canUseDenseArrayForward(CpuStorageView[] inputs, CpuStorageView output) {
        if (!output.isArray() || !isDenseZeroOffset(output)) {
            return false;
        }
        for (CpuStorageView input : inputs) {
            if (!input.isArray() || !isDenseZeroOffset(input)) {
                return false;
            }
        }
        return true;
    }

    private static boolean isDenseZeroOffset(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[i]);
        }
        return true;
    }

    private static ResolvedScaledDotProductAttentionPlan requirePlan(CpuKernelContext context, Tensor node) {
        if (context == null || context.attentionPlan() == null) {
            throw new IllegalStateException("Missing prepared attention plan for node " + node.getLabel());
        }
        return context.attentionPlan();
    }
}
