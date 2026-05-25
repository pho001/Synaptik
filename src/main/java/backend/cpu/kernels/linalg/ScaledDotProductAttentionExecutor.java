package backend.cpu.kernels.linalg;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.plan.linalg.attention.ResolvedAttentionHints;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.common.MatMulBatchingSupport;
import backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend;
import backend.cpu.kernels.linalg.matmul.f64.F64MatMulJavaBackend;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionBackward;
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
    private static final int ATTENTION_MICRO_DIM = 32;
    private static final int ATTENTION_F32_VALUE_BLOCK_KEYS = 4;
    private static final int ATTENTION_F64_VALUE_BLOCK_KEYS = 2;
    private static final ThreadLocal<double[]> F64_ATTENTION_SCORES = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_ATTENTION_SCORES = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<double[]> F64_ATTENTION_DWEIGHTS = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_ATTENTION_DWEIGHTS = ThreadLocal.withInitial(() -> new float[0]);
    private static final ThreadLocal<double[]> F64_ATTENTION_QUERY_GRAD_ROW = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<float[]> F32_ATTENTION_QUERY_GRAD_ROW = ThreadLocal.withInitial(() -> new float[0]);

    private ScaledDotProductAttentionExecutor() {}

    static void executeF64(scaledDotProductAttention attention, Tensor[] inputs, Tensor node, CpuKernelContext context) {
        Tensor query = inputs[0];
        Tensor key = inputs[1];
        Tensor value = inputs[2];
        Tensor mask = inputs.length == 4 ? inputs[3] : null;
        validate(attention, query, key, value, mask, node);
        int[] scoresShape = scoreShape(query.getShapeUnsafe(), key.getShapeUnsafe());
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        executeDirectF64(
                TensorInternalAccess.float64Data(query),
                query.getShapeUnsafe(),
                TensorInternalAccess.float64Data(key),
                key.getShapeUnsafe(),
                TensorInternalAccess.float64Data(value),
                value.getShapeUnsafe(),
                mask == null ? null : TensorInternalAccess.boolData(mask),
                mask == null ? null : mask.getShapeUnsafe(),
                TensorInternalAccess.float64Data(node),
                node.getShapeUnsafe(),
                attention.getScale(),
                plan.forwardDirectHints(),
                context.useFastExpApprox(),
                runtimeCache == null ? null : TensorInternalAccess.float64Data(runtimeCache.weights())
        );
    }

    static void executeF32(scaledDotProductAttention attention, Tensor[] inputs, Tensor node, CpuKernelContext context) {
        Tensor query = inputs[0];
        Tensor key = inputs[1];
        Tensor value = inputs[2];
        Tensor mask = inputs.length == 4 ? inputs[3] : null;
        validate(attention, query, key, value, mask, node);
        int[] scoresShape = scoreShape(query.getShapeUnsafe(), key.getShapeUnsafe());
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        executeDirectF32(
                TensorInternalAccess.float32Data(query),
                query.getShapeUnsafe(),
                TensorInternalAccess.float32Data(key),
                key.getShapeUnsafe(),
                TensorInternalAccess.float32Data(value),
                value.getShapeUnsafe(),
                mask == null ? null : TensorInternalAccess.boolData(mask),
                mask == null ? null : mask.getShapeUnsafe(),
                TensorInternalAccess.float32Data(node),
                node.getShapeUnsafe(),
                (float) attention.getScale(),
                plan.forwardDirectHints(),
                context.useFastExpApprox(),
                runtimeCache == null ? null : TensorInternalAccess.float32Data(runtimeCache.weights()),
                null
        );
    }

    static void executeBF16(scaledDotProductAttention attention, Tensor[] inputs, Tensor node, CpuKernelContext context) {
        Tensor query = inputs[0];
        Tensor key = inputs[1];
        Tensor value = inputs[2];
        Tensor mask = inputs.length == 4 ? inputs[3] : null;
        validate(attention, query, key, value, mask, node);

        float[] queryF32 = resolveBF16InputF32(context, 0, query, "attention_query_f32");
        float[] keyF32 = resolveBF16InputF32(context, 1, key, "attention_key_f32");
        float[] valueF32 = resolveBF16InputF32(context, 2, value, "attention_value_f32");
        int[] queryShape = query.getShapeUnsafe();
        int[] keyShape = key.getShapeUnsafe();
        int[] valueShape = value.getShapeUnsafe();
        int[] scoresShape = scoreShape(queryShape, keyShape);
        ScaledDotProductAttentionRuntimeCache runtimeCache = prepareRuntimeCache(context, node, scoresShape);
        ResolvedScaledDotProductAttentionPlan plan = requirePlan(context, node);
        float[] outF32 = context.publishFloatContinuation() && context.cpuWorkspace() != null
                ? context.cpuWorkspace().requireFloatWorkspace()
                : new float[node.getFlatDataSize()];
        executeDirectF32(
                queryF32,
                queryShape,
                keyF32,
                keyShape,
                valueF32,
                valueShape,
                mask == null ? null : TensorInternalAccess.boolData(mask),
                mask == null ? null : mask.getShapeUnsafe(),
                outF32,
                node.getShapeUnsafe(),
                (float) attention.getScale(),
                plan.forwardDirectHints(),
                context.useFastExpApprox(),
                runtimeCache == null ? null : TensorInternalAccess.float32Data(runtimeCache.weights()),
                null
        );
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return;
        }
        writeBF16(outF32, TensorInternalAccess.bfloat16Data(node));
    }

    static void executeBackwardF64(
            scaledDotProductAttentionBackward.OutputKind outputKind,
            Tensor attentionOut,
            Tensor outGrad,
            Tensor node,
            CpuKernelContext context
    ) {
        AttentionBackwardSpec spec = validateBackward(outputKind, attentionOut, outGrad, node);
        ScaledDotProductAttentionRuntimeCache runtimeCache = requireRuntimeCache(context, attentionOut);
        ensureBackwardGradsF64(spec, attentionOut, outGrad, runtimeCache, requirePlan(context, node));
        Tensor cached = switch (outputKind) {
            case QUERY -> runtimeCache.queryGrad();
            case KEY -> runtimeCache.keyGrad();
            case VALUE -> runtimeCache.valueGrad();
        };
        System.arraycopy(TensorInternalAccess.float64Data(cached), 0, TensorInternalAccess.float64Data(node), 0, node.getFlatDataSize());
    }

    static void executeBackwardF32(
            scaledDotProductAttentionBackward.OutputKind outputKind,
            Tensor attentionOut,
            Tensor outGrad,
            Tensor node,
            CpuKernelContext context
    ) {
        AttentionBackwardSpec spec = validateBackward(outputKind, attentionOut, outGrad, node);
        ScaledDotProductAttentionRuntimeCache runtimeCache = requireRuntimeCache(context, attentionOut);
        ensureBackwardGradsF32(spec, attentionOut, outGrad, runtimeCache, requirePlan(context, node));
        Tensor cached = switch (outputKind) {
            case QUERY -> runtimeCache.queryGrad();
            case KEY -> runtimeCache.keyGrad();
            case VALUE -> runtimeCache.valueGrad();
        };
        System.arraycopy(TensorInternalAccess.float32Data(cached), 0, TensorInternalAccess.float32Data(node), 0, node.getFlatDataSize());
    }

    static void executeBackwardBF16(
            scaledDotProductAttentionBackward.OutputKind outputKind,
            Tensor attentionOut,
            Tensor outGrad,
            Tensor node,
            CpuKernelContext context
    ) {
        AttentionBackwardSpec spec = validateBackward(outputKind, attentionOut, outGrad, node);
        ScaledDotProductAttentionRuntimeCache runtimeCache = requireRuntimeCache(context, attentionOut);
        ensureBackwardGradsBF16(spec, attentionOut, outGrad, runtimeCache, requirePlan(context, node), context);
        Tensor cached = switch (outputKind) {
            case QUERY -> runtimeCache.queryGrad();
            case KEY -> runtimeCache.keyGrad();
            case VALUE -> runtimeCache.valueGrad();
        };
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            System.arraycopy(TensorInternalAccess.float32Data(cached), 0, out, 0, node.getFlatDataSize());
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return;
        }
        writeBF16(TensorInternalAccess.float32Data(cached), TensorInternalAccess.bfloat16Data(node));
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

    private static double[] ensureF64GradientCapacity(int length) {
        double[] buffer = F64_ATTENTION_DWEIGHTS.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        double[] resized = new double[newLength];
        F64_ATTENTION_DWEIGHTS.set(resized);
        return resized;
    }

    private static double[] ensureF64QueryGradRowCapacity(int length) {
        double[] buffer = F64_ATTENTION_QUERY_GRAD_ROW.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        double[] resized = new double[newLength];
        F64_ATTENTION_QUERY_GRAD_ROW.set(resized);
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

    private static float[] ensureF32GradientCapacity(int length) {
        float[] buffer = F32_ATTENTION_DWEIGHTS.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        float[] resized = new float[newLength];
        F32_ATTENTION_DWEIGHTS.set(resized);
        return resized;
    }

    private static float[] ensureF32QueryGradRowCapacity(int length) {
        float[] buffer = F32_ATTENTION_QUERY_GRAD_ROW.get();
        if (buffer.length >= length) {
            return buffer;
        }
        int newLength = Math.max(length, buffer.length == 0 ? length : buffer.length * 2);
        float[] resized = new float[newLength];
        F32_ATTENTION_QUERY_GRAD_ROW.set(resized);
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

    private static void ensureBackwardGradsF64(
            AttentionBackwardSpec spec,
            Tensor attentionOut,
            Tensor outGrad,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            ResolvedScaledDotProductAttentionPlan plan
    ) {
        if (runtimeCache.hasBackwardGrads()) {
            return;
        }
        Tensor queryGrad = runtimeCache.requireQueryGrad(rawQueryGradShape(attentionOut.getShapeUnsafe(), spec.query().getShapeUnsafe()), DataType.FLOAT64);
        Tensor keyGrad = runtimeCache.requireKeyGrad(rawKeyGradShape(attentionOut.getShapeUnsafe(), spec.key().getShapeUnsafe()), DataType.FLOAT64);
        Tensor valueGrad = runtimeCache.requireValueGrad(rawValueGradShape(attentionOut.getShapeUnsafe(), spec.value().getShapeUnsafe()), DataType.FLOAT64);
        computeBackwardF64(spec, runtimeCache, outGrad, queryGrad, keyGrad, valueGrad, plan);
        runtimeCache.markBackwardGradsReady();
    }

    private static void ensureBackwardGradsF32(
            AttentionBackwardSpec spec,
            Tensor attentionOut,
            Tensor outGrad,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            ResolvedScaledDotProductAttentionPlan plan
    ) {
        if (runtimeCache.hasBackwardGrads()) {
            return;
        }
        Tensor queryGrad = runtimeCache.requireQueryGrad(rawQueryGradShape(attentionOut.getShapeUnsafe(), spec.query().getShapeUnsafe()), DataType.FLOAT32);
        Tensor keyGrad = runtimeCache.requireKeyGrad(rawKeyGradShape(attentionOut.getShapeUnsafe(), spec.key().getShapeUnsafe()), DataType.FLOAT32);
        Tensor valueGrad = runtimeCache.requireValueGrad(rawValueGradShape(attentionOut.getShapeUnsafe(), spec.value().getShapeUnsafe()), DataType.FLOAT32);
        computeBackwardF32(spec, runtimeCache, outGrad, queryGrad, keyGrad, valueGrad, plan);
        runtimeCache.markBackwardGradsReady();
    }

    private static void ensureBackwardGradsBF16(
            AttentionBackwardSpec spec,
            Tensor attentionOut,
            Tensor outGrad,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            ResolvedScaledDotProductAttentionPlan plan,
            CpuKernelContext context
    ) {
        if (runtimeCache.hasBackwardGrads()) {
            return;
        }
        Tensor queryGrad = runtimeCache.requireQueryGrad(rawQueryGradShape(attentionOut.getShapeUnsafe(), spec.query().getShapeUnsafe()), DataType.FLOAT32);
        Tensor keyGrad = runtimeCache.requireKeyGrad(rawKeyGradShape(attentionOut.getShapeUnsafe(), spec.key().getShapeUnsafe()), DataType.FLOAT32);
        Tensor valueGrad = runtimeCache.requireValueGrad(rawValueGradShape(attentionOut.getShapeUnsafe(), spec.value().getShapeUnsafe()), DataType.FLOAT32);
        computeBackwardBF16(spec, runtimeCache, outGrad, queryGrad, keyGrad, valueGrad, plan, context);
        runtimeCache.markBackwardGradsReady();
    }

    private static void computeBackwardF64(
            AttentionBackwardSpec spec,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            Tensor outGrad,
            Tensor queryGrad,
            Tensor keyGrad,
            Tensor valueGrad,
            ResolvedScaledDotProductAttentionPlan plan
    ) {
        Tensor weights = runtimeCache.weights();
        Tensor dWeights = runtimeCache.requireDWeights(weights.getShapeUnsafe(), DataType.FLOAT64);
        runMatMulRightTransposedF64(outGrad, spec.value(), dWeights, requireMatMulHints(plan.backwardDWeightsMatMulHints(), "attention backward dWeights"));

        Tensor dScores = runtimeCache.requireDScores(weights.getShapeUnsafe(), DataType.FLOAT64);
        computeSoftmaxGradRowsF64(
                TensorInternalAccess.float64Data(weights),
                TensorInternalAccess.float64Data(dWeights),
                TensorInternalAccess.float64Data(dScores),
                weights.getShapeUnsafe(),
                spec.scale(),
                requireAttentionHints(plan.backwardSoftmaxGradHints(), "attention backward softmax")
        );

        runMatMulF64(dScores, spec.key(), queryGrad, requireMatMulHints(plan.backwardQueryGradMatMulHints(), "attention backward queryGrad"));
        runMatMulLeftTransposedF64(weights, outGrad, valueGrad, requireMatMulHints(plan.backwardValueGradMatMulHints(), "attention backward valueGrad"));
        runMatMulLeftTransposedF64(dScores, spec.query(), keyGrad, requireMatMulHints(plan.backwardKeyGradMatMulHints(), "attention backward keyGrad"));
    }

    private static void computeBackwardF32(
            AttentionBackwardSpec spec,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            Tensor outGrad,
            Tensor queryGrad,
            Tensor keyGrad,
            Tensor valueGrad,
            ResolvedScaledDotProductAttentionPlan plan
    ) {
        Tensor weights = runtimeCache.weights();
        Tensor dWeights = runtimeCache.requireDWeights(weights.getShapeUnsafe(), DataType.FLOAT32);
        runMatMulRightTransposedF32(outGrad, spec.value(), dWeights, requireMatMulHints(plan.backwardDWeightsMatMulHints(), "attention backward dWeights"));

        Tensor dScores = runtimeCache.requireDScores(weights.getShapeUnsafe(), DataType.FLOAT32);
        computeSoftmaxGradRowsF32(
                TensorInternalAccess.float32Data(weights),
                TensorInternalAccess.float32Data(dWeights),
                TensorInternalAccess.float32Data(dScores),
                weights.getShapeUnsafe(),
                (float) spec.scale(),
                requireAttentionHints(plan.backwardSoftmaxGradHints(), "attention backward softmax")
        );

        runMatMulF32(dScores, spec.key(), queryGrad, requireMatMulHints(plan.backwardQueryGradMatMulHints(), "attention backward queryGrad"));
        runMatMulLeftTransposedF32(weights, outGrad, valueGrad, requireMatMulHints(plan.backwardValueGradMatMulHints(), "attention backward valueGrad"));
        runMatMulLeftTransposedF32(dScores, spec.query(), keyGrad, requireMatMulHints(plan.backwardKeyGradMatMulHints(), "attention backward keyGrad"));
    }

    private static void computeBackwardBF16(
            AttentionBackwardSpec spec,
            ScaledDotProductAttentionRuntimeCache runtimeCache,
            Tensor outGrad,
            Tensor queryGrad,
            Tensor keyGrad,
            Tensor valueGrad,
            ResolvedScaledDotProductAttentionPlan plan,
            CpuKernelContext context
    ) {
        Tensor weights = runtimeCache.weights();
        Tensor dWeights = runtimeCache.requireDWeights(weights.getShapeUnsafe(), DataType.FLOAT32);
        Tensor outGradF32 = toPreparedF32(outGrad, context == null ? null : context.inputFloatContinuation(1, outGrad.getFlatDataSize()), "attention_out_grad_f32");
        Tensor queryF32 = toPreparedF32(spec.query(), null, "attention_query_f32");
        Tensor keyF32 = toPreparedF32(spec.key(), null, "attention_key_f32");
        Tensor valueF32 = toPreparedF32(spec.value(), null, "attention_value_f32");

        runMatMulRightTransposedF32(outGradF32, valueF32, dWeights, requireMatMulHints(plan.backwardDWeightsMatMulHints(), "attention backward dWeights"));

        Tensor dScores = runtimeCache.requireDScores(weights.getShapeUnsafe(), DataType.FLOAT32);
        computeSoftmaxGradRowsF32(
                TensorInternalAccess.float32Data(weights),
                TensorInternalAccess.float32Data(dWeights),
                TensorInternalAccess.float32Data(dScores),
                weights.getShapeUnsafe(),
                (float) spec.scale(),
                requireAttentionHints(plan.backwardSoftmaxGradHints(), "attention backward softmax")
        );

        runMatMulF32(dScores, keyF32, queryGrad, requireMatMulHints(plan.backwardQueryGradMatMulHints(), "attention backward queryGrad"));
        runMatMulLeftTransposedF32(weights, outGradF32, valueGrad, requireMatMulHints(plan.backwardValueGradMatMulHints(), "attention backward valueGrad"));
        runMatMulLeftTransposedF32(dScores, queryF32, keyGrad, requireMatMulHints(plan.backwardKeyGradMatMulHints(), "attention backward keyGrad"));
    }

    private static void computeBackwardBatchF64(
            double scale,
            double[] query,
            double[] key,
            double[] value,
            byte[] mask,
            double[] weights,
            double[] outGrad,
            double[] queryGrad,
            double[] keyGrad,
            double[] valueGrad,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] weightsBatchOffsets,
            int[] maskBatchOffsets,
            int batch,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            boolean vectorized
    ) {
        double[] dWeightsRow = ensureF64GradientCapacity(keyLen);
        int queryBatchBase = queryBatchOffsets[batch];
        int keyBatchBase = keyBatchOffsets[batch];
        int valueBatchBase = valueBatchOffsets[batch];
        int weightsBatchBase = weightsBatchOffsets[batch];
        int maskBatchBase = mask == null ? -1 : maskBatchOffsets[batch];
        int outGradBatchBase = batch * queryLen * valueDim;
        int queryGradBatchBase = batch * queryLen * depth;
        int keyGradBatchBase = batch * keyLen * depth;
        int valueGradBatchBase = batch * keyLen * valueDim;
        for (int queryIndex = 0; queryIndex < queryLen; queryIndex++) {
            int queryBase = queryBatchBase + queryIndex * depth;
            int outGradBase = outGradBatchBase + queryIndex * valueDim;
            int queryGradBase = queryGradBatchBase + queryIndex * depth;
            int weightsBase = weightsBatchBase + queryIndex * keyLen;
            int maskBase = mask == null ? -1 : maskBatchBase + queryIndex * keyLen;
            double weightedDot = vectorized && valueDim == ATTENTION_MICRO_DIM
                    ? computeValueGradAndDWeights32F64(
                    weights,
                    weightsBase,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase,
                    valueGrad,
                    valueGradBatchBase,
                    dWeightsRow,
                    keyLen
            )
                    : computeValueGradAndDWeightsF64(
                    weights,
                    weightsBase,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase,
                    valueGrad,
                    valueGradBatchBase,
                    dWeightsRow,
                    keyLen,
                    valueDim,
                    vectorized
            );
            if (vectorized && depth == ATTENTION_MICRO_DIM) {
                computeScoreGrads32F64(
                        scale,
                        mask,
                        maskBase,
                        weights,
                        weightsBase,
                        dWeightsRow,
                        weightedDot,
                        query,
                        queryBase,
                        key,
                        keyBatchBase,
                        queryGrad,
                        queryGradBase,
                        keyGrad,
                        keyGradBatchBase,
                        keyLen
                );
            } else {
                for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                    double dScore = weights[weightsBase + keyIndex] * (dWeightsRow[keyIndex] - weightedDot);
                    if (mask != null && mask[maskBase + keyIndex] == 0) {
                        dScore = 0.0d;
                    }
                    dScore *= scale;
                    if (dScore == 0.0d) {
                        continue;
                    }
                    int keyBase = keyBatchBase + keyIndex * depth;
                    if (vectorized) {
                        accumulateWeightedF64(queryGrad, queryGradBase, key, keyBase, dScore, depth, true);
                        accumulateWeightedF64(keyGrad, keyGradBatchBase + keyIndex * depth, query, queryBase, dScore, depth, true);
                    } else {
                        accumulateWeightedF64(queryGrad, queryGradBase, key, keyBase, dScore, depth, false);
                        accumulateWeightedF64(keyGrad, keyGradBatchBase + keyIndex * depth, query, queryBase, dScore, depth, false);
                    }
                }
            }
        }
    }

    private static void computeBackwardBatchF32(
            float scale,
            float[] query,
            float[] key,
            float[] value,
            byte[] mask,
            float[] weights,
            float[] outGrad,
            float[] queryGrad,
            float[] keyGrad,
            float[] valueGrad,
            int[] queryBatchOffsets,
            int[] keyBatchOffsets,
            int[] valueBatchOffsets,
            int[] weightsBatchOffsets,
            int[] maskBatchOffsets,
            int batch,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            boolean vectorized
    ) {
        float[] dWeightsRow = ensureF32GradientCapacity(keyLen);
        int queryBatchBase = queryBatchOffsets[batch];
        int keyBatchBase = keyBatchOffsets[batch];
        int valueBatchBase = valueBatchOffsets[batch];
        int weightsBatchBase = weightsBatchOffsets[batch];
        int maskBatchBase = mask == null ? -1 : maskBatchOffsets[batch];
        int outGradBatchBase = batch * queryLen * valueDim;
        int queryGradBatchBase = batch * queryLen * depth;
        int keyGradBatchBase = batch * keyLen * depth;
        int valueGradBatchBase = batch * keyLen * valueDim;
        for (int queryIndex = 0; queryIndex < queryLen; queryIndex++) {
            int queryBase = queryBatchBase + queryIndex * depth;
            int outGradBase = outGradBatchBase + queryIndex * valueDim;
            int queryGradBase = queryGradBatchBase + queryIndex * depth;
            int weightsBase = weightsBatchBase + queryIndex * keyLen;
            int maskBase = mask == null ? -1 : maskBatchBase + queryIndex * keyLen;
            float weightedDot = vectorized && valueDim == ATTENTION_MICRO_DIM
                    ? computeValueGradAndDWeights32F32(
                    weights,
                    weightsBase,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase,
                    valueGrad,
                    valueGradBatchBase,
                    dWeightsRow,
                    keyLen
            )
                    : computeValueGradAndDWeightsF32(
                    weights,
                    weightsBase,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase,
                    valueGrad,
                    valueGradBatchBase,
                    dWeightsRow,
                    keyLen,
                    valueDim,
                    vectorized
            );
            if (vectorized && depth == ATTENTION_MICRO_DIM) {
                computeScoreGrads32F32(
                        scale,
                        mask,
                        maskBase,
                        weights,
                        weightsBase,
                        dWeightsRow,
                        weightedDot,
                        query,
                        queryBase,
                        key,
                        keyBatchBase,
                        queryGrad,
                        queryGradBase,
                        keyGrad,
                        keyGradBatchBase,
                        keyLen
                );
            } else {
                for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                    float dScore = weights[weightsBase + keyIndex] * (dWeightsRow[keyIndex] - weightedDot);
                    if (mask != null && mask[maskBase + keyIndex] == 0) {
                        dScore = 0.0f;
                    }
                    dScore *= scale;
                    if (dScore == 0.0f) {
                        continue;
                    }
                    int keyBase = keyBatchBase + keyIndex * depth;
                    if (vectorized) {
                        accumulateWeightedF32(queryGrad, queryGradBase, key, keyBase, dScore, depth, true);
                        accumulateWeightedF32(keyGrad, keyGradBatchBase + keyIndex * depth, query, queryBase, dScore, depth, true);
                    } else {
                        accumulateWeightedF32(queryGrad, queryGradBase, key, keyBase, dScore, depth, false);
                        accumulateWeightedF32(keyGrad, keyGradBatchBase + keyIndex * depth, query, queryBase, dScore, depth, false);
                    }
                }
            }
        }
    }

    private static double computeValueGradAndDWeightsF64(
            double[] weights,
            int weightsBase,
            double[] outGrad,
            int outGradBase,
            double[] value,
            int valueBatchBase,
            double[] valueGrad,
            int valueGradBatchBase,
            double[] dWeightsRow,
            int keyLen,
            int valueDim,
            boolean vectorized
    ) {
        double weightedDot = 0.0d;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            int valueBase = valueBatchBase + keyIndex * valueDim;
            double dWeight = dotF64(outGrad, outGradBase, value, valueBase, valueDim, vectorized);
            dWeightsRow[keyIndex] = dWeight;
            double weight = weights[weightsBase + keyIndex];
            weightedDot += weight * dWeight;
            if (weight != 0.0d) {
                accumulateWeightedF64(valueGrad, valueGradBatchBase + keyIndex * valueDim, outGrad, outGradBase, weight, valueDim, vectorized);
            }
        }
        return weightedDot;
    }

    private static float computeValueGradAndDWeightsF32(
            float[] weights,
            int weightsBase,
            float[] outGrad,
            int outGradBase,
            float[] value,
            int valueBatchBase,
            float[] valueGrad,
            int valueGradBatchBase,
            float[] dWeightsRow,
            int keyLen,
            int valueDim,
            boolean vectorized
    ) {
        float weightedDot = 0.0f;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            int valueBase = valueBatchBase + keyIndex * valueDim;
            float dWeight = dotF32(outGrad, outGradBase, value, valueBase, valueDim, vectorized);
            dWeightsRow[keyIndex] = dWeight;
            float weight = weights[weightsBase + keyIndex];
            weightedDot += weight * dWeight;
            if (weight != 0.0f) {
                accumulateWeightedF32(valueGrad, valueGradBatchBase + keyIndex * valueDim, outGrad, outGradBase, weight, valueDim, vectorized);
            }
        }
        return weightedDot;
    }

    private static double computeValueGradAndDWeights32F64(
            double[] weights,
            int weightsBase,
            double[] outGrad,
            int outGradBase,
            double[] value,
            int valueBatchBase,
            double[] valueGrad,
            int valueGradBatchBase,
            double[] dWeightsRow,
            int keyLen
    ) {
        double weightedDot = 0.0d;
        int keyIndex = 0;
        for (; keyIndex + ATTENTION_F64_VALUE_BLOCK_KEYS - 1 < keyLen; keyIndex += ATTENTION_F64_VALUE_BLOCK_KEYS) {
            weightedDot += computeValueGradBlock32F64(
                    weights,
                    weightsBase + keyIndex,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                    valueGrad,
                    valueGradBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                    dWeightsRow,
                    keyIndex
            );
        }
        for (; keyIndex < keyLen; keyIndex++) {
            int valueBase = valueBatchBase + keyIndex * ATTENTION_MICRO_DIM;
            double dWeight = dotF64(outGrad, outGradBase, value, valueBase, ATTENTION_MICRO_DIM, true);
            dWeightsRow[keyIndex] = dWeight;
            double weight = weights[weightsBase + keyIndex];
            weightedDot += weight * dWeight;
            if (weight != 0.0d) {
                accumulateWeightedF64(
                        valueGrad,
                        valueGradBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                        outGrad,
                        outGradBase,
                        weight,
                        ATTENTION_MICRO_DIM,
                        true
                );
            }
        }
        return weightedDot;
    }

    private static float computeValueGradAndDWeights32F32(
            float[] weights,
            int weightsBase,
            float[] outGrad,
            int outGradBase,
            float[] value,
            int valueBatchBase,
            float[] valueGrad,
            int valueGradBatchBase,
            float[] dWeightsRow,
            int keyLen
    ) {
        float weightedDot = 0.0f;
        int keyIndex = 0;
        for (; keyIndex + ATTENTION_F32_VALUE_BLOCK_KEYS - 1 < keyLen; keyIndex += ATTENTION_F32_VALUE_BLOCK_KEYS) {
            weightedDot += computeValueGradBlock32F32(
                    weights,
                    weightsBase + keyIndex,
                    outGrad,
                    outGradBase,
                    value,
                    valueBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                    valueGrad,
                    valueGradBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                    dWeightsRow,
                    keyIndex
            );
        }
        for (; keyIndex < keyLen; keyIndex++) {
            int valueBase = valueBatchBase + keyIndex * ATTENTION_MICRO_DIM;
            float dWeight = dotF32(outGrad, outGradBase, value, valueBase, ATTENTION_MICRO_DIM, true);
            dWeightsRow[keyIndex] = dWeight;
            float weight = weights[weightsBase + keyIndex];
            weightedDot += weight * dWeight;
            if (weight != 0.0f) {
                accumulateWeightedF32(
                        valueGrad,
                        valueGradBatchBase + keyIndex * ATTENTION_MICRO_DIM,
                        outGrad,
                        outGradBase,
                        weight,
                        ATTENTION_MICRO_DIM,
                        true
                );
            }
        }
        return weightedDot;
    }

    private static double computeValueGradBlock32F64(
            double[] weights,
            int weightsBase,
            double[] outGrad,
            int outGradBase,
            double[] value,
            int valueBase,
            double[] valueGrad,
            int valueGradBase,
            double[] dWeightsRow,
            int dWeightsOffset
    ) {
        double weight0 = weights[weightsBase];
        double weight1 = weights[weightsBase + 1];
        DoubleVector scale0 = DoubleVector.broadcast(F64, weight0);
        DoubleVector scale1 = DoubleVector.broadcast(F64, weight1);
        DoubleVector acc0 = DoubleVector.zero(F64);
        DoubleVector acc1 = DoubleVector.zero(F64);
        for (int i = 0; i < ATTENTION_MICRO_DIM; i += F64.length()) {
            DoubleVector outGradVector = DoubleVector.fromArray(F64, outGrad, outGradBase + i);
            DoubleVector value0 = DoubleVector.fromArray(F64, value, valueBase + i);
            DoubleVector value1 = DoubleVector.fromArray(F64, value, valueBase + ATTENTION_MICRO_DIM + i);
            acc0 = acc0.add(outGradVector.mul(value0));
            acc1 = acc1.add(outGradVector.mul(value1));
            if (weight0 != 0.0d) {
                DoubleVector.fromArray(F64, valueGrad, valueGradBase + i)
                        .add(outGradVector.mul(scale0))
                        .intoArray(valueGrad, valueGradBase + i);
            }
            if (weight1 != 0.0d) {
                DoubleVector.fromArray(F64, valueGrad, valueGradBase + ATTENTION_MICRO_DIM + i)
                        .add(outGradVector.mul(scale1))
                        .intoArray(valueGrad, valueGradBase + ATTENTION_MICRO_DIM + i);
            }
        }
        double dWeight0 = acc0.reduceLanes(VectorOperators.ADD);
        double dWeight1 = acc1.reduceLanes(VectorOperators.ADD);
        dWeightsRow[dWeightsOffset] = dWeight0;
        dWeightsRow[dWeightsOffset + 1] = dWeight1;
        return weight0 * dWeight0 + weight1 * dWeight1;
    }

    private static float computeValueGradBlock32F32(
            float[] weights,
            int weightsBase,
            float[] outGrad,
            int outGradBase,
            float[] value,
            int valueBase,
            float[] valueGrad,
            int valueGradBase,
            float[] dWeightsRow,
            int dWeightsOffset
    ) {
        float weight0 = weights[weightsBase];
        float weight1 = weights[weightsBase + 1];
        float weight2 = weights[weightsBase + 2];
        float weight3 = weights[weightsBase + 3];
        FloatVector scale0 = FloatVector.broadcast(F32, weight0);
        FloatVector scale1 = FloatVector.broadcast(F32, weight1);
        FloatVector scale2 = FloatVector.broadcast(F32, weight2);
        FloatVector scale3 = FloatVector.broadcast(F32, weight3);
        FloatVector acc0 = FloatVector.zero(F32);
        FloatVector acc1 = FloatVector.zero(F32);
        FloatVector acc2 = FloatVector.zero(F32);
        FloatVector acc3 = FloatVector.zero(F32);
        for (int i = 0; i < ATTENTION_MICRO_DIM; i += F32.length()) {
            FloatVector outGradVector = FloatVector.fromArray(F32, outGrad, outGradBase + i);
            FloatVector value0 = FloatVector.fromArray(F32, value, valueBase + i);
            FloatVector value1 = FloatVector.fromArray(F32, value, valueBase + ATTENTION_MICRO_DIM + i);
            FloatVector value2 = FloatVector.fromArray(F32, value, valueBase + 2 * ATTENTION_MICRO_DIM + i);
            FloatVector value3 = FloatVector.fromArray(F32, value, valueBase + 3 * ATTENTION_MICRO_DIM + i);
            acc0 = acc0.add(outGradVector.mul(value0));
            acc1 = acc1.add(outGradVector.mul(value1));
            acc2 = acc2.add(outGradVector.mul(value2));
            acc3 = acc3.add(outGradVector.mul(value3));
            if (weight0 != 0.0f) {
                FloatVector.fromArray(F32, valueGrad, valueGradBase + i)
                        .add(outGradVector.mul(scale0))
                        .intoArray(valueGrad, valueGradBase + i);
            }
            if (weight1 != 0.0f) {
                FloatVector.fromArray(F32, valueGrad, valueGradBase + ATTENTION_MICRO_DIM + i)
                        .add(outGradVector.mul(scale1))
                        .intoArray(valueGrad, valueGradBase + ATTENTION_MICRO_DIM + i);
            }
            if (weight2 != 0.0f) {
                FloatVector.fromArray(F32, valueGrad, valueGradBase + 2 * ATTENTION_MICRO_DIM + i)
                        .add(outGradVector.mul(scale2))
                        .intoArray(valueGrad, valueGradBase + 2 * ATTENTION_MICRO_DIM + i);
            }
            if (weight3 != 0.0f) {
                FloatVector.fromArray(F32, valueGrad, valueGradBase + 3 * ATTENTION_MICRO_DIM + i)
                        .add(outGradVector.mul(scale3))
                        .intoArray(valueGrad, valueGradBase + 3 * ATTENTION_MICRO_DIM + i);
            }
        }
        float dWeight0 = acc0.reduceLanes(VectorOperators.ADD);
        float dWeight1 = acc1.reduceLanes(VectorOperators.ADD);
        float dWeight2 = acc2.reduceLanes(VectorOperators.ADD);
        float dWeight3 = acc3.reduceLanes(VectorOperators.ADD);
        dWeightsRow[dWeightsOffset] = dWeight0;
        dWeightsRow[dWeightsOffset + 1] = dWeight1;
        dWeightsRow[dWeightsOffset + 2] = dWeight2;
        dWeightsRow[dWeightsOffset + 3] = dWeight3;
        return weight0 * dWeight0 + weight1 * dWeight1 + weight2 * dWeight2 + weight3 * dWeight3;
    }

    private static void computeScoreGrads32F64(
            double scale,
            byte[] mask,
            int maskBase,
            double[] weights,
            int weightsBase,
            double[] dWeightsRow,
            double weightedDot,
            double[] query,
            int queryBase,
            double[] key,
            int keyBatchBase,
            double[] queryGrad,
            int queryGradBase,
            double[] keyGrad,
            int keyGradBatchBase,
            int keyLen
    ) {
        double[] queryGradRow = ensureF64QueryGradRowCapacity(ATTENTION_MICRO_DIM);
        Arrays.fill(queryGradRow, 0, ATTENTION_MICRO_DIM, 0.0d);
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            double dScore = weights[weightsBase + keyIndex] * (dWeightsRow[keyIndex] - weightedDot);
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                dScore = 0.0d;
            }
            dScore *= scale;
            if (dScore == 0.0d) {
                continue;
            }
            int keyBase = keyBatchBase + keyIndex * ATTENTION_MICRO_DIM;
            accumulateWeightedF64(queryGradRow, 0, key, keyBase, dScore, ATTENTION_MICRO_DIM, true);
            accumulateWeightedF64(keyGrad, keyGradBatchBase + keyIndex * ATTENTION_MICRO_DIM, query, queryBase, dScore, ATTENTION_MICRO_DIM, true);
        }
        for (int i = 0; i < ATTENTION_MICRO_DIM; i++) {
            queryGrad[queryGradBase + i] += queryGradRow[i];
        }
    }

    private static void computeScoreGrads32F32(
            float scale,
            byte[] mask,
            int maskBase,
            float[] weights,
            int weightsBase,
            float[] dWeightsRow,
            float weightedDot,
            float[] query,
            int queryBase,
            float[] key,
            int keyBatchBase,
            float[] queryGrad,
            int queryGradBase,
            float[] keyGrad,
            int keyGradBatchBase,
            int keyLen
    ) {
        float[] queryGradRow = ensureF32QueryGradRowCapacity(ATTENTION_MICRO_DIM);
        Arrays.fill(queryGradRow, 0, ATTENTION_MICRO_DIM, 0.0f);
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            float dScore = weights[weightsBase + keyIndex] * (dWeightsRow[keyIndex] - weightedDot);
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                dScore = 0.0f;
            }
            dScore *= scale;
            if (dScore == 0.0f) {
                continue;
            }
            int keyBase = keyBatchBase + keyIndex * ATTENTION_MICRO_DIM;
            accumulateWeightedF32(queryGradRow, 0, key, keyBase, dScore, ATTENTION_MICRO_DIM, true);
            accumulateWeightedF32(keyGrad, keyGradBatchBase + keyIndex * ATTENTION_MICRO_DIM, query, queryBase, dScore, ATTENTION_MICRO_DIM, true);
        }
        for (int i = 0; i < ATTENTION_MICRO_DIM; i++) {
            queryGrad[queryGradBase + i] += queryGradRow[i];
        }
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

    private static void softmaxAttentionRowF64(double[] values, byte[] mask, int base, int length, double scale, boolean fastExp) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (mask != null && mask[base + i] == 0) {
                continue;
            }
            double candidate = values[base + i] * scale;
            if (candidate > max) {
                max = candidate;
            }
        }
        if (max == Double.NEGATIVE_INFINITY) {
            Arrays.fill(values, base, base + length, 1.0d / length);
            return;
        }
        double sum = 0.0d;
        for (int i = 0; i < length; i++) {
            int index = base + i;
            if (mask != null && mask[index] == 0) {
                values[index] = 0.0d;
                continue;
            }
            double value = values[index] * scale - max;
            double exp = fastExp ? FastTranscendentals.fastExpF64(value) : Math.exp(value);
            values[index] = exp;
            sum += exp;
        }
        double inv = 1.0d / sum;
        for (int i = 0; i < length; i++) {
            values[base + i] *= inv;
        }
    }

    private static void softmaxAttentionRowF32(float[] values, byte[] mask, int base, int length, float scale, boolean fastExp) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (mask != null && mask[base + i] == 0) {
                continue;
            }
            float candidate = values[base + i] * scale;
            if (candidate > max) {
                max = candidate;
            }
        }
        if (max == Float.NEGATIVE_INFINITY) {
            Arrays.fill(values, base, base + length, 1.0f / length);
            return;
        }
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            int index = base + i;
            if (mask != null && mask[index] == 0) {
                values[index] = 0.0f;
                continue;
            }
            float value = values[index] * scale - max;
            float exp = fastExp ? FastTranscendentals.fastExpF32(value) : (float) Math.exp(value);
            values[index] = exp;
            sum += exp;
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < length; i++) {
            values[base + i] *= inv;
        }
    }

    private static void runMatMulF64(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        double[] ad = TensorInternalAccess.float64Data(a);
        double[] bd = TensorInternalAccess.float64Data(b);
        double[] od = TensorInternalAccess.float64Data(out);
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int n = bs[bs.length - 1];
        int k = as[as.length - 1];
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasF64(ad, bd, od, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasF64(ad, as, bd, bs, od, out.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(od, 0.0d);
        F64MatMulJavaBackend.run(ad, as, bd, bs, od, out.getShapeUnsafe(), hints);
    }

    private static void runMatMulF32(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        float[] ad = TensorInternalAccess.float32Data(a);
        float[] bd = TensorInternalAccess.float32Data(b);
        float[] od = TensorInternalAccess.float32Data(out);
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int n = bs[bs.length - 1];
        int k = as[as.length - 1];
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasF32(ad, bd, od, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasF32(ad, as, bd, bs, od, out.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(od, 0.0f);
        F32MatMulJavaBackend.run(ad, as, bd, bs, od, out.getShapeUnsafe(), hints);
    }

    private static void runMatMulRightTransposedF64(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        double[] ad = TensorInternalAccess.float64Data(a);
        double[] bd = TensorInternalAccess.float64Data(b);
        double[] od = TensorInternalAccess.float64Data(out);
        Arrays.fill(od, 0.0d);
        F64MatMulJavaBackend.runRightTransposed(ad, a.getShapeUnsafe(), bd, b.getShapeUnsafe(), od, out.getShapeUnsafe(), hints);
    }

    private static void runMatMulRightTransposedF32(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        float[] ad = TensorInternalAccess.float32Data(a);
        float[] bd = TensorInternalAccess.float32Data(b);
        float[] od = TensorInternalAccess.float32Data(out);
        Arrays.fill(od, 0.0f);
        F32MatMulJavaBackend.runRightTransposed(ad, a.getShapeUnsafe(), bd, b.getShapeUnsafe(), od, out.getShapeUnsafe(), hints);
    }

    private static void runMatMulLeftTransposedF64(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        double[] ad = TensorInternalAccess.float64Data(a);
        double[] bd = TensorInternalAccess.float64Data(b);
        double[] od = TensorInternalAccess.float64Data(out);
        Arrays.fill(od, 0.0d);
        F64MatMulJavaBackend.runLeftTransposed(ad, a.getShapeUnsafe(), bd, b.getShapeUnsafe(), od, out.getShapeUnsafe(), hints);
    }

    private static void runMatMulLeftTransposedF32(Tensor a, Tensor b, Tensor out, ResolvedMatMulHints hints) {
        float[] ad = TensorInternalAccess.float32Data(a);
        float[] bd = TensorInternalAccess.float32Data(b);
        float[] od = TensorInternalAccess.float32Data(out);
        Arrays.fill(od, 0.0f);
        F32MatMulJavaBackend.runLeftTransposed(ad, a.getShapeUnsafe(), bd, b.getShapeUnsafe(), od, out.getShapeUnsafe(), hints);
    }

    private static void computeSoftmaxGradRowsF64(
            double[] weights,
            double[] dWeights,
            double[] dScores,
            int[] shape,
            double scale,
            ResolvedAttentionHints hints
    ) {
        int rowLength = shape[shape.length - 1];
        int rowCount = product(shape) / rowLength;
        if (hints.parallel() && rowCount > 1) {
            int rowsPerChunk = hints.taskChunkSize();
            int chunks = (rowCount + rowsPerChunk - 1) / rowsPerChunk;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * rowsPerChunk;
                int end = Math.min(start + rowsPerChunk, rowCount);
                computeSoftmaxGradRowsF64Range(weights, dWeights, dScores, rowLength, scale, start, end, hints.vectorized());
            });
            return;
        }
        computeSoftmaxGradRowsF64Range(weights, dWeights, dScores, rowLength, scale, 0, rowCount, hints.vectorized());
    }

    private static void computeSoftmaxGradRowsF64Range(
            double[] weights,
            double[] dWeights,
            double[] dScores,
            int rowLength,
            double scale,
            int rowStart,
            int rowEnd,
            boolean vectorized
    ) {
        int upper = F64.loopBound(rowLength);
        DoubleVector scaleVector = DoubleVector.broadcast(F64, scale);
        for (int row = rowStart; row < rowEnd; row++) {
            int base = row * rowLength;
            double dot = dotF64(weights, base, dWeights, base, rowLength, vectorized);
            DoubleVector dotVector = DoubleVector.broadcast(F64, dot);
            int i = 0;
            if (vectorized) {
                for (; i < upper; i += F64.length()) {
                    DoubleVector.fromArray(F64, weights, base + i)
                            .mul(DoubleVector.fromArray(F64, dWeights, base + i).sub(dotVector))
                            .mul(scaleVector)
                            .intoArray(dScores, base + i);
                }
            }
            for (; i < rowLength; i++) {
                dScores[base + i] = weights[base + i] * (dWeights[base + i] - dot) * scale;
            }
        }
    }

    private static void computeSoftmaxGradRowsF32(
            float[] weights,
            float[] dWeights,
            float[] dScores,
            int[] shape,
            float scale,
            ResolvedAttentionHints hints
    ) {
        int rowLength = shape[shape.length - 1];
        int rowCount = product(shape) / rowLength;
        if (hints.parallel() && rowCount > 1) {
            int rowsPerChunk = hints.taskChunkSize();
            int chunks = (rowCount + rowsPerChunk - 1) / rowsPerChunk;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * rowsPerChunk;
                int end = Math.min(start + rowsPerChunk, rowCount);
                computeSoftmaxGradRowsF32Range(weights, dWeights, dScores, rowLength, scale, start, end, hints.vectorized());
            });
            return;
        }
        computeSoftmaxGradRowsF32Range(weights, dWeights, dScores, rowLength, scale, 0, rowCount, hints.vectorized());
    }

    private static void computeSoftmaxGradRowsF32Range(
            float[] weights,
            float[] dWeights,
            float[] dScores,
            int rowLength,
            float scale,
            int rowStart,
            int rowEnd,
            boolean vectorized
    ) {
        int upper = F32.loopBound(rowLength);
        FloatVector scaleVector = FloatVector.broadcast(F32, scale);
        for (int row = rowStart; row < rowEnd; row++) {
            int base = row * rowLength;
            float dot = dotF32(weights, base, dWeights, base, rowLength, vectorized);
            FloatVector dotVector = FloatVector.broadcast(F32, dot);
            int i = 0;
            if (vectorized) {
                for (; i < upper; i += F32.length()) {
                    FloatVector.fromArray(F32, weights, base + i)
                            .mul(FloatVector.fromArray(F32, dWeights, base + i).sub(dotVector))
                            .mul(scaleVector)
                            .intoArray(dScores, base + i);
                }
            }
            for (; i < rowLength; i++) {
                dScores[base + i] = weights[base + i] * (dWeights[base + i] - dot) * scale;
            }
        }
    }

    private static void transposeLastTwoAxesF64(double[] src, int[] shape, double[] dst) {
        int batch = batchCount(shape);
        int tk = shape[shape.length - 2];
        int depth = shape[shape.length - 1];
        int block = tk * depth;
        for (int b = 0; b < batch; b++) {
            int srcBase = b * block;
            int dstBase = b * block;
            for (int t = 0; t < tk; t++) {
                int srcRow = srcBase + t * depth;
                for (int d = 0; d < depth; d++) {
                    dst[dstBase + d * tk + t] = src[srcRow + d];
                }
            }
        }
    }

    private static void transposeLastTwoAxesF32(float[] src, int[] shape, float[] dst) {
        int batch = batchCount(shape);
        int tk = shape[shape.length - 2];
        int depth = shape[shape.length - 1];
        int block = tk * depth;
        for (int b = 0; b < batch; b++) {
            int srcBase = b * block;
            int dstBase = b * block;
            for (int t = 0; t < tk; t++) {
                int srcRow = srcBase + t * depth;
                for (int d = 0; d < depth; d++) {
                    dst[dstBase + d * tk + t] = src[srcRow + d];
                }
            }
        }
    }

    private static float[] resolveBF16InputF32(CpuKernelContext context, int inputIndex, Tensor tensor, String label) {
        float[] continuation = context == null ? null : context.inputFloatContinuation(inputIndex, tensor.getFlatDataSize());
        if (continuation != null) {
            return continuation;
        }
        return TensorInternalAccess.float32Data(toPreparedF32(tensor, null, label));
    }

    private static Tensor toPreparedF32(Tensor tensor, float[] continuation, String label) {
        if (tensor.getDataType() == DataType.FLOAT32 && continuation == null) {
            return tensor;
        }
        int[] shape = tensor.getShapeUnsafe().clone();
        if (continuation != null) {
            return new Tensor(continuation, shape, List.of(), label, DataType.FLOAT32);
        }
        if (tensor.getDataType() == DataType.BFLOAT16) {
            return new Tensor(toF32(TensorInternalAccess.bfloat16Data(tensor)), shape, List.of(), label, DataType.FLOAT32);
        }
        throw new IllegalArgumentException("Expected BF16/F32 tensor for attention F32 preparation, got " + tensor.getDataType());
    }

    private static float[] toF32(short[] src) {
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(src[i]);
        }
        return out;
    }

    private static void writeBF16(float[] src, short[] dst) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] = TensorDTypeOps.toBFloat16Bits(src[i]);
        }
    }

    private static byte[] denseBoolData(Tensor tensor) {
        boolean[] logical = tensor.toBooleanArrayCopy();
        byte[] out = new byte[logical.length];
        for (int i = 0; i < logical.length; i++) {
            out[i] = (byte) (logical[i] ? 1 : 0);
        }
        return out;
    }

    private static int[] transposedKeyShape(int[] keyShape) {
        int[] out = keyShape.clone();
        int tmp = out[out.length - 1];
        out[out.length - 1] = out[out.length - 2];
        out[out.length - 2] = tmp;
        return out;
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

    private static int[] rawQueryGradShape(int[] outShape, int[] queryShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = queryShape[queryShape.length - 2];
        out[out.length - 1] = queryShape[queryShape.length - 1];
        return out;
    }

    private static int[] rawKeyGradShape(int[] outShape, int[] keyShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = keyShape[keyShape.length - 2];
        out[out.length - 1] = keyShape[keyShape.length - 1];
        return out;
    }

    private static int[] rawValueGradShape(int[] outShape, int[] valueShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = valueShape[valueShape.length - 2];
        out[out.length - 1] = valueShape[valueShape.length - 1];
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

    private static int product(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static double maskFillValue(DataType dtype) {
        return switch (dtype) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("attention mask fill requires floating dtype.");
        };
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

    private static AttentionBackwardSpec validateBackward(
            scaledDotProductAttentionBackward.OutputKind outputKind,
            Tensor attentionOut,
            Tensor outGrad,
            Tensor node
    ) {
        if (outputKind == null || attentionOut == null || outGrad == null || node == null) {
            throw new IllegalArgumentException("attention backward execution arguments cannot be null");
        }
        if (!(attentionOut.getOperation() instanceof scaledDotProductAttention attention)) {
            throw new IllegalArgumentException("attention backward expects attention output produced by scaledDotProductAttention");
        }
        List<Tensor> inputs = attentionOut.getPrevTensors();
        if (inputs == null || (inputs.size() != 3 && inputs.size() != 4)) {
            throw new IllegalArgumentException("attention backward expects attention output with q/k/v[/mask] inputs");
        }
        Tensor query = inputs.get(0);
        Tensor key = inputs.get(1);
        Tensor value = inputs.get(2);
        Tensor mask = inputs.size() == 4 ? inputs.get(3) : null;
        validate(attention, query, key, value, mask, attentionOut);
        if (attentionOut.getDataType() != outGrad.getDataType() || attentionOut.getDataType() != node.getDataType()) {
            throw new IllegalArgumentException("attention backward requires matching floating dtypes");
        }
        if (!Arrays.equals(attentionOut.getShapeUnsafe(), outGrad.getShapeUnsafe())) {
            throw new IllegalArgumentException("attention backward requires outGrad shape to match attention output");
        }
        int[] expectedShape = switch (outputKind) {
            case QUERY -> rawQueryGradShape(attentionOut.getShapeUnsafe(), query.getShapeUnsafe());
            case KEY -> rawKeyGradShape(attentionOut.getShapeUnsafe(), key.getShapeUnsafe());
            case VALUE -> rawValueGradShape(attentionOut.getShapeUnsafe(), value.getShapeUnsafe());
        };
        if (!Arrays.equals(expectedShape, node.getShapeUnsafe())) {
            throw new IllegalArgumentException("attention backward output shape mismatch");
        }
        return new AttentionBackwardSpec(attention.getScale(), query, key, value, mask);
    }

    private static ScaledDotProductAttentionRuntimeCache requireRuntimeCache(CpuKernelContext context, Tensor attentionOut) {
        if (context == null) {
            throw new IllegalStateException("attention backward requires CpuKernelContext");
        }
        ScaledDotProductAttentionRuntimeCache runtimeCache =
                context.runtimeStateFor(attentionOut, ScaledDotProductAttentionRuntimeCache.class);
        if (runtimeCache == null) {
            throw new IllegalStateException("attention backward requires forward runtime cache on attention output");
        }
        return runtimeCache;
    }

    private static ResolvedScaledDotProductAttentionPlan requirePlan(CpuKernelContext context, Tensor node) {
        if (context == null || context.attentionPlan() == null) {
            throw new IllegalStateException("Missing prepared attention plan for node " + node.getLabel());
        }
        return context.attentionPlan();
    }

    private static ResolvedAttentionHints requireAttentionHints(ResolvedAttentionHints hints, String stage) {
        if (hints == null) {
            throw new IllegalStateException("Missing prepared attention hints for " + stage + ".");
        }
        return hints;
    }

    private static ResolvedMatMulHints requireMatMulHints(ResolvedMatMulHints hints, String stage) {
        if (hints == null) {
            throw new IllegalStateException("Missing prepared matmul hints for " + stage + ".");
        }
        return hints;
    }

    private record AttentionBackwardSpec(double scale, Tensor query, Tensor key, Tensor value, Tensor mask) {}
}
