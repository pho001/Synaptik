package backend.cpu1.kernels.linalg.attention;

import backend.cpu1.exec.Cpu1AttentionWeightsCache;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;
import utils.FastTranscendentals;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense scaled dot-product attention loops for cpu1.
 */
public final class Cpu1AttentionLoops {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Cpu1AttentionLoops() {
    }

    public static void runAttentionArray(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (unit.dataType() == DataType.FLOAT64) {
            runF64AttentionArray(unit, context);
        } else {
            runF32OrBf16AttentionArray(unit, context);
        }
    }

    public static void runAttentionSegment(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (unit.dataType() == DataType.FLOAT64) {
            runF64AttentionSegment(unit, context);
        } else {
            runF32OrBf16AttentionSegment(unit, context);
        }
    }

    public static void runAttentionWeightsArray(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor attentionOutput = context.runtimeTensorForNodeId(unit.attentionOutputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = context.runtimeStateFor(attentionOutput, Cpu1AttentionWeightsCache.class);
        if (cache == null) {
            throw new IllegalStateException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS requires cached forward "
                    + "weights on attention output nodeId=" + unit.attentionOutputNodeId());
        }
        requireShape("attention weights output", outputTensor.getShapeUnsafe(), cache.shape());
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("attention weights output", output, Cpu1StorageKind.JAVA_ARRAY);
        publishWeightsToArray(unit, cache, output);
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS wrote CPU array");
    }

    public static void runAttentionWeightsSegment(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor attentionOutput = context.runtimeTensorForNodeId(unit.attentionOutputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = context.runtimeStateFor(attentionOutput, Cpu1AttentionWeightsCache.class);
        if (cache == null) {
            throw new IllegalStateException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS requires cached forward "
                    + "weights on attention output nodeId=" + unit.attentionOutputNodeId());
        }
        requireShape("attention weights output", outputTensor.getShapeUnsafe(), cache.shape());
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-attention-weights-node-" + unit.nodeId()
        );
        publishWeightsToSegment(unit, cache, nativeOutput.segment());
        nativeOutput.markModified();
        context.attachNativeStorage(
                unit.nodeId(),
                nativeOutput,
                "cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS wrote native CPU segment"
        );
    }

    private static void runF32OrBf16AttentionArray(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = prepareRuntimeCache(unit, context, outputTensor);
        Cpu1ScratchBuffer scratchBuffer = requireScratch(unit, context);
        float[] rowScores = scratchBuffer.requireF32Array(Math.multiplyExact(unit.scratchSlotCount(), unit.keyLen()));
        F32ArrayInputs inputs = f32ArrayInputs(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        requireShape("output", output.shape(), unit.outputShape());
        if (unit.dataType() == DataType.BFLOAT16) {
            runBf16ArrayRows(unit, inputs, output.bfloat16Array(), cache, rowScores);
        } else {
            runF32ArrayRows(unit, inputs, output.float32Array(), cache, rowScores);
        }
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 attention wrote CPU array");
    }

    private static void runF32OrBf16AttentionSegment(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = prepareRuntimeCache(unit, context, outputTensor);
        Cpu1ScratchBuffer scratchBuffer = requireScratch(unit, context);
        float[] rowScores = scratchBuffer.requireF32Array(Math.multiplyExact(unit.scratchSlotCount(), unit.keyLen()));
        F32SegmentInputs inputs = f32SegmentInputs(unit, context);
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-attention-node-" + unit.nodeId()
        );
        MemorySegment output = nativeOutput.segment();
        if (unit.dataType() == DataType.BFLOAT16) {
            runBf16SegmentRows(unit, inputs, output, cache, rowScores);
        } else {
            runF32SegmentRows(unit, inputs, output, cache, rowScores);
        }
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 attention wrote native CPU segment");
    }

    private static void runF64AttentionArray(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = prepareRuntimeCache(unit, context, outputTensor);
        Cpu1ScratchBuffer scratchBuffer = requireScratch(unit, context);
        double[] rowScores = scratchBuffer.requireF64Array(Math.multiplyExact(unit.scratchSlotCount(), unit.keyLen()));
        F64ArrayInputs inputs = f64ArrayInputs(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        requireShape("output", output.shape(), unit.outputShape());
        runF64ArrayRows(unit, inputs, output.float64Array(), cache, rowScores);
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 attention wrote CPU array");
    }

    private static void runF64AttentionSegment(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1AttentionWeightsCache cache = prepareRuntimeCache(unit, context, outputTensor);
        Cpu1ScratchBuffer scratchBuffer = requireScratch(unit, context);
        double[] rowScores = scratchBuffer.requireF64Array(Math.multiplyExact(unit.scratchSlotCount(), unit.keyLen()));
        F64SegmentInputs inputs = f64SegmentInputs(unit, context);
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-attention-node-" + unit.nodeId()
        );
        runF64SegmentRows(unit, inputs, nativeOutput.segment(), cache, rowScores);
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 attention wrote native CPU segment");
    }

    private static void runF32ArrayRows(
            Cpu1PreparedAttentionUnit unit,
            F32ArrayInputs inputs,
            float[] output,
            Cpu1AttentionWeightsCache cache,
            float[] rowScores
    ) {
        float[] cachedWeights = cache == null ? null : cache.requireF32Weights();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                if (vector) {
                    computeF32ArrayRowVector(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                } else {
                    computeF32ArrayRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                }
            }
        });
    }

    private static void runBf16ArrayRows(
            Cpu1PreparedAttentionUnit unit,
            F32ArrayInputs inputs,
            short[] output,
            Cpu1AttentionWeightsCache cache,
            float[] rowScores
    ) {
        float[] cachedWeights = cache == null ? null : cache.requireF32Weights();
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                computeBf16ArrayRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
            }
        });
    }

    private static void runF64ArrayRows(
            Cpu1PreparedAttentionUnit unit,
            F64ArrayInputs inputs,
            double[] output,
            Cpu1AttentionWeightsCache cache,
            double[] rowScores
    ) {
        double[] cachedWeights = cache == null ? null : cache.requireF64Weights();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                if (vector) {
                    computeF64ArrayRowVector(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                } else {
                    computeF64ArrayRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                }
            }
        });
    }

    private static void runF32SegmentRows(
            Cpu1PreparedAttentionUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            Cpu1AttentionWeightsCache cache,
            float[] rowScores
    ) {
        float[] cachedWeights = cache == null ? null : cache.requireF32Weights();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                if (vector) {
                    computeF32SegmentRowVector(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                } else {
                    computeF32SegmentRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                }
            }
        });
    }

    private static void runBf16SegmentRows(
            Cpu1PreparedAttentionUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            Cpu1AttentionWeightsCache cache,
            float[] rowScores
    ) {
        float[] cachedWeights = cache == null ? null : cache.requireF32Weights();
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                computeBf16SegmentRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
            }
        });
    }

    private static void runF64SegmentRows(
            Cpu1PreparedAttentionUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            Cpu1AttentionWeightsCache cache,
            double[] rowScores
    ) {
        double[] cachedWeights = cache == null ? null : cache.requireF64Weights();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        Cpu1RangeLauncher.launchIndexed(unit.batchCount() * unit.queryLen(), unit.launchConfig(), (slot, start, end) -> {
            int scoreBase = slot * unit.keyLen();
            for (int row = start; row < end; row++) {
                if (vector) {
                    computeF64SegmentRowVector(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                } else {
                    computeF64SegmentRow(unit, inputs, output, cachedWeights, rowScores, scoreBase, row);
                }
            }
        });
    }

    private static void computeF32ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            F32ArrayInputs inputs,
            float[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotF32(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(), unit.depth())
                    * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF32ArrayRow(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max, anyValid);
    }

    private static void computeF32ArrayRowVector(
            Cpu1PreparedAttentionUnit unit,
            F32ArrayInputs inputs,
            float[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotF32Vector(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF32ArrayRowVector(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeBf16ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            F32ArrayInputs inputs,
            short[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotBf16(inputs.queryBf16, queryBase, inputs.keyBf16, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishBf16ArrayRow(unit, inputs.valueBf16, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF64ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            F64ArrayInputs inputs,
            double[] output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                rowScores[scoreBase + keyIndex] = Double.NaN;
                continue;
            }
            double score = dotF64(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(), unit.depth())
                    * unit.scale();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF64ArrayRow(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF64ArrayRowVector(
            Cpu1PreparedAttentionUnit unit,
            F64ArrayInputs inputs,
            double[] output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                rowScores[scoreBase + keyIndex] = Double.NaN;
                continue;
            }
            double score = dotF64Vector(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scale();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF64ArrayRowVector(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF32SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask.get(JAVA_BYTE, maskBase + keyIndex) == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotF32Segment(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF32SegmentRow(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF32SegmentRowVector(
            Cpu1PreparedAttentionUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask.get(JAVA_BYTE, maskBase + keyIndex) == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotF32SegmentVector(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF32SegmentRowVector(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeBf16SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask.get(JAVA_BYTE, maskBase + keyIndex) == 0) {
                rowScores[scoreBase + keyIndex] = Float.NaN;
                continue;
            }
            float score = dotBf16Segment(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scaleF32();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishBf16SegmentRow(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF64SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask.get(JAVA_BYTE, maskBase + keyIndex) == 0) {
                rowScores[scoreBase + keyIndex] = Double.NaN;
                continue;
            }
            double score = dotF64Segment(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scale();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF64SegmentRow(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void computeF64SegmentRowVector(
            Cpu1PreparedAttentionUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row
    ) {
        int batch = row / unit.queryLen();
        int queryIndex = row - batch * unit.queryLen();
        int queryBase = unit.queryBatchOffset(batch) + queryIndex * unit.depth();
        int keyBase = unit.keyBatchOffset(batch);
        int valueBase = unit.valueBatchOffset(batch);
        int maskBase = unit.hasMask() ? unit.maskBatchOffset(batch) + queryIndex * unit.keyLen() : -1;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (inputs.mask != null && inputs.mask.get(JAVA_BYTE, maskBase + keyIndex) == 0) {
                rowScores[scoreBase + keyIndex] = Double.NaN;
                continue;
            }
            double score = dotF64SegmentVector(inputs.query, queryBase, inputs.key, keyBase + keyIndex * unit.depth(),
                    unit.depth()) * unit.scale();
            rowScores[scoreBase + keyIndex] = score;
            max = Math.max(max, score);
            anyValid = true;
        }
        finishF64SegmentRowVector(unit, inputs.value, output, cachedWeights, rowScores, scoreBase, row, valueBase, max,
                anyValid);
    }

    private static void finishF32ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            float[] value,
            float[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        Arrays.fill(output, outBase, outBase + unit.valueDim(), 0.0f);
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF32(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0f) {
                accumulateF32(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static void finishF32ArrayRowVector(
            Cpu1PreparedAttentionUnit unit,
            float[] value,
            float[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        Arrays.fill(output, outBase, outBase + unit.valueDim(), 0.0f);
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF32Vector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0f) {
                accumulateF32Vector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static void finishBf16ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            short[] value,
            short[] output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        for (int col = 0; col < unit.valueDim(); col++) {
            output[outBase + col] = 0;
        }
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                rowScores[scoreBase + keyIndex] = uniform;
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
            }
            for (int col = 0; col < unit.valueDim(); col++) {
                float sum = 0.0f;
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    sum += TensorDTypeOps.fromBFloat16Bits(value[valueBase + keyIndex * unit.valueDim() + col])
                            * rowScores[scoreBase + keyIndex];
                }
                output[outBase + col] = TensorDTypeOps.toBFloat16Bits(sum);
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            rowScores[scoreBase + keyIndex] = weight;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
        }
        for (int col = 0; col < unit.valueDim(); col++) {
            float sum = 0.0f;
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                float weight = rowScores[scoreBase + keyIndex];
                if (weight != 0.0f) {
                    sum += TensorDTypeOps.fromBFloat16Bits(value[valueBase + keyIndex * unit.valueDim() + col])
                            * weight;
                }
            }
            output[outBase + col] = TensorDTypeOps.toBFloat16Bits(sum);
        }
    }

    private static void finishF64ArrayRow(
            Cpu1PreparedAttentionUnit unit,
            double[] value,
            double[] output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            double max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        Arrays.fill(output, outBase, outBase + unit.valueDim(), 0.0d);
        if (!anyValid) {
            double uniform = 1.0d / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF64(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        double inv = softmaxF64(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double weight = rowScores[scoreBase + keyIndex] * inv;
            publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0d) {
                accumulateF64(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static void finishF64ArrayRowVector(
            Cpu1PreparedAttentionUnit unit,
            double[] value,
            double[] output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            double max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        Arrays.fill(output, outBase, outBase + unit.valueDim(), 0.0d);
        if (!anyValid) {
            double uniform = 1.0d / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF64Vector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        double inv = softmaxF64(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double weight = rowScores[scoreBase + keyIndex] * inv;
            publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0d) {
                accumulateF64Vector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static void finishF32SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            MemorySegment value,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                rowScores[scoreBase + keyIndex] = uniform;
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
            }
            for (int col = 0; col < unit.valueDim(); col++) {
                float sum = 0.0f;
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    sum += value.get(JAVA_FLOAT, (long) (valueBase + keyIndex * unit.valueDim() + col) * Float.BYTES)
                            * rowScores[scoreBase + keyIndex];
                }
                output.set(JAVA_FLOAT, (long) (outBase + col) * Float.BYTES, sum);
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            rowScores[scoreBase + keyIndex] = weight;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
        }
        for (int col = 0; col < unit.valueDim(); col++) {
            float sum = 0.0f;
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                float weight = rowScores[scoreBase + keyIndex];
                if (weight != 0.0f) {
                    sum += value.get(JAVA_FLOAT, (long) (valueBase + keyIndex * unit.valueDim() + col) * Float.BYTES)
                            * weight;
                }
            }
            output.set(JAVA_FLOAT, (long) (outBase + col) * Float.BYTES, sum);
        }
    }

    private static void finishF32SegmentRowVector(
            Cpu1PreparedAttentionUnit unit,
            MemorySegment value,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        zeroF32Segment(output, outBase, unit.valueDim());
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF32SegmentVector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0f) {
                accumulateF32SegmentVector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static void finishBf16SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            MemorySegment value,
            MemorySegment output,
            float[] cachedWeights,
            float[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            float max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        if (!anyValid) {
            float uniform = 1.0f / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                rowScores[scoreBase + keyIndex] = uniform;
                publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
            }
            for (int col = 0; col < unit.valueDim(); col++) {
                float sum = 0.0f;
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    short bits = value.get(JAVA_SHORT, (long) (valueBase + keyIndex * unit.valueDim() + col)
                            * Short.BYTES);
                    sum += TensorDTypeOps.fromBFloat16Bits(bits) * rowScores[scoreBase + keyIndex];
                }
                output.set(JAVA_SHORT, (long) (outBase + col) * Short.BYTES, TensorDTypeOps.toBFloat16Bits(sum));
            }
            return;
        }
        float inv = softmaxF32(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float weight = rowScores[scoreBase + keyIndex] * inv;
            rowScores[scoreBase + keyIndex] = weight;
            publishF32Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
        }
        for (int col = 0; col < unit.valueDim(); col++) {
            float sum = 0.0f;
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                float weight = rowScores[scoreBase + keyIndex];
                if (weight != 0.0f) {
                    short bits = value.get(JAVA_SHORT, (long) (valueBase + keyIndex * unit.valueDim() + col)
                            * Short.BYTES);
                    sum += TensorDTypeOps.fromBFloat16Bits(bits) * weight;
                }
            }
            output.set(JAVA_SHORT, (long) (outBase + col) * Short.BYTES, TensorDTypeOps.toBFloat16Bits(sum));
        }
    }

    private static void finishF64SegmentRow(
            Cpu1PreparedAttentionUnit unit,
            MemorySegment value,
            MemorySegment output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            double max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        if (!anyValid) {
            double uniform = 1.0d / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                rowScores[scoreBase + keyIndex] = uniform;
                publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
            }
            for (int col = 0; col < unit.valueDim(); col++) {
                double sum = 0.0d;
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    sum += value.get(JAVA_DOUBLE, (long) (valueBase + keyIndex * unit.valueDim() + col)
                            * Double.BYTES) * rowScores[scoreBase + keyIndex];
                }
                output.set(JAVA_DOUBLE, (long) (outBase + col) * Double.BYTES, sum);
            }
            return;
        }
        double inv = softmaxF64(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double weight = rowScores[scoreBase + keyIndex] * inv;
            rowScores[scoreBase + keyIndex] = weight;
            publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
        }
        for (int col = 0; col < unit.valueDim(); col++) {
            double sum = 0.0d;
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                double weight = rowScores[scoreBase + keyIndex];
                if (weight != 0.0d) {
                    sum += value.get(JAVA_DOUBLE, (long) (valueBase + keyIndex * unit.valueDim() + col)
                            * Double.BYTES) * weight;
                }
            }
            output.set(JAVA_DOUBLE, (long) (outBase + col) * Double.BYTES, sum);
        }
    }

    private static void finishF64SegmentRowVector(
            Cpu1PreparedAttentionUnit unit,
            MemorySegment value,
            MemorySegment output,
            double[] cachedWeights,
            double[] rowScores,
            int scoreBase,
            int row,
            int valueBase,
            double max,
            boolean anyValid
    ) {
        int outBase = row * unit.valueDim();
        zeroF64Segment(output, outBase, unit.valueDim());
        if (!anyValid) {
            double uniform = 1.0d / unit.keyLen();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, uniform);
                accumulateF64SegmentVector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), uniform,
                        unit.valueDim());
            }
            return;
        }
        double inv = softmaxF64(unit, rowScores, scoreBase, max);
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double weight = rowScores[scoreBase + keyIndex] * inv;
            publishF64Weight(cachedWeights, row, unit.keyLen(), keyIndex, weight);
            if (weight != 0.0d) {
                accumulateF64SegmentVector(output, outBase, value, valueBase + keyIndex * unit.valueDim(), weight,
                        unit.valueDim());
            }
        }
    }

    private static float softmaxF32(Cpu1PreparedAttentionUnit unit, float[] rowScores, int scoreBase, float max) {
        float sum = 0.0f;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (Float.isNaN(rowScores[scoreBase + keyIndex])) {
                rowScores[scoreBase + keyIndex] = 0.0f;
                continue;
            }
            float shifted = rowScores[scoreBase + keyIndex] - max;
            float exp = unit.useFastExpApprox()
                    ? FastTranscendentals.fastExpF32(shifted)
                    : (float) Math.exp(shifted);
            rowScores[scoreBase + keyIndex] = exp;
            sum += exp;
        }
        return 1.0f / sum;
    }

    private static double softmaxF64(Cpu1PreparedAttentionUnit unit, double[] rowScores, int scoreBase, double max) {
        double sum = 0.0d;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            if (Double.isNaN(rowScores[scoreBase + keyIndex])) {
                rowScores[scoreBase + keyIndex] = 0.0d;
                continue;
            }
            double shifted = rowScores[scoreBase + keyIndex] - max;
            double exp = unit.useFastExpApprox()
                    ? FastTranscendentals.fastExpF64(shifted)
                    : Math.exp(shifted);
            rowScores[scoreBase + keyIndex] = exp;
            sum += exp;
        }
        return 1.0d / sum;
    }

    private static float dotF32(float[] left, int leftBase, float[] right, int rightBase, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static float dotF32Vector(float[] left, int leftBase, float[] right, int rightBase, int length) {
        FloatVector sum = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector leftVector = FloatVector.fromArray(F32, left, leftBase + i);
            FloatVector rightVector = FloatVector.fromArray(F32, right, rightBase + i);
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * right[rightBase + i];
        }
        return scalarSum;
    }

    private static double dotF64(double[] left, int leftBase, double[] right, int rightBase, int length) {
        double sum = 0.0d;
        for (int i = 0; i < length; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static double dotF64Vector(double[] left, int leftBase, double[] right, int rightBase, int length) {
        DoubleVector sum = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromArray(F64, left, leftBase + i);
            DoubleVector rightVector = DoubleVector.fromArray(F64, right, rightBase + i);
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * right[rightBase + i];
        }
        return scalarSum;
    }

    private static float dotBf16(short[] left, int leftBase, short[] right, int rightBase, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += TensorDTypeOps.fromBFloat16Bits(left[leftBase + i])
                    * TensorDTypeOps.fromBFloat16Bits(right[rightBase + i]);
        }
        return sum;
    }

    private static float dotF32Segment(MemorySegment left, int leftBase, MemorySegment right, int rightBase, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += left.get(JAVA_FLOAT, (long) (leftBase + i) * Float.BYTES)
                    * right.get(JAVA_FLOAT, (long) (rightBase + i) * Float.BYTES);
        }
        return sum;
    }

    private static float dotF32SegmentVector(
            MemorySegment left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        FloatVector sum = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector leftVector = FloatVector.fromMemorySegment(
                    F32,
                    left,
                    (long) (leftBase + i) * Float.BYTES,
                    ORDER
            );
            FloatVector rightVector = FloatVector.fromMemorySegment(
                    F32,
                    right,
                    (long) (rightBase + i) * Float.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left.get(JAVA_FLOAT, (long) (leftBase + i) * Float.BYTES)
                    * right.get(JAVA_FLOAT, (long) (rightBase + i) * Float.BYTES);
        }
        return scalarSum;
    }

    private static double dotF64Segment(MemorySegment left, int leftBase, MemorySegment right, int rightBase, int length) {
        double sum = 0.0d;
        for (int i = 0; i < length; i++) {
            sum += left.get(JAVA_DOUBLE, (long) (leftBase + i) * Double.BYTES)
                    * right.get(JAVA_DOUBLE, (long) (rightBase + i) * Double.BYTES);
        }
        return sum;
    }

    private static double dotF64SegmentVector(
            MemorySegment left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        DoubleVector sum = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromMemorySegment(
                    F64,
                    left,
                    (long) (leftBase + i) * Double.BYTES,
                    ORDER
            );
            DoubleVector rightVector = DoubleVector.fromMemorySegment(
                    F64,
                    right,
                    (long) (rightBase + i) * Double.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left.get(JAVA_DOUBLE, (long) (leftBase + i) * Double.BYTES)
                    * right.get(JAVA_DOUBLE, (long) (rightBase + i) * Double.BYTES);
        }
        return scalarSum;
    }

    private static float dotBf16Segment(MemorySegment left, int leftBase, MemorySegment right, int rightBase, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += TensorDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, (long) (leftBase + i) * Short.BYTES))
                    * TensorDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, (long) (rightBase + i) * Short.BYTES));
        }
        return sum;
    }

    private static void accumulateF32(float[] output, int outputBase, float[] value, int valueBase, float weight,
                                      int length) {
        for (int i = 0; i < length; i++) {
            output[outputBase + i] += value[valueBase + i] * weight;
        }
    }

    private static void accumulateF32Vector(
            float[] output,
            int outputBase,
            float[] value,
            int valueBase,
            float weight,
            int length
    ) {
        FloatVector weightVector = FloatVector.broadcast(F32, weight);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector outputVector = FloatVector.fromArray(F32, output, outputBase + i);
            FloatVector valueVector = FloatVector.fromArray(F32, value, valueBase + i);
            valueVector.fma(weightVector, outputVector).intoArray(output, outputBase + i);
        }
        for (; i < length; i++) {
            output[outputBase + i] += value[valueBase + i] * weight;
        }
    }

    private static void accumulateF64(double[] output, int outputBase, double[] value, int valueBase, double weight,
                                      int length) {
        for (int i = 0; i < length; i++) {
            output[outputBase + i] += value[valueBase + i] * weight;
        }
    }

    private static void accumulateF64Vector(
            double[] output,
            int outputBase,
            double[] value,
            int valueBase,
            double weight,
            int length
    ) {
        DoubleVector weightVector = DoubleVector.broadcast(F64, weight);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector outputVector = DoubleVector.fromArray(F64, output, outputBase + i);
            DoubleVector valueVector = DoubleVector.fromArray(F64, value, valueBase + i);
            valueVector.fma(weightVector, outputVector).intoArray(output, outputBase + i);
        }
        for (; i < length; i++) {
            output[outputBase + i] += value[valueBase + i] * weight;
        }
    }

    private static void accumulateF32SegmentVector(
            MemorySegment output,
            int outputBase,
            MemorySegment value,
            int valueBase,
            float weight,
            int length
    ) {
        FloatVector weightVector = FloatVector.broadcast(F32, weight);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            long outputOffset = (long) (outputBase + i) * Float.BYTES;
            long valueOffset = (long) (valueBase + i) * Float.BYTES;
            FloatVector outputVector = FloatVector.fromMemorySegment(F32, output, outputOffset, ORDER);
            FloatVector valueVector = FloatVector.fromMemorySegment(F32, value, valueOffset, ORDER);
            valueVector.fma(weightVector, outputVector).intoMemorySegment(output, outputOffset, ORDER);
        }
        for (; i < length; i++) {
            long outputOffset = (long) (outputBase + i) * Float.BYTES;
            float current = output.get(JAVA_FLOAT, outputOffset);
            float next = current + value.get(JAVA_FLOAT, (long) (valueBase + i) * Float.BYTES) * weight;
            output.set(JAVA_FLOAT, outputOffset, next);
        }
    }

    private static void accumulateF64SegmentVector(
            MemorySegment output,
            int outputBase,
            MemorySegment value,
            int valueBase,
            double weight,
            int length
    ) {
        DoubleVector weightVector = DoubleVector.broadcast(F64, weight);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            long outputOffset = (long) (outputBase + i) * Double.BYTES;
            long valueOffset = (long) (valueBase + i) * Double.BYTES;
            DoubleVector outputVector = DoubleVector.fromMemorySegment(F64, output, outputOffset, ORDER);
            DoubleVector valueVector = DoubleVector.fromMemorySegment(F64, value, valueOffset, ORDER);
            valueVector.fma(weightVector, outputVector).intoMemorySegment(output, outputOffset, ORDER);
        }
        for (; i < length; i++) {
            long outputOffset = (long) (outputBase + i) * Double.BYTES;
            double current = output.get(JAVA_DOUBLE, outputOffset);
            double next = current + value.get(JAVA_DOUBLE, (long) (valueBase + i) * Double.BYTES) * weight;
            output.set(JAVA_DOUBLE, outputOffset, next);
        }
    }

    private static void zeroF32Segment(MemorySegment output, int outputBase, int length) {
        FloatVector zero = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            zero.intoMemorySegment(output, (long) (outputBase + i) * Float.BYTES, ORDER);
        }
        for (; i < length; i++) {
            output.set(JAVA_FLOAT, (long) (outputBase + i) * Float.BYTES, 0.0f);
        }
    }

    private static void zeroF64Segment(MemorySegment output, int outputBase, int length) {
        DoubleVector zero = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            zero.intoMemorySegment(output, (long) (outputBase + i) * Double.BYTES, ORDER);
        }
        for (; i < length; i++) {
            output.set(JAVA_DOUBLE, (long) (outputBase + i) * Double.BYTES, 0.0d);
        }
    }

    private static void publishF32Weight(float[] cachedWeights, int row, int keyLen, int keyIndex, float weight) {
        if (cachedWeights != null) {
            cachedWeights[row * keyLen + keyIndex] = weight;
        }
    }

    private static void publishF64Weight(double[] cachedWeights, int row, int keyLen, int keyIndex, double weight) {
        if (cachedWeights != null) {
            cachedWeights[row * keyLen + keyIndex] = weight;
        }
    }

    private static Cpu1AttentionWeightsCache prepareRuntimeCache(
            Cpu1PreparedAttentionUnit unit,
            ExecutionContext context,
            Tensor outputTensor
    ) {
        if (!outputTensor.getRequiresGrad()) {
            context.clearRuntimeState(outputTensor);
            return null;
        }
        DataType cacheType = unit.dataType() == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32;
        Cpu1AttentionWeightsCache existing = context.runtimeStateFor(outputTensor, Cpu1AttentionWeightsCache.class);
        if (existing != null && existing.matches(cacheType, unit.scoresShape())) {
            return existing;
        }
        Cpu1AttentionWeightsCache cache = cacheType == DataType.FLOAT64
                ? Cpu1AttentionWeightsCache.f64(unit.scoresShape())
                : Cpu1AttentionWeightsCache.f32(unit.scoresShape());
        context.putRuntimeState(outputTensor, cache);
        return cache;
    }

    private static Cpu1ScratchBuffer requireScratch(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        Cpu1ScratchBuffer scratchBuffer = context.requireWorkspace(unit.nodeId(), Cpu1ScratchBuffer.class);
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 attention nodeId=" + unit.nodeId()
                    + " requires prepared row scratch buffer.");
        }
        return scratchBuffer;
    }

    private static F32ArrayInputs f32ArrayInputs(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        Cpu1TensorView query = inputArrayView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context);
        Cpu1TensorView key = inputArrayView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context);
        Cpu1TensorView value = inputArrayView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context);
        Cpu1TensorView mask = unit.hasMask()
                ? inputArrayView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return unit.dataType() == DataType.BFLOAT16
                ? new F32ArrayInputs(null, null, null, query.bfloat16Array(), key.bfloat16Array(),
                value.bfloat16Array(), mask == null ? null : mask.boolArray())
                : new F32ArrayInputs(query.float32Array(), key.float32Array(), value.float32Array(), null, null,
                null, mask == null ? null : mask.boolArray());
    }

    private static F64ArrayInputs f64ArrayInputs(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        Cpu1TensorView query = inputArrayView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context);
        Cpu1TensorView key = inputArrayView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context);
        Cpu1TensorView value = inputArrayView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context);
        Cpu1TensorView mask = unit.hasMask()
                ? inputArrayView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F64ArrayInputs(query.float64Array(), key.float64Array(), value.float64Array(),
                mask == null ? null : mask.boolArray());
    }

    private static F32SegmentInputs f32SegmentInputs(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        Cpu1TensorView query = inputSegmentView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context);
        Cpu1TensorView key = inputSegmentView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context);
        Cpu1TensorView value = inputSegmentView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context);
        Cpu1TensorView mask = unit.hasMask()
                ? inputSegmentView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F32SegmentInputs(query.segment(), key.segment(), value.segment(), mask == null ? null : mask.segment());
    }

    private static F64SegmentInputs f64SegmentInputs(Cpu1PreparedAttentionUnit unit, ExecutionContext context) {
        Cpu1TensorView query = inputSegmentView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context);
        Cpu1TensorView key = inputSegmentView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context);
        Cpu1TensorView value = inputSegmentView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context);
        Cpu1TensorView mask = unit.hasMask()
                ? inputSegmentView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F64SegmentInputs(query.segment(), key.segment(), value.segment(), mask == null ? null : mask.segment());
    }

    private static Cpu1TensorView inputArrayView(
            String role,
            int nodeId,
            int[] expectedShape,
            DataType expectedDataType,
            ExecutionContext context
    ) {
        context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != expectedDataType) {
            throw new UnsupportedOperationException("cpu1 attention " + role + " dtype mismatch. expected="
                    + expectedDataType + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromTensor(tensor);
        requireDenseNoOffset(role, view, Cpu1StorageKind.JAVA_ARRAY);
        requireShape(role, view.shape(), expectedShape);
        return view;
    }

    private static Cpu1TensorView inputSegmentView(
            String role,
            int nodeId,
            int[] expectedShape,
            DataType expectedDataType,
            ExecutionContext context
    ) {
        NativeTensorStorage storage = context.requireNativeReadable(
                nodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != expectedDataType) {
            throw new UnsupportedOperationException("cpu1 attention " + role + " dtype mismatch. expected="
                    + expectedDataType + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromNativeStorage(tensor, storage);
        requireDenseNoOffset(role, view, Cpu1StorageKind.MEMORY_SEGMENT);
        requireShape(role, view.shape(), expectedShape);
        return view;
    }

    private static void publishWeightsToArray(
            Cpu1PreparedAttentionUnit unit,
            Cpu1AttentionWeightsCache cache,
            Cpu1TensorView output
    ) {
        requireShape("attention weights output", output.shape(), unit.outputShape());
        requireWeightsCacheDType(unit, cache);
        int size = unit.outputElementCount();
        switch (unit.dataType()) {
            case FLOAT32 -> System.arraycopy(cache.requireF32Weights(), 0, output.float32Array(), output.storageOffset(), size);
            case FLOAT64 -> System.arraycopy(cache.requireF64Weights(), 0, output.float64Array(), output.storageOffset(), size);
            case BFLOAT16 -> {
                float[] source = cache.requireF32Weights();
                short[] target = output.bfloat16Array();
                for (int i = 0; i < size; i++) {
                    target[output.storageOffset() + i] = TensorDTypeOps.toBFloat16Bits(source[i]);
                }
            }
            case INT32, INT64, BOOL -> throw unsupportedWeightsDType(unit.dataType());
        }
    }

    private static void publishWeightsToSegment(
            Cpu1PreparedAttentionUnit unit,
            Cpu1AttentionWeightsCache cache,
            MemorySegment output
    ) {
        requireWeightsCacheDType(unit, cache);
        int size = unit.outputElementCount();
        switch (unit.dataType()) {
            case FLOAT32 -> {
                float[] source = cache.requireF32Weights();
                for (int i = 0; i < size; i++) {
                    output.set(JAVA_FLOAT, (long) i * Float.BYTES, source[i]);
                }
            }
            case FLOAT64 -> {
                double[] source = cache.requireF64Weights();
                for (int i = 0; i < size; i++) {
                    output.set(JAVA_DOUBLE, (long) i * Double.BYTES, source[i]);
                }
            }
            case BFLOAT16 -> {
                float[] source = cache.requireF32Weights();
                for (int i = 0; i < size; i++) {
                    output.set(JAVA_SHORT, (long) i * Short.BYTES, TensorDTypeOps.toBFloat16Bits(source[i]));
                }
            }
            case INT32, INT64, BOOL -> throw unsupportedWeightsDType(unit.dataType());
        }
    }

    private static void requireWeightsCacheDType(Cpu1PreparedAttentionUnit unit, Cpu1AttentionWeightsCache cache) {
        DataType outputType = unit.dataType();
        if (outputType == DataType.FLOAT64 && cache.dataType() == DataType.FLOAT64) {
            return;
        }
        if ((outputType == DataType.FLOAT32 || outputType == DataType.BFLOAT16)
                && cache.dataType() == DataType.FLOAT32) {
            return;
        }
        throw new IllegalStateException("cpu1 attention weights cache dtype mismatch: cache=" + cache.dataType()
                + ", output=" + outputType);
    }

    private static UnsupportedOperationException unsupportedWeightsDType(DataType dataType) {
        return new UnsupportedOperationException("cpu1 attention weights does not support output dtype " + dataType);
    }

    private static void requireDenseNoOffset(String role, Cpu1TensorView view, Cpu1StorageKind expectedStorageKind) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 attention dense slice supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 attention dense slice supports only dense contiguous "
                    + "no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 attention " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }

    private record F32ArrayInputs(
            float[] query,
            float[] key,
            float[] value,
            short[] queryBf16,
            short[] keyBf16,
            short[] valueBf16,
            byte[] mask
    ) {
    }

    private record F64ArrayInputs(double[] query, double[] key, double[] value, byte[] mask) {
    }

    private record F32SegmentInputs(MemorySegment query, MemorySegment key, MemorySegment value, MemorySegment mask) {
    }

    private record F64SegmentInputs(MemorySegment query, MemorySegment key, MemorySegment value, MemorySegment mask) {
    }
}
