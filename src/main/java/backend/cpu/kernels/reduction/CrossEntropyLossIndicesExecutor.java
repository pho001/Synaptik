package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import backend.cpu.storage.CpuStorageView;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.loss.crossEntropyLossIndices;
import tensor.DataType;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;
import tensor.loss.LossReduction;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class CrossEntropyLossIndicesExecutor {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;

    private CrossEntropyLossIndicesExecutor() {}

    static void executeF64(crossEntropyLossIndices loss, CpuStorageView logits, CpuStorageView targetIndices, CpuStorageView output, CpuKernelContext context) {
        validate(loss, logits, targetIndices, output, context, DataType.FLOAT64);
        if (loss.getReduction() == LossReduction.NONE) {
            if (logits.isArray() && output.isArray()) {
                double[] logitsData = logits.requireF64Array();
                double[] out = output.requireF64Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = result.valid() ? result.loss() : 0.0d;
                });
            } else if (logits.isArray()) {
                double[] logitsData = logits.requireF64Array();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_DOUBLE, (long) outOffset * Double.BYTES, result.valid() ? result.loss() : 0.0d);
                });
            } else if (output.isArray()) {
                MemorySegment logitsData = logits.requireSegment();
                double[] out = output.requireF64Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = result.valid() ? result.loss() : 0.0d;
                });
            } else {
                MemorySegment logitsData = logits.requireSegment();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_DOUBLE, (long) outOffset * Double.BYTES, result.valid() ? result.loss() : 0.0d);
                });
            }
            return;
        }
        ReductionResult result;
        if (logits.isArray()) {
            double[] logitsData = logits.requireF64Array();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        } else {
            MemorySegment logitsData = logits.requireSegment();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossF64(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        }
        writeF64(output, output.storageOffset(), finalizeReduction(result, loss.getReduction(), targetIndices.logicalSize()));
    }

    static void executeF32(crossEntropyLossIndices loss, CpuStorageView logits, CpuStorageView targetIndices, CpuStorageView output, CpuKernelContext context) {
        validate(loss, logits, targetIndices, output, context, DataType.FLOAT32);
        if (loss.getReduction() == LossReduction.NONE) {
            if (logits.isArray() && output.isArray()) {
                float[] logitsData = logits.requireF32Array();
                float[] out = output.requireF32Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = result.valid() ? (float) result.loss() : 0.0f;
                });
            } else if (logits.isArray()) {
                float[] logitsData = logits.requireF32Array();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_FLOAT, (long) outOffset * Float.BYTES, result.valid() ? (float) result.loss() : 0.0f);
                });
            } else if (output.isArray()) {
                MemorySegment logitsData = logits.requireSegment();
                float[] out = output.requireF32Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = result.valid() ? (float) result.loss() : 0.0f;
                });
            } else {
                MemorySegment logitsData = logits.requireSegment();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_FLOAT, (long) outOffset * Float.BYTES, result.valid() ? (float) result.loss() : 0.0f);
                });
            }
            return;
        }
        ReductionResult result;
        if (logits.isArray()) {
            float[] logitsData = logits.requireF32Array();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        } else {
            MemorySegment logitsData = logits.requireSegment();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        }
        writeF32(output, output.storageOffset(), (float) finalizeReduction(result, loss.getReduction(), targetIndices.logicalSize()));
    }

    static void executeBF16(crossEntropyLossIndices loss, CpuStorageView logits, CpuStorageView targetIndices, CpuStorageView output, CpuKernelContext context) {
        validate(loss, logits, targetIndices, output, context, DataType.BFLOAT16);
        if (loss.getReduction() == LossReduction.NONE) {
            if (logits.isArray() && output.isArray()) {
                short[] logitsData = logits.requireBF16Array();
                short[] out = output.requireBF16Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f);
                });
            } else if (logits.isArray()) {
                short[] logitsData = logits.requireBF16Array();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_SHORT, (long) outOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f));
                });
            } else if (output.isArray()) {
                MemorySegment logitsData = logits.requireSegment();
                short[] out = output.requireBF16Array();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f);
                });
            } else {
                MemorySegment logitsData = logits.requireSegment();
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, logits.storageOffset(), (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_SHORT, (long) outOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f));
                });
            }
            return;
        }
        ReductionResult result;
        if (logits.isArray()) {
            short[] logitsData = logits.requireBF16Array();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        } else {
            MemorySegment logitsData = logits.requireSegment();
            result = reduceGroups(logits, targetIndices, loss, context, logits.storageOffset(), (baseLogits, targetOffset, axisStride, axisSize) ->
                    computeLossBF16(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        }
        writeBF16(output, output.storageOffset(), TensorDTypeOps.toBFloat16Bits((float) finalizeReduction(result, loss.getReduction(), targetIndices.logicalSize())));
    }

    static void executeF32ToBF16(crossEntropyLossIndices loss, CpuStorageView logits, float[] logitsData, CpuStorageView targetIndices, CpuStorageView output, CpuKernelContext context) {
        validate(loss, logits, targetIndices, output, context, DataType.BFLOAT16);
        if (logitsData == null) {
            throw new IllegalArgumentException("Float continuation logits cannot be null");
        }
        if (loss.getReduction() == LossReduction.NONE) {
            if (output.isArray()) {
                short[] out = output.requireBF16Array();
                runGroups(logits, targetIndices, output, loss, context, 0, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out[outOffset] = TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f);
                });
            } else {
                MemorySegment out = output.requireSegment();
                runGroups(logits, targetIndices, output, loss, context, 0, (baseLogits, targetOffset, outOffset, axisStride, axisSize) -> {
                    SampleResult result = computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex());
                    out.set(JAVA_SHORT, (long) outOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits(result.valid() ? (float) result.loss() : 0.0f));
                });
            }
            return;
        }
        ReductionResult result = reduceGroups(logits, targetIndices, loss, context, 0, (baseLogits, targetOffset, axisStride, axisSize) ->
                computeLossF32(logitsData, baseLogits, axisStride, axisSize, readIndex(targetIndices, targetOffset), loss.getIgnoreIndex()));
        writeBF16(output, output.storageOffset(), TensorDTypeOps.toBFloat16Bits((float) finalizeReduction(result, loss.getReduction(), targetIndices.logicalSize())));
    }

    private static void validate(crossEntropyLossIndices loss, CpuStorageView logits, CpuStorageView targetIndices, CpuStorageView output, CpuKernelContext context, DataType dtype) {
        if (loss == null || logits == null || targetIndices == null || output == null || context == null) {
            throw new IllegalArgumentException("cross entropy loss from indices execution arguments cannot be null");
        }
        if (logits.dtype() != dtype || output.dtype() != dtype) {
            throw new IllegalArgumentException("Cross entropy loss from indices requires " + dtype
                    + " logits/output storage views, logits=" + logits.dtype() + ", output=" + output.dtype());
        }
        if (targetIndices.dtype() == DataType.BOOL) {
            throw new IllegalArgumentException("Target indices must be numeric integral values");
        }
        int[] logitsShape = logits.shape();
        int classDimension = loss.getClassDimension();
        if (classDimension < 0 || classDimension >= logitsShape.length) {
            throw new IllegalArgumentException("Class dimension out of bounds: " + classDimension);
        }
        int[] expectedTargetShape = reduceShape(logitsShape, classDimension);
        validateShape(targetIndices.shape(), expectedTargetShape, "Target indices shape must equal logits shape without class axis");
        if (loss.getReduction() == LossReduction.NONE) {
            validateShape(output.shape(), expectedTargetShape, "NONE reduction output shape must equal target indices shape");
        } else if (output.shape().length != 1 || output.shape()[0] != 1) {
            throw new IllegalArgumentException("Reduced loss output shape must be [1]");
        }
    }

    private static void runGroups(
            CpuStorageView logits,
            CpuStorageView targetIndices,
            CpuStorageView output,
            crossEntropyLossIndices loss,
            CpuKernelContext context,
            int logitsBaseOffset,
            GroupWriter writer
    ) {
        int[] logitsShape = logits.shape();
        int axis = loss.getClassDimension();
        int[] reducedShape = reduceShape(logitsShape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = logitsShape[axis];
        int[] logitsStrides = logits.strides();
        int[] targetStrides = targetIndices.strides();
        int[] outputStrides = output.strides();
        int axisStride = logitsStrides[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(logits, targetIndices, output, axis)) {
            runDenseContiguousGroups(groupCount, axisSize, logitsBaseOffset, targetIndices.storageOffset(), output.storageOffset(), hints,
                    (group, baseLogits, targetOffset, outOffset) -> writer.write(baseLogits, targetOffset, outOffset, 1, axisSize));
            return;
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, logitsShape, logitsStrides, logitsBaseOffset,
                            targetStrides, targetIndices.storageOffset(),
                            outputStrides, output.storageOffset(), axis, reducedDenseStrides, axisSize, axisStride);
                    writer.write(state.baseLogits(), state.targetOffset(), state.outOffset(), state.axisStride(), state.axisSize());
                }
            });
            return;
        }

        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, logitsShape, logitsStrides, logitsBaseOffset,
                    targetStrides, targetIndices.storageOffset(),
                    outputStrides, output.storageOffset(), axis, reducedDenseStrides, axisSize, axisStride);
            writer.write(state.baseLogits(), state.targetOffset(), state.outOffset(), state.axisStride(), state.axisSize());
        }
    }

    private static ReductionResult reduceGroups(
            CpuStorageView logits,
            CpuStorageView targetIndices,
            crossEntropyLossIndices loss,
            CpuKernelContext context,
            int logitsBaseOffset,
            GroupReducer reducer
    ) {
        int[] logitsShape = logits.shape();
        int axis = loss.getClassDimension();
        int[] reducedShape = reduceShape(logitsShape, axis);
        int[] reducedDenseStrides = TensorMetadata.computeStrides(reducedShape);
        int groupCount = logicalSize(reducedShape);
        int axisSize = logitsShape[axis];
        int[] logitsStrides = logits.strides();
        int[] targetStrides = targetIndices.strides();
        int axisStride = logitsStrides[axis];
        ResolvedReductionHints hints = context.reductionHints();

        if (canUseDenseContiguousLastAxisFastPath(logits, targetIndices, null, axis)) {
            return reduceDenseContiguousGroups(groupCount, axisSize, logitsBaseOffset, targetIndices.storageOffset(), hints, reducer);
        }

        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partialLosses = new double[chunks];
            int[] partialValid = new int[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partialLoss = 0.0d;
                int partialCount = 0;
                for (int group = start; group < end; group++) {
                    GroupState state = groupState(group, logitsShape, logitsStrides, logitsBaseOffset,
                            targetStrides, targetIndices.storageOffset(),
                            null, 0, axis, reducedDenseStrides, axisSize, axisStride);
                    SampleResult result = reducer.compute(state.baseLogits(), state.targetOffset(), state.axisStride(), state.axisSize());
                    if (result.valid()) {
                        partialLoss += result.loss();
                        partialCount++;
                    }
                }
                partialLosses[chunk] = partialLoss;
                partialValid[chunk] = partialCount;
            });
            double totalLoss = 0.0d;
            int totalValid = 0;
            for (int i = 0; i < chunks; i++) {
                totalLoss += partialLosses[i];
                totalValid += partialValid[i];
            }
            return new ReductionResult(totalLoss, totalValid);
        }

        double totalLoss = 0.0d;
        int totalValid = 0;
        for (int group = 0; group < groupCount; group++) {
            GroupState state = groupState(group, logitsShape, logitsStrides, logitsBaseOffset,
                    targetStrides, targetIndices.storageOffset(),
                    null, 0, axis, reducedDenseStrides, axisSize, axisStride);
            SampleResult result = reducer.compute(state.baseLogits(), state.targetOffset(), state.axisStride(), state.axisSize());
            if (result.valid()) {
                totalLoss += result.loss();
                totalValid++;
            }
        }
        return new ReductionResult(totalLoss, totalValid);
    }

    private static ReductionResult reduceDenseContiguousGroups(
            int groupCount,
            int axisSize,
            int logitsBaseOffset,
            int targetBaseOffset,
            ResolvedReductionHints hints,
            GroupReducer reducer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            double[] partialLosses = new double[chunks];
            int[] partialValid = new int[chunks];
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                double partialLoss = 0.0d;
                int partialCount = 0;
                for (int group = start; group < end; group++) {
                    SampleResult result = reducer.compute(logitsBaseOffset + group * axisSize, targetBaseOffset + group, 1, axisSize);
                    if (result.valid()) {
                        partialLoss += result.loss();
                        partialCount++;
                    }
                }
                partialLosses[chunk] = partialLoss;
                partialValid[chunk] = partialCount;
            });
            double totalLoss = 0.0d;
            int totalValid = 0;
            for (int i = 0; i < chunks; i++) {
                totalLoss += partialLosses[i];
                totalValid += partialValid[i];
            }
            return new ReductionResult(totalLoss, totalValid);
        }

        double totalLoss = 0.0d;
        int totalValid = 0;
        for (int group = 0; group < groupCount; group++) {
            SampleResult result = reducer.compute(logitsBaseOffset + group * axisSize, targetBaseOffset + group, 1, axisSize);
            if (result.valid()) {
                totalLoss += result.loss();
                totalValid++;
            }
        }
        return new ReductionResult(totalLoss, totalValid);
    }

    private static void runDenseContiguousGroups(
            int groupCount,
            int axisSize,
            int logitsBaseOffset,
            int targetBaseOffset,
            int outBaseOffset,
            ResolvedReductionHints hints,
            DenseGroupWriter writer
    ) {
        if (hints != null && hints.parallel() && groupCount > 1) {
            int chunkSize = Math.max(1, hints.chunkSize());
            int chunks = (groupCount + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, groupCount);
                for (int group = start; group < end; group++) {
                    writer.write(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, outBaseOffset + group);
                }
            });
            return;
        }
        for (int group = 0; group < groupCount; group++) {
            writer.write(group, logitsBaseOffset + group * axisSize, targetBaseOffset + group, outBaseOffset + group);
        }
    }

    private static GroupState groupState(
            int reducedIndex,
            int[] logitsShape,
            int[] logitsStrides,
            int logitsBaseOffset,
            int[] targetStrides,
            int targetBaseOffset,
            int[] outStrides,
            int outBaseOffset,
            int axis,
            int[] reducedDenseStrides,
            int axisSize,
            int axisStride
    ) {
        int rem = reducedIndex;
        int baseLogits = logitsBaseOffset;
        int targetOffset = targetBaseOffset;
        int outputOffset = outStrides == null ? -1 : outBaseOffset;
        for (int d = 0, rd = 0; d < logitsShape.length; d++) {
            if (d == axis) {
                continue;
            }
            int coord = rem / reducedDenseStrides[rd];
            rem %= reducedDenseStrides[rd];
            baseLogits += coord * logitsStrides[d];
            targetOffset += coord * targetStrides[rd];
            if (outStrides != null) {
                outputOffset += coord * outStrides[rd];
            }
            rd++;
        }
        return new GroupState(baseLogits, targetOffset, outputOffset, axisStride, axisSize);
    }

    private static boolean canUseDenseContiguousLastAxisFastPath(CpuStorageView logits, CpuStorageView targetIndices, CpuStorageView out, int axis) {
        int[] logitsShape = logits.shape();
        int[] logitsStrides = logits.strides();
        return axis == logitsShape.length - 1
                && isDenseContiguous(logitsShape, logitsStrides)
                && logitsStrides[axis] == 1
                && isDenseContiguous(targetIndices.shape(), targetIndices.strides())
                && (out == null || isDenseContiguous(out.shape(), out.strides()));
    }

    private static boolean isDenseContiguous(int[] shape, int[] strides) {
        if (shape.length != strides.length) {
            return false;
        }
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    private static SampleResult computeLossF64(double[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        double targetLogit = logits[baseLogits + targetIndex * axisStride];
        if (axisStride == 1 && axisSize >= F64_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            return new SampleResult(logSumExpContiguousF64(logits, baseLogits, axisSize) - targetLogit, true);
        }
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossF64(MemorySegment logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        double targetLogit = logits.get(JAVA_DOUBLE, (long) (baseLogits + targetIndex * axisStride) * Double.BYTES);
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits.get(JAVA_DOUBLE, (long) offset * Double.BYTES));
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits.get(JAVA_DOUBLE, (long) offset * Double.BYTES) - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossF32(float[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = logits[baseLogits + targetIndex * axisStride];
        if (axisStride == 1 && axisSize >= F32_SPECIES.length() * MIN_VECTOR_AXIS_MULTIPLIER) {
            return new SampleResult(logSumExpContiguousF32(logits, baseLogits, axisSize) - targetLogit, true);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossF32(MemorySegment logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = logits.get(JAVA_FLOAT, (long) (baseLogits + targetIndex * axisStride) * Float.BYTES);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, logits.get(JAVA_FLOAT, (long) offset * Float.BYTES));
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(logits.get(JAVA_FLOAT, (long) offset * Float.BYTES) - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossBF16(short[] logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = TensorDTypeOps.fromBFloat16Bits(logits[baseLogits + targetIndex * axisStride]);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(logits[offset]));
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(TensorDTypeOps.fromBFloat16Bits(logits[offset]) - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleResult computeLossBF16(MemorySegment logits, int baseLogits, int axisStride, int axisSize, int targetIndex, Integer ignoreIndex) {
        if (ignoreIndex != null && targetIndex == ignoreIndex) {
            return SampleResult.ignored();
        }
        validateTargetIndex(targetIndex, axisSize);
        float targetLogit = readBF16(logits, baseLogits + targetIndex * axisStride);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            max = Math.max(max, readBF16(logits, offset));
        }
        double sumExp = 0.0d;
        for (int i = 0, offset = baseLogits; i < axisSize; i++, offset += axisStride) {
            sumExp += Math.exp(readBF16(logits, offset) - max);
        }
        return new SampleResult(max + Math.log(sumExp) - targetLogit, true);
    }

    private static double logSumExpContiguousF64(double[] logits, int base, int length) {
        double max = maxContiguousF64(logits, base, length);
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector maxVector = DoubleVector.broadcast(F64_SPECIES, max);
        DoubleVector sumVector = DoubleVector.zero(F64_SPECIES);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            DoubleVector values = DoubleVector.fromArray(F64_SPECIES, logits, base + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += Math.exp(logits[base + i] - max);
        }
        return max + Math.log(sum);
    }

    private static double logSumExpContiguousF32(float[] logits, int base, int length) {
        float max = maxContiguousF32(logits, base, length);
        int upper = F32_SPECIES.loopBound(length);
        FloatVector maxVector = FloatVector.broadcast(F32_SPECIES, max);
        FloatVector sumVector = FloatVector.zero(F32_SPECIES);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            FloatVector values = FloatVector.fromArray(F32_SPECIES, logits, base + i).sub(maxVector).lanewise(VectorOperators.EXP);
            sumVector = sumVector.add(values);
        }
        double sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            sum += Math.exp(logits[base + i] - max);
        }
        return max + Math.log(sum);
    }

    private static double maxContiguousF64(double[] logits, int base, int length) {
        int upper = F64_SPECIES.loopBound(length);
        DoubleVector vectorMax = DoubleVector.broadcast(F64_SPECIES, Double.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F64_SPECIES.length()) {
            vectorMax = vectorMax.max(DoubleVector.fromArray(F64_SPECIES, logits, base + i));
        }
        double max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, logits[base + i]);
        }
        return max;
    }

    private static float maxContiguousF32(float[] logits, int base, int length) {
        int upper = F32_SPECIES.loopBound(length);
        FloatVector vectorMax = FloatVector.broadcast(F32_SPECIES, Float.NEGATIVE_INFINITY);
        int i = 0;
        for (; i < upper; i += F32_SPECIES.length()) {
            vectorMax = vectorMax.max(FloatVector.fromArray(F32_SPECIES, logits, base + i));
        }
        float max = vectorMax.reduceLanes(VectorOperators.MAX);
        for (; i < length; i++) {
            max = Math.max(max, logits[base + i]);
        }
        return max;
    }

    private static int readIndex(CpuStorageView targetIndices, int storageOffset) {
        return switch (targetIndices.dtype()) {
            case INT32 -> targetIndices.isArray()
                    ? targetIndices.requireI32Array()[storageOffset]
                    : targetIndices.requireSegment().get(JAVA_INT, (long) storageOffset * Integer.BYTES);
            case INT64 -> Math.toIntExact(targetIndices.isArray()
                    ? targetIndices.requireI64Array()[storageOffset]
                    : targetIndices.requireSegment().get(JAVA_LONG, (long) storageOffset * Long.BYTES));
            case FLOAT64 -> toIntegralIndex(targetIndices.isArray()
                    ? targetIndices.requireF64Array()[storageOffset]
                    : targetIndices.requireSegment().get(JAVA_DOUBLE, (long) storageOffset * Double.BYTES));
            case FLOAT32 -> toIntegralIndex(targetIndices.isArray()
                    ? targetIndices.requireF32Array()[storageOffset]
                    : targetIndices.requireSegment().get(JAVA_FLOAT, (long) storageOffset * Float.BYTES));
            case BFLOAT16 -> toIntegralIndex(targetIndices.isArray()
                    ? TensorDTypeOps.fromBFloat16Bits(targetIndices.requireBF16Array()[storageOffset])
                    : readBF16(targetIndices.requireSegment(), storageOffset));
            case BOOL -> throw new IllegalArgumentException("Target indices must be numeric integral values");
        };
    }

    private static float readBF16(MemorySegment segment, int offset) {
        return TensorDTypeOps.fromBFloat16Bits(segment.get(JAVA_SHORT, (long) offset * Short.BYTES));
    }

    private static void writeF64(CpuStorageView output, int offset, double value) {
        if (output.isArray()) {
            output.requireF64Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static void writeF32(CpuStorageView output, int offset, float value) {
        if (output.isArray()) {
            output.requireF32Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static void writeBF16(CpuStorageView output, int offset, short value) {
        if (output.isArray()) {
            output.requireBF16Array()[offset] = value;
        } else {
            output.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, value);
        }
    }

    private static int toIntegralIndex(double raw) {
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return Math.toIntExact(integral);
    }

    private static int toIntegralIndex(float raw) {
        if (!Float.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value");
        }
        int integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-6f) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return integral;
    }

    private static void validateTargetIndex(int targetIndex, int axisSize) {
        if (targetIndex < 0 || targetIndex >= axisSize) {
            throw new IllegalArgumentException("Target index out of range: " + targetIndex + " for classes=" + axisSize);
        }
    }

    private static double finalizeReduction(ReductionResult result, LossReduction reduction, int totalGroups) {
        return switch (reduction) {
            case NONE -> throw new IllegalStateException("NONE reduction does not produce scalar output");
            case SUM -> result.totalLoss();
            case MEAN -> result.totalLoss() / Math.max(1, result.validCount() == 0 ? 0 : result.validCount());
        };
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int logicalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    private static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    @FunctionalInterface
    private interface GroupWriter {
        void write(int baseLogits, int targetOffset, int outOffset, int axisStride, int axisSize);
    }

    @FunctionalInterface
    private interface DenseGroupWriter {
        void write(int group, int baseLogits, int targetOffset, int outOffset);
    }

    @FunctionalInterface
    private interface GroupReducer {
        SampleResult compute(int baseLogits, int targetOffset, int axisStride, int axisSize);
    }

    private record GroupState(int baseLogits, int targetOffset, int outOffset, int axisStride, int axisSize) {}

    private record SampleResult(double loss, boolean valid) {
        static SampleResult ignored() {
            return new SampleResult(0.0d, false);
        }
    }

    private record ReductionResult(double totalLoss, int validCount) {}
}
