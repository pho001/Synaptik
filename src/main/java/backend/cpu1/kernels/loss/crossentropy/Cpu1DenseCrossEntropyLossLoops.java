package backend.cpu1.kernels.loss.crossentropy;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedDenseCrossEntropyLossUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class Cpu1DenseCrossEntropyLossLoops {
    private Cpu1DenseCrossEntropyLossLoops() {
    }

    public static void runF32DenseArray(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossF32(
                unit,
                logits.float32Array(),
                targets.float32Array(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.float32Array()[output.storageOffset()] = (float) loss;
        markOutputWritten(unit, output, context);
    }

    public static void runF64DenseArray(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossF64(
                unit,
                logits.float64Array(),
                targets.float64Array(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.float64Array()[output.storageOffset()] = loss;
        markOutputWritten(unit, output, context);
    }

    public static void runBf16DenseArray(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossBf16(
                unit,
                logits.bfloat16Array(),
                targets.bfloat16Array(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.bfloat16Array()[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) loss);
        markOutputWritten(unit, output, context);
    }

    public static void runF32DenseSegment(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossF32Segment(
                unit,
                logits.segment(),
                targets.segment(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.segment().set(JAVA_FLOAT, (long) output.storageOffset() * Float.BYTES, (float) loss);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    public static void runF64DenseSegment(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossF64Segment(
                unit,
                logits.segment(),
                targets.segment(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.segment().set(JAVA_DOUBLE, (long) output.storageOffset() * Double.BYTES, loss);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    public static void runBf16DenseSegment(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossBf16Segment(
                unit,
                logits.segment(),
                targets.segment(),
                logits.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.segment().set(
                JAVA_SHORT,
                (long) output.storageOffset() * Short.BYTES,
                TensorDTypeOps.toBFloat16Bits((float) loss)
        );
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static double reduce(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            ExecutionContext context,
            SampleComputer sampleComputer
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            for (int group = 0; group < unit.groupCount(); group++) {
                total += sampleComputer.compute(group);
            }
            return total / unit.groupCount();
        }
        int slotCount = Cpu1RangeLauncher.slotCount(unit.groupCount(), unit.launchConfig());
        Cpu1ScratchBuffer scratchBuffer = context.cpu1ScratchBufferForNodeId(unit.nodeId());
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 CROSS_ENTROPY_LOSS parallel nodeId=" + unit.nodeId()
                    + " requires prepared F64 scratch buffers.");
        }
        double[] partialLosses = scratchBuffer.requireF64Array(slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.groupCount(), unit.launchConfig(), (slot, start, end) -> {
            double total = 0.0d;
            for (int group = start; group < end; group++) {
                total += sampleComputer.compute(group);
            }
            partialLosses[slot] = total;
        });
        double total = 0.0d;
        for (int slot = 0; slot < slotCount; slot++) {
            total += partialLosses[slot];
        }
        return total / unit.groupCount();
    }

    private static double sampleLossF32(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            float[] logits,
            float[] targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits[logitsBaseOffset + offset]);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = targets[targetsBaseOffset + offset];
            double logit = logits[logitsBaseOffset + offset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double sampleLossF64(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            double[] logits,
            double[] targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double max = Double.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits[logitsBaseOffset + offset]);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = targets[targetsBaseOffset + offset];
            double logit = logits[logitsBaseOffset + offset];
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double sampleLossBf16(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            short[] logits,
            short[] targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(logits[logitsBaseOffset + offset]));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = TensorDTypeOps.fromBFloat16Bits(targets[targetsBaseOffset + offset]);
            double logit = TensorDTypeOps.fromBFloat16Bits(logits[logitsBaseOffset + offset]);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double sampleLossF32Segment(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            MemorySegment logits,
            MemorySegment targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits.get(JAVA_FLOAT, (long) (logitsBaseOffset + offset) * Float.BYTES));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = targets.get(JAVA_FLOAT, (long) (targetsBaseOffset + offset) * Float.BYTES);
            double logit = logits.get(JAVA_FLOAT, (long) (logitsBaseOffset + offset) * Float.BYTES);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double sampleLossF64Segment(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            MemorySegment logits,
            MemorySegment targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double max = Double.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits.get(JAVA_DOUBLE, (long) (logitsBaseOffset + offset) * Double.BYTES));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = targets.get(JAVA_DOUBLE, (long) (targetsBaseOffset + offset) * Double.BYTES);
            double logit = logits.get(JAVA_DOUBLE, (long) (logitsBaseOffset + offset) * Double.BYTES);
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static double sampleLossBf16Segment(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            MemorySegment logits,
            MemorySegment targets,
            int logitsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(
                    logits.get(JAVA_SHORT, (long) (logitsBaseOffset + offset) * Short.BYTES)
            ));
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            double target = TensorDTypeOps.fromBFloat16Bits(
                    targets.get(JAVA_SHORT, (long) (targetsBaseOffset + offset) * Short.BYTES)
            );
            double logit = TensorDTypeOps.fromBFloat16Bits(
                    logits.get(JAVA_SHORT, (long) (logitsBaseOffset + offset) * Short.BYTES)
            );
            sumExp += Math.exp(logit - max);
            weightedLogits += target * logit;
            targetSum += target;
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static int baseOffset(Cpu1PreparedDenseCrossEntropyLossUnit unit, int group) {
        int[] shape = unit.logitsShape();
        int remaining = group;
        int offset = 0;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim == unit.classAxis()) {
                continue;
            }
            int reducedStride = reducedDenseStride(shape, dim, unit.classAxis());
            int coord = remaining / reducedStride;
            remaining %= reducedStride;
            offset += coord * denseStride(shape, dim);
        }
        return offset;
    }

    private static int reducedDenseStride(int[] shape, int dim, int classAxis) {
        int stride = 1;
        for (int index = dim + 1; index < shape.length; index++) {
            if (index != classAxis) {
                stride = Math.multiplyExact(stride, shape[index]);
            }
        }
        return stride;
    }

    private static int denseStride(int[] shape, int dim) {
        int stride = 1;
        for (int index = dim + 1; index < shape.length; index++) {
            stride = Math.multiplyExact(stride, shape[index]);
        }
        return stride;
    }

    private static Cpu1TensorView inputArrayView(int inputNodeId, ExecutionContext context) {
        context.requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(inputNodeId);
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView inputSegmentView(int inputNodeId, ExecutionContext context) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(inputNodeId);
        return Cpu1TensorView.fromNativeStorage(input, nativeInput);
    }

    private static Cpu1TensorView outputArrayView(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            ExecutionContext context
    ) {
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                1,
                "cpu1-node-" + unit.nodeId() + ":dense-cross-entropy-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 CROSS_ENTROPY_LOSS CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedDenseCrossEntropyLossUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 CROSS_ENTROPY_LOSS native CPU segment");
    }

    @FunctionalInterface
    private interface SampleComputer {
        double compute(int group);
    }
}
