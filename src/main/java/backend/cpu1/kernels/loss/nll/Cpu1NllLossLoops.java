package backend.cpu1.kernels.loss.nll;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedNllLossUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class Cpu1NllLossLoops {
    private Cpu1NllLossLoops() {
    }

    public static void runF32DenseArray(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputArrayView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossF32(
                unit,
                logProbs.float32Array(),
                targets.float32Array(),
                logProbs.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.float32Array()[output.storageOffset()] = (float) loss;
        markOutputWritten(unit, output, context);
    }

    public static void runF64DenseArray(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputArrayView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossF64(
                unit,
                logProbs.float64Array(),
                targets.float64Array(),
                logProbs.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.float64Array()[output.storageOffset()] = loss;
        markOutputWritten(unit, output, context);
    }

    public static void runBf16DenseArray(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputArrayView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double loss = reduce(unit, context, group -> sampleLossBf16(
                unit,
                logProbs.bfloat16Array(),
                targets.bfloat16Array(),
                logProbs.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.bfloat16Array()[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) loss);
        markOutputWritten(unit, output, context);
    }

    public static void runF32DenseSegment(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputSegmentView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossF32Segment(
                unit,
                logProbs.segment(),
                targets.segment(),
                logProbs.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.segment().set(JAVA_FLOAT, (long) output.storageOffset() * Float.BYTES, (float) loss);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    public static void runF64DenseSegment(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputSegmentView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossF64Segment(
                unit,
                logProbs.segment(),
                targets.segment(),
                logProbs.storageOffset(),
                targets.storageOffset(),
                group
        ));
        output.segment().set(JAVA_DOUBLE, (long) output.storageOffset() * Double.BYTES, loss);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    public static void runBf16DenseSegment(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logProbs = inputSegmentView(unit.logProbsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        double loss = reduce(unit, context, group -> sampleLossBf16Segment(
                unit,
                logProbs.segment(),
                targets.segment(),
                logProbs.storageOffset(),
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
            Cpu1PreparedNllLossUnit unit,
            ExecutionContext context,
            SampleComputer sampleComputer
    ) {
        if (unit.groupCount() == 0) {
            return 0.0d;
        }
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
            throw new IllegalStateException("cpu1 NLL_LOSS parallel nodeId=" + unit.nodeId()
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
            Cpu1PreparedNllLossUnit unit,
            float[] logProbs,
            float[] targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= targets[targetsBaseOffset + offset] * logProbs[logProbsBaseOffset + offset];
        }
        return loss;
    }

    private static double sampleLossF64(
            Cpu1PreparedNllLossUnit unit,
            double[] logProbs,
            double[] targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= targets[targetsBaseOffset + offset] * logProbs[logProbsBaseOffset + offset];
        }
        return loss;
    }

    private static double sampleLossBf16(
            Cpu1PreparedNllLossUnit unit,
            short[] logProbs,
            short[] targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= TensorDTypeOps.fromBFloat16Bits(targets[targetsBaseOffset + offset])
                    * TensorDTypeOps.fromBFloat16Bits(logProbs[logProbsBaseOffset + offset]);
        }
        return loss;
    }

    private static double sampleLossF32Segment(
            Cpu1PreparedNllLossUnit unit,
            MemorySegment logProbs,
            MemorySegment targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= targets.get(JAVA_FLOAT, (long) (targetsBaseOffset + offset) * Float.BYTES)
                    * logProbs.get(JAVA_FLOAT, (long) (logProbsBaseOffset + offset) * Float.BYTES);
        }
        return loss;
    }

    private static double sampleLossF64Segment(
            Cpu1PreparedNllLossUnit unit,
            MemorySegment logProbs,
            MemorySegment targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= targets.get(JAVA_DOUBLE, (long) (targetsBaseOffset + offset) * Double.BYTES)
                    * logProbs.get(JAVA_DOUBLE, (long) (logProbsBaseOffset + offset) * Double.BYTES);
        }
        return loss;
    }

    private static double sampleLossBf16Segment(
            Cpu1PreparedNllLossUnit unit,
            MemorySegment logProbs,
            MemorySegment targets,
            int logProbsBaseOffset,
            int targetsBaseOffset,
            int group
    ) {
        int base = baseOffset(unit, group);
        double loss = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            loss -= TensorDTypeOps.fromBFloat16Bits(
                    targets.get(JAVA_SHORT, (long) (targetsBaseOffset + offset) * Short.BYTES)
            ) * TensorDTypeOps.fromBFloat16Bits(
                    logProbs.get(JAVA_SHORT, (long) (logProbsBaseOffset + offset) * Short.BYTES)
            );
        }
        return loss;
    }

    private static int baseOffset(Cpu1PreparedNllLossUnit unit, int group) {
        int[] shape = unit.logProbsShape();
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

    private static Cpu1TensorView outputArrayView(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(Cpu1PreparedNllLossUnit unit, ExecutionContext context) {
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                1,
                "cpu1-node-" + unit.nodeId() + ":nll-loss-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedNllLossUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 NLL_LOSS CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedNllLossUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 NLL_LOSS native CPU segment");
    }

    @FunctionalInterface
    private interface SampleComputer {
        double compute(int group);
    }
}
