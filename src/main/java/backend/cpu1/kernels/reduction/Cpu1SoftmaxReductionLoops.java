package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense scalar SOFTMAX and LOG_SOFTMAX kernels for cpu1.
 */
public final class Cpu1SoftmaxReductionLoops {
    private Cpu1SoftmaxReductionLoops() {
    }

    public static void softmaxF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeF32(unit, context, false);
    }

    public static void softmaxF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeF64(unit, context, false);
    }

    public static void softmaxBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeBf16(unit, context, false);
    }

    public static void logSoftmaxF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeF32(unit, context, true);
    }

    public static void logSoftmaxF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeF64(unit, context, true);
    }

    public static void logSoftmaxBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        computeBf16(unit, context, true);
    }

    private static void computeF32(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            computeF32Segment(unit, context, log);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeF32ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit,
                            log, start, end));
        } else {
            computeF32ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit, log,
                    0, groupCount);
        }
        markOutputWritten(unit, output, context);
    }

    private static void computeF64(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            computeF64Segment(unit, context, log);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeF64ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit,
                            log, start, end));
        } else {
            computeF64ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit, log,
                    0, groupCount);
        }
        markOutputWritten(unit, output, context);
    }

    private static void computeBf16(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            computeBf16Segment(unit, context, log);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        short[] inputArray = input.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeBf16ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit,
                            log, start, end));
        } else {
            computeBf16ArrayRange(inputArray, outputArray, input.storageOffset(), output.storageOffset(), unit, log,
                    0, groupCount);
        }
        markOutputWritten(unit, output, context);
    }

    private static void computeF32ArrayRange(
            float[] inputArray,
            float[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                max = Math.max(max, inputArray[inputOuterBase + index * unit.innerSize() + inner]);
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                sum += Math.exp(inputArray[inputOuterBase + index * unit.innerSize() + inner] - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = inputArray[inputOffset] - max;
                outputArray[outputOffset] = log
                        ? (float) (shifted - logDenominator)
                        : (float) (Math.exp(shifted) / sum);
            }
        }
    }

    private static void computeF64ArrayRange(
            double[] inputArray,
            double[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                max = Math.max(max, inputArray[inputOuterBase + index * unit.innerSize() + inner]);
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                sum += Math.exp(inputArray[inputOuterBase + index * unit.innerSize() + inner] - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = inputArray[inputOffset] - max;
                outputArray[outputOffset] = log ? shifted - logDenominator : Math.exp(shifted) / sum;
            }
        }
    }

    private static void computeBf16ArrayRange(
            short[] inputArray,
            short[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                float value = TensorDTypeOps.fromBFloat16Bits(
                        inputArray[inputOuterBase + index * unit.innerSize() + inner]
                );
                max = Math.max(max, value);
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                float value = TensorDTypeOps.fromBFloat16Bits(
                        inputArray[inputOuterBase + index * unit.innerSize() + inner]
                );
                sum += Math.exp(value - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = TensorDTypeOps.fromBFloat16Bits(inputArray[inputOffset]) - max;
                float value = log ? (float) (shifted - logDenominator) : (float) (Math.exp(shifted) / sum);
                outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits(value);
            }
        }
    }

    private static void computeF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeF32SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(),
                            unit, log, start, end));
        } else {
            computeF32SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(), unit,
                    log, 0, groupCount);
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void computeF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeF64SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(),
                            unit, log, start, end));
        } else {
            computeF64SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(), unit,
                    log, 0, groupCount);
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void computeBf16Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean log) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int groupCount = groupCount(unit);
        if (shouldUseGroupParallel(unit, groupCount)) {
            Cpu1RangeLauncher.launch(groupCount, unit.launchConfig(), (start, end) ->
                    computeBf16SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(),
                            unit, log, start, end));
        } else {
            computeBf16SegmentRange(inputSegment, outputSegment, input.storageOffset(), output.storageOffset(), unit,
                    log, 0, groupCount);
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void computeF32SegmentRange(
            MemorySegment inputSegment,
            MemorySegment outputSegment,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                max = Math.max(max, inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES));
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                sum += Math.exp(inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES) - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES) - max;
                float value = log ? (float) (shifted - logDenominator) : (float) (Math.exp(shifted) / sum);
                outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, value);
            }
        }
    }

    private static void computeF64SegmentRange(
            MemorySegment inputSegment,
            MemorySegment outputSegment,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                max = Math.max(max, inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES));
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                sum += Math.exp(inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES) - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES) - max;
                double value = log ? shifted - logDenominator : Math.exp(shifted) / sum;
                outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
            }
        }
    }

    private static void computeBf16SegmentRange(
            MemorySegment inputSegment,
            MemorySegment outputSegment,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log,
            int startGroupInclusive,
            int endGroupExclusive
    ) {
        for (int group = startGroupInclusive; group < endGroupExclusive; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                float value = TensorDTypeOps.fromBFloat16Bits(
                        inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                );
                max = Math.max(max, value);
            }
            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                float value = TensorDTypeOps.fromBFloat16Bits(
                        inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                );
                sum += Math.exp(value - max);
            }
            double logDenominator = log ? Math.log(sum) : 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = TensorDTypeOps.fromBFloat16Bits(
                        inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                ) - max;
                float value = log ? (float) (shifted - logDenominator) : (float) (Math.exp(shifted) / sum);
                outputSegment.set(
                        JAVA_SHORT,
                        (long) outputOffset * Short.BYTES,
                        TensorDTypeOps.toBFloat16Bits(value)
                );
            }
        }
    }

    private static int groupCount(Cpu1PreparedReductionUnit unit) {
        return Math.multiplyExact(unit.outerSize(), unit.innerSize());
    }

    private static boolean shouldUseGroupParallel(Cpu1PreparedReductionUnit unit, int groupCount) {
        return unit.launchConfig().workerCount() > 1 && groupCount > 1;
    }

    private static Cpu1TensorView inputArrayView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView inputSegmentView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(
                unit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        return Cpu1TensorView.fromNativeStorage(input, nativeInput);
    }

    private static Cpu1TensorView outputArrayView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-node-" + unit.nodeId() + ":softmax-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedReductionUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " normalized CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedReductionUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(
                unit.nodeId(),
                nativeOutput,
                "cpu1 " + unit.opType() + " normalized native CPU segment"
        );
    }
}
