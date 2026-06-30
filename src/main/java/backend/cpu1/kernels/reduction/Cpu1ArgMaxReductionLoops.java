package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense scalar ARGMAX reductions for cpu1.
 */
public final class Cpu1ArgMaxReductionLoops {
    private Cpu1ArgMaxReductionLoops() {
    }

    public static void argMaxF32ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF32Segment(unit, context);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceF32(input.float32Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxF64ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF64Segment(unit, context);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceF64(input.float64Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxBf16ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceBf16Segment(unit, context);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceBf16(input.bfloat16Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxI32ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceI32Segment(unit, context);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceI32(input.int32Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxI64ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceI64Segment(unit, context);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceI64(input.int64Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    private static void reduceF32(
            float[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * reduction * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    double value = inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputArray[outputOuterBase + inner] = bestIndex;
            }
        }
    }

    private static void reduceF64(
            double[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * reduction * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    double value = inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputArray[outputOuterBase + inner] = bestIndex;
            }
        }
    }

    private static void reduceBf16(
            short[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * reduction * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    double value = TensorDTypeOps.fromBFloat16Bits(
                            inputArray[inputOuterBase + index * unit.innerSize() + inner]
                    );
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputArray[outputOuterBase + inner] = bestIndex;
            }
        }
    }

    private static void reduceI32(
            int[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * reduction * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                int bestValue = Integer.MIN_VALUE;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int value = inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    if (!seen || value > bestValue || (unit.argMaxLastIndexWins() && value == bestValue)) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputArray[outputOuterBase + inner] = bestIndex;
            }
        }
    }

    private static void reduceI64(
            long[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * reduction * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                long bestValue = Long.MIN_VALUE;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    long value = inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    if (!seen || value > bestValue || (unit.argMaxLastIndexWins() && value == bestValue)) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputArray[outputOuterBase + inner] = bestIndex;
            }
        }
    }

    private static boolean isBetter(double value, double bestValue, boolean seen, boolean lastIndexWins) {
        return !seen || value > bestValue || (lastIndexWins && Double.compare(value, bestValue) == 0);
    }

    private static void reduceF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    double value = inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputSegment.set(JAVA_LONG, (long) (outputOuterBase + inner) * Long.BYTES, bestIndex);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    double value = inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputSegment.set(JAVA_LONG, (long) (outputOuterBase + inner) * Long.BYTES, bestIndex);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceBf16Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    double value = TensorDTypeOps.fromBFloat16Bits(
                            inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                    );
                    if (isBetter(value, bestValue, seen, unit.argMaxLastIndexWins())) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputSegment.set(JAVA_LONG, (long) (outputOuterBase + inner) * Long.BYTES, bestIndex);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceI32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                int bestValue = Integer.MIN_VALUE;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    int value = inputSegment.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    if (!seen || value > bestValue || (unit.argMaxLastIndexWins() && value == bestValue)) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputSegment.set(JAVA_LONG, (long) (outputOuterBase + inner) * Long.BYTES, bestIndex);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceI64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int bestIndex = 0;
                long bestValue = Long.MIN_VALUE;
                boolean seen = false;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    long value = inputSegment.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    if (!seen || value > bestValue || (unit.argMaxLastIndexWins() && value == bestValue)) {
                        seen = true;
                        bestValue = value;
                        bestIndex = index;
                    }
                }
                outputSegment.set(JAVA_LONG, (long) (outputOuterBase + inner) * Long.BYTES, bestIndex);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static Cpu1TensorView inputView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
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

    private static Cpu1TensorView outputView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-node-" + unit.nodeId() + ":argmax-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedReductionUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " reduced CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedReductionUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 " + unit.opType() + " reduced native CPU segment");
    }
}
