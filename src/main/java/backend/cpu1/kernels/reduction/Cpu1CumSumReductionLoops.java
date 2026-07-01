package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
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
 * Dense scalar CUMSUM scans for cpu1.
 */
public final class Cpu1CumSumReductionLoops {
    private Cpu1CumSumReductionLoops() {
    }

    public static void cumSumF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanF32(input.float32Array(), output.float32Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumF32DenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        scanF32Segment(unit, context);
    }

    public static void cumSumF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanF64(input.float64Array(), output.float64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumF64DenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        scanF64Segment(unit, context);
    }

    public static void cumSumBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanBf16(input.bfloat16Array(), output.bfloat16Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumBf16DenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        scanBf16Segment(unit, context);
    }

    public static void cumSumI32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanI32(input.int32Array(), output.int32Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumI32DenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        scanI32Segment(unit, context);
    }

    public static void cumSumI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanI64(input.int64Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumI64DenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        scanI64Segment(unit, context);
    }

    private static void scanF32(
            float[] inputArray,
            float[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float acc = 0.0f;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void scanF64(
            double[] inputArray,
            double[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double acc = 0.0d;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        double value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        double value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void scanBf16(
            short[] inputArray,
            short[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float acc = 0.0f;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = TensorDTypeOps.fromBFloat16Bits(inputArray[inputOffset]);
                        if (exclusive) {
                            outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits(acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits(acc);
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = TensorDTypeOps.fromBFloat16Bits(inputArray[inputOffset]);
                        if (exclusive) {
                            outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits(acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits(acc);
                        }
                    }
                }
            }
        }
    }

    private static void scanI32(
            int[] inputArray,
            int[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int acc = 0;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        int value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        int value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void scanI64(
            long[] inputArray,
            long[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit
    ) {
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                long acc = 0L;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        long value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        long value = inputArray[inputOffset];
                        if (exclusive) {
                            outputArray[outputOffset] = acc;
                            acc += value;
                        } else {
                            acc += value;
                            outputArray[outputOffset] = acc;
                        }
                    }
                }
            }
        }
    }

    private static void scanF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float acc = 0.0f;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, acc);
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, acc);
                        }
                    }
                }
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void scanF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double acc = 0.0d;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        double value = inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, acc);
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        double value = inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, acc);
                        }
                    }
                }
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void scanBf16Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float acc = 0.0f;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = TensorDTypeOps.fromBFloat16Bits(
                                inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                        );
                        if (exclusive) {
                            outputSegment.set(JAVA_SHORT, (long) outputOffset * Short.BYTES,
                                    TensorDTypeOps.toBFloat16Bits(acc));
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_SHORT, (long) outputOffset * Short.BYTES,
                                    TensorDTypeOps.toBFloat16Bits(acc));
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        float value = TensorDTypeOps.fromBFloat16Bits(
                                inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                        );
                        if (exclusive) {
                            outputSegment.set(JAVA_SHORT, (long) outputOffset * Short.BYTES,
                                    TensorDTypeOps.toBFloat16Bits(acc));
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_SHORT, (long) outputOffset * Short.BYTES,
                                    TensorDTypeOps.toBFloat16Bits(acc));
                        }
                    }
                }
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void scanI32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                int acc = 0;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        int value = inputSegment.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_INT, (long) outputOffset * Integer.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_INT, (long) outputOffset * Integer.BYTES, acc);
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        int value = inputSegment.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_INT, (long) outputOffset * Integer.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_INT, (long) outputOffset * Integer.BYTES, acc);
                        }
                    }
                }
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void scanI64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        boolean exclusive = unit.cumSumExclusive();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                long acc = 0L;
                if (unit.cumSumReverse()) {
                    for (int index = unit.axisSize() - 1; index >= 0; index--) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        long value = inputSegment.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_LONG, (long) outputOffset * Long.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_LONG, (long) outputOffset * Long.BYTES, acc);
                        }
                    }
                } else {
                    for (int index = 0; index < unit.axisSize(); index++) {
                        int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                        int outputOffset = outputOuterBase + index * unit.innerSize() + inner;
                        long value = inputSegment.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                        if (exclusive) {
                            outputSegment.set(JAVA_LONG, (long) outputOffset * Long.BYTES, acc);
                            acc += value;
                        } else {
                            acc += value;
                            outputSegment.set(JAVA_LONG, (long) outputOffset * Long.BYTES, acc);
                        }
                    }
                }
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
                "cpu1-node-" + unit.nodeId() + ":cumsum-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedReductionUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " scanned CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedReductionUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 " + unit.opType() + " scanned native CPU segment");
    }
}
