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
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense scalar MIN/MAX/PROD reductions for cpu1.
 */
public final class Cpu1MinMaxProdReductionLoops {
    private Cpu1MinMaxProdReductionLoops() {
    }

    public static void minF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, NumericOp.MIN);
    }

    public static void maxF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, NumericOp.MAX);
    }

    public static void prodF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, NumericOp.PROD);
    }

    public static void minF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, NumericOp.MIN);
    }

    public static void maxF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, NumericOp.MAX);
    }

    public static void prodF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, NumericOp.PROD);
    }

    public static void minBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, NumericOp.MIN);
    }

    public static void maxBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, NumericOp.MAX);
    }

    public static void prodBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, NumericOp.PROD);
    }

    private static void reduceF32(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF32Segment(unit, context, op);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float value = initialF32(op);
                for (int index = 0; index < reduction; index++) {
                    value = applyF32(op, value, inputArray[inputOuterBase + index * unit.innerSize() + inner]);
                }
                outputArray[outputOuterBase + inner] = value;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF64(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF64Segment(unit, context, op);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double value = initialF64(op);
                for (int index = 0; index < reduction; index++) {
                    value = applyF64(op, value, inputArray[inputOuterBase + index * unit.innerSize() + inner]);
                }
                outputArray[outputOuterBase + inner] = value;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceBf16(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceBf16Segment(unit, context, op);
            return;
        }
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        short[] inputArray = input.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float value = initialF32(op);
                for (int index = 0; index < reduction; index++) {
                    float inputValue = TensorDTypeOps.fromBFloat16Bits(
                            inputArray[inputOuterBase + index * unit.innerSize() + inner]
                    );
                    value = applyF32(op, value, inputValue);
                }
                outputArray[outputOuterBase + inner] = TensorDTypeOps.toBFloat16Bits(value);
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float value = initialF32(op);
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    value = applyF32(op, value, inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES));
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, value);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double value = initialF64(op);
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    value = applyF64(op, value, inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES));
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceBf16Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, NumericOp op) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float value = initialF32(op);
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    float inputValue = TensorDTypeOps.fromBFloat16Bits(
                            inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                    );
                    value = applyF32(op, value, inputValue);
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(
                        JAVA_SHORT,
                        (long) outputOffset * Short.BYTES,
                        TensorDTypeOps.toBFloat16Bits(value)
                );
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static float initialF32(NumericOp op) {
        return switch (op) {
            case MIN -> Float.POSITIVE_INFINITY;
            case MAX -> Float.NEGATIVE_INFINITY;
            case PROD -> 1.0f;
        };
    }

    private static double initialF64(NumericOp op) {
        return switch (op) {
            case MIN -> Double.POSITIVE_INFINITY;
            case MAX -> Double.NEGATIVE_INFINITY;
            case PROD -> 1.0d;
        };
    }

    private static float applyF32(NumericOp op, float current, float value) {
        return switch (op) {
            case MIN -> Math.min(current, value);
            case MAX -> Math.max(current, value);
            case PROD -> current * value;
        };
    }

    private static double applyF64(NumericOp op, double current, double value) {
        return switch (op) {
            case MIN -> Math.min(current, value);
            case MAX -> Math.max(current, value);
            case PROD -> current * value;
        };
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
                "cpu1-node-" + unit.nodeId() + ":reduction-native-segment"
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
        context.attachNativeStorage(
                unit.nodeId(),
                nativeOutput,
                "cpu1 " + unit.opType() + " reduced native CPU segment"
        );
    }

    private enum NumericOp {
        MIN,
        MAX,
        PROD
    }
}
