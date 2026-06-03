package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Dense scalar SUM/MEAN reductions for the first cpu1 reduction phase.
 */
public final class Cpu1SumMeanReductionLoops {
    private Cpu1SumMeanReductionLoops() {
    }

    public static void sumF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, false);
    }

    public static void meanF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, true);
    }

    public static void sumF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, false);
    }

    public static void meanF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, true);
    }

    public static void sumBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, false);
    }

    public static void meanBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, true);
    }

    private static void reduceF32(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF32Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                }
                outputArray[outputOuterBase + inner] = mean ? sum / reduction : sum;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF64(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF64Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                }
                outputArray[outputOuterBase + inner] = mean ? sum / reduction : sum;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceBf16(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        short[] inputArray = input.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    sum += TensorDTypeOps.fromBFloat16Bits(inputArray[inputOuterBase + index * unit.innerSize() + inner]);
                }
                float value = mean ? sum / reduction : sum;
                outputArray[outputOuterBase + inner] = TensorDTypeOps.toBFloat16Bits(value);
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
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
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    sum += inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, mean ? sum / reduction : sum);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
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
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    sum += inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, mean ? sum / reduction : sum);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
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
        Cpu1Workspace workspace = context.cpu1WorkspaceForNodeId(unit.nodeId());
        if (workspace == null) {
            throw new IllegalStateException("cpu1 " + unit.opType() + " native reduction nodeId=" + unit.nodeId()
                    + " requires prepared native output workspace.");
        }
        return workspace.requireNativeOutputStorage(
                unit.dataType(),
                unit.outputElementCount(),
                unit.nodeId(),
                context,
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
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 " + unit.opType() + " reduced native CPU segment");
    }
}
