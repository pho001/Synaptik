package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

/**
 * Dense scalar boolean reductions for cpu1.
 */
public final class Cpu1BoolReductionLoops {
    private Cpu1BoolReductionLoops() {
    }

    public static void allBoolDenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBoolArray(unit, context, true);
    }

    public static void allBoolDenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBoolSegment(unit, context, true);
    }

    public static void anyBoolDenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBoolArray(unit, context, false);
    }

    public static void anyBoolDenseScalarSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBoolSegment(unit, context, false);
    }

    private static void reduceBoolArray(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean all) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        byte[] inputArray = input.boolArray();
        byte[] outputArray = output.boolArray();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                boolean value = all;
                for (int index = 0; index < reduction; index++) {
                    boolean inputValue = inputArray[inputOuterBase + index * unit.innerSize() + inner] != 0;
                    if (all) {
                        if (!inputValue) {
                            value = false;
                            break;
                        }
                    } else if (inputValue) {
                        value = true;
                        break;
                    }
                }
                outputArray[outputOuterBase + inner] = value ? (byte) 1 : (byte) 0;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceBoolSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean all) {
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
                boolean value = all;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    boolean inputValue = inputSegment.get(JAVA_BYTE, inputOffset) != 0;
                    if (all) {
                        if (!inputValue) {
                            value = false;
                            break;
                        }
                    } else if (inputValue) {
                        value = true;
                        break;
                    }
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_BYTE, outputOffset, value ? (byte) 1 : (byte) 0);
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
}
