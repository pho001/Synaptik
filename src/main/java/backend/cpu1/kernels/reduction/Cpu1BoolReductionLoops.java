package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;

/**
 * Dense scalar boolean reductions for cpu1.
 */
public final class Cpu1BoolReductionLoops {
    private Cpu1BoolReductionLoops() {
    }

    public static void allBoolDenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBool(unit, context, true);
    }

    public static void anyBoolDenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBool(unit, context, false);
    }

    private static void reduceBool(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean all) {
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

    private static Cpu1TensorView inputView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView outputView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static void markOutputWritten(
            Cpu1PreparedReductionUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " reduced CPU array");
    }
}
