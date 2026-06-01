package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

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
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
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
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
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
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
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
