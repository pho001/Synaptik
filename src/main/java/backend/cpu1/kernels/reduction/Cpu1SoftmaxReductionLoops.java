package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

/**
 * Dense scalar SOFTMAX and LOG_SOFTMAX kernels for cpu1.
 */
public final class Cpu1SoftmaxReductionLoops {
    private Cpu1SoftmaxReductionLoops() {
    }

    public static void softmaxF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeF32(input.float32Array(), output.float32Array(), input.storageOffset(), output.storageOffset(), unit, false);
        markOutputWritten(unit, output, context);
    }

    public static void softmaxF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeF64(input.float64Array(), output.float64Array(), input.storageOffset(), output.storageOffset(), unit, false);
        markOutputWritten(unit, output, context);
    }

    public static void softmaxBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeBf16(input.bfloat16Array(), output.bfloat16Array(), input.storageOffset(), output.storageOffset(), unit, false);
        markOutputWritten(unit, output, context);
    }

    public static void logSoftmaxF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeF32(input.float32Array(), output.float32Array(), input.storageOffset(), output.storageOffset(), unit, true);
        markOutputWritten(unit, output, context);
    }

    public static void logSoftmaxF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeF64(input.float64Array(), output.float64Array(), input.storageOffset(), output.storageOffset(), unit, true);
        markOutputWritten(unit, output, context);
    }

    public static void logSoftmaxBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        computeBf16(input.bfloat16Array(), output.bfloat16Array(), input.storageOffset(), output.storageOffset(), unit, true);
        markOutputWritten(unit, output, context);
    }

    private static void computeF32(
            float[] inputArray,
            float[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log
    ) {
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
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
    }

    private static void computeF64(
            double[] inputArray,
            double[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log
    ) {
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
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
    }

    private static void computeBf16(
            short[] inputArray,
            short[] outputArray,
            int inputStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedReductionUnit unit,
            boolean log
    ) {
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = inputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputStorageOffset + outer * unit.axisSize() * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
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
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " normalized CPU array");
    }
}
