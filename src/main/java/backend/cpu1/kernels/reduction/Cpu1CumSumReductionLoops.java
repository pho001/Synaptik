package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

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

    public static void cumSumF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanF64(input.float64Array(), output.float64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanBf16(input.bfloat16Array(), output.bfloat16Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumI32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanI32(input.int32Array(), output.int32Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void cumSumI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        scanI64(input.int64Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
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
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " scanned CPU array");
    }
}
