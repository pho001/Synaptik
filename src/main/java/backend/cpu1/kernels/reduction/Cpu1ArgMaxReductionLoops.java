package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

/**
 * Dense scalar ARGMAX reductions for cpu1.
 */
public final class Cpu1ArgMaxReductionLoops {
    private Cpu1ArgMaxReductionLoops() {
    }

    public static void argMaxF32ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceF32(input.float32Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxF64ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceF64(input.float64Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxBf16ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceBf16(input.bfloat16Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxI32ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Cpu1TensorView input = inputView(unit, context);
        Cpu1TensorView output = outputView(unit, context);
        reduceI32(input.int32Array(), output.int64Array(), input.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void argMaxI64ToI64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
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
