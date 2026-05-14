package backend.cpu.fused.exec;

import backend.cpu.fused.codegen.FusedDTypeOps;
import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import backend.cpu.fused.codegen.ScalarDoubleAttribute;
import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.fused.FusedExecutionOptions;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import utils.FastTranscendentals;

import java.util.List;

/**
 * Conservative interpreted fused executable used when generated ASM cannot be prepared.
 *
 * <p>The interpreter keeps the same fused execution contract as generated kernels: it evaluates the
 * lowered fused expression over a half-open output range and writes directly into the output tensor.
 * It is intentionally scalar and correctness-oriented. Hot numeric paths still use generated ASM
 * whenever bytecode preparation succeeds.</p>
 */
public final class InterpretedPreparedFusedExecutable implements PreparedFusedExecutable {
    private final FusedExpressionPlan plan;
    private final int precisionMode;

    /**
     * Creates an interpreter for one lowered fused expression plan.
     *
     * @param plan lowered fused expression
     * @param precisionMode fused compute precision mode
     */
    public InterpretedPreparedFusedExecutable(FusedExpressionPlan plan, int precisionMode) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        this.plan = plan;
        this.precisionMode = precisionMode;
    }

    @Override
    public void applyRangeScalar(
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            int startInclusive,
            int endExclusive,
            FusedExecutionOptions options
    ) {
        int inputCount = plan.inputCount();
        double[] numericValues = new double[plan.nodeCount()];
        boolean[] boolValues = new boolean[plan.nodeCount()];
        for (int index = startInclusive; index < endExclusive; index++) {
            for (FusedNodePlan node : plan.nodes()) {
                if (node.outputType() == DataType.BOOL) {
                    boolValues[node.index()] = evalBool(node, inputs, numericValues, boolValues, index, options);
                } else {
                    numericValues[node.index()] = evalNumeric(node, inputs, numericValues, boolValues, index, options);
                }
            }
            FusedNodePlan output = plan.outputNode();
            int storageIndex = outputStorageIndex(out, index);
            if (output.outputType() == DataType.BOOL) {
                out.getBoolData()[storageIndex] = boolValues[output.index()] ? (byte) 1 : (byte) 0;
            } else {
                storeNumeric(out, storageIndex, numericValues[output.index()]);
            }
        }
    }

    private double evalNumeric(
            FusedNodePlan node,
            List<Tensor> inputs,
            double[] numericValues,
            boolean[] boolValues,
            int index,
            FusedExecutionOptions options
    ) {
        Operation.OpType op = node.opType();
        return switch (op) {
            case ADD -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    + numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case SUB -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    - numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case MUL -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    * numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case DIV -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    / numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case MIN -> Math.min(
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index),
                    numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index)
            );
            case MAX -> Math.max(
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index),
                    numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index)
            );
            case NEG -> -numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index);
            case INV -> 1.0d / numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index);
            case LOG -> Math.log(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case EXP -> exp(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index), options);
            case FAST_EXP -> fastExp(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case TANH -> tanh(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index), options);
            case FAST_TANH -> fastTanh(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case POW -> Math.pow(
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index),
                    ((ScalarDoubleAttribute) node.attributes()).value()
            );
            case POW_TENSOR -> Math.pow(
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index),
                    numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index)
            );
            case SQRT -> Math.sqrt(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case ABS -> Math.abs(numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case CONST_SCALAR -> ((ScalarDoubleAttribute) node.attributes()).value();
            case MUL_SCALAR -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    * ((ScalarDoubleAttribute) node.attributes()).value();
            case RELU -> Math.max(0.0d, numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index));
            case CLAMP_MIN -> Math.max(
                    ((ScalarDoubleAttribute) node.attributes()).value(),
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
            );
            case CLAMP_MAX -> Math.min(
                    ((ScalarDoubleAttribute) node.attributes()).value(),
                    numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
            );
            case SIGMOID -> {
                double value = numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index);
                yield 1.0d / (1.0d + Math.exp(-value));
            }
            case NOOP -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index);
            case WHERE -> boolRef(node.inputRefs().get(0), inputs, boolValues, index)
                    ? numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index)
                    : numericRef(node.inputRefs().get(2), inputs, numericValues, boolValues, index);
            default -> throw new UnsupportedOperationException("Operation " + op + " is not supported by fused interpreter.");
        };
    }

    private boolean evalBool(
            FusedNodePlan node,
            List<Tensor> inputs,
            double[] numericValues,
            boolean[] boolValues,
            int index,
            FusedExecutionOptions options
    ) {
        Operation.OpType op = node.opType();
        return switch (op) {
            case GT -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    > numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case GE -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    >= numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case LT -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    < numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case LE -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    <= numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case EQ -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    == numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case NE -> numericRef(node.inputRefs().get(0), inputs, numericValues, boolValues, index)
                    != numericRef(node.inputRefs().get(1), inputs, numericValues, boolValues, index);
            case LOGICAL_AND -> boolRef(node.inputRefs().get(0), inputs, boolValues, index)
                    && boolRef(node.inputRefs().get(1), inputs, boolValues, index);
            case LOGICAL_OR -> boolRef(node.inputRefs().get(0), inputs, boolValues, index)
                    || boolRef(node.inputRefs().get(1), inputs, boolValues, index);
            case LOGICAL_NOT -> !boolRef(node.inputRefs().get(0), inputs, boolValues, index);
            case WHERE -> boolRef(node.inputRefs().get(0), inputs, boolValues, index)
                    ? boolRef(node.inputRefs().get(1), inputs, boolValues, index)
                    : boolRef(node.inputRefs().get(2), inputs, boolValues, index);
            default -> evalNumeric(node, inputs, numericValues, boolValues, index, options) != 0.0d;
        };
    }

    private double numericRef(
            int ref,
            List<Tensor> inputs,
            double[] numericValues,
            boolean[] boolValues,
            int index
    ) {
        if (ref < plan.inputCount()) {
            Tensor input = inputs.get(ref);
            return loadNumeric(input, storageIndex(plan.inputs().get(ref), index));
        }
        int nodeIndex = ref - plan.inputCount();
        FusedNodePlan node = plan.nodes().get(nodeIndex);
        return node.outputType() == DataType.BOOL ? (boolValues[nodeIndex] ? 1.0d : 0.0d) : numericValues[nodeIndex];
    }

    private boolean boolRef(int ref, List<Tensor> inputs, boolean[] boolValues, int index) {
        if (ref < plan.inputCount()) {
            Tensor input = inputs.get(ref);
            if (input.getDataType() != DataType.BOOL) {
                throw new UnsupportedOperationException("Fused bool ref must use BOOL external input.");
            }
            return input.getBoolData()[storageIndex(plan.inputs().get(ref), index)] != 0;
        }
        int nodeIndex = ref - plan.inputCount();
        if (plan.nodes().get(nodeIndex).outputType() != DataType.BOOL) {
            throw new UnsupportedOperationException("Fused bool ref points to non-BOOL node.");
        }
        return boolValues[nodeIndex];
    }

    private int storageIndex(FusedExternalInputPlan meta, int logicalIndex) {
        if (meta.isLinearAccess()) {
            return meta.storageOffset() + logicalIndex;
        }
        int storageIndex = meta.storageOffset();
        int remaining = logicalIndex;
        int[] dense = meta.logicalOutputDenseStrides();
        int[] strides = meta.effectiveStrides();
        for (int dim = 0; dim < dense.length; dim++) {
            int stride = dense[dim];
            int coord = stride == 0 ? 0 : remaining / stride;
            remaining = stride == 0 ? remaining : remaining % stride;
            storageIndex += coord * strides[dim];
        }
        return storageIndex;
    }

    private int outputStorageIndex(Tensor out, int logicalIndex) {
        if (out.isContiguous()) {
            return out.getStorageOffsetUnsafe() + logicalIndex;
        }
        int storageIndex = out.getStorageOffsetUnsafe();
        int remaining = logicalIndex;
        int[] dense = tensor.TensorMetadata.computeStrides(out.getShape());
        int[] strides = out.getStridesUnsafe();
        for (int dim = 0; dim < dense.length; dim++) {
            int coord = dense[dim] == 0 ? 0 : remaining / dense[dim];
            remaining = dense[dim] == 0 ? remaining : remaining % dense[dim];
            storageIndex += coord * strides[dim];
        }
        return storageIndex;
    }

    private double loadNumeric(Tensor input, int storageIndex) {
        return switch (input.getDataType()) {
            case FLOAT64 -> input.getFloat64Data()[storageIndex];
            case FLOAT32 -> input.getFloat32Data()[storageIndex];
            case BFLOAT16 -> CpuDTypeOps.fromBFloat16Bits(input.getBFloat16Data()[storageIndex]);
            case BOOL -> input.getBoolData()[storageIndex] == 0 ? 0.0d : 1.0d;
            case INT32 -> input.getInt32Data()[storageIndex];
            case INT64 -> input.getInt64Data()[storageIndex];
        };
    }

    private void storeNumeric(Tensor out, int storageIndex, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> out.getFloat64Data()[storageIndex] = value;
            case FLOAT32 -> out.getFloat32Data()[storageIndex] = (float) value;
            case BFLOAT16 -> out.getBFloat16Data()[storageIndex] = CpuDTypeOps.toBFloat16Bits((float) value);
            case BOOL -> out.getBoolData()[storageIndex] = value == 0.0d ? (byte) 0 : (byte) 1;
            case INT32 -> out.getInt32Data()[storageIndex] = (int) value;
            case INT64 -> out.getInt64Data()[storageIndex] = (long) value;
        }
    }

    private double exp(double value, FusedExecutionOptions options) {
        if (options != null && options.useFastExpApprox()) {
            return fastExp(value);
        }
        return Math.exp(value);
    }

    private double fastExp(double value) {
        return precisionMode == FusedDTypeOps.MODE_F64
                ? FastTranscendentals.fastExpF64(value)
                : FastTranscendentals.fastExpF32((float) value);
    }

    private double tanh(double value, FusedExecutionOptions options) {
        if (options != null && options.useFastTanhApprox()) {
            return fastTanh(value);
        }
        return Math.tanh(value);
    }

    private double fastTanh(double value) {
        return precisionMode == FusedDTypeOps.MODE_F64
                ? FastTranscendentals.fastTanhF64(value)
                : FastTranscendentals.fastTanhF32((float) value);
    }
}
