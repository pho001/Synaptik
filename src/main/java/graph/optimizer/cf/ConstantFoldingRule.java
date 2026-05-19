package graph.optimizer.cf;

import graph.optimizer.rewrite.LocalTensorRewriteRule;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.DataType;
import tensor.Tensor;
import utils.SpecialFunctions;

import java.util.Arrays;
import java.util.List;

/**
 * Deterministic compile-time evaluation of small pure constant subgraphs.
 */
public final class ConstantFoldingRule extends LocalTensorRewriteRule {
    private static final int MAX_FOLD_ELEMENTS = Integer.getInteger("cg.optimizer.cf.maxElements", 4096);

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation operation = tensor.getOperation();
        if (operation == null || tensor.getRequiresGrad() || tensor.isBackward()) {
            return tensor;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.isEmpty() || !inputs.stream().allMatch(ConstantFoldingRule::isFoldableConstant)) {
            return tensor;
        }
        if (tensor.getFlatDataSize() > MAX_FOLD_ELEMENTS) {
            return tensor;
        }
        return switch (operation.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR,
                 GT, GE, LT, LE, EQ, NE,
                 LOGICAL_AND, LOGICAL_OR,
                 NEG, INV, LOG, EXP, FAST_EXP, ERF, TANH, FAST_TANH, SQRT, ABS,
                 FLOOR, CEIL, SIGN, MUL_SCALAR, POW, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID,
                 LOGICAL_NOT, WHERE -> foldElementwise(tensor, operation, inputs);
            default -> tensor;
        };
    }

    private static boolean isFoldableConstant(Tensor tensor) {
        return tensor != null
                && tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getFlatDataSize() == 1
                && isKnownConstantLabel(tensor.getLabel());
    }

    private static boolean isKnownConstantLabel(String label) {
        return label != null && (label.equals("scalar_const") || label.endsWith("_const"));
    }

    private static Tensor foldElementwise(Tensor original, Operation operation, List<Tensor> inputs) {
        try {
            return foldElementwiseUnchecked(original, operation, inputs);
        } catch (IllegalArgumentException | UnsupportedOperationException ex) {
            return original;
        }
    }

    private static Tensor foldElementwiseUnchecked(Tensor original, Operation operation, List<Tensor> inputs) {
        int outSize = original.getFlatDataSize();
        int[] outShape = original.getShapeUnsafe().clone();
        DataType outType = original.getDataType();

        if (outType == DataType.BOOL) {
            byte[] out = new byte[outSize];
            for (int i = 0; i < outSize; i++) {
                out[i] = (byte) (evalBoolean(operation, inputs, outShape, i) ? 1 : 0);
            }
            return new Tensor(out, outShape, null, constantLabel(original), DataType.BOOL);
        }
        if (outType == DataType.INT32) {
            return original;
        }

        double[] out = new double[outSize];
        for (int i = 0; i < outSize; i++) {
            out[i] = evalNumeric(operation, inputs, outShape, i);
        }
        return new Tensor(out, outShape, null, constantLabel(original), outType);
    }

    private static String constantLabel(Tensor original) {
        String label = original.getLabel();
        return label == null || label.isBlank() ? "const_fold" : label + "_const";
    }

    private static double evalNumeric(Operation operation, List<Tensor> inputs, int[] outShape, int outFlatIndex) {
        return switch (operation.opType()) {
            case ADD -> numeric(inputs.get(0), outShape, outFlatIndex) + numeric(inputs.get(1), outShape, outFlatIndex);
            case SUB -> numeric(inputs.get(0), outShape, outFlatIndex) - numeric(inputs.get(1), outShape, outFlatIndex);
            case MUL -> numeric(inputs.get(0), outShape, outFlatIndex) * numeric(inputs.get(1), outShape, outFlatIndex);
            case DIV -> numeric(inputs.get(0), outShape, outFlatIndex) / numeric(inputs.get(1), outShape, outFlatIndex);
            case MIN -> Math.min(numeric(inputs.get(0), outShape, outFlatIndex), numeric(inputs.get(1), outShape, outFlatIndex));
            case MAX -> Math.max(numeric(inputs.get(0), outShape, outFlatIndex), numeric(inputs.get(1), outShape, outFlatIndex));
            case POW_TENSOR -> Math.pow(numeric(inputs.get(0), outShape, outFlatIndex), numeric(inputs.get(1), outShape, outFlatIndex));
            case NEG -> -numeric(inputs.get(0), outShape, outFlatIndex);
            case INV -> 1.0d / numeric(inputs.get(0), outShape, outFlatIndex);
            case LOG -> Math.log(numeric(inputs.get(0), outShape, outFlatIndex));
            case EXP, FAST_EXP -> Math.exp(numeric(inputs.get(0), outShape, outFlatIndex));
            case ERF -> SpecialFunctions.erf(numeric(inputs.get(0), outShape, outFlatIndex));
            case TANH, FAST_TANH -> Math.tanh(numeric(inputs.get(0), outShape, outFlatIndex));
            case SQRT -> Math.sqrt(numeric(inputs.get(0), outShape, outFlatIndex));
            case ABS -> Math.abs(numeric(inputs.get(0), outShape, outFlatIndex));
            case FLOOR -> Math.floor(numeric(inputs.get(0), outShape, outFlatIndex));
            case CEIL -> Math.ceil(numeric(inputs.get(0), outShape, outFlatIndex));
            case SIGN -> Math.signum(numeric(inputs.get(0), outShape, outFlatIndex));
            case MUL_SCALAR -> numeric(inputs.get(0), outShape, outFlatIndex) * ((mulScalar) operation).getScalar();
            case POW -> Math.pow(numeric(inputs.get(0), outShape, outFlatIndex), ((pow) operation).getExponent());
            case RELU -> Math.max(0.0d, numeric(inputs.get(0), outShape, outFlatIndex));
            case CLAMP_MIN -> Math.max(((clampMin) operation).getMinValue(), numeric(inputs.get(0), outShape, outFlatIndex));
            case CLAMP_MAX -> Math.min(((clampMax) operation).getMaxValue(), numeric(inputs.get(0), outShape, outFlatIndex));
            case SIGMOID -> {
                double x = numeric(inputs.get(0), outShape, outFlatIndex);
                yield 1.0d / (1.0d + Math.exp(-x));
            }
            case WHERE -> evalBooleanInput(inputs.get(0), outShape, outFlatIndex)
                    ? numeric(inputs.get(1), outShape, outFlatIndex)
                    : numeric(inputs.get(2), outShape, outFlatIndex);
            default -> throw new IllegalArgumentException("Not a numeric fold op: " + operation.opType());
        };
    }

    private static boolean evalBoolean(Operation operation, List<Tensor> inputs, int[] outShape, int outFlatIndex) {
        return switch (operation.opType()) {
            case GT -> numeric(inputs.get(0), outShape, outFlatIndex) > numeric(inputs.get(1), outShape, outFlatIndex);
            case GE -> numeric(inputs.get(0), outShape, outFlatIndex) >= numeric(inputs.get(1), outShape, outFlatIndex);
            case LT -> numeric(inputs.get(0), outShape, outFlatIndex) < numeric(inputs.get(1), outShape, outFlatIndex);
            case LE -> numeric(inputs.get(0), outShape, outFlatIndex) <= numeric(inputs.get(1), outShape, outFlatIndex);
            case EQ -> numeric(inputs.get(0), outShape, outFlatIndex) == numeric(inputs.get(1), outShape, outFlatIndex);
            case NE -> numeric(inputs.get(0), outShape, outFlatIndex) != numeric(inputs.get(1), outShape, outFlatIndex);
            case LOGICAL_AND -> evalBooleanInput(inputs.get(0), outShape, outFlatIndex)
                    && evalBooleanInput(inputs.get(1), outShape, outFlatIndex);
            case LOGICAL_OR -> evalBooleanInput(inputs.get(0), outShape, outFlatIndex)
                    || evalBooleanInput(inputs.get(1), outShape, outFlatIndex);
            case LOGICAL_NOT -> !evalBooleanInput(inputs.get(0), outShape, outFlatIndex);
            case WHERE -> evalBooleanInput(inputs.get(0), outShape, outFlatIndex)
                    ? evalBooleanInput(inputs.get(1), outShape, outFlatIndex)
                    : evalBooleanInput(inputs.get(2), outShape, outFlatIndex);
            default -> throw new IllegalArgumentException("Not a boolean fold op: " + operation.opType());
        };
    }

    private static double numeric(Tensor input, int[] outShape, int outFlatIndex) {
        return input.getByFlatIndex(inputFlatIndex(input, outShape, outFlatIndex));
    }

    private static boolean evalBooleanInput(Tensor input, int[] outShape, int outFlatIndex) {
        int inputIndex = inputFlatIndex(input, outShape, outFlatIndex);
        if (input.getDataType() == DataType.BOOL) {
            return input.toBooleanArrayCopy()[inputIndex];
        }
        return input.getByFlatIndex(inputIndex) != 0.0d;
    }

    private static int inputFlatIndex(Tensor input, int[] outShape, int outFlatIndex) {
        int[] inputShape = input.getShapeUnsafe();
        if (input.getFlatDataSize() == 1) {
            return 0;
        }
        if (Arrays.equals(inputShape, outShape)) {
            return outFlatIndex;
        }
        int[] coords = unravel(outFlatIndex, outShape);
        int rankDelta = outShape.length - inputShape.length;
        if (rankDelta < 0) {
            throw new IllegalArgumentException("Cannot fold broadcast from higher-rank input.");
        }
        int flat = 0;
        for (int i = 0; i < inputShape.length; i++) {
            int outAxis = i + rankDelta;
            int coord = inputShape[i] == 1 ? 0 : coords[outAxis];
            if (coord >= inputShape[i]) {
                throw new IllegalArgumentException("Input is not broadcast-compatible with output.");
            }
            flat = flat * inputShape[i] + coord;
        }
        return flat;
    }

    private static int[] unravel(int flatIndex, int[] shape) {
        int[] coords = new int[shape.length];
        int remaining = flatIndex;
        for (int i = shape.length - 1; i >= 0; i--) {
            coords[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        return coords;
    }
}
