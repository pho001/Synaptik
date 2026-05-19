package graph.optimizer.rewrite.canonical;

import config.optimizer.PiecewiseLoweringConfig;
import operations.Operation;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Shared matcher for optional piecewise canonical forms.
 */
public final class PiecewisePatternLowerer {
    public static final int UNBOUNDED_ZERO_SCAN = -1;

    private final PiecewiseLoweringConfig config;
    private final int zeroTensorScanLimit;

    public PiecewisePatternLowerer(PiecewiseLoweringConfig config) {
        this(config, UNBOUNDED_ZERO_SCAN);
    }

    public PiecewisePatternLowerer(PiecewiseLoweringConfig config, int zeroTensorScanLimit) {
        this.config = config == null ? PiecewiseLoweringConfig.defaults() : config;
        this.zeroTensorScanLimit = zeroTensorScanLimit;
    }

    public Tensor lower(Tensor tensor) {
        Operation op = tensor == null ? null : tensor.getOperation();
        if (op == null) {
            return null;
        }
        return switch (op.opType()) {
            case INV -> config.canonicalSigmoid() ? lowerCanonicalSigmoid(tensor) : null;
            case WHERE -> lowerWhere(tensor);
            default -> null;
        };
    }

    public Tensor lowerWhere(Tensor tensor) {
        List<Tensor> inputs = tensor == null ? null : tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 3) {
            return null;
        }
        return lowerWhere(inputs.get(0), inputs.get(1), inputs.get(2));
    }

    public Tensor lowerWhere(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (config.reluLikeWhere()) {
            Tensor relu = lowerRelu(condition, ifTrue, ifFalse);
            if (relu != null) {
                return relu;
            }
        }
        if (config.clampLikeWhere()) {
            Tensor clamp = lowerClamp(condition, ifTrue, ifFalse);
            if (clamp != null) {
                return clamp;
            }
        }
        return null;
    }

    private Tensor lowerCanonicalSigmoid(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        Tensor add = inputs.get(0);
        if (!isOp(add, Operation.OpType.ADD)) {
            return null;
        }
        Tensor left = add.getPrevTensors().get(0);
        Tensor right = add.getPrevTensors().get(1);
        Tensor expNode = isConstant(left, 1.0) ? right : isConstant(right, 1.0) ? left : null;
        if (expNode == null || !isOp(expNode, Operation.OpType.EXP)) {
            return null;
        }
        Tensor source = extractNegatedSource(expNode.getPrevTensors().get(0));
        return source == null ? null : source.sigmoid();
    }

    private Tensor lowerRelu(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (!isOp(condition, Operation.OpType.GT)) {
            return null;
        }
        Tensor source = condition.getPrevTensors().get(0);
        Tensor threshold = condition.getPrevTensors().get(1);
        if (source == ifTrue && isConstant(threshold, 0.0) && isZeroTensorLike(ifFalse, source)) {
            return source.relu();
        }
        return null;
    }

    private Tensor lowerClamp(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (isOp(condition, Operation.OpType.LT)) {
            Tensor source = condition.getPrevTensors().get(0);
            Tensor threshold = condition.getPrevTensors().get(1);
            if (source == ifFalse && isScalarConstantLike(ifTrue, threshold)) {
                return source.clampMin(threshold.scalarAsDouble());
            }
        }
        if (isOp(condition, Operation.OpType.GT)) {
            Tensor source = condition.getPrevTensors().get(0);
            Tensor threshold = condition.getPrevTensors().get(1);
            if (source == ifFalse && isScalarConstantLike(ifTrue, threshold)) {
                return source.clampMax(threshold.scalarAsDouble());
            }
        }
        return null;
    }

    private static Tensor extractNegatedSource(Tensor tensor) {
        if (isOp(tensor, Operation.OpType.NEG)) {
            return tensor.getPrevTensors().get(0);
        }
        if (isOp(tensor, Operation.OpType.MUL_SCALAR)
                && tensor.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar
                && Math.abs(mulScalar.getScalar() + 1.0) < 1e-12) {
            return tensor.getPrevTensors().get(0);
        }
        return null;
    }

    private static boolean isScalarConstantLike(Tensor candidate, Tensor reference) {
        return candidate != null
                && reference != null
                && candidate.getOperation() == null
                && reference.getOperation() == null
                && candidate.getFlatDataSize() == 1
                && reference.getFlatDataSize() == 1
                && candidate.getDataType() == reference.getDataType()
                && Math.abs(candidate.scalarAsDouble() - reference.scalarAsDouble()) < 1e-12;
    }

    private boolean isZeroTensorLike(Tensor candidate, Tensor reference) {
        if (candidate == null || reference == null) {
            return false;
        }
        if (candidate.getOperation() != null || candidate.getDataType() != reference.getDataType()) {
            return false;
        }
        if (!Arrays.equals(candidate.getShapeUnsafe(), reference.getShapeUnsafe())) {
            return false;
        }
        if (zeroTensorScanLimit >= 0 && candidate.getFlatDataSize() > zeroTensorScanLimit) {
            return false;
        }
        double[] values = candidate.toDoubleArrayCopy();
        for (double value : values) {
            if (Math.abs(value) > 1e-12) {
                return false;
            }
        }
        return true;
    }

    private static boolean isConstant(Tensor tensor, double expected) {
        return tensor != null
                && tensor.getOperation() == null
                && tensor.getFlatDataSize() == 1
                && Math.abs(tensor.scalarAsDouble() - expected) < 1e-12;
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }
}
