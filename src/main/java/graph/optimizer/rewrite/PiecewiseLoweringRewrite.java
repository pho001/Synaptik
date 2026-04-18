package graph.optimizer.rewrite;

import config.optimizer.PiecewiseLoweringConfig;
import operations.Operation;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Optional canonicalization pass for externally imported or manually decomposed graphs.
 * Internal Tensor builders should prefer creating the specialized surface op directly
 * instead of relying on this rewrite as a repair step.
 */
public final class PiecewiseLoweringRewrite extends AbstractRewriteRule {
    private final PiecewiseLoweringConfig config;

    public PiecewiseLoweringRewrite(PiecewiseLoweringConfig config) {
        this.config = config == null ? PiecewiseLoweringConfig.defaults() : config;
    }

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (op == null) {
            return tensor;
        }
        return switch (op.opType()) {
            case INV -> config.canonicalSigmoid() ? rewriteCanonicalSigmoid(tensor) : tensor;
            case WHERE -> rewriteWherePattern(tensor);
            default -> tensor;
        };
    }

    private Tensor rewriteCanonicalSigmoid(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return tensor;
        }
        Tensor add = inputs.get(0);
        if (!isOp(add, Operation.OpType.ADD)) {
            return tensor;
        }
        Tensor left = add.getPrevTensors().get(0);
        Tensor right = add.getPrevTensors().get(1);
        Tensor expNode = isConstant(left, 1.0) ? right : isConstant(right, 1.0) ? left : null;
        if (expNode == null || !isOp(expNode, Operation.OpType.EXP)) {
            return tensor;
        }
        Tensor source = extractNegatedSource(expNode.getPrevTensors().get(0));
        if (source == null) {
            return tensor;
        }
        return source.sigmoid();
    }

    private Tensor rewriteWherePattern(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 3) {
            return tensor;
        }
        Tensor condition = inputs.get(0);
        Tensor ifTrue = inputs.get(1);
        Tensor ifFalse = inputs.get(2);

        if (config.reluLikeWhere()) {
            Tensor relu = tryLowerRelu(condition, ifTrue, ifFalse);
            if (relu != condition) {
                return relu;
            }
        }
        if (config.clampLikeWhere()) {
            Tensor clamp = tryLowerClamp(condition, ifTrue, ifFalse);
            if (clamp != condition) {
                return clamp;
            }
        }
        return tensor;
    }

    private Tensor tryLowerRelu(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (!isOp(condition, Operation.OpType.GT)) {
            return condition;
        }
        Tensor source = condition.getPrevTensors().get(0);
        Tensor threshold = condition.getPrevTensors().get(1);
        if (source == ifTrue && isConstant(threshold, 0.0) && isZeroTensorLike(ifFalse, source)) {
            return source.relu();
        }
        return condition;
    }

    private Tensor tryLowerClamp(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
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
        return condition;
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

    private static boolean isZeroTensorLike(Tensor candidate, Tensor reference) {
        if (candidate == null || reference == null) {
            return false;
        }
        if (candidate.getOperation() != null) {
            return false;
        }
        if (candidate.getDataType() != reference.getDataType()) {
            return false;
        }
        if (!Arrays.equals(candidate.getShapeUnsafe(), reference.getShapeUnsafe())) {
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
