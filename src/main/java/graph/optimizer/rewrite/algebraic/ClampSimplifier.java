package graph.optimizer.rewrite.algebraic;

import operations.Operation;
import tensor.Tensor;

final class ClampSimplifier {
    Tensor simplify(Tensor tensor) {
        return switch (tensor.getOperation().opType()) {
            case CLAMP_MIN -> simplifyClampMin(tensor);
            case CLAMP_MAX -> simplifyClampMax(tensor);
            default -> tensor;
        };
    }

    private Tensor simplifyClampMin(Tensor tensor) {
        Tensor input = tensor.getPrevTensors().get(0);
        if (!(tensor.getOperation() instanceof operations.elementwise.unary.clampMin clamp)) return tensor;

        if (!AlgebraicRewriteSwitches.DISABLE_CLAMPMIN_IDENTITY
                && clamp.getMinValue() == Double.NEGATIVE_INFINITY) {
            return input;
        }
        if (!AlgebraicRewriteSwitches.DISABLE_CLAMPMIN_FLATTEN
                && AlgebraicPatterns.isOp(input, Operation.OpType.CLAMP_MIN)
                && input.getOperation() instanceof operations.elementwise.unary.clampMin inner) {
            return input.getPrevTensors().get(0).clampMin(Math.max(inner.getMinValue(), clamp.getMinValue()));
        }
        return tensor;
    }

    private Tensor simplifyClampMax(Tensor tensor) {
        Tensor input = tensor.getPrevTensors().get(0);
        if (!(tensor.getOperation() instanceof operations.elementwise.unary.clampMax clamp)) return tensor;

        if (!AlgebraicRewriteSwitches.DISABLE_CLAMPMAX_IDENTITY
                && clamp.getMaxValue() == Double.POSITIVE_INFINITY) {
            return input;
        }
        if (!AlgebraicRewriteSwitches.DISABLE_CLAMPMAX_FLATTEN
                && AlgebraicPatterns.isOp(input, Operation.OpType.CLAMP_MAX)
                && input.getOperation() instanceof operations.elementwise.unary.clampMax inner) {
            return input.getPrevTensors().get(0).clampMax(Math.min(inner.getMaxValue(), clamp.getMaxValue()));
        }
        return tensor;
    }
}
