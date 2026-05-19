package graph.optimizer.rewrite.algebraic;

import graph.optimizer.rewrite.LocalTensorRewriteRule;
import operations.Operation;
import tensor.Tensor;

/**
 * Algebraic rewrite pass for local arithmetic simplifications.
 *
 * <p>The pass replaces recognized algebraic forms with cheaper equivalent tensor expressions while preserving gradient
 * and backend intent metadata through the base rewrite contract. Individual transforms can be disabled with
 * {@code cg.optimizer.ar.*} system properties.
 */
public final class AlgebraicSimplificationRule extends LocalTensorRewriteRule {
    private final ArithmeticSimplifier arithmetic = new ArithmeticSimplifier();
    private final UnarySimplifier unary = new UnarySimplifier();
    private final ClampSimplifier clamp = new ClampSimplifier();

    @Override
    protected boolean rebuildClosure() {
        return !AlgebraicRewriteSwitches.DISABLE_REBUILD_TOPO_CLOSURE;
    }

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        if (AlgebraicRewriteSwitches.DISABLE_ALL_TRANSFORMS) {
            return tensor;
        }
        if (tensor.getOperation() == null || tensor.getOperation().opType() == Operation.OpType.FUSED) {
            return tensor;
        }

        return switch (tensor.getOperation().opType()) {
            case ADD, SUB, MUL, MUL_SCALAR, DIV -> arithmetic.simplify(tensor);
            case POW, NEG, LOG, EXP, INV, SQRT -> unary.simplify(tensor);
            case CLAMP_MIN, CLAMP_MAX -> clamp.simplify(tensor);
            default -> tensor;
        };
    }
}
