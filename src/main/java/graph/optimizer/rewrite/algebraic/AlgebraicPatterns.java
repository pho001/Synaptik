package graph.optimizer.rewrite.algebraic;

import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

final class AlgebraicPatterns {
    private AlgebraicPatterns() {
    }

    static boolean isConstant(Tensor tensor) {
        return tensor != null
                && tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getDataType() != DataType.BOOL
                && tensor.getDataType() != DataType.INT32
                && tensor.getFlatDataSize() == 1;
    }

    static boolean isConstant(Tensor tensor, double expected) {
        return isConstant(tensor) && Math.abs(tensor.scalarAsDouble() - expected) < 1e-12;
    }

    static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    static boolean isNegOf(Tensor maybeNeg, Tensor base) {
        return isOp(maybeNeg, Operation.OpType.NEG) && maybeNeg.getPrevTensors().get(0) == base;
    }

    static Double getMulScalarIfBase(Tensor maybeMulScalar, Tensor base) {
        if (!isOp(maybeMulScalar, Operation.OpType.MUL_SCALAR)) {
            return null;
        }
        if (!(maybeMulScalar.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar)) {
            return null;
        }
        if (maybeMulScalar.getPrevTensors().get(0) != base) {
            return null;
        }
        return mulScalar.getScalar();
    }
}
