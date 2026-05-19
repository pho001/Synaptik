package graph.optimizer.rewrite.algebraic;

import operations.Operation;
import tensor.Tensor;

final class ArithmeticSimplifier {
    Tensor simplify(Tensor tensor) {
        return switch (tensor.getOperation().opType()) {
            case ADD -> simplifyAdd(tensor);
            case SUB -> simplifySub(tensor);
            case MUL -> simplifyMul(tensor);
            case MUL_SCALAR -> simplifyMulScalar(tensor);
            case DIV -> simplifyDiv(tensor);
            default -> tensor;
        };
    }

    private Tensor simplifyAdd(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        Tensor b = tensor.getPrevTensors().get(1);

        if (AlgebraicPatterns.isConstant(a, 0.0)) return b;
        if (AlgebraicPatterns.isConstant(b, 0.0)) return a;
        if (!AlgebraicRewriteSwitches.DISABLE_ADD_SELF_TO_MUL2 && a == b) return a.mul(2.0);

        if (!AlgebraicRewriteSwitches.DISABLE_ADD_NEG_TO_ZERO
                && (AlgebraicPatterns.isNegOf(a, b) || AlgebraicPatterns.isNegOf(b, a))) {
            return Tensor.zerosLike(a);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = AlgebraicPatterns.getMulScalarIfBase(a, b);
            if (c1 != null) return b.mul(1.0 + c1);
            Double c2 = AlgebraicPatterns.getMulScalarIfBase(b, a);
            if (c2 != null) return a.mul(1.0 + c2);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_ADD_NEGNEG_TO_NEGADD
                && AlgebraicPatterns.isOp(a, Operation.OpType.NEG)
                && AlgebraicPatterns.isOp(b, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).neg();
        }

        if (!AlgebraicRewriteSwitches.DISABLE_ADD_LOGLOG_TO_LOGMUL
                && AlgebraicPatterns.isOp(a, Operation.OpType.LOG)
                && AlgebraicPatterns.isOp(b, Operation.OpType.LOG)) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0)).log();
        }

        return tensor;
    }

    private Tensor simplifySub(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        Tensor b = tensor.getPrevTensors().get(1);

        if (AlgebraicPatterns.isConstant(b, 0.0)) return a;
        if (AlgebraicPatterns.isConstant(a, 0.0)) return b.neg();
        if (a == b) return Tensor.zerosLike(a);

        if (!AlgebraicRewriteSwitches.DISABLE_SUB_NEG_TO_ADD && AlgebraicPatterns.isOp(b, Operation.OpType.NEG)) {
            return a.add(b.getPrevTensors().get(0));
        }

        if (!AlgebraicRewriteSwitches.DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = AlgebraicPatterns.getMulScalarIfBase(b, a);
            if (c1 != null) return a.mul(1.0 - c1);
            Double c2 = AlgebraicPatterns.getMulScalarIfBase(a, b);
            if (c2 != null) return b.mul(c2 - 1.0);
        }

        return tensor;
    }

    private Tensor simplifyMul(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        Tensor b = tensor.getPrevTensors().get(1);

        if (AlgebraicPatterns.isConstant(a, 0.0) || AlgebraicPatterns.isConstant(b, 0.0)) return Tensor.zerosLike(tensor);
        if (AlgebraicPatterns.isConstant(a, 1.0)) return b;
        if (AlgebraicPatterns.isConstant(b, 1.0)) return a;
        if (AlgebraicPatterns.isConstant(a, -1.0)) return b.neg();
        if (AlgebraicPatterns.isConstant(b, -1.0)) return a.neg();

        if (!AlgebraicRewriteSwitches.DISABLE_MUL_INV_TO_ONE
                && AlgebraicPatterns.isOp(a, Operation.OpType.INV)
                && a.getPrevTensors().get(0) == b) {
            return Tensor.onesLike(b);
        }
        if (!AlgebraicRewriteSwitches.DISABLE_MUL_INV_TO_ONE
                && AlgebraicPatterns.isOp(b, Operation.OpType.INV)
                && b.getPrevTensors().get(0) == a) {
            return Tensor.onesLike(a);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_MUL_NEGNEG_TO_MUL
                && AlgebraicPatterns.isOp(a, Operation.OpType.NEG)
                && AlgebraicPatterns.isOp(b, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0));
        }

        if (!AlgebraicRewriteSwitches.DISABLE_MUL_EXPEXP_TO_EXPADD
                && AlgebraicPatterns.isOp(a, Operation.OpType.EXP)
                && AlgebraicPatterns.isOp(b, Operation.OpType.EXP)) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).exp();
        }

        return tensor;
    }

    private Tensor simplifyMulScalar(Tensor tensor) {
        Tensor input = tensor.getPrevTensors().get(0);
        if (!(tensor.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar)) return tensor;

        double scalar = mulScalar.getScalar();
        if (scalar == 0.0) return Tensor.zerosLike(input);
        if (scalar == 1.0) return input;
        if (scalar == -1.0) return input.neg();

        if (!AlgebraicRewriteSwitches.DISABLE_MULSCALAR_ASSOC
                && AlgebraicPatterns.isOp(input, Operation.OpType.MUL_SCALAR)
                && input.getOperation() instanceof operations.elementwise.unary.mulScalar inner) {
            return input.getPrevTensors().get(0).mul(inner.getScalar() * scalar);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_MULSCALAR_NEG_PUSH && AlgebraicPatterns.isOp(input, Operation.OpType.NEG)) {
            return input.getPrevTensors().get(0).mul(-scalar);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_MULSCALAR_CONST_FOLD && AlgebraicPatterns.isConstant(input)) {
            return Tensor.scalar(input.scalarAsDouble() * scalar, input.getDataType());
        }

        return tensor;
    }

    private Tensor simplifyDiv(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        Tensor b = tensor.getPrevTensors().get(1);

        if (AlgebraicPatterns.isConstant(a, 0.0)) return Tensor.zerosLike(tensor);
        if (!AlgebraicRewriteSwitches.DISABLE_DIV_ONE_TO_INV && AlgebraicPatterns.isConstant(a, 1.0)) return b.inv();
        if (AlgebraicPatterns.isConstant(b, 1.0)) return a;
        if (AlgebraicPatterns.isConstant(b, -1.0)) return a.neg();

        if (!AlgebraicRewriteSwitches.DISABLE_DIV_INV_TO_MUL && AlgebraicPatterns.isOp(b, Operation.OpType.INV)) {
            return a.mul(b.getPrevTensors().get(0));
        }

        if (!AlgebraicRewriteSwitches.DISABLE_DIV_MULSCALAR_BY_CONST
                && AlgebraicPatterns.isOp(a, Operation.OpType.MUL_SCALAR)
                && a.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar
                && mulScalar.getScalar() != 0.0
                && AlgebraicPatterns.isConstant(b, mulScalar.getScalar())) {
            return a.getPrevTensors().get(0);
        }

        if (!AlgebraicRewriteSwitches.DISABLE_DIV_MULSCALAR_BY_CONST
                && AlgebraicPatterns.isOp(a, Operation.OpType.MUL_SCALAR)
                && a.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar
                && AlgebraicPatterns.isConstant(b)) {
            return a.getPrevTensors().get(0).mul(mulScalar.getScalar() / b.scalarAsDouble());
        }

        if (!AlgebraicRewriteSwitches.DISABLE_DIV_CONST_TO_MULRECIP && AlgebraicPatterns.isConstant(b)) {
            return a.mul(1.0 / b.scalarAsDouble());
        }

        return tensor;
    }
}
