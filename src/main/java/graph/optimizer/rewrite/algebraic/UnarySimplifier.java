package graph.optimizer.rewrite.algebraic;

import operations.Operation;
import tensor.Tensor;

final class UnarySimplifier {
    Tensor simplify(Tensor tensor) {
        return switch (tensor.getOperation().opType()) {
            case POW -> simplifyPow(tensor);
            case NEG -> simplifyNeg(tensor);
            case LOG -> simplifyLog(tensor);
            case EXP -> simplifyExp(tensor);
            case INV -> simplifyInv(tensor);
            case SQRT -> simplifySqrt(tensor);
            default -> tensor;
        };
    }

    private Tensor simplifyPow(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (!(tensor.getOperation() instanceof operations.elementwise.unary.pow pow)) return tensor;
        double exponent = pow.getExponent();

        if (exponent == 0.0) return Tensor.onesLike(a);
        if (exponent == 1.0) return a;
        if (exponent == -1.0) return a.inv();
        if (!AlgebraicRewriteSwitches.DISABLE_POW2_TO_MUL && exponent == 2.0) return a.mul(a);
        if (!AlgebraicRewriteSwitches.DISABLE_POW_NEG2_TO_MUL_INV && exponent == -2.0) return a.mul(a).inv();
        if (!AlgebraicRewriteSwitches.DISABLE_POW_INV_TO_NEGEXP && AlgebraicPatterns.isOp(a, Operation.OpType.INV)) {
            return a.getPrevTensors().get(0).pow(-exponent);
        }
        if (!AlgebraicRewriteSwitches.DISABLE_POW_POW_FLATTEN
                && AlgebraicPatterns.isOp(a, Operation.OpType.POW)
                && a.getOperation() instanceof operations.elementwise.unary.pow inner) {
            return a.getPrevTensors().get(0).pow(inner.getExponent() * exponent);
        }

        return tensor;
    }

    private Tensor simplifyNeg(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (AlgebraicPatterns.isConstant(a, 0.0)) return a;
        if (AlgebraicPatterns.isOp(a, Operation.OpType.NEG)) return a.getPrevTensors().get(0);
        if (!AlgebraicRewriteSwitches.DISABLE_NEG_SUB_SWAP && AlgebraicPatterns.isOp(a, Operation.OpType.SUB)) {
            return a.getPrevTensors().get(1).sub(a.getPrevTensors().get(0));
        }
        if (!AlgebraicRewriteSwitches.DISABLE_NEG_MULSCALAR_PUSH
                && AlgebraicPatterns.isOp(a, Operation.OpType.MUL_SCALAR)
                && a.getOperation() instanceof operations.elementwise.unary.mulScalar mulScalar) {
            return a.getPrevTensors().get(0).mul(-mulScalar.getScalar());
        }
        return tensor;
    }

    private Tensor simplifyLog(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (!AlgebraicRewriteSwitches.DISABLE_LOG_POW_TO_MULLOG
                && AlgebraicPatterns.isOp(a, Operation.OpType.POW)
                && a.getOperation() instanceof operations.elementwise.unary.pow pow) {
            return a.getPrevTensors().get(0).log().mul(pow.getExponent());
        }
        if (!AlgebraicRewriteSwitches.DISABLE_LOG_INV_TO_NEGLOG && AlgebraicPatterns.isOp(a, Operation.OpType.INV)) {
            return a.getPrevTensors().get(0).log().neg();
        }
        if (!AlgebraicRewriteSwitches.DISABLE_LOG_SQRT_TO_HALFLOG && AlgebraicPatterns.isOp(a, Operation.OpType.SQRT)) {
            return a.getPrevTensors().get(0).log().mul(0.5);
        }
        return tensor;
    }

    private Tensor simplifyExp(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (!AlgebraicRewriteSwitches.DISABLE_EXP_LOG_CANCEL && AlgebraicPatterns.isOp(a, Operation.OpType.LOG)) {
            return a.getPrevTensors().get(0);
        }
        return tensor;
    }

    private Tensor simplifyInv(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (AlgebraicPatterns.isOp(a, Operation.OpType.INV)) return a.getPrevTensors().get(0);
        if (!AlgebraicRewriteSwitches.DISABLE_INV_POW_TO_NEGEXP
                && AlgebraicPatterns.isOp(a, Operation.OpType.POW)
                && a.getOperation() instanceof operations.elementwise.unary.pow pow) {
            return a.getPrevTensors().get(0).pow(-pow.getExponent());
        }
        if (!AlgebraicRewriteSwitches.DISABLE_INV_EXP_TO_EXPNEG && AlgebraicPatterns.isOp(a, Operation.OpType.EXP)) {
            return a.getPrevTensors().get(0).neg().exp();
        }
        if (!AlgebraicRewriteSwitches.DISABLE_INV_NEG_PUSH && AlgebraicPatterns.isOp(a, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).inv().neg();
        }
        if (!AlgebraicRewriteSwitches.DISABLE_INV_SIGMOID_PATTERN && AlgebraicPatterns.isOp(a, Operation.OpType.SIGMOID)) {
            Tensor x = a.getPrevTensors().get(0);
            return Tensor.onesLike(x).add(x.neg().exp());
        }
        return tensor;
    }

    private Tensor simplifySqrt(Tensor tensor) {
        Tensor a = tensor.getPrevTensors().get(0);
        if (AlgebraicPatterns.isConstant(a, 0.0)) return a;
        if (AlgebraicPatterns.isConstant(a, 1.0)) return a;
        return tensor;
    }
}
