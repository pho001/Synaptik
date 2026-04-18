package graph.optimizer.rewrite;

import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

public class AlgebraicRewrite extends AbstractRewriteRule {
    private static final boolean DISABLE_ALL_TRANSFORMS =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAllTransforms", "false"));
    private static final boolean DISABLE_REBUILD_TOPO_CLOSURE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableRebuildTopologicalClosure", "false"));
    private static final boolean DISABLE_POW2_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePow2ToMul", "false"));
    private static final boolean DISABLE_ADD_SELF_TO_MUL2 =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddSelfToMul2", "false"));
    private static final boolean DISABLE_ADD_NEG_TO_ZERO =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddNegToZero", "false"));
    private static final boolean DISABLE_ADD_NEGNEG_TO_NEGADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddNegNegToNegAdd", "false"));
    private static final boolean DISABLE_ADD_LOGLOG_TO_LOGMUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddLogLogToLogMul", "false"));
    private static final boolean DISABLE_SUB_NEG_TO_ADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableSubNegToAdd", "false"));
    private static final boolean DISABLE_DIV_CONST_TO_MULRECIP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivConstToMulRecip", "false"));
    private static final boolean DISABLE_DIV_MULSCALAR_BY_CONST =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivMulScalarByConst", "false"));
    private static final boolean DISABLE_DIV_INV_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivInvToMul", "false"));
    private static final boolean DISABLE_DIV_ONE_TO_INV =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableDivOneToInv", "false"));
    private static final boolean DISABLE_MULSCALAR_ASSOC =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarAssoc", "false"));
    private static final boolean DISABLE_MULSCALAR_NEG_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarNegPush", "false"));
    private static final boolean DISABLE_MULSCALAR_CONST_FOLD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulScalarConstFold", "false"));
    private static final boolean DISABLE_ADD_SUB_FACTORIZE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableAddSubFactorize", "false"));
    private static final boolean DISABLE_MUL_INV_TO_ONE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulInvToOne", "false"));
    private static final boolean DISABLE_MUL_NEGNEG_TO_MUL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulNegNegToMul", "false"));
    private static final boolean DISABLE_MUL_EXPEXP_TO_EXPADD =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableMulExpExpToExpAdd", "false"));
    private static final boolean DISABLE_NEG_SUB_SWAP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableNegSubSwap", "false"));
    private static final boolean DISABLE_NEG_MULSCALAR_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableNegMulScalarPush", "false"));
    private static final boolean DISABLE_POW_POW_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePowPowFlatten", "false"));
    private static final boolean DISABLE_POW_INV_TO_NEGEXP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disablePowInvToNegExp", "false"));
    private static final boolean DISABLE_LOG_POW_TO_MULLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogPowToMulLog", "false"));
    private static final boolean DISABLE_LOG_INV_TO_NEGLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogInvToNegLog", "false"));
    private static final boolean DISABLE_LOG_SQRT_TO_HALFLOG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableLogSqrtToHalfLog", "false"));
    private static final boolean DISABLE_EXP_LOG_CANCEL =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableExpLogCancel", "false"));
    private static final boolean DISABLE_INV_SIGMOID_PATTERN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvSigmoidPattern", "false"));
    private static final boolean DISABLE_INV_POW_TO_NEGEXP =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvPowToNegExp", "false"));
    private static final boolean DISABLE_INV_EXP_TO_EXPNEG =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvExpToExpNeg", "false"));
    private static final boolean DISABLE_INV_NEG_PUSH =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableInvNegPush", "false"));
    private static final boolean DISABLE_CLAMPMIN_IDENTITY =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMinIdentity", "false"));
    private static final boolean DISABLE_CLAMPMIN_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMinFlatten", "false"));
    private static final boolean DISABLE_CLAMPMAX_IDENTITY =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMaxIdentity", "false"));
    private static final boolean DISABLE_CLAMPMAX_FLATTEN =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.ar.disableClampMaxFlatten", "false"));

    @Override
    protected boolean rebuildClosure() {
        return !DISABLE_REBUILD_TOPO_CLOSURE;
    }

    @Override
    protected Tensor rewriteTensor(Tensor t) {
        if (DISABLE_ALL_TRANSFORMS) return t;
        if (t.getOperation() == null) return t;
        if (t.getOperation().opType() == Operation.OpType.FUSED) return t;

        return switch (t.getOperation().opType()) {
            case ADD -> simplifyAdd(t);
            case SUB -> simplifySub(t);
            case MUL -> simplifyMul(t);
            case MUL_SCALAR -> simplifyMulScalar(t);
            case DIV -> simplifyDiv(t);
            case POW -> simplifyPow(t);
            case NEG -> simplifyNeg(t);
            case LOG -> simplifyLog(t);
            case EXP -> simplifyExp(t);
            case INV -> simplifyInv(t);
            case SQRT -> simplifySqrt(t);
            case WHERE -> simplifyWhere(t);
            case CLAMP_MIN -> simplifyClampMin(t);
            case CLAMP_MAX -> simplifyClampMax(t);
            default -> t;
        };
    }

    private Tensor simplifyAdd(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(a, 0.0)) return b;
        if (isConstant(b, 0.0)) return a;
        if (!DISABLE_ADD_SELF_TO_MUL2 && a == b) return a.mul(2.0);

        if (!DISABLE_ADD_NEG_TO_ZERO && (isNegOf(a, b) || isNegOf(b, a))) return Tensor.zerosLike(a);

        if (!DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = getMulScalarIfBase(a, b);
            if (c1 != null) return b.mul(1.0 + c1);
            Double c2 = getMulScalarIfBase(b, a);
            if (c2 != null) return a.mul(1.0 + c2);
        }

        if (!DISABLE_ADD_NEGNEG_TO_NEGADD && isOp(a, Operation.OpType.NEG) && isOp(b, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).neg();
        }

        if (!DISABLE_ADD_LOGLOG_TO_LOGMUL && isOp(a, Operation.OpType.LOG) && isOp(b, Operation.OpType.LOG)) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0)).log();
        }

        return t;
    }

    private Tensor simplifySub(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(b, 0.0)) return a;
        if (isConstant(a, 0.0)) return b.neg();
        if (a == b) return Tensor.zerosLike(a);

        if (!DISABLE_SUB_NEG_TO_ADD && isOp(b, Operation.OpType.NEG)) return a.add(b.getPrevTensors().get(0));

        if (!DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = getMulScalarIfBase(b, a);
            if (c1 != null) return a.mul(1.0 - c1);
            Double c2 = getMulScalarIfBase(a, b);
            if (c2 != null) return b.mul(c2 - 1.0);
        }

        return t;
    }

    private Tensor simplifyMul(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(a, 0.0) || isConstant(b, 0.0)) return Tensor.zerosLike(t);
        if (isConstant(a, 1.0)) return b;
        if (isConstant(b, 1.0)) return a;
        if (isConstant(a, -1.0)) return b.neg();
        if (isConstant(b, -1.0)) return a.neg();

        if (!DISABLE_MUL_INV_TO_ONE && isOp(a, Operation.OpType.INV) && a.getPrevTensors().get(0) == b) return Tensor.onesLike(b);
        if (!DISABLE_MUL_INV_TO_ONE && isOp(b, Operation.OpType.INV) && b.getPrevTensors().get(0) == a) return Tensor.onesLike(a);

        if (!DISABLE_MUL_NEGNEG_TO_MUL && isOp(a, Operation.OpType.NEG) && isOp(b, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0));
        }

        if (!DISABLE_MUL_EXPEXP_TO_EXPADD && isOp(a, Operation.OpType.EXP) && isOp(b, Operation.OpType.EXP)) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).exp();
        }

        return t;
    }

    private Tensor simplifyMulScalar(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof operations.elementwise.unary.mulScalar m)) return t;

        double s = m.getScalar();
        if (s == 0.0) return Tensor.zerosLike(input);
        if (s == 1.0) return input;
        if (s == -1.0) return input.neg();

        if (!DISABLE_MULSCALAR_ASSOC && isOp(input, Operation.OpType.MUL_SCALAR) && input.getOperation() instanceof operations.elementwise.unary.mulScalar in) {
            return input.getPrevTensors().get(0).mul(in.getScalar() * s);
        }

        if (!DISABLE_MULSCALAR_NEG_PUSH && isOp(input, Operation.OpType.NEG)) {
            return input.getPrevTensors().get(0).mul(-s);
        }

        if (!DISABLE_MULSCALAR_CONST_FOLD && isConstant(input)) {
            return Tensor.scalar(input.scalarAsDouble() * s, input.getDataType());
        }

        return t;
    }

    private Tensor simplifyDiv(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(a, 0.0)) return Tensor.zerosLike(t);
        if (!DISABLE_DIV_ONE_TO_INV && isConstant(a, 1.0)) return b.inv();
        if (isConstant(b, 1.0)) return a;
        if (isConstant(b, -1.0)) return a.neg();

        if (!DISABLE_DIV_INV_TO_MUL && isOp(b, Operation.OpType.INV)) return a.mul(b.getPrevTensors().get(0));

        if (!DISABLE_DIV_MULSCALAR_BY_CONST
                && isOp(a, Operation.OpType.MUL_SCALAR)
                && a.getOperation() instanceof operations.elementwise.unary.mulScalar ms
                && ms.getScalar() != 0.0
                && isConstant(b, ms.getScalar())) {
            return a.getPrevTensors().get(0);
        }

        if (!DISABLE_DIV_MULSCALAR_BY_CONST
                && isOp(a, Operation.OpType.MUL_SCALAR)
                && a.getOperation() instanceof operations.elementwise.unary.mulScalar ms
                && isConstant(b)) {
            return a.getPrevTensors().get(0).mul(ms.getScalar() / b.scalarAsDouble());
        }

        if (!DISABLE_DIV_CONST_TO_MULRECIP && isConstant(b)) {
            return a.mul(1.0 / b.scalarAsDouble());
        }

        return t;
    }

    private Tensor simplifyPow(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof operations.elementwise.unary.pow p)) return t;
        double e = p.getExponent();

        if (e == 0.0) return Tensor.onesLike(a);
        if (e == 1.0) return a;
        if (e == -1.0) return a.inv();
        if (!DISABLE_POW2_TO_MUL && e == 2.0) return a.mul(a);
        if (!DISABLE_POW_INV_TO_NEGEXP && isOp(a, Operation.OpType.INV)) return a.getPrevTensors().get(0).pow(-e);
        if (!DISABLE_POW_POW_FLATTEN && isOp(a, Operation.OpType.POW) && a.getOperation() instanceof operations.elementwise.unary.pow inner) {
            return a.getPrevTensors().get(0).pow(inner.getExponent() * e);
        }

        return t;
    }

    private Tensor simplifyNeg(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (isConstant(a, 0.0)) return a;
        if (isOp(a, Operation.OpType.NEG)) return a.getPrevTensors().get(0);
        if (!DISABLE_NEG_SUB_SWAP && isOp(a, Operation.OpType.SUB)) return a.getPrevTensors().get(1).sub(a.getPrevTensors().get(0));
        if (!DISABLE_NEG_MULSCALAR_PUSH && isOp(a, Operation.OpType.MUL_SCALAR) && a.getOperation() instanceof operations.elementwise.unary.mulScalar ms) {
            return a.getPrevTensors().get(0).mul(-ms.getScalar());
        }
        return t;
    }

    private Tensor simplifyLog(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (!DISABLE_LOG_POW_TO_MULLOG && isOp(a, Operation.OpType.POW) && a.getOperation() instanceof operations.elementwise.unary.pow p) {
            return a.getPrevTensors().get(0).log().mul(p.getExponent());
        }
        if (!DISABLE_LOG_INV_TO_NEGLOG && isOp(a, Operation.OpType.INV)) {
            return a.getPrevTensors().get(0).log().neg();
        }
        if (!DISABLE_LOG_SQRT_TO_HALFLOG && isOp(a, Operation.OpType.SQRT)) {
            return a.getPrevTensors().get(0).log().mul(0.5);
        }
        return t;
    }

    private Tensor simplifyExp(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (!DISABLE_EXP_LOG_CANCEL && isOp(a, Operation.OpType.LOG)) {
            return a.getPrevTensors().get(0);
        }
        return t;
    }

    private Tensor simplifyInv(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (isOp(a, Operation.OpType.INV)) return a.getPrevTensors().get(0);
        if (!DISABLE_INV_POW_TO_NEGEXP && isOp(a, Operation.OpType.POW) && a.getOperation() instanceof operations.elementwise.unary.pow p) {
            return a.getPrevTensors().get(0).pow(-p.getExponent());
        }
        if (!DISABLE_INV_EXP_TO_EXPNEG && isOp(a, Operation.OpType.EXP)) {
            return a.getPrevTensors().get(0).neg().exp();
        }
        if (!DISABLE_INV_NEG_PUSH && isOp(a, Operation.OpType.NEG)) {
            return a.getPrevTensors().get(0).inv().neg();
        }
        if (!DISABLE_INV_SIGMOID_PATTERN && isOp(a, Operation.OpType.SIGMOID)) {
            Tensor x = a.getPrevTensors().get(0);
            return Tensor.onesLike(x).add(x.neg().exp());
        }
        return t;
    }

    private Tensor simplifySqrt(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (isConstant(a, 0.0)) return a;
        if (isConstant(a, 1.0)) return a;
        return t;
    }

    private Tensor simplifyWhere(Tensor t) {
        return t;
    }

    private Tensor simplifyClampMin(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof operations.elementwise.unary.clampMin clamp)) return t;

        if (!DISABLE_CLAMPMIN_IDENTITY && clamp.getMinValue() == Double.NEGATIVE_INFINITY) {
            return a;
        }
        if (!DISABLE_CLAMPMIN_FLATTEN && isOp(a, Operation.OpType.CLAMP_MIN) && a.getOperation() instanceof operations.elementwise.unary.clampMin inner) {
            return a.getPrevTensors().get(0).clampMin(Math.max(inner.getMinValue(), clamp.getMinValue()));
        }
        return t;
    }

    private Tensor simplifyClampMax(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof operations.elementwise.unary.clampMax clamp)) return t;

        if (!DISABLE_CLAMPMAX_IDENTITY && clamp.getMaxValue() == Double.POSITIVE_INFINITY) {
            return a;
        }
        if (!DISABLE_CLAMPMAX_FLATTEN && isOp(a, Operation.OpType.CLAMP_MAX) && a.getOperation() instanceof operations.elementwise.unary.clampMax inner) {
            return a.getPrevTensors().get(0).clampMax(Math.min(inner.getMaxValue(), clamp.getMaxValue()));
        }
        return t;
    }

    private static boolean isConstant(Tensor tensor) {
        return tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getDataType() != DataType.BOOL
                && tensor.getDataType() != DataType.INT32
                && tensor.getFlatDataSize() == 1;
    }

    private static boolean isConstant(Tensor tensor, double expected) {
        return isConstant(tensor) && Math.abs(tensor.scalarAsDouble() - expected) < 1e-12;
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    private static boolean isNegOf(Tensor maybeNeg, Tensor base) {
        return isOp(maybeNeg, Operation.OpType.NEG) && maybeNeg.getPrevTensors().get(0) == base;
    }

    private static Double getMulScalarIfBase(Tensor maybeMulScalar, Tensor base) {
        if (!isOp(maybeMulScalar, Operation.OpType.MUL_SCALAR)) return null;
        if (!(maybeMulScalar.getOperation() instanceof operations.elementwise.unary.mulScalar ms)) return null;
        if (maybeMulScalar.getPrevTensors().get(0) != base) return null;
        return ms.getScalar();
    }
}
