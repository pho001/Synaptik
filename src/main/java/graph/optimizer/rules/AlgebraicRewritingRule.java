package graph.optimizer.rules;

import graph.optimizer.OptimizerGraphSupport;
import graph.optimizer.OptimizationRule;
import operations.*;
import tensor.Tensor;

import java.util.*;

public class AlgebraicRewritingRule implements OptimizationRule {
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

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> optimized = new ArrayList<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (Tensor t : sortedGraph) {
            OptimizerGraphSupport.rewriteInputs(t, replacements);

            Tensor simplified = trySimplify(t);
            if (simplified != t) {
                if (t.isBackward()) {
                    simplified.setBackward(true);
                }
                replacements.put(t, simplified);
            } else {
                optimized.add(t);
            }
        }

        if (DISABLE_REBUILD_TOPO_CLOSURE) {
            return optimized;
        }
        return OptimizerGraphSupport.rebuildTopologicalClosure(optimized);
    }

    private Tensor trySimplify(Tensor t) {
        if (DISABLE_ALL_TRANSFORMS) return t;
        if (t.getOperation() == null) return t;
        if (t.getOperation().opType() == Operation.OpType.FUSED) return t;

        String op = t.getOperation().getClass().getSimpleName().toLowerCase(Locale.ROOT);
        return switch (op) {
            case "add" -> simplifyAdd(t);
            case "sub" -> simplifySub(t);
            case "mul" -> simplifyMul(t);
            case "mulscalar" -> simplifyMulScalar(t);
            case "div" -> simplifyDiv(t);
            case "pow" -> simplifyPow(t);
            case "neg" -> simplifyNeg(t);
            case "log" -> simplifyLog(t);
            case "exp" -> simplifyExp(t);
            case "inv" -> simplifyInv(t);
            case "sqrt" -> simplifySqrt(t);
            default -> t;
        };
    }

    private Tensor simplifyAdd(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(a, 0.0)) return b;
        if (isConstant(b, 0.0)) return a;
        if (!DISABLE_ADD_SELF_TO_MUL2 && a == b) return a.mul(2.0);

        // x + (-x) -> 0
        if (!DISABLE_ADD_NEG_TO_ZERO && (isNegOf(a, b) || isNegOf(b, a))) return Tensor.zerosLike(a);

        // x + x*c -> x*(1+c)
        if (!DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = getMulScalarIfBase(a, b);
            if (c1 != null) return b.mul(1.0 + c1);
            Double c2 = getMulScalarIfBase(b, a);
            if (c2 != null) return a.mul(1.0 + c2);
        }

        // (-x) + (-y) -> -(x+y)
        if (!DISABLE_ADD_NEGNEG_TO_NEGADD && isOp(a, "neg") && isOp(b, "neg")) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).neg();
        }

        // log(a) + log(b) -> log(a*b) (doména: a,b > 0)
        if (!DISABLE_ADD_LOGLOG_TO_LOGMUL && isOp(a, "log") && isOp(b, "log")) {
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

        // x - (-y) -> x + y
        if (!DISABLE_SUB_NEG_TO_ADD && isOp(b, "neg")) return a.add(b.getPrevTensors().get(0));

        // x - x*c -> x*(1-c)
        if (!DISABLE_ADD_SUB_FACTORIZE) {
            Double c1 = getMulScalarIfBase(b, a);
            if (c1 != null) return a.mul(1.0 - c1);

            // x*c - x -> x*(c-1)
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

        // x * (1/x) -> 1 (jen přesný inv(x))
        if (!DISABLE_MUL_INV_TO_ONE && isOp(a, "inv") && a.getPrevTensors().get(0) == b) return Tensor.onesLike(b);
        if (!DISABLE_MUL_INV_TO_ONE && isOp(b, "inv") && b.getPrevTensors().get(0) == a) return Tensor.onesLike(a);

        // (-x) * (-y) -> x*y
        if (!DISABLE_MUL_NEGNEG_TO_MUL && isOp(a, "neg") && isOp(b, "neg")) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0));
        }

        // exp(a) * exp(b) -> exp(a+b)
        if (!DISABLE_MUL_EXPEXP_TO_EXPADD && isOp(a, "exp") && isOp(b, "exp")) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).exp();
        }

        return t;
    }

    private Tensor simplifyMulScalar(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof mulScalar m)) return t;

        double s = m.getScalar();
        if (s == 0.0) return Tensor.zerosLike(input);
        if (s == 1.0) return input;
        if (s == -1.0) return input.neg();

        // (x*a)*b -> x*(a*b)
        if (!DISABLE_MULSCALAR_ASSOC && isOp(input, "mulscalar") && input.getOperation() instanceof mulScalar in) {
            return input.getPrevTensors().get(0).mul(in.getScalar() * s);
        }

        // (-x)*a -> x*(-a)
        if (!DISABLE_MULSCALAR_NEG_PUSH && isOp(input, "neg")) {
            return input.getPrevTensors().get(0).mul(-s);
        }

        // constant folding pro scalar konstantu
        if (!DISABLE_MULSCALAR_CONST_FOLD && isConstant(input)) {
            return Tensor.scalar(input.scalarAsDouble() * s, input.getDataType());
        }

        return t;
    }

    private Tensor simplifyDiv(Tensor t) {
        Tensor a = t.getPrevTensors().get(0);
        Tensor b = t.getPrevTensors().get(1);

        if (isConstant(a, 0.0)) return Tensor.zerosLike(t);
        if (isConstant(b, 1.0)) return a;
        if (isConstant(b, -1.0)) return a.neg();

        // x / (1/y) -> x*y
        if (!DISABLE_DIV_INV_TO_MUL && isOp(b, "inv")) return a.mul(b.getPrevTensors().get(0));

        // (x*c)/c -> x
        if (!DISABLE_DIV_MULSCALAR_BY_CONST
                && isOp(a, "mulscalar")
                && a.getOperation() instanceof mulScalar ms
                && ms.getScalar() != 0.0
                && isConstant(b, ms.getScalar())) {
            return a.getPrevTensors().get(0);
        }

        // (x*c)/d -> x*(c/d)
        if (!DISABLE_DIV_MULSCALAR_BY_CONST
                && isOp(a, "mulscalar")
                && a.getOperation() instanceof mulScalar ms
                && isConstant(b)) {
            return a.getPrevTensors().get(0).mul(ms.getScalar() / b.scalarAsDouble());
        }

        if (!DISABLE_DIV_CONST_TO_MULRECIP && isConstant(b)) return a.mul(1.0 / b.scalarAsDouble());
        return t;
    }

    private Tensor simplifyNeg(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isOp(input, "neg")) return input.getPrevTensors().get(0);

        // -(x - y) -> y - x
        if (!DISABLE_NEG_SUB_SWAP && isOp(input, "sub")) {
            Tensor x = input.getPrevTensors().get(0);
            Tensor y = input.getPrevTensors().get(1);
            return y.sub(x);
        }

        // -(x*c) -> x*(-c)
        if (!DISABLE_NEG_MULSCALAR_PUSH && isOp(input, "mulscalar") && input.getOperation() instanceof mulScalar m) {
            return input.getPrevTensors().get(0).mul(-m.getScalar());
        }

        return t;
    }

    private Tensor simplifyPow(Tensor t) {
        Tensor base = t.getPrevTensors().get(0);
        if (!(t.getOperation() instanceof pow p)) return t;

        double exponent = p.getExponent();
        if (exponent == 0.0) return Tensor.onesLike(base);
        if (exponent == 1.0) return base;
        if (exponent == -1.0) return base.inv();
        if (exponent == 0.5) return base.sqrt();
        if (exponent == -0.5) return base.sqrt().inv();

        // (x^a)^b -> x^(a*b)
        if (!DISABLE_POW_POW_FLATTEN && isOp(base, "pow") && base.getOperation() instanceof pow pInner) {
            return base.getPrevTensors().get(0).pow(pInner.getExponent() * exponent);
        }

        // (1/x)^a -> x^(-a)
        if (!DISABLE_POW_INV_TO_NEGEXP && isOp(base, "inv")) {
            return base.getPrevTensors().get(0).pow(-exponent);
        }

        // x^2 -> x*x (numerically can differ from pow kernel, keep switchable)
        if (!DISABLE_POW2_TO_MUL && exponent == 2.0) return base.mul(base);

        return t;
    }

    private Tensor simplifyLog(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isOp(input, "exp")) return input.getPrevTensors().get(0);
        if (!DISABLE_LOG_POW_TO_MULLOG && isOp(input, "pow") && input.getOperation() instanceof pow p) {
            return input.getPrevTensors().get(0).log().mul(p.getExponent());
        }
        if (!DISABLE_LOG_INV_TO_NEGLOG && isOp(input, "inv")) {
            return input.getPrevTensors().get(0).log().neg();
        }
        if (!DISABLE_LOG_SQRT_TO_HALFLOG && isOp(input, "sqrt")) {
            return input.getPrevTensors().get(0).log().mul(0.5);
        }

        return t;
    }

    private Tensor simplifyExp(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (!DISABLE_EXP_LOG_CANCEL && isOp(input, "log")) return input.getPrevTensors().get(0);
        return t;
    }

    private Tensor simplifyInv(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isConstant(input, 1.0)) return input;
        if (isOp(input, "inv")) return input.getPrevTensors().get(0);
        if (!DISABLE_INV_SIGMOID_PATTERN && !t.getRequiresGrad()) {
            Tensor sigmoidInput = matchSigmoidInput(input);
            if (sigmoidInput != null) return sigmoidInput.sigmoid();
        }
        if (!DISABLE_INV_POW_TO_NEGEXP && isOp(input, "pow") && input.getOperation() instanceof pow p) {
            return input.getPrevTensors().get(0).pow(-p.getExponent());
        }
        if (!DISABLE_INV_EXP_TO_EXPNEG && isOp(input, "exp")) {
            return input.getPrevTensors().get(0).neg().exp();
        }
        if (!DISABLE_INV_NEG_PUSH && isOp(input, "neg")) {
            return input.getPrevTensors().get(0).inv().neg();
        }

        return t;
    }

    private Tensor matchSigmoidInput(Tensor candidate) {
        if (!isOp(candidate, "add") || candidate.getPrevTensors().size() != 2) {
            return null;
        }
        Tensor left = candidate.getPrevTensors().get(0);
        Tensor right = candidate.getPrevTensors().get(1);

        Tensor fromLeft = matchExpNeg(right, left);
        if (fromLeft != null) {
            return fromLeft;
        }
        return matchExpNeg(left, right);
    }

    private Tensor matchExpNeg(Tensor constantCandidate, Tensor expCandidate) {
        if (!isConstant(constantCandidate, 1.0)) {
            return null;
        }
        if (!isOp(expCandidate, "exp") || expCandidate.getPrevTensors().size() != 1) {
            return null;
        }
        Tensor negLikeInput = unwrapNegLike(expCandidate.getPrevTensors().get(0));
        if (negLikeInput == null) return null;
        return negLikeInput;
    }

    private Tensor unwrapNegLike(Tensor t) {
        if (isOp(t, "neg") && t.getPrevTensors().size() == 1) {
            return t.getPrevTensors().get(0);
        }
        if (isOp(t, "mulscalar")
                && t.getOperation() instanceof mulScalar ms
                && ms.getScalar() == -1.0
                && t.getPrevTensors().size() == 1) {
            return t.getPrevTensors().get(0);
        }
        return null;
    }

    private Tensor simplifySqrt(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (isConstant(input, 1.0) || isConstant(input, 0.0)) return input;
        return t;
    }

    private boolean isConstant(Tensor t) {
        return t.getOperation() == null
                && t.getFlatDataSize() == 1
                && !t.getRequiresGrad();
    }

    private boolean isConstant(Tensor t, double val) {
        return isConstant(t) && t.scalarAsDouble() == val;
    }

    private boolean isOp(Tensor t, String opName) {
        return t.getOperation() != null
                && t.getOperation().getClass().getSimpleName().equalsIgnoreCase(opName);
    }

    private boolean isNegOf(Tensor a, Tensor b) {
        return isOp(a, "neg") && a.getPrevTensors().get(0) == b;
    }

    private Double getMulScalarIfBase(Tensor candidate, Tensor base) {
        if (!isOp(candidate, "mulscalar")) return null;
        if (!(candidate.getOperation() instanceof mulScalar m)) return null;
        if (candidate.getPrevTensors().isEmpty()) return null;
        return candidate.getPrevTensors().get(0) == base ? m.getScalar() : null;
    }

}
