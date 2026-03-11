package Graph.optimizer.rules;

import Graph.optimizer.OptimizationRule;
import Operations.*;
import Tensor.Tensor;

import java.util.*;

public class AlgebraicRewritingRule implements OptimizationRule {

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> optimized = new ArrayList<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();

        for (Tensor t : sortedGraph) {
            updateInputs(t, replacements);

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

        return rebuildTopologicalClosure(optimized);
    }

    private Tensor trySimplify(Tensor t) {
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
        if (a == b) return a.mul(2.0);

        // x + (-x) -> 0
        if (isNegOf(a, b) || isNegOf(b, a)) return Tensor.zerosLike(a);

        // x + x*c -> x*(1+c)
        Double c1 = getMulScalarIfBase(a, b);
        if (c1 != null) return b.mul(1.0 + c1);
        Double c2 = getMulScalarIfBase(b, a);
        if (c2 != null) return a.mul(1.0 + c2);

        // (-x) + (-y) -> -(x+y)
        if (isOp(a, "neg") && isOp(b, "neg")) {
            return a.getPrevTensors().get(0).add(b.getPrevTensors().get(0)).neg();
        }

        // log(a) + log(b) -> log(a*b) (doména: a,b > 0)
        if (isOp(a, "log") && isOp(b, "log")) {
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
        if (isOp(b, "neg")) return a.add(b.getPrevTensors().get(0));

        // x - x*c -> x*(1-c)
        Double c1 = getMulScalarIfBase(b, a);
        if (c1 != null) return a.mul(1.0 - c1);

        // x*c - x -> x*(c-1)
        Double c2 = getMulScalarIfBase(a, b);
        if (c2 != null) return b.mul(c2 - 1.0);

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
        if (isOp(a, "inv") && a.getPrevTensors().get(0) == b) return Tensor.onesLike(b);
        if (isOp(b, "inv") && b.getPrevTensors().get(0) == a) return Tensor.onesLike(a);

        // (-x) * (-y) -> x*y
        if (isOp(a, "neg") && isOp(b, "neg")) {
            return a.getPrevTensors().get(0).mul(b.getPrevTensors().get(0));
        }

        // exp(a) * exp(b) -> exp(a+b)
        if (isOp(a, "exp") && isOp(b, "exp")) {
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
        if (isOp(input, "mulscalar") && input.getOperation() instanceof mulScalar in) {
            return input.getPrevTensors().get(0).mul(in.getScalar() * s);
        }

        // (-x)*a -> x*(-a)
        if (isOp(input, "neg")) {
            return input.getPrevTensors().get(0).mul(-s);
        }

        // constant folding pro scalar konstantu
        if (isConstant(input)) {
            return Tensor.scalar(input.getData()[0] * s);
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
        if (isOp(b, "inv")) return a.mul(b.getPrevTensors().get(0));

        // (x*c)/c -> x
        if (isOp(a, "mulscalar")
                && a.getOperation() instanceof mulScalar ms
                && ms.getScalar() != 0.0
                && isConstant(b, ms.getScalar())) {
            return a.getPrevTensors().get(0);
        }

        // (x*c)/d -> x*(c/d)
        if (isOp(a, "mulscalar")
                && a.getOperation() instanceof mulScalar ms
                && isConstant(b)) {
            return a.getPrevTensors().get(0).mul(ms.getScalar() / b.getData()[0]);
        }

        if (isConstant(b)) return a.mul(1.0 / b.getData()[0]);
        return t;
    }

    private Tensor simplifyNeg(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isOp(input, "neg")) return input.getPrevTensors().get(0);

        // -(x - y) -> y - x
        if (isOp(input, "sub")) {
            Tensor x = input.getPrevTensors().get(0);
            Tensor y = input.getPrevTensors().get(1);
            return y.sub(x);
        }

        // -(x*c) -> x*(-c)
        if (isOp(input, "mulscalar") && input.getOperation() instanceof mulScalar m) {
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
        if (isOp(base, "pow") && base.getOperation() instanceof pow pInner) {
            return base.getPrevTensors().get(0).pow(pInner.getExponent() * exponent);
        }

        // (1/x)^a -> x^(-a)
        if (isOp(base, "inv")) {
            return base.getPrevTensors().get(0).pow(-exponent);
        }

        // x^2 -> x*x
        if (exponent == 2.0) return base.mul(base);

        return t;
    }

    private Tensor simplifyLog(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isOp(input, "exp")) return input.getPrevTensors().get(0);
        if (isOp(input, "pow") && input.getOperation() instanceof pow p) {
            return input.getPrevTensors().get(0).log().mul(p.getExponent());
        }
        if (isOp(input, "inv")) {
            return input.getPrevTensors().get(0).log().neg();
        }
        if (isOp(input, "sqrt")) {
            return input.getPrevTensors().get(0).log().mul(0.5);
        }

        return t;
    }

    private Tensor simplifyExp(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (isOp(input, "log")) return input.getPrevTensors().get(0);
        return t;
    }

    private Tensor simplifyInv(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);

        if (isConstant(input, 1.0)) return input;
        if (isOp(input, "inv")) return input.getPrevTensors().get(0);
        if (isOp(input, "pow") && input.getOperation() instanceof pow p) {
            return input.getPrevTensors().get(0).pow(-p.getExponent());
        }
        if (isOp(input, "exp")) {
            return input.getPrevTensors().get(0).neg().exp();
        }
        if (isOp(input, "neg")) {
            return input.getPrevTensors().get(0).inv().neg();
        }

        return t;
    }

    private Tensor simplifySqrt(Tensor t) {
        Tensor input = t.getPrevTensors().get(0);
        if (isConstant(input, 1.0) || isConstant(input, 0.0)) return input;
        return t;
    }

    private boolean isConstant(Tensor t) {
        return t.getOperation() == null
                && t.getData() != null
                && t.getData().length == 1
                && !t.getRequiresGrad();
    }

    private boolean isConstant(Tensor t, double val) {
        return isConstant(t) && t.getData()[0] == val;
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

    private void updateInputs(Tensor t, Map<Tensor, Tensor> replacements) {
        if (t.getPrevTensors() == null) return;
        List<Tensor> inputs = t.getPrevTensors();
        for (int i = 0; i < inputs.size(); i++) {
            Tensor currentInput = inputs.get(i);
            Tensor replacement = replacements.get(currentInput);
            if (replacement != null) {
                inputs.set(i, replacement);
            }
        }
    }

    private List<Tensor> rebuildTopologicalClosure(List<Tensor> graph) {
        if (graph.isEmpty()) return graph;

        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (Tensor t : graph) {
            if (t.getPrevTensors() != null) {
                for (Tensor p : t.getPrevTensors()) {
                    consumerCounts.put(p, consumerCounts.getOrDefault(p, 0) + 1);
                }
            }
        }

        List<Tensor> sinks = new ArrayList<>();
        for (Tensor t : graph) {
            if (consumerCounts.getOrDefault(t, 0) == 0) {
                sinks.add(t);
            }
        }

        Set<Tensor> visited = new HashSet<>();
        List<Tensor> rebuilt = new ArrayList<>();
        for (Tensor sink : sinks) {
            dfsPostOrder(sink, visited, rebuilt);
        }
        return rebuilt;
    }

    private void dfsPostOrder(Tensor node, Set<Tensor> visited, List<Tensor> out) {
        if (node == null || visited.contains(node)) return;
        visited.add(node);

        if (node.getPrevTensors() != null) {
            for (Tensor p : node.getPrevTensors()) {
                dfsPostOrder(p, visited, out);
            }
        }

        out.add(node);
    }
}
