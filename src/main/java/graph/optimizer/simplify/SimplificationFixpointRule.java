package graph.optimizer.simplify;

import graph.optimizer.OptimizationRule;
import graph.optimizer.cost.CostComparison;
import graph.optimizer.state.OptimizerState;
import trace.compile.OptimizerTrace;
import trace.compile.CostExplanationTrace;
import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostExplanation;

import java.util.List;
import java.util.Objects;

/**
 * Repeats graph simplification rules until the graph is stable or no structural improvement is made.
 */
public final class SimplificationFixpointRule implements OptimizationRule {
    private static final int DEFAULT_MAX_ITERATIONS = 8;

    private final List<OptimizationRule> rules;
    private final int maxIterations;

    public SimplificationFixpointRule(List<OptimizationRule> rules) {
        this(rules, Integer.getInteger("cg.optimizer.simplify.maxIterations", DEFAULT_MAX_ITERATIONS));
    }

    public SimplificationFixpointRule(List<OptimizationRule> rules, int maxIterations) {
        this.rules = List.copyOf(Objects.requireNonNull(rules, "rules cannot be null"));
        if (this.rules.isEmpty()) {
            throw new IllegalArgumentException("rules cannot be empty");
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be >= 1");
        }
        this.maxIterations = maxIterations;
    }

    @Override
    public OptimizerState apply(OptimizerState state) {
        OptimizerState current = Objects.requireNonNull(state, "state cannot be null");
        GraphOptimizationFingerprint currentFingerprint = GraphOptimizationFingerprint.capture(
                current.graph(),
                current.forwardOutput()
        );
        GraphOptimizationScore currentScore = GraphOptimizationScore.capture(current.graph());
        current = appendCostTrace(current, currentScore, "simplification-initial", CostComparison.INCOMPARABLE, 0);

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            OptimizerState next = applyOnce(current);
            GraphOptimizationFingerprint nextFingerprint = GraphOptimizationFingerprint.capture(
                    next.graph(),
                    next.forwardOutput()
            );
            GraphOptimizationScore nextScore = GraphOptimizationScore.capture(next.graph());
            CostComparison comparison = nextScore.toCostScore().compare(currentScore.toCostScore());
            if (nextFingerprint.equals(currentFingerprint)) {
                return appendCostTrace(next, nextScore, "simplification-stable", comparison, iteration + 1);
            }

            if (nextScore.compareTo(currentScore) >= 0) {
                return appendCostTrace(current, nextScore, "simplification-rejected", comparison, iteration + 1);
            }

            next = appendCostTrace(next, nextScore, "simplification-improved", comparison, iteration + 1);
            current = next;
            currentFingerprint = nextFingerprint;
            currentScore = nextScore;
        }

        return appendCostTrace(current, currentScore, "simplification-max-iterations", CostComparison.UNCHANGED, maxIterations);
    }

    private OptimizerState applyOnce(OptimizerState state) {
        OptimizerState current = state;
        for (OptimizationRule rule : rules) {
            current = Objects.requireNonNull(rule.apply(current), rule.getClass().getSimpleName() + " returned null");
        }
        return current;
    }

    private static OptimizerState appendCostTrace(
            OptimizerState state,
            GraphOptimizationScore score,
            String reasonCode,
            CostComparison comparison,
            int iteration
    ) {
        OptimizerTrace trace = state.trace()
                .withEvent("simplification-cost iteration=" + iteration
                        + " reason=" + reasonCode
                        + " comparison=" + comparison.name()
                        + " weightedOperationCost=" + score.weightedOperationCost()
                        + " nodeCount=" + score.nodeCount()
                        + " edgeCount=" + score.edgeCount())
                .withCostExplanation(traceExplanation(score.toCostScore().explain(reasonCode, comparison)));
        return state.withTrace(trace);
    }

    private static CostExplanationTrace traceExplanation(CostExplanation source) {
        return new CostExplanationTrace(
                source.modelName(), source.inputKind(), source.comparison().name(), source.reasonCode(),
                source.topContributors().stream().map(SimplificationFixpointRule::traceComponent).toList(),
                source.rawComponents().stream().map(SimplificationFixpointRule::traceComponent).toList()
        );
    }

    private static CostExplanationTrace.Component traceComponent(CostComponent source) {
        return new CostExplanationTrace.Component(
                source.name(), source.value(), source.direction().name(), source.reason()
        );
    }
}
