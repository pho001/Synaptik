package tuning.benchmark.report;

import graph.optimizer.cost.CostComparison;
import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostDirection;
import graph.optimizer.cost.CostExplanation;
import trace.compile.CostExplanationTrace;

final class TraceCostExplanationAdapter {
    private TraceCostExplanationAdapter() {
    }

    static CostExplanation toCostExplanation(CostExplanationTrace source) {
        return new CostExplanation(
                source.modelName(), source.inputKind(), CostComparison.valueOf(source.comparison()), source.reasonCode(),
                source.topContributors().stream().map(TraceCostExplanationAdapter::toComponent).toList(),
                source.rawComponents().stream().map(TraceCostExplanationAdapter::toComponent).toList()
        );
    }

    private static CostComponent toComponent(CostExplanationTrace.Component source) {
        return new CostComponent(
                source.name(), source.value(), CostDirection.valueOf(source.direction()), source.reason()
        );
    }
}
