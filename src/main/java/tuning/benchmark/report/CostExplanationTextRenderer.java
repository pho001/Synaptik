package tuning.benchmark.report;

import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostExplanation;

import java.util.Locale;
import java.util.stream.Collectors;

final class CostExplanationTextRenderer {
    private CostExplanationTextRenderer() {
    }

    static String renderCompact(CostExplanation explanation) {
        if (explanation == null) {
            return "";
        }
        return "cost: model=" + explanation.modelName()
                + " input=" + explanation.inputKind()
                + " reason=" + explanation.reasonCode()
                + " comparison=" + explanation.comparison().name()
                + " top=" + topContributors(explanation);
    }

    private static String topContributors(CostExplanation explanation) {
        if (explanation.topContributors().isEmpty()) {
            return "[]";
        }
        return explanation.topContributors().stream()
                .map(CostExplanationTextRenderer::component)
                .collect(Collectors.joining(", ", "[", "]"));
    }

    private static String component(CostComponent component) {
        return component.name()
                + "=" + format(component.value())
                + " " + component.direction().name();
    }

    private static String format(double value) {
        if (!Double.isFinite(value)) {
            return "n/a";
        }
        return String.format(Locale.US, "%.6f", value);
    }
}
