package graph.optimizer.cost;

import java.util.List;

/**
 * Report-only explanation emitted by a specialized cost model.
 *
 * @param modelName specialized model name
 * @param inputKind kind of input scored by the model
 * @param comparison optional comparison result, if this explanation compares against another score
 * @param reasonCode stable reason code from the specialized model
 * @param topContributors most important non-informational components
 * @param rawComponents all raw components used for explanation
 */
public record CostExplanation(
        String modelName,
        String inputKind,
        CostComparison comparison,
        String reasonCode,
        List<CostComponent> topContributors,
        List<CostComponent> rawComponents
) {
    public CostExplanation {
        modelName = modelName == null ? "" : modelName;
        inputKind = inputKind == null ? "" : inputKind;
        comparison = comparison == null ? CostComparison.INCOMPARABLE : comparison;
        reasonCode = reasonCode == null ? "" : reasonCode;
        topContributors = List.copyOf(topContributors == null ? List.of() : topContributors);
        rawComponents = List.copyOf(rawComponents == null ? List.of() : rawComponents);
    }
}
