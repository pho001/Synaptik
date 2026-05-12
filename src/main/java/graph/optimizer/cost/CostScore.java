package graph.optimizer.cost;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Shared report-only score vocabulary for specialized cost models.
 *
 * <p>This type does not impose one global score formula. It only provides common component
 * reporting and a deterministic lexicographic comparison for scores from the same model.</p>
 *
 * @param modelName specialized model name
 * @param inputKind kind of input scored by the model
 * @param components ordered cost components; comparable components should appear in priority order
 */
public record CostScore(
        String modelName,
        String inputKind,
        List<CostComponent> components
) {
    private static final int DEFAULT_TOP_CONTRIBUTOR_COUNT = 5;

    public CostScore {
        modelName = modelName == null ? "" : modelName;
        inputKind = inputKind == null ? "" : inputKind;
        components = List.copyOf(components == null ? List.of() : components);
    }

    public static CostScore of(String modelName, String inputKind, List<CostComponent> components) {
        return new CostScore(modelName, inputKind, components);
    }

    /**
     * Compares this score to an older or alternative score from the same specialized model.
     *
     * <p>The comparison is intentionally conservative: different model names, input kinds,
     * component counts, names, or directions are treated as incomparable. Informational
     * components do not decide ordering.</p>
     *
     * @param other score to compare against
     * @return comparison of this score relative to {@code other}
     */
    public CostComparison compare(CostScore other) {
        if (other == null
                || !Objects.equals(modelName, other.modelName)
                || !Objects.equals(inputKind, other.inputKind)
                || components.size() != other.components.size()) {
            return CostComparison.INCOMPARABLE;
        }
        for (int i = 0; i < components.size(); i++) {
            CostComponent current = components.get(i);
            CostComponent previous = other.components.get(i);
            if (!Objects.equals(current.name(), previous.name()) || current.direction() != previous.direction()) {
                return CostComparison.INCOMPARABLE;
            }
            int comparison = Double.compare(current.value(), previous.value());
            if (comparison == 0 || current.direction() == CostDirection.INFORMATIONAL) {
                continue;
            }
            if (current.direction() == CostDirection.LOWER_IS_BETTER) {
                return comparison < 0 ? CostComparison.IMPROVED : CostComparison.WORSE;
            }
            return comparison > 0 ? CostComparison.IMPROVED : CostComparison.WORSE;
        }
        return CostComparison.UNCHANGED;
    }

    public CostExplanation explain(String reasonCode) {
        return explain(reasonCode, CostComparison.INCOMPARABLE);
    }

    public CostExplanation explain(String reasonCode, CostComparison comparison) {
        return new CostExplanation(
                modelName,
                inputKind,
                comparison,
                reasonCode,
                topContributors(DEFAULT_TOP_CONTRIBUTOR_COUNT),
                components
        );
    }

    public List<CostComponent> topContributors(int limit) {
        int resolvedLimit = Math.max(0, limit);
        if (resolvedLimit == 0) {
            return List.of();
        }
        return components.stream()
                .filter(component -> component.direction() != CostDirection.INFORMATIONAL)
                .sorted(Comparator.comparingDouble((CostComponent component) -> Math.abs(component.value())).reversed())
                .limit(resolvedLimit)
                .toList();
    }
}
