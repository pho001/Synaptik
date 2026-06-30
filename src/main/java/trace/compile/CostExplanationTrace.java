package trace.compile;

import java.util.List;

/** Lossless diagnostic snapshot of an optimizer-owned cost explanation. */
public record CostExplanationTrace(
        String modelName,
        String inputKind,
        String comparison,
        String reasonCode,
        List<Component> topContributors,
        List<Component> rawComponents
) {
    public CostExplanationTrace {
        modelName = modelName == null ? "" : modelName;
        inputKind = inputKind == null ? "" : inputKind;
        comparison = comparison == null ? "INCOMPARABLE" : comparison;
        reasonCode = reasonCode == null ? "" : reasonCode;
        topContributors = List.copyOf(topContributors == null ? List.of() : topContributors);
        rawComponents = List.copyOf(rawComponents == null ? List.of() : rawComponents);
    }

    public record Component(String name, double value, String direction, String reason) {
        public Component {
            name = name == null ? "" : name;
            direction = direction == null ? "INFORMATIONAL" : direction;
            reason = reason == null ? "" : reason;
        }
    }
}
