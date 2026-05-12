package graph.optimizer.cost;

/**
 * One explainable component of a specialized cost model.
 *
 * @param name stable component name used in traces and reports
 * @param value numeric value for unified reporting
 * @param direction whether lower, higher, or neither is preferred
 * @param reason human-readable explanation for the component
 */
public record CostComponent(
        String name,
        double value,
        CostDirection direction,
        String reason
) {
    public CostComponent {
        name = name == null ? "" : name;
        direction = direction == null ? CostDirection.INFORMATIONAL : direction;
        reason = reason == null ? "" : reason;
    }

    public static CostComponent lowerIsBetter(String name, double value, String reason) {
        return new CostComponent(name, value, CostDirection.LOWER_IS_BETTER, reason);
    }

    public static CostComponent higherIsBetter(String name, double value, String reason) {
        return new CostComponent(name, value, CostDirection.HIGHER_IS_BETTER, reason);
    }

    public static CostComponent informational(String name, double value, String reason) {
        return new CostComponent(name, value, CostDirection.INFORMATIONAL, reason);
    }
}
