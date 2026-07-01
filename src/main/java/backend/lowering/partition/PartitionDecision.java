package backend.lowering.partition;

public record PartitionDecision(
        boolean selected,
        String route,
        String reason
) {
    public PartitionDecision {
        route = route == null ? "" : route;
        reason = reason == null ? "" : reason;
    }

    public static PartitionDecision selected(String route, String reason) {
        return new PartitionDecision(true, route, reason);
    }

    public static PartitionDecision rejected(String route, String reason) {
        return new PartitionDecision(false, route, reason);
    }
}
