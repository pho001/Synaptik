package backend.lowering.region;

public record RegionDecision(
        boolean selected,
        String route,
        String reason
) {
    public RegionDecision {
        route = route == null ? "" : route;
        reason = reason == null ? "" : reason;
    }

    public static RegionDecision selected(String route, String reason) {
        return new RegionDecision(true, route, reason);
    }

    public static RegionDecision rejected(String route, String reason) {
        return new RegionDecision(false, route, reason);
    }
}
