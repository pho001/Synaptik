package tuning.session;

public record BaselinePolicy(
        boolean includeNoOptBaseline,
        boolean includeNoOptConservativeRuntimeBaseline
) {
    public static BaselinePolicy defaults() {
        return new BaselinePolicy(true, true);
    }

    public static BaselinePolicy disabled() {
        return new BaselinePolicy(false, false);
    }
}
