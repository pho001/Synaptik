package graph.execution.plan;

/**
 * Backend-neutral output residency effect for a prepared execution step.
 */
public record OutputResidencyEffect(Mode mode, String reason) {
    public enum Mode {
        NONE,
        CPU_CURRENT_PRESERVE_NATIVE,
        CPU_CURRENT_IF_UNSET
    }

    public OutputResidencyEffect {
        mode = mode == null ? Mode.NONE : mode;
        reason = reason == null || reason.isBlank() ? "backend wrote CPU array" : reason;
    }

    public static OutputResidencyEffect none() {
        return new OutputResidencyEffect(Mode.NONE, "");
    }

    public static OutputResidencyEffect cpuCurrentPreserveNative() {
        return new OutputResidencyEffect(Mode.CPU_CURRENT_PRESERVE_NATIVE, "backend wrote CPU array");
    }

    public static OutputResidencyEffect cpuCurrentIfUnset(String reason) {
        return new OutputResidencyEffect(Mode.CPU_CURRENT_IF_UNSET, reason);
    }
}
