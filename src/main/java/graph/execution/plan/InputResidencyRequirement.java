package graph.execution.plan;

/**
 * Backend-neutral input residency requirement for a prepared execution step.
 */
public record InputResidencyRequirement(Mode mode) {
    public enum Mode {
        NONE,
        CPU_READABLE_ALL,
        CPU_READABLE_FIRST
    }

    public InputResidencyRequirement {
        mode = mode == null ? Mode.NONE : mode;
    }

    public static InputResidencyRequirement none() {
        return new InputResidencyRequirement(Mode.NONE);
    }

    public static InputResidencyRequirement cpuReadableAll() {
        return new InputResidencyRequirement(Mode.CPU_READABLE_ALL);
    }

    public static InputResidencyRequirement cpuReadableFirst() {
        return new InputResidencyRequirement(Mode.CPU_READABLE_FIRST);
    }
}
