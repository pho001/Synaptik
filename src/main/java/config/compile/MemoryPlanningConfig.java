package config.compile;

import config.optimizer.MemoryConfig;

/**
 * Compile-time memory planning policy.
 */
public record MemoryPlanningConfig(
        boolean enabled,
        MemoryConfig memory
) {
    public MemoryPlanningConfig {
        memory = memory == null ? MemoryConfig.defaults() : memory;
    }

    public static MemoryPlanningConfig defaults() {
        return new MemoryPlanningConfig(true, MemoryConfig.defaults());
    }

    public static MemoryPlanningConfig trainingDefaults() {
        return defaults();
    }

    public static MemoryPlanningConfig disabledUnlessRequired() {
        return new MemoryPlanningConfig(false, MemoryConfig.defaults());
    }

    public MemoryPlanningConfig withMemory(MemoryConfig newMemory) {
        return new MemoryPlanningConfig(enabled, newMemory);
    }
}
