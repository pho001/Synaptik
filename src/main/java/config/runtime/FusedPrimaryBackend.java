package config.runtime;

/**
 * Primary backend family for fused elementwise regions.
 */
public enum FusedPrimaryBackend {
    /**
     * Generated JVM bytecode/ASM fused loop backend.
     */
    ASM
}
