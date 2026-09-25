package io.github.pho001.synaptik.config.compile;

/**
 * Immutable graph-wide numerical-profile identity vocabulary for compilation configuration.
 *
 * <p>Model owns the meaning of each profile and its recursive allowed-result contract; see the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">Model-owned
 * numerical-profile contract</a>. Config owns only this declarative identity. Compiler and
 * Planning transport it through cold capability queries, artifacts, preparation, and backend
 * plan identity; Engine captures one selection at construction; Runtime performs no per-run
 * lookup. It is not a default, capability, backend route, or execution behavior.</p>
 */
public enum NumericalProfile {
    /** Identifies every operation's unchanged current Model-owned allowed-result set. */
    STRICT_IEEE,

    /** Identifies the strict set plus Model's total recursive FLOAT32 accelerator floors. */
    ACCELERATOR
}
