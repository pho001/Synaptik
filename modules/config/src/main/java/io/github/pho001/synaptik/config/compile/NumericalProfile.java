package io.github.pho001.synaptik.config.compile;

/**
 * Immutable graph-wide numerical-profile identity vocabulary for compilation configuration.
 *
 * <p>Model owns the meaning of each profile and its operation-specific allowed-result contract;
 * see the <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">Model-owned
 * numerical-profile contract</a>. Config owns only this declarative identity. The identity is not
 * a default, capability, backend mode, route, or execution behavior, and it is not currently
 * consumed by a compiler, Planning, Engine, or backend.</p>
 */
public enum NumericalProfile {
    /** Identifies each operation's current Model-owned allowed-result set. */
    STRICT_IEEE,

    /** Identifies only the bounded opt-in superset owned by the Model contract. */
    ACCELERATOR
}
