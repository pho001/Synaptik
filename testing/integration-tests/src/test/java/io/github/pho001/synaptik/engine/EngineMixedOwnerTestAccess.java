package io.github.pho001.synaptik.engine;

import java.util.Objects;

/** Test-only access to the focused post-prepare registry poison seam. */
public final class EngineMixedOwnerTestAccess {
    private EngineMixedOwnerTestAccess() {}

    /** Makes any later registry owner lookup fail without closing captured backend adapters. */
    public static void poisonBackendLookup(Engine engine) {
        Objects.requireNonNull(engine, "engine").poisonBackendLookupForTesting();
    }
}
