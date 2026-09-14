package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import java.util.Objects;

/**
 * Opaque immutable handle for compile artifacts owned by one {@link AdvancedEngine}.
 *
 * <p>The handle is thread-safe and exposes no inward Compiler recipe. Only the exact open Engine
 * that created it may prepare it. Closing that Engine makes the handle unusable; the handle has
 * no independent close operation and owns no closeable resource.</p>
 */
public final class AdvancedCompiledGraph {
    private final AdvancedEngine owner;
    private final CompileArtifacts artifacts;

    /**
     * Retains the exact owner and immutable Compiler result.
     *
     * @param owner non-null exact Engine that created and exclusively consumes this handle
     * @param artifacts non-null immutable Compiler result retained without copying
     * @throws NullPointerException if either argument is null
     */
    AdvancedCompiledGraph(AdvancedEngine owner, CompileArtifacts artifacts) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.artifacts = Objects.requireNonNull(artifacts, "artifacts");
    }

    /**
     * Returns the exact creating Engine for package-private identity validation.
     *
     * @return the non-null exact owner
     */
    AdvancedEngine owner() {
        return owner;
    }

    /**
     * Returns the retained Compiler recipe only to package-private orchestration.
     *
     * @return the non-null immutable artifacts
     */
    CompileArtifacts artifacts() {
        return artifacts;
    }
}
