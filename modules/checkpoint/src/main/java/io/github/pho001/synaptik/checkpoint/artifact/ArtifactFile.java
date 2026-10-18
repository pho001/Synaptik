package io.github.pho001.synaptik.checkpoint.artifact;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * File boundary for the opaque version-one envelope. A read validates the entire image before
 * returning it. A save forces and reopens a sibling temporary file before one atomic move, giving
 * live readers the prior complete file (or absence for a new target) or the new complete file when
 * that move succeeds. This does not portably make the rename survive power loss. Paths are
 * caller-owned and require a trusted local directory and quiescent target; no non-atomic fallback,
 * file locking, or exclusive no-replace promise exists.
 */
public final class ArtifactFile {
    private ArtifactFile() {
    }

    /**
     * Reads and fully validates a regular artifact file against caller-selected bounds.
     *
     * @param path non-null path to a quiescent regular file; symbolic links are rejected
     * @param limits non-null resource ceilings applied before size-driven allocation
     * @return detached immutable image; never {@code null} or partially validated
     * @throws ArtifactFormatException if the path is not a regular file or the image is malformed,
     *     corrupt, unsupported, or over limit
     * @throws IOException if the file cannot be accessed
     * @throws NullPointerException if an argument is null
     */
    public static ArtifactEnvelope read(Path path, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(limits, "limits");
        return ArtifactWire.read(path, limits);
    }

    /**
     * Writes, forces, closes, reopens, and validates a sibling temporary file, then attempts one
     * atomic move to the target. A pre-existing regular target may be replaced; a symlink or
     * directory target is refused before writing. A failed atomic move has provider-dependent
     * target state, so callers must inspect the target rather than assume rollback. Force or
     * validation failures attempt to clean the owned temporary file before any move attempt. An
     * explicit unsupported-atomic-move result permits attempted cleanup; other move failures leave the
     * temporary path untouched because the result is uncertain. No target is deliberately
     * deleted, and no filesystem operation follows a successful move. Successful return
     * establishes atomic live visibility, not portable power-loss durability of the rename.
     *
     * @param target non-null caller-owned target in an existing trusted local directory with no
     *     concurrent external mutation
     * @param image non-null immutable source image, which is not modified
     * @param limits non-null resource ceilings for the written and reopened file
     * @throws ArtifactFormatException if the image exceeds configured limits or validation fails
     * @throws IOException if target type, I/O, force, or atomic move fails
     * @throws NullPointerException if an argument is null
     */
    public static void save(Path target, ArtifactEnvelope image, ArtifactLimits limits)
            throws IOException {
        ArtifactPublication.save(target, image, limits, ArtifactPublication.NIO);
    }
}
