package io.github.pho001.synaptik.checkpoint.artifact;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Owns the sibling temporary file through write, file force, reopen, validation, and one atomic
 * move. A successful move provides live visibility of the prior complete file (or absence) or
 * the new complete file on the supported trusted local filesystem; forcing the temporary file
 * does not portably make the rename survive power loss.
 * Before the move attempt, failures attempt to clean the owned temporary file. After an attempted
 * move, only an explicit unsupported-atomic-move result permits attempted cleanup; an ambiguous
 * failure performs no path probes or cleanup. No path operation follows a successful move.
 */
final class ArtifactPublication {
    interface FileOperations {
        /**
         * Forces the temporary file's channel writes before validation and publication.
         *
         * @param channel the open writable temporary-file channel; never {@code null}
         * @throws IOException if the file cannot be forced
         */
        void force(FileChannel channel) throws IOException;

        /**
         * Attempts the one atomic publication operation; failure may leave the target's state
         * uncertain, so callers must not infer rollback.
         *
         * @param temporary the owned validated sibling file; never {@code null}
         * @param target the requested publication path; never {@code null}
         * @throws IOException if atomic publication fails or its result cannot be confirmed
         */
        void move(Path temporary, Path target) throws IOException;

        /**
         * Removes an owned temporary file only before any move attempt, or after a specific
         * unsupported-atomic-move result. It must not be called after successful or ambiguous
         * publication.
         *
         * @param temporary the owned temporary path; never {@code null}
         * @throws IOException if that temporary file cannot be removed
         */
        default void cleanupTemporary(Path temporary) throws IOException {
            Files.deleteIfExists(temporary);
        }
    }

    static final FileOperations NIO = new FileOperations() {
        @Override
        public void force(FileChannel channel) throws IOException {
            channel.force(true);
        }

        @Override
        public void move(Path temporary, Path target) throws IOException {
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
        }
    };

    private ArtifactPublication() {
    }

    /**
     * Publishes a fully validated image through one atomic move. Cleanup attempts are limited to failures
     * before the move and a definite unsupported-atomic-move result; an ambiguous result leaves
     * both paths untouched for caller inspection. Successful publication performs no later file
     * operation.
     *
     * @param target the non-null caller-owned destination in an existing trusted directory
     * @param image the non-null immutable source image
     * @param limits the non-null resource ceilings for writing and validation
     * @param operations the non-null file operations, replaceable only by package tests
     * @throws IOException if preflight, force, validation, publication, or temporary cleanup fails
     * @throws NullPointerException if an argument is null
     */
    static void save(Path target, ArtifactEnvelope image, ArtifactLimits limits,
                     FileOperations operations) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(limits, "limits");
        Objects.requireNonNull(operations, "operations");
        Path absolute = target.toAbsolutePath().normalize();
        Path parent = absolute.getParent();
        if (parent == null || !Files.isDirectory(parent, LinkOption.NOFOLLOW_LINKS)) {
            throw new IOException("Artifact parent must be an existing trusted directory");
        }
        if (Files.exists(absolute, LinkOption.NOFOLLOW_LINKS)
                && !Files.isRegularFile(absolute, LinkOption.NOFOLLOW_LINKS)) {
            throw new IOException("Artifact target must be absent or a regular file");
        }
        Path temporary = Files.createTempFile(parent, ".synaptik-artifact-", ".tmp");
        try {
            try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING, LinkOption.NOFOLLOW_LINKS)) {
                ArtifactWire.write(channel, image, limits);
                operations.force(channel);
            }
            ArtifactWire.validate(temporary, limits, image);
        } catch (IOException | RuntimeException | Error ex) {
            cleanupTemporary(temporary, operations, ex);
            throw ex;
        }
        try {
            operations.move(temporary, absolute);
        } catch (AtomicMoveNotSupportedException ex) {
            IOException failure = new IOException("Atomic artifact move unsupported; inspect the "
                    + "target before retrying", ex);
            cleanupTemporary(temporary, operations, failure);
            throw failure;
        } catch (IOException | RuntimeException ex) {
            throw new IOException("Atomic artifact move failed; inspect the target because the "
                    + "filesystem provider may have changed it; the temporary file may remain", ex);
        }
        // No filesystem operation after a successful atomic move.
    }

    /**
     * Attempts to remove only the owned pre-publication temporary path while preserving the
     * original failure if cleanup also fails.
     *
     * @param temporary the owned non-null temporary path
     * @param operations the non-null file operations used for cleanup
     * @param failure the non-null original failure to which cleanup trouble is suppressed
     */
    private static void cleanupTemporary(Path temporary, FileOperations operations, Throwable failure) {
        try {
            operations.cleanupTemporary(temporary);
        } catch (IOException | RuntimeException ex) {
            failure.addSuppressed(ex);
        }
    }
}
