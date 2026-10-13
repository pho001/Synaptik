package io.github.pho001.synaptik.checkpoint.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.FileSystemException;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ArtifactPublicationTest {
    private static final ArtifactLimits LIMITS = new ArtifactLimits(4096, 2048, 8, 64, 1024, 2048);
    private static final ArtifactEnvelope OLD = new ArtifactEnvelope("old", List.of(
            new ArtifactEnvelope.Entry("state", new byte[] {1})));
    private static final ArtifactEnvelope NEW = new ArtifactEnvelope("new", List.of(
            new ArtifactEnvelope.Entry("state", new byte[] {2, 3})));

    @TempDir Path directory;

    @Test
    void replacesExistingRegularTargetWithFullyValidatedImage() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        ArtifactFile.save(target, NEW, LIMITS);
        assertEquals("new", ArtifactFile.read(target, LIMITS).kind());
        assertArrayEquals(new byte[] {2, 3}, bytes(ArtifactFile.read(target, LIMITS)));
        assertEquals(1, fileCount());
    }

    @Test
    void successfulMoveDoesNoTemporaryCleanupAfterPublication() throws IOException {
        Path target = directory.resolve("state.art");
        AtomicBoolean cleanupCalled = new AtomicBoolean();
        ArtifactPublication.save(target, NEW, LIMITS, new ArtifactPublication.FileOperations() {
            @Override public void force(FileChannel channel) throws IOException {
                channel.force(true);
            }

            @Override public void move(Path temporary, Path destination) throws IOException {
                Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
            }

            @Override public void cleanupTemporary(Path temporary) {
                cleanupCalled.set(true);
            }
        });
        assertFalse(cleanupCalled.get());
        assertEquals("new", ArtifactFile.read(target, LIMITS).kind());
        assertEquals(1, fileCount());
    }

    @Test
    void refusesDirectoryAndMissingParentBeforeWork() throws IOException {
        Path real = directory.resolve("real.art");
        ArtifactFile.save(real, OLD, LIMITS);
        assertThrows(IOException.class, () -> ArtifactFile.save(directory, NEW, LIMITS));
        assertThrows(IOException.class, () -> ArtifactFile.save(directory.resolve("missing/file.art"), NEW,
                LIMITS));
        assertEquals("old", ArtifactFile.read(real, LIMITS).kind());
        assertEquals(1, fileCount());
    }

    @Test
    void refusesSymlinkWhenProviderSupportsCreatingOne() throws IOException {
        Path real = directory.resolve("real.art");
        ArtifactFile.save(real, OLD, LIMITS);
        Path link = directory.resolve("link.art");
        try {
            Files.createSymbolicLink(link, real.getFileName());
        } catch (UnsupportedOperationException | FileSystemException | SecurityException ex) {
            Assumptions.abort("Symbolic links unavailable in this test environment: " + ex);
        }
        assertThrows(IOException.class, () -> ArtifactFile.save(link, NEW, LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ArtifactFile.read(link, LIMITS));
        assertEquals("old", ArtifactFile.read(real, LIMITS).kind());
        assertEquals(2, fileCount());
    }

    @Test
    void forceFailureLeavesOldTargetAndCleansOwnedTemporaryFile() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        AtomicBoolean moved = new AtomicBoolean();
        IOException failure = assertThrows(IOException.class, () -> ArtifactPublication.save(target,
                NEW, LIMITS, new ArtifactPublication.FileOperations() {
                    @Override public void force(FileChannel channel) throws IOException {
                        throw new IOException("force unavailable");
                    }

                    @Override public void move(Path temporary, Path destination) {
                        moved.set(true);
                    }
                }));
        assertEquals("force unavailable", failure.getMessage());
        assertFalse(moved.get());
        assertEquals("old", ArtifactFile.read(target, LIMITS).kind());
        assertEquals(1, fileCount());
    }

    @Test
    void corruptedTemporaryImageFailsValidationBeforeMove() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        AtomicBoolean moved = new AtomicBoolean();
        assertThrows(ArtifactFormatException.class, () -> ArtifactPublication.save(target,
                NEW, LIMITS, new ArtifactPublication.FileOperations() {
                    @Override public void force(FileChannel channel) throws IOException {
                        channel.write(ByteBuffer.wrap(new byte[] {'X'}), 0);
                        channel.force(true);
                    }

                    @Override public void move(Path temporary, Path destination) {
                        moved.set(true);
                    }
                }));
        assertFalse(moved.get());
        assertEquals("old", ArtifactFile.read(target, LIMITS).kind());
        assertEquals(1, fileCount());
    }

    @Test
    void unsupportedAtomicMoveHasNoFallbackAndLeavesOldTarget() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        AtomicBoolean forced = new AtomicBoolean();
        IOException failure = assertThrows(IOException.class, () -> ArtifactPublication.save(target,
                NEW, LIMITS, new ArtifactPublication.FileOperations() {
                    @Override public void force(FileChannel channel) throws IOException {
                        channel.force(true);
                        forced.set(true);
                    }

                    @Override public void move(Path temporary, Path destination) throws IOException {
                        throw new java.nio.file.AtomicMoveNotSupportedException(
                                temporary.toString(), destination.toString(), "test provider");
                    }
                }));
        assertTrue(forced.get());
        assertTrue(failure.getMessage().contains("inspect the target"));
        assertTrue(failure.getCause() instanceof java.nio.file.AtomicMoveNotSupportedException);
        assertEquals("old", ArtifactFile.read(target, LIMITS).kind());
        assertEquals(1, fileCount());
    }

    @Test
    void providerFailureAfterMoveKeepsNewTargetAndReportsUncertainState() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        AtomicBoolean cleanupCalled = new AtomicBoolean();
        IOException failure = assertThrows(IOException.class, () -> ArtifactPublication.save(target,
                NEW, LIMITS, new ArtifactPublication.FileOperations() {
                    @Override public void force(FileChannel channel) throws IOException {
                        channel.force(true);
                    }

                    @Override public void move(Path temporary, Path destination) throws IOException {
                        Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
                        throw new IOException("provider lost confirmation");
                    }

                    @Override public void cleanupTemporary(Path temporary) {
                        cleanupCalled.set(true);
                    }
                }));
        assertTrue(failure.getMessage().contains("inspect the target"));
        assertFalse(cleanupCalled.get());
        assertEquals("new", ArtifactFile.read(target, LIMITS).kind());
        assertArrayEquals(new byte[] {2, 3}, bytes(ArtifactFile.read(target, LIMITS)));
        assertEquals(1, fileCount());
    }

    @Test
    void ambiguousMoveFailureLeavesTemporaryPathUnprobedAndUntouched() throws IOException {
        Path target = directory.resolve("state.art");
        ArtifactFile.save(target, OLD, LIMITS);
        AtomicReference<Path> temporaryPath = new AtomicReference<>();
        AtomicBoolean cleanupCalled = new AtomicBoolean();
        IOException failure = assertThrows(IOException.class, () -> ArtifactPublication.save(target,
                NEW, LIMITS, new ArtifactPublication.FileOperations() {
                    @Override public void force(FileChannel channel) throws IOException {
                        channel.force(true);
                    }

                    @Override public void move(Path temporary, Path destination) throws IOException {
                        temporaryPath.set(temporary);
                        throw new IOException("provider outcome unknown");
                    }

                    @Override public void cleanupTemporary(Path temporary) {
                        cleanupCalled.set(true);
                    }
                }));
        assertTrue(failure.getMessage().contains("temporary file may remain"));
        assertFalse(cleanupCalled.get());
        assertEquals("old", ArtifactFile.read(target, LIMITS).kind());
        assertTrue(Files.isRegularFile(temporaryPath.get()));
        assertEquals(2, fileCount());
    }

    private byte[] bytes(ArtifactEnvelope image) {
        ByteBuffer payload = image.entries().get(0).payload();
        byte[] result = new byte[payload.remaining()];
        payload.get(result);
        return result;
    }

    private long fileCount() throws IOException {
        try (var files = Files.list(directory)) {
            return files.count();
        }
    }
}
