package io.github.pho001.synaptik.backend.cpu.execution;

import static java.nio.file.StandardCopyOption.ATOMIC_MOVE;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.Objects;

/**
 * CPU-private durable store for deterministic generated class bytes. Each instance is rooted at
 * one explicit absolute normalized trusted-local path and creates only its fixed
 * {@code generated-kernels/v1/sha256} namespace on first use. A SHA-256 digest of a domain
 * separator plus complete canonical compatibility metadata selects a sharded {@code .cpuclass}
 * path. The digest locates a candidate; only byte-for-byte metadata comparison establishes
 * compatibility.
 *
 * <p>Each final file is one big-endian, length-delimited envelope containing the eight-byte
 * magic, format version, bounded metadata and class lengths, class-byte SHA-256 checksum, exact
 * metadata, and deterministic class bytes, with trailing data rejected. Missing, incompatible,
 * corrupt, or invalid final entries are safe misses. Other filesystem failures fail the cold
 * request. Publication forces a unique temporary regular file, atomically replaces the final
 * entry without a non-atomic fallback, then re-reads and fully validates the visible result
 * before definition. Generator-owned verification checks the Java class file and its exact class
 * shape. Checksums detect accidental corruption; they do not authenticate executable bytes. The
 * caller must prevent untrusted writers from modifying the root or replacing its ancestors.</p>
 *
 * <p>Equal requests across store instances share one process-local in-flight attempt and one live
 * weakly interned loaded artifact. No completed artifact is strongly retained by the store, no
 * age or access history affects correctness, and no background thread, eviction, or cleanup
 * policy exists. Compatible age and access history never invalidate an entry. Callers retain
 * returned artifacts strongly for as long as they need the hidden class and exact handle. The
 * store performs no route selection, tuning, Runtime binding, or Runtime execution.</p>
 */
final class CpuGeneratedKernelArtifactStore {
    private static final int CHECKSUM_BYTES = 32;
    private static final int FIXED_ENVELOPE_BYTES = 8 + 4 + 4 + 4 + CHECKSUM_BYTES;
    private static final int MAX_METADATA_BYTES = 1 << 20;
    private static final int MAX_CLASS_BYTES = 16 << 20;
    private static final int MAX_ENVELOPE_BYTES =
            FIXED_ENVELOPE_BYTES + MAX_METADATA_BYTES + MAX_CLASS_BYTES;
    private static final int MAX_METADATA_FIELD_BYTES = 1 << 16;
    private static final String NAMESPACE = "generated-kernels/v1/sha256";
    private static final Object COORDINATION = new Object();
    private static final Map<ProcessKey, Attempt> IN_FLIGHT = new HashMap<>();
    private static final Map<ProcessKey, KernelReference> INTERNED = new HashMap<>();
    private static final ReferenceQueue<CpuGeneratedKernel> STALE_KERNELS = new ReferenceQueue<>();

    private final Path root;
    private final CpuClassFileKernelGenerator generator = new CpuClassFileKernelGenerator();

    /**
     * Creates a store identity without touching the filesystem.
     *
     * @param root explicit trusted local root; converted to an absolute normalized path and not
     *     accessed until the first load or generation request
     * @throws NullPointerException if {@code root} is {@code null}
     */
    CpuGeneratedKernelArtifactStore(Path root) {
        this.root = Objects.requireNonNull(root, "root").toAbsolutePath().normalize();
    }

    /**
     * Loads or emits one exact compatible generated kernel during cold CPU finalization.
     * Filesystem and atomic-publication failures fail the request with a contextual unchecked
     * exception. Missing or invalid entries are regenerated and replaced atomically. Waiters do
     * not cancel a shared attempt; interruption is restored before the shared result or exact
     * unchecked failure is observed.
     *
     * @param specialization complete non-null specialization selected before finalization
     * @param familyEmitter non-null emitter with the exact matching lowering fingerprint
     * @return one exact live weakly interned or newly loaded artifact that the caller must retain
     *     strongly for its required lifetime; never {@code null}
     * @throws NullPointerException if an input is {@code null}
     * @throws IllegalArgumentException if the emitter fingerprint does not match
     * @throws IllegalStateException if durable store access, forced atomic publication, final
     *     validation, class definition, or exact entry resolution cannot complete
     */
    CpuGeneratedKernel loadOrGenerate(CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(familyEmitter, "familyEmitter");
        if (!specialization.loweringFingerprint().equals(familyEmitter.loweringFingerprint())) {
            throw new IllegalArgumentException(
                    "familyEmitter lowering fingerprint does not match specialization");
        }

        byte[] metadata = metadata(specialization);
        ProcessKey key = new ProcessKey(root, metadata);
        Attempt attempt;
        boolean owner;
        synchronized (COORDINATION) {
            drainStaleReferences();
            KernelReference reference = INTERNED.get(key);
            CpuGeneratedKernel live = reference == null ? null : reference.get();
            if (live != null) return live;
            if (reference != null) INTERNED.remove(key, reference);
            attempt = IN_FLIGHT.get(key);
            if (attempt == null) {
                attempt = new Attempt();
                IN_FLIGHT.put(key, attempt);
                owner = true;
            } else {
                owner = false;
            }
        }
        if (!owner) return attempt.await();

        try {
            CpuGeneratedKernel loaded = loadOrPublish(specialization, familyEmitter, metadata);
            synchronized (COORDINATION) {
                drainStaleReferences();
                INTERNED.put(key, new KernelReference(key, loaded));
                attempt.succeed(loaded);
                IN_FLIGHT.remove(key, attempt);
            }
            return loaded;
        } catch (RuntimeException | Error failure) {
            synchronized (COORDINATION) {
                attempt.fail(failure);
                IN_FLIGHT.remove(key, attempt);
            }
            throw failure;
        }
    }

    private CpuGeneratedKernel loadOrPublish(CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter, byte[] metadata) {
        Path entry = entryPath(metadata);
        CpuGeneratedKernel stored = readValid(entry, metadata, specialization);
        if (stored != null) return stored;

        byte[] generated = generator.generateClassBytes(specialization, familyEmitter);
        byte[] envelope = new ArtifactEnvelope(metadata, generated).bytes();
        stored = readValid(entry, metadata, specialization);
        if (stored != null) return stored;
        publish(entry, envelope);
        CpuGeneratedKernel published = readValid(entry, metadata, specialization);
        if (published == null) {
            throw new ArtifactStoreException("published generated-kernel artifact is invalid: "
                    + entry, null);
        }
        return published;
    }

    private CpuGeneratedKernel readValid(Path entry, byte[] expectedMetadata,
            CpuKernelSpecialization specialization) {
        if (!Files.exists(entry, LinkOption.NOFOLLOW_LINKS)) return null;
        if (!Files.isRegularFile(entry, LinkOption.NOFOLLOW_LINKS)) return null;
        try (FileChannel channel = FileChannel.open(entry, READ, LinkOption.NOFOLLOW_LINKS)) {
            long size = channel.size();
            if (size < FIXED_ENVELOPE_BYTES || size > MAX_ENVELOPE_BYTES) return null;
            ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(size));
            while (bytes.hasRemaining()) {
                if (channel.read(bytes) < 0) return null;
            }
            if (channel.read(ByteBuffer.allocate(1)) != -1 || channel.size() != size) return null;
            bytes.flip();
            byte[] expectedMagic = CpuGeneratorSchema.ARTIFACT_MAGIC
                    .getBytes(StandardCharsets.US_ASCII);
            byte[] magic = new byte[expectedMagic.length];
            bytes.get(magic);
            if (!Arrays.equals(magic, expectedMagic)) return null;
            if (bytes.getInt() != CpuGeneratorSchema.ARTIFACT_FORMAT_VERSION) return null;
            long metadataLength = Integer.toUnsignedLong(bytes.getInt());
            long classLength = Integer.toUnsignedLong(bytes.getInt());
            if (metadataLength > MAX_METADATA_BYTES || classLength > MAX_CLASS_BYTES) return null;
            long expectedSize = Math.addExact(FIXED_ENVELOPE_BYTES,
                    Math.addExact(metadataLength, classLength));
            if (expectedSize != size) return null;
            byte[] checksum = new byte[CHECKSUM_BYTES];
            bytes.get(checksum);
            byte[] metadata = new byte[(int) metadataLength];
            bytes.get(metadata);
            if (!Arrays.equals(metadata, expectedMetadata)) return null;
            byte[] classBytes = new byte[(int) classLength];
            bytes.get(classBytes);
            if (bytes.hasRemaining() || !MessageDigest.isEqual(checksum, sha256(classBytes))) {
                return null;
            }
            try {
                return generator.defineClassBytes(specialization,
                        new ArtifactEnvelope(metadata, classBytes).classBytes());
            } catch (IllegalArgumentException invalidClass) {
                return null;
            }
        } catch (java.nio.file.NoSuchFileException missingRace) {
            return null;
        } catch (ArithmeticException impossibleLength) {
            return null;
        } catch (IOException failure) {
            throw new ArtifactStoreException("failed to read generated-kernel artifact: " + entry,
                    failure);
        }
    }

    private void publish(Path entry, byte[] envelope) {
        Path directory = entry.getParent();
        Path temporary = null;
        boolean published = false;
        try {
            Files.createDirectories(directory);
            if (!Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS)) {
                throw new IOException("artifact directory is not a directory");
            }
            temporary = Files.createTempFile(directory, ".cpuclass-", ".tmp");
            if (!Files.isRegularFile(temporary, LinkOption.NOFOLLOW_LINKS)) {
                throw new IOException("artifact temporary path is not a regular file");
            }
            try (FileChannel channel = FileChannel.open(temporary, WRITE,
                    LinkOption.NOFOLLOW_LINKS)) {
                ByteBuffer bytes = ByteBuffer.wrap(envelope);
                while (bytes.hasRemaining()) channel.write(bytes);
                channel.force(true);
            }
            Files.move(temporary, entry, ATOMIC_MOVE, REPLACE_EXISTING);
            published = true;
        } catch (IOException | UnsupportedOperationException failure) {
            throw new ArtifactStoreException(
                    "failed to atomically publish generated-kernel artifact: " + entry, failure);
        } finally {
            if (!published && temporary != null) {
                try {
                    Files.deleteIfExists(temporary);
                } catch (IOException ignored) {
                    // Preserve the publication failure; only this attempt's unpublished file is touched.
                }
            }
        }
    }

    private Path entryPath(byte[] metadata) {
        byte[] domain = CpuGeneratorSchema.ARTIFACT_KEY_DOMAIN.getBytes(StandardCharsets.US_ASCII);
        byte[] domainAndMetadata = new byte[domain.length + 1 + metadata.length];
        System.arraycopy(domain, 0, domainAndMetadata, 0, domain.length);
        System.arraycopy(metadata, 0, domainAndMetadata,
                domain.length + 1, metadata.length);
        String digest = HexFormat.of().formatHex(sha256(domainAndMetadata));
        return root.resolve(NAMESPACE).resolve(digest.substring(0, 2))
                .resolve(digest.substring(2) + ".cpuclass");
    }

    private static byte[] metadata(CpuKernelSpecialization specialization) {
        try {
            var buffer = new ByteArrayOutputStream();
            var output = new DataOutputStream(buffer);
            output.writeInt(CpuGeneratorSchema.ARTIFACT_FORMAT_VERSION);
            output.writeInt(CpuGeneratorSchema.ARTIFACT_KEY_DOMAIN_VERSION);
            output.writeInt(CpuGeneratorSchema.CURRENT_VERSION);
            writeBytes(output, CpuGeneratorSchema.generatedBinaryName(specialization)
                    .getBytes(StandardCharsets.UTF_8));
            writeBytes(output, CpuGeneratorSchema.GENERATED_ENTRY_NAME
                    .getBytes(StandardCharsets.US_ASCII));
            writeBytes(output, specialization.entryType().descriptorString()
                    .getBytes(StandardCharsets.US_ASCII));
            output.writeInt(specialization.classFileMajorVersion());
            output.writeInt(CpuGeneratorSchema.CLASSFILE_MINOR_VERSION);
            output.writeInt(CpuGeneratorSchema.JAVA_FEATURE_VERSION);
            writeBytes(output, specialization.artifactCompatibilityBytes());
            output.flush();
            byte[] metadata = buffer.toByteArray();
            if (metadata.length > MAX_METADATA_BYTES) {
                throw new IllegalArgumentException("artifact metadata exceeds maximum length");
            }
            return metadata;
        } catch (IOException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static void writeBytes(DataOutputStream output, byte[] value) throws IOException {
        if (value.length > MAX_METADATA_FIELD_BYTES) {
            throw new IllegalArgumentException("artifact metadata field exceeds maximum length");
        }
        output.writeInt(value.length);
        output.write(value);
    }

    private static byte[] sha256(byte[] bytes) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("SHA-256 is required by the Java platform", impossible);
        }
    }

    private static void drainStaleReferences() {
        KernelReference stale;
        while ((stale = (KernelReference) STALE_KERNELS.poll()) != null) {
            INTERNED.remove(stale.key, stale);
        }
    }

    private record ProcessKey(Path root, byte[] metadata) {
        private ProcessKey {
            Objects.requireNonNull(root, "root");
            metadata = Objects.requireNonNull(metadata, "metadata").clone();
        }

        @Override public byte[] metadata() { return metadata.clone(); }

        @Override public boolean equals(Object other) {
            return this == other || other instanceof ProcessKey that
                    && root.equals(that.root) && Arrays.equals(metadata, that.metadata);
        }

        @Override public int hashCode() {
            return 31 * root.hashCode() + Arrays.hashCode(metadata);
        }
    }

    private record ArtifactEnvelope(byte[] metadata, byte[] classBytes) {
        private ArtifactEnvelope {
            metadata = Objects.requireNonNull(metadata, "metadata").clone();
            classBytes = Objects.requireNonNull(classBytes, "classBytes").clone();
            if (metadata.length > MAX_METADATA_BYTES || classBytes.length > MAX_CLASS_BYTES) {
                throw new IllegalArgumentException(
                        "generated-kernel artifact exceeds maximum length");
            }
        }

        @Override public byte[] metadata() { return metadata.clone(); }
        @Override public byte[] classBytes() { return classBytes.clone(); }

        private byte[] bytes() {
            try {
                var buffer = new ByteArrayOutputStream(FIXED_ENVELOPE_BYTES
                        + metadata.length + classBytes.length);
                var output = new DataOutputStream(buffer);
                output.write(CpuGeneratorSchema.ARTIFACT_MAGIC
                        .getBytes(StandardCharsets.US_ASCII));
                output.writeInt(CpuGeneratorSchema.ARTIFACT_FORMAT_VERSION);
                output.writeInt(metadata.length);
                output.writeInt(classBytes.length);
                output.write(sha256(classBytes));
                output.write(metadata);
                output.write(classBytes);
                output.flush();
                return buffer.toByteArray();
            } catch (IOException impossible) {
                throw new AssertionError(impossible);
            }
        }
    }

    private static final class KernelReference extends WeakReference<CpuGeneratedKernel> {
        private final ProcessKey key;

        private KernelReference(ProcessKey key, CpuGeneratedKernel artifact) {
            super(artifact, STALE_KERNELS);
            this.key = key;
        }
    }

    private static final class Attempt {
        private CpuGeneratedKernel result;
        private Throwable failure;
        private boolean complete;

        private synchronized CpuGeneratedKernel await() {
            boolean interrupted = false;
            try {
                while (!complete) {
                    try {
                        wait();
                    } catch (InterruptedException ignored) {
                        interrupted = true;
                    }
                }
                if (failure instanceof RuntimeException runtime) throw runtime;
                if (failure instanceof Error error) throw error;
                return result;
            } finally {
                if (interrupted) Thread.currentThread().interrupt();
            }
        }

        private synchronized void succeed(CpuGeneratedKernel artifact) {
            result = artifact;
            complete = true;
            notifyAll();
        }

        private synchronized void fail(Throwable throwable) {
            failure = throwable;
            complete = true;
            notifyAll();
        }
    }

    private static final class ArtifactStoreException extends IllegalStateException {
        private ArtifactStoreException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
