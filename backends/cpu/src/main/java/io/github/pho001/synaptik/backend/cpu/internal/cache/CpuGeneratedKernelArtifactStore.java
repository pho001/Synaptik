package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import java.io.*;
import java.lang.ref.WeakReference;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Optional bounded trusted-root persistence of verified generated class bytes.
 *
 * <p>One realization performs at most one process-local weak-intern lookup and, when a root is
 * present, one current-schema envelope lookup. Missing, incompatible, corrupt, oversized, or
 * inaccessible entries are safe misses: verified in-memory generation remains the correctness
 * path, and optional publication failure is non-critical. A persisted hit reuses class bytes only;
 * it defines a fresh hidden class and preserves no Java Virtual Machine just-in-time (JIT)
 * machine code or profiling state. This store is also distinct from the future workload tuning
 * cache, which records route/configuration decisions.
 *
 * <p>The supplied root is trusted local storage. Checksums and class-shape validation detect
 * corruption and incompatibility but do not authenticate hostile bytes.
 */
public final class CpuGeneratedKernelArtifactStore {
    /** Maximum accepted or published complete envelope size in bytes. */
    static final int MAX_ENVELOPE_BYTES = 2 * 1024 * 1024;
    /** Maximum accepted or published compatibility-metadata size in bytes. */
    static final int MAX_METADATA_BYTES = 64 * 1024;
    /** Maximum accepted or published generated-class size in bytes. */
    static final int MAX_CLASS_BYTES = 1024 * 1024;
    private static final int MAGIC = 0x53435055;
    private static final ConcurrentHashMap<String, WeakReference<CpuGeneratedKernel>> LOADED =
            new ConcurrentHashMap<>();
    private final Optional<Path> trustedRoot;
    private final CpuClassFileKernelGenerator generator = new CpuClassFileKernelGenerator();

    /** Development-evidence classification of one realization path. */
    enum RealizationSource {
        /** Process-local compatible weak-intern hit. */ WEAK_INTERN_HIT,
        /** Verified trusted-root envelope hit. */ PERSISTED_HIT,
        /** Verified fresh generation, with optional non-critical publication. */ GENERATED
    }
    /**
     * Development-evidence result pairing the artifact with its observed source.
     *
     * @param artifact non-null verified realized artifact
     * @param source non-null observed realization classification
     */
    record ObservedRealization(CpuGeneratedKernel artifact, RealizationSource source) { }

    /** Creates the default persistence-free, in-memory-only store. */
    public CpuGeneratedKernelArtifactStore() { this(Optional.empty()); }
    /**
     * Creates a store with optional trusted-root persistence.
     *
     * @param trustedRoot non-null optional trusted local root; a present path is made absolute and
     *     normalized without creating or reading it
     * @throws NullPointerException if {@code trustedRoot} is {@code null}
     */
    public CpuGeneratedKernelArtifactStore(Optional<Path> trustedRoot) {
        this.trustedRoot = Objects.requireNonNull(trustedRoot, "trustedRoot")
                .map(path -> path.toAbsolutePath().normalize());
    }

    /**
     * Reuses or realizes one artifact matching the selected specialization and canonical IR.
     *
     * @param specialization non-null structural specialization and compatibility identity
     * @param kernelIr non-null canonical IR whose structural key must match the specialization
     * @return a verified artifact strongly retained by the caller; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if IR and specialization disagree or verified generation
     *     or definition fails
     */
    public CpuGeneratedKernel loadOrGenerate(CpuKernelSpecialization specialization,
            CpuKernelIr kernelIr) {
        return loadOrGenerateObserved(specialization, kernelIr).artifact();
    }

    /**
     * Realizes one artifact while exposing the path classification to focused evidence code.
     *
     * @param specialization non-null structural specialization
     * @param kernelIr non-null matching canonical IR
     * @return the verified artifact and its non-null observed source
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if compatibility or verified realization fails
     */
    ObservedRealization loadOrGenerateObserved(CpuKernelSpecialization specialization,
            CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        String key = specialization.structuralKey();
        synchronized (LOADED) {
            WeakReference<CpuGeneratedKernel> reference = LOADED.get(key);
            CpuGeneratedKernel interned = reference == null ? null : reference.get();
            if (interned != null) {
                return new ObservedRealization(interned, RealizationSource.WEAK_INTERN_HIT);
            }
            byte[] metadata = specialization.compatibilityBytes();
            byte[] bytes = trustedRoot.flatMap(root -> readEnvelope(root, key, metadata)).orElse(null);
            CpuGeneratedKernel artifact = null;
            if (bytes != null) try { artifact = generator.defineClassBytes(specialization, bytes); }
            catch (IllegalArgumentException malformed) { artifact = null; }
            RealizationSource source = RealizationSource.PERSISTED_HIT;
            if (artifact == null) {
                bytes = generator.generateClassBytes(specialization, kernelIr);
                artifact = generator.defineClassBytes(specialization, bytes);
                byte[] generated = bytes;
                trustedRoot.ifPresent(root -> writeEnvelope(root, key, metadata, generated));
                source = RealizationSource.GENERATED;
            }
            LOADED.put(key, new WeakReference<>(artifact));
            return new ObservedRealization(artifact, source);
        }
    }

    private static Optional<byte[]> readEnvelope(Path root, String key, byte[] expectedMetadata) {
        Path file = root.resolve(key + ".artifact");
        try {
            long size = Files.size(file);
            if (size < 0 || size > MAX_ENVELOPE_BYTES) return Optional.empty();
            byte[] envelope = Files.readAllBytes(file);
            if (envelope.length > MAX_ENVELOPE_BYTES) return Optional.empty();
            try (var input = new DataInputStream(new ByteArrayInputStream(envelope))) {
                if (input.readInt() != MAGIC || input.readInt() != CpuGeneratorSchema.CURRENT_VERSION
                        || !input.readUTF().equals(key)) return Optional.empty();
                int metadataLength = input.readInt();
                if (metadataLength < 0 || metadataLength > MAX_METADATA_BYTES) return Optional.empty();
                byte[] metadata = input.readNBytes(metadataLength);
                if (metadata.length != metadataLength || !Arrays.equals(metadata, expectedMetadata))
                    return Optional.empty();
                int classLength = input.readInt();
                if (classLength < 0 || classLength > MAX_CLASS_BYTES) return Optional.empty();
                byte[] classBytes = input.readNBytes(classLength);
                byte[] checksum = input.readNBytes(32);
                if (classBytes.length != classLength || checksum.length != 32 || input.read() != -1
                        || !MessageDigest.isEqual(checksum, checksum(key, metadata, classBytes)))
                    return Optional.empty();
                return Optional.of(classBytes);
            }
        } catch (IOException | RuntimeException invalid) {
            return Optional.empty();
        }
    }

    private static void writeEnvelope(Path root, String key, byte[] metadata, byte[] classBytes) {
        if (metadata.length > MAX_METADATA_BYTES || classBytes.length > MAX_CLASS_BYTES) return;
        Path temporary = null;
        try {
            byte[] envelope = envelope(key, metadata, classBytes);
            if (envelope.length > MAX_ENVELOPE_BYTES) return;
            Files.createDirectories(root);
            temporary = Files.createTempFile(root, key, ".tmp");
            try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                channel.write(java.nio.ByteBuffer.wrap(envelope));
                channel.force(true);
            }
            Files.move(temporary, root.resolve(key + ".artifact"),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            temporary = null;
        } catch (IOException | SecurityException ignored) {
            // Optional persistence never changes correctness.
        } finally {
            if (temporary != null) try { Files.deleteIfExists(temporary); }
            catch (IOException | SecurityException ignored) { }
        }
    }

    private static byte[] envelope(String key, byte[] metadata, byte[] classBytes) throws IOException {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) {
            output.writeInt(MAGIC); output.writeInt(CpuGeneratorSchema.CURRENT_VERSION);
            output.writeUTF(key); output.writeInt(metadata.length); output.write(metadata);
            output.writeInt(classBytes.length); output.write(classBytes);
            output.write(checksum(key, metadata, classBytes));
        }
        return bytes.toByteArray();
    }

    private static byte[] checksum(String key, byte[] metadata, byte[] classBytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(key.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            digest.update(metadata); return digest.digest(classBytes);
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    /** Clears process-local weak interning for isolated tests and evidence forks. */
    static void clearLoadedForTests() { synchronized (LOADED) { LOADED.clear(); } }
}
