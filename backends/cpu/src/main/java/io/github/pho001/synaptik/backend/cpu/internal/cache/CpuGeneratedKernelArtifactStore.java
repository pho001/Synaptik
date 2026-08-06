package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Optional trusted-root class-byte persistence and process-local compatibility interning.
 * Persistence is cold policy only; absence or corruption deterministically falls back to memory.
 */
public final class CpuGeneratedKernelArtifactStore {
    private static final ConcurrentHashMap<String, WeakReference<CpuGeneratedKernel>> LOADED =
            new ConcurrentHashMap<>();
    private final Optional<Path> trustedRoot;
    private final CpuClassFileKernelGenerator generator = new CpuClassFileKernelGenerator();

    /** Creates the default in-memory-only policy. */
    public CpuGeneratedKernelArtifactStore() { trustedRoot = Optional.empty(); }
    /**
     * Creates an optional trusted-root policy.
     *
     * @param trustedRoot non-null optional trusted local root; empty selects in-memory-only
     *     realization, and a present path is normalized without creating it
     * @throws NullPointerException if {@code trustedRoot} is {@code null}
     */
    public CpuGeneratedKernelArtifactStore(Optional<Path> trustedRoot) {
        this.trustedRoot = Objects.requireNonNull(trustedRoot, "trustedRoot")
                .map(path -> path.toAbsolutePath().normalize());
    }

    /**
     * Reuses or realizes one exact compatible artifact.
     * Filesystem absence, corruption, or publication failure is treated as a cache miss and cannot
     * change correctness.
     *
     * @param specialization non-null structural specialization used for compatibility identity
     * @param kernelIr non-null canonical IR whose structural key must match the specialization
     * @return a verified, defined artifact retained strongly by the caller; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if the IR and specialization disagree or verified class
     *     realization fails
     */
    public CpuGeneratedKernel loadOrGenerate(
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        String key = specialization.structuralKey();
        synchronized (LOADED) {
            WeakReference<CpuGeneratedKernel> reference = LOADED.get(key);
            CpuGeneratedKernel loaded = reference == null ? null : reference.get();
            if (loaded != null) {
                trustedRoot.ifPresent(root -> writeIfAbsent(root, key, loaded.classBytes()));
                return loaded;
            }
            byte[] bytes = trustedRoot.flatMap(root -> read(root, key)).orElse(null);
            CpuGeneratedKernel artifact = null;
            if (bytes != null) {
                try { artifact = generator.defineClassBytes(specialization, bytes); }
                catch (IllegalArgumentException corrupt) { artifact = null; }
            }
            if (artifact == null) {
                bytes = generator.generateClassBytes(specialization, kernelIr);
                artifact = generator.defineClassBytes(specialization, bytes);
                byte[] publish = bytes;
                trustedRoot.ifPresent(root -> write(root, key, publish));
            }
            LOADED.put(key, new WeakReference<>(artifact));
            return artifact;
        }
    }

    private static Optional<byte[]> read(Path root, String key) {
        try { return Optional.of(Files.readAllBytes(root.resolve(key + ".class"))); }
        catch (IOException | SecurityException missing) { return Optional.empty(); }
    }
    private static void write(Path root, String key, byte[] bytes) {
        try {
            Files.createDirectories(root);
            Path temporary = Files.createTempFile(root, key, ".tmp");
            Files.write(temporary, bytes);
            try { Files.move(temporary, root.resolve(key + ".class"),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
            catch (IOException noAtomicMove) { Files.move(temporary, root.resolve(key + ".class"),
                    StandardCopyOption.REPLACE_EXISTING); }
        } catch (IOException | SecurityException ignored) {
            // Persistence is optional and never correctness-critical.
        }
    }

    private static void writeIfAbsent(Path root, String key, byte[] bytes) {
        if (!Files.isRegularFile(root.resolve(key + ".class"))) write(root, key, bytes);
    }

    /** Clears only process-local weak interning state for isolated focused tests. */
    static void clearLoadedForTests() { synchronized (LOADED) { LOADED.clear(); } }
}
