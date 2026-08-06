package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/**
 * Structural portable-kernel specialization. Compatible extents, element count, graph values,
 * slots, segments, addresses, run identity, and artifact-root identity are deliberately absent.
 *
 * @param loweringFingerprint non-null canonical lowering fingerprint
 * @param numericalMode non-null selected numerical mode; currently exact/default only
 * @param executionStrategy non-null selected strategy; currently scalar/single-thread only
 */
public record CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
        NumericalMode numericalMode,
        CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy) {
    /** Numerical modes currently admissible. */
    public enum NumericalMode {
        /** Ordinary exact/default operation contract with no relaxed permission. */ EXACT_DEFAULT
    }
    /**
     * Validates the exact scalar/default slice.
     *
     * @throws NullPointerException if a component is {@code null}
     * @throws IllegalArgumentException if the mode or strategy is outside CPU 0005A
     */
    public CpuKernelSpecialization {
        Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        Objects.requireNonNull(numericalMode, "numericalMode");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        if (numericalMode != NumericalMode.EXACT_DEFAULT
                || !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)) {
            throw new IllegalArgumentException("CPU 0005A supports only exact/default scalar strategy");
        }
    }
    /** Returns the generated entry signature.
     * @return the immutable universal method type */
    public MethodType entryType() {
        return MethodType.methodType(void.class, MemorySegment.class, MemorySegment.class,
                MemorySegment.class, MemorySegment.class, long.class, long.class);
    }
    /** Returns compatibility metadata.
     * @return a new deterministic schema byte array */
    public byte[] compatibilityBytes() {
        return (CpuGeneratorSchema.CURRENT_VERSION + "|" + loweringFingerprint.hex() + "|"
                + numericalMode + "|" + executionStrategy).getBytes(StandardCharsets.US_ASCII);
    }
    /** Returns artifact identity.
     * @return the deterministic lowercase hexadecimal key */
    public String structuralKey() {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(compatibilityBytes())); }
        catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }
}
