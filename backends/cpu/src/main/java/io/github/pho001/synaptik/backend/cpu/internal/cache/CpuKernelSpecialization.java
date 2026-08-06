package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.List;

/**
 * Structural portable-kernel specialization. Compatible extents, element count, graph values,
 * slots, segments, addresses, run identity, and artifact-root identity are deliberately absent.
 *
 * @param loweringFingerprint non-null canonical lowering fingerprint
 * @param numericalMode non-null selected numerical mode; currently exact/default only
 * @param executionStrategy non-null selected strategy; currently scalar/single-thread only
 * @param carrierPattern non-null immutable ordered carrier form for the current four boundaries
 */
public record CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
        NumericalMode numericalMode,
        CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
        List<CarrierAccess> carrierPattern) {
    /** Direct carrier form at one ordered materialized boundary. */
    public enum CarrierAccess {
        /** Observable direct {@code double[]} access. */ DOUBLE_ARRAY,
        /** Exact selected {@link MemorySegment} access. */ MEMORY_SEGMENT
    }
    /** Numerical modes currently admissible. */
    public enum NumericalMode {
        /** Ordinary exact/default operation contract with no relaxed permission. */ EXACT_DEFAULT
    }
    /**
     * Validates the exact scalar/default slice.
     *
     * @throws NullPointerException if a component is {@code null}
     * @throws IllegalArgumentException if the mode, strategy, or boundary count is outside the
     *     current CPU proving slice
     */
    public CpuKernelSpecialization {
        Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        Objects.requireNonNull(numericalMode, "numericalMode");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        carrierPattern = List.copyOf(carrierPattern);
        if (carrierPattern.size() != 4) throw new IllegalArgumentException(
                "current fused specialization requires four carrier entries");
        if (numericalMode != NumericalMode.EXACT_DEFAULT
                || !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)) {
            throw new IllegalArgumentException("current CPU route supports only exact/default scalar strategy");
        }
    }
    /** Returns the generated entry signature.
     * @return the immutable universal method type */
    public MethodType entryType() {
        Class<?>[] parameters = new Class<?>[7];
        for (int i = 0; i < 4; i++) parameters[i] = carrierPattern.get(i)
                == CarrierAccess.DOUBLE_ARRAY ? double[].class : MemorySegment.class;
        parameters[4] = long[].class;
        parameters[5] = long.class;
        parameters[6] = long.class;
        return MethodType.methodType(void.class, parameters);
    }
    /** Returns compatibility metadata.
     * @return a new deterministic schema byte array */
    public byte[] compatibilityBytes() {
        return (CpuGeneratorSchema.CURRENT_VERSION + "|" + loweringFingerprint.hex() + "|"
                + numericalMode + "|" + executionStrategy + "|" + carrierPattern)
                .getBytes(StandardCharsets.US_ASCII);
    }
    /** Returns artifact identity.
     * @return the deterministic lowercase hexadecimal key */
    public String structuralKey() {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(compatibilityBytes())); }
        catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }
}
