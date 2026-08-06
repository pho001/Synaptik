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
 * Structural portable-kernel specialization for one generated scalar or vector body.
 * Compatible extents, element count, parallel orchestration, graph values, slots, carriers,
 * addresses, run identity, worker identity, and artifact-root identity are deliberately absent.
 *
 * @param loweringFingerprint non-null canonical lowering fingerprint
 * @param numericalMode non-null selected numerical mode; currently exact/default only
 * @param executionStrategy non-null generated compute strategy; orchestration is single-thread
 *     because parallel plans reuse the corresponding scalar or vector artifact
 * @param carrierPattern non-null immutable ordered carrier form for the current four boundaries
 * @param vectorSpeciesBitSize exact positive preferred FLOAT64 species size in bits for vector
 *     compute, or zero for scalar compute
 * @param materializedSourcePosition copied input position {@code 0} through {@code 2}, or
 *     {@code -1} for direct access; a copied position must use a segment carrier in the generated
 *     pattern
 */
public record CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
        NumericalMode numericalMode,
        CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
        List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
        int materializedSourcePosition) {
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
     * Creates a direct, non-materialized compatibility specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param carrierPattern non-null ordered four-boundary generated carrier pattern
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species size for vector compute, or
     *     zero for scalar compute
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the components are outside the current proving slice
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize) {
        this(loweringFingerprint, numericalMode, executionStrategy, carrierPattern,
                vectorSpeciesBitSize, -1);
    }
    /**
     * Validates the exact/default scalar-or-vector generated specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null single-thread generated compute strategy
     * @param carrierPattern non-null ordered four-boundary carrier pattern; copied defensively
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species bit size for vector compute,
     *     or zero for scalar compute
     * @param materializedSourcePosition copied input position, or {@code -1} for direct access
     * @throws NullPointerException if a component is {@code null}
     * @throws IllegalArgumentException if the mode, compute/species relationship, or boundary
     *     count is outside the current CPU proving slice
     */
    public CpuKernelSpecialization {
        Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        Objects.requireNonNull(numericalMode, "numericalMode");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        carrierPattern = List.copyOf(carrierPattern);
        if (carrierPattern.size() != 4) throw new IllegalArgumentException(
                "current fused specialization requires four carrier entries");
        boolean vector = executionStrategy.compute()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
        if (numericalMode != NumericalMode.EXACT_DEFAULT
                || vector != (vectorSpeciesBitSize > 0)
                || materializedSourcePosition < -1 || materializedSourcePosition > 2
                || (materializedSourcePosition >= 0
                    && carrierPattern.get(materializedSourcePosition) != CarrierAccess.MEMORY_SEGMENT)) {
            throw new IllegalArgumentException("current CPU route requires exact strategy/species facts");
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
                + numericalMode + "|" + executionStrategy.compute() + "|" + carrierPattern
                + "|" + vectorSpeciesBitSize + "|materialized=" + materializedSourcePosition)
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
