package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.List;
import java.util.Optional;
import io.github.pho001.synaptik.model.datatype.DataType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;

/**
 * Structural portable-kernel specialization for one generated scalar or vector body.
 * Compatible extents, element count, parallel orchestration, graph values, slots, carriers,
 * addresses, run identity, worker identity, and artifact-root identity are deliberately absent.
 *
 * @param loweringFingerprint non-null canonical lowering fingerprint
 * @param numericalMode non-null selected numerical mode; currently exact/default only
 * @param executionStrategy non-null generated compute strategy; orchestration is single-thread
 *     because parallel plans reuse the corresponding scalar or vector artifact
 * @param boundaryDataTypes non-null immutable ordered data type for every derived boundary
 * @param carrierPattern non-null immutable ordered carrier form for every derived boundary
 * @param vectorSpeciesBitSize exact positive preferred typed species size in bits for
 *     vector compute, or zero for scalar compute
 * @param materializedSourcePosition copied input position before the final output, or
 *     {@code -1} for direct access; a copied position must use a segment carrier in the generated
 *     pattern
 * @param scalarPowerRealizations non-null ordered realization facts for every scalar-power
 *     instruction in canonical instruction order; copied defensively
 * @param scratchParameter whether the generated entry accepts one exact CPU scratch segment
 * @param classIdentitySchema schema projection used only for the generated class identity;
 *     {@code 52} for unchanged families, {@code 54} for MATMUL, {@code 55} for Pool2d,
 *     {@code 56} for Pool3d, {@code 57} for attention, {@code 58} for direct loss bodies,
 *     {@code 59} for scalar BFLOAT16 pointwise bodies without cross-type CAST, and {@code 60}
 *     exactly for pointwise bodies containing cross-type CAST, and {@code 61} only for vector
 *     pointwise bodies with a dense FLOAT32/FLOAT64 BOOL mask boundary
 * @param matmulIr exact typed MATMUL code-shaping facts, or empty for every unchanged family
 */
public record CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
        NumericalMode numericalMode,
        CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
        List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
        int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
        boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr) {
    /** Generated loop/address representation selected from structural proof facts. */
    public enum LoopAddressing {
        /** Checked {@code long} state for segments, mixed carriers, or general access plans. */
        GENERAL_LONG,
        /** One-time narrowed {@code int} indexes for dense/all-zero heap-array boundaries. */
        DENSE_HEAP_ARRAY_INT
    }
    /** Direct carrier form at one ordered materialized boundary. */
    public enum CarrierAccess {
        /** Observable direct {@code double[]} access. */ DOUBLE_ARRAY,
        /** Observable direct {@code float[]} access. */ FLOAT_ARRAY,
        /**
         * Observable direct {@code short[]} access for represented-bit BFLOAT16 boundaries.
         * Scalar pointwise bodies may decode these bits for numerical work; no BFLOAT16 vector
         * lane semantics are implied.
         */ SHORT_ARRAY,
        /** Observable direct {@code int[]} access. */ INT_ARRAY,
        /** Observable direct {@code long[]} access. */ LONG_ARRAY,
        /** Observable direct canonical {@code byte[]} access. */ BYTE_ARRAY,
        /** Exact selected {@link MemorySegment} access. */ MEMORY_SEGMENT
    }
    /** Numerical modes currently admissible. */
    public enum NumericalMode {
        /** Ordinary exact/default operation contract with no relaxed permission. */ EXACT_DEFAULT
    }

    /**
     * Preserves the pre-0008F constructor and byte-identical schema-52 class projection.
     *
     * @param loweringFingerprint canonical lowering fingerprint
     * @param numericalMode exact numerical mode
     * @param executionStrategy generated compute strategy
     * @param boundaryDataTypes ordered boundary types
     * @param carrierPattern ordered carriers
     * @param vectorSpeciesBitSize vector species bits, or zero
     * @param materializedSourcePosition copied source, or minus one
     * @param scalarPowerRealizations scalar-power facts
     * @param scratchParameter whether scratch is present
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter) {
        this(loweringFingerprint, numericalMode, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, scratchParameter, 52, Optional.empty());
    }

    /**
     * Creates a specialization with an explicit class-identity projection and no MATMUL facts.
     *
     * @param loweringFingerprint canonical lowering fingerprint
     * @param numericalMode exact numerical mode
     * @param executionStrategy generated compute strategy
     * @param boundaryDataTypes ordered boundary types
     * @param carrierPattern ordered carriers
     * @param vectorSpeciesBitSize vector species bits, or zero
     * @param materializedSourcePosition copied source, or minus one
     * @param scalarPowerRealizations scalar-power facts
     * @param scratchParameter whether scratch is present
     * @param classIdentitySchema class projection schema; must be one of the accepted current
     *     family projections, including {@code 60} for matching cross-type CAST IR and
     *     {@code 61} for matching dense floating-mask vector IR
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema) {
        this(loweringFingerprint, numericalMode, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, scratchParameter, classIdentitySchema, Optional.empty());
    }
    /**
     * Creates a direct, non-materialized compatibility specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param carrierPattern non-null ordered FLOAT64 compatibility carrier pattern
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species size for vector compute, or
     *     zero for scalar compute
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the components are outside the compatibility
     *     constructor's FLOAT64 boundary contract
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize) {
        this(loweringFingerprint, numericalMode, executionStrategy,
                java.util.Collections.nCopies(carrierPattern.size(), DataType.FLOAT64), carrierPattern,
                vectorSpeciesBitSize, -1, List.of(), false);
    }

    /**
     * Creates a FLOAT64 compatibility specialization with optional one-input materialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param carrierPattern non-null ordered FLOAT64 compatibility carrier pattern
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species size for vector compute, or
     *     zero for scalar compute
     * @param materializedSourcePosition copied input position, or {@code -1} for direct access
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if strategy, species, carrier, or materialization facts
     *     disagree
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition) {
        this(loweringFingerprint, numericalMode, executionStrategy,
                java.util.Collections.nCopies(carrierPattern.size(), DataType.FLOAT64),
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition, List.of(), false);
    }

    /**
     * Creates a specialization without scalar-power realization facts.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param boundaryDataTypes non-null ordered boundary data types
     * @param carrierPattern non-null ordered boundary carrier forms
     * @param vectorSpeciesBitSize positive preferred species size for vector compute, or zero
     * @param materializedSourcePosition copied input position, or {@code -1}
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if specialization facts disagree
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition) {
        this(loweringFingerprint, numericalMode, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition, List.of(), false);
    }

    /**
     * Creates a specialization with scalar-power facts and no generated scratch parameter.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param boundaryDataTypes non-null ordered boundary data types; copied defensively
     * @param carrierPattern non-null ordered boundary carrier forms; copied defensively
     * @param vectorSpeciesBitSize positive preferred species size for vector compute, or zero
     * @param materializedSourcePosition copied input position, or {@code -1} for direct access
     * @param scalarPowerRealizations non-null ordered scalar-power realization facts; copied
     *     defensively
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if specialization facts disagree
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            NumericalMode numericalMode,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations) {
        this(loweringFingerprint, numericalMode, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, false);
    }
    /**
     * Validates the exact/default scalar-or-vector generated specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param numericalMode non-null selected exact/default numerical mode
     * @param executionStrategy non-null single-thread generated compute strategy
     * @param boundaryDataTypes non-null ordered boundary types; copied defensively
     * @param carrierPattern non-null ordered boundary carrier pattern; copied defensively
     * @param vectorSpeciesBitSize positive preferred typed species bit size for vector
     *     compute, or zero for scalar compute
     * @param materializedSourcePosition copied input position, or {@code -1} for direct access
     * @param scalarPowerRealizations non-null ordered scalar-power realization facts; copied
     * @param scratchParameter whether the direct generated entry accepts one scratch segment
     * @throws NullPointerException if a component is {@code null}
     * @throws IllegalArgumentException if the mode, compute/species relationship, boundary/type
     *     mapping, or materialized position is inconsistent
     */
    public CpuKernelSpecialization {
        Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        Objects.requireNonNull(numericalMode, "numericalMode");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        boundaryDataTypes = List.copyOf(boundaryDataTypes);
        carrierPattern = List.copyOf(carrierPattern);
        scalarPowerRealizations = List.copyOf(scalarPowerRealizations);
        matmulIr = Objects.requireNonNull(matmulIr, "matmulIr");
        if (classIdentitySchema != 52 && classIdentitySchema != 54 && classIdentitySchema != 55
                && classIdentitySchema != 56 && classIdentitySchema != 57
                && classIdentitySchema != 58 && classIdentitySchema != 59 && classIdentitySchema != 60
                && classIdentitySchema != 61) {
            throw new IllegalArgumentException("class identity schema must be 52, 54, 55, 56, 57, 58, 59, 60, or 61");
        }
        if ((classIdentitySchema == 54) != matmulIr.isPresent()
                || classIdentitySchema == 55 && matmulIr.isPresent()) {
            throw new IllegalArgumentException(
                    "class identity schema and typed MATMUL facts disagree");
        }
        if (carrierPattern.isEmpty() || carrierPattern.size() != boundaryDataTypes.size()) {
            throw new IllegalArgumentException("boundary type and carrier entries must agree");
        }
        for (int i = 0; i < carrierPattern.size(); i++) {
            if (carrierPattern.get(i) != expectedCarrier(boundaryDataTypes.get(i))
                    && carrierPattern.get(i) != CarrierAccess.MEMORY_SEGMENT) {
                throw new IllegalArgumentException("heap carrier does not match boundary data type");
            }
        }
        boolean vector = executionStrategy.compute()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
        int expectedSpeciesBitSize = preferredSpeciesBitSize(boundaryDataTypes);
        if (numericalMode != NumericalMode.EXACT_DEFAULT
                || vector != (vectorSpeciesBitSize > 0)
                || (vector && vectorSpeciesBitSize != expectedSpeciesBitSize)
                || materializedSourcePosition < -1
                || materializedSourcePosition >= carrierPattern.size() - 1
                || (materializedSourcePosition >= 0
                    && carrierPattern.get(materializedSourcePosition) != CarrierAccess.MEMORY_SEGMENT)
                || scratchParameter && materializedSourcePosition >= 0) {
            throw new IllegalArgumentException("current CPU route requires exact strategy/species facts");
        }
    }
    /** Returns the generated entry signature.
     * @return the immutable universal method type */
    public MethodType entryType() {
        Class<?>[] parameters = new Class<?>[carrierPattern.size() + 3 + (scratchParameter ? 1 : 0)];
        for (int i = 0; i < carrierPattern.size(); i++) parameters[i] = carrierClass(carrierPattern.get(i));
        int next = carrierPattern.size();
        if (scratchParameter) parameters[next++] = MemorySegment.class;
        parameters[next++] = long[].class;
        parameters[next++] = long.class;
        parameters[next] = long.class;
        return MethodType.methodType(void.class, parameters);
    }

    /**
     * Resolves the typed loop-address proof category encoded by this specialization and IR.
     * Compatible extents do not affect the result because heap-array binding independently
     * validates every concrete address span before invocation.
     *
     * @param kernelIr non-null canonical IR matching {@link #loweringFingerprint()}
     * @return the deterministic emitted loop-address form; never {@code null}
     * @throws NullPointerException if {@code kernelIr} is {@code null}
     * @throws IllegalArgumentException if the IR does not match this specialization
     */
    public LoopAddressing loopAddressing(CpuKernelIr kernelIr) {
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!loweringFingerprint.hex().equals(kernelIr.structuralKey()))
            throw new IllegalArgumentException("kernel IR does not match specialization");
        boolean denseArrays = !carrierPattern.contains(CarrierAccess.MEMORY_SEGMENT)
                && kernelIr.values().stream().filter(value -> value.kind()
                    != CpuKernelIr.Value.Kind.VIRTUAL).allMatch(value -> value.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR
                    || value.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
        return denseArrays ? LoopAddressing.DENSE_HEAP_ARRAY_INT : LoopAddressing.GENERAL_LONG;
    }
    /** Returns compatibility metadata.
     * @return a new deterministic schema byte array */
    public byte[] compatibilityBytes() {
        return (CpuGeneratorSchema.CURRENT_VERSION + "|" + loweringFingerprint.hex() + "|"
                + numericalMode + "|" + executionStrategy.compute() + "|" + boundaryDataTypes
                + "|" + carrierPattern
                + "|" + vectorSpeciesBitSize + "|materialized=" + materializedSourcePosition
                + "|power=" + scalarPowerRealizations + "|scratch=" + scratchParameter)
                .getBytes(StandardCharsets.US_ASCII);
    }
    /**
     * Returns the stable generated-class identity projection.
     *
     * <p>Schema 54 adds MATMUL forms. Retaining schema 52 for every unchanged family in this
 * projection preserves byte-exact binary names and class bytes for otherwise unchanged
 * pointwise specializations, while {@link #compatibilityBytes()} invalidates every schema-53
 * persisted envelope.</p>
     *
     * @return a new deterministic schema-52 or MATMUL schema-54 identity byte array
     */
    public byte[] classIdentityBytes() {
        return (classIdentitySchema + "|" + loweringFingerprint.hex() + "|"
                + numericalMode + "|" + executionStrategy.compute() + "|" + boundaryDataTypes
                + "|" + carrierPattern
                + "|" + vectorSpeciesBitSize + "|materialized=" + materializedSourcePosition
                + "|power=" + scalarPowerRealizations + "|scratch=" + scratchParameter)
                .getBytes(StandardCharsets.US_ASCII);
    }
    /** Returns artifact identity.
     * @return the deterministic lowercase hexadecimal key */
    public String structuralKey() {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(classIdentityBytes())); }
        catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    private static CarrierAccess expectedCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
    }

    private static int preferredSpeciesBitSize(List<DataType> boundaryTypes) {
        List<DataType> numeric = boundaryTypes.stream()
                .filter(type -> type != DataType.BOOL).distinct().toList();
        if (numeric.size() > 1) return 0;
        DataType laneType = numeric.isEmpty() ? DataType.BOOL : numeric.getFirst();
        return switch (laneType) {
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> LongVector.SPECIES_PREFERRED.vectorBitSize();
            case BOOL -> ByteVector.SPECIES_PREFERRED.vectorBitSize();
            case BFLOAT16 -> 0;
        };
    }

    private static Class<?> carrierClass(CarrierAccess carrier) {
        return switch (carrier) {
            case DOUBLE_ARRAY -> double[].class;
            case FLOAT_ARRAY -> float[].class;
            case SHORT_ARRAY -> short[].class;
            case INT_ARRAY -> int[].class;
            case LONG_ARRAY -> long[].class;
            case BYTE_ARRAY -> byte[].class;
            case MEMORY_SEGMENT -> MemorySegment.class;
        };
    }
}
