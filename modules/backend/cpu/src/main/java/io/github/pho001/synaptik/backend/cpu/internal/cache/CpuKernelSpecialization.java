package io.github.pho001.synaptik.backend.cpu.internal.cache;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
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
 * For unchanged families compatible extents, element count, parallel orchestration, graph values, slots, carriers,
 * addresses, run identity, worker identity, and artifact-root identity are deliberately absent.
 * The selective ordinary aggregate-fusion payload additionally binds its baked static normalized
 * access geometry and typed sites in both identity streams; it never retains invocation carriers,
 * actual bases, assigned slots or mutable state. Unrelated families append no fusion payload.
 *
 * @param loweringFingerprint non-null canonical lowering fingerprint
 * @param executionStrategy non-null generated compute strategy; ordinary parallel plans reuse
 *     scalar/vector artifacts, while selected row exec-v1 binds orchestration and helper layout
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
 * @param classIdentitySchema generated family projection; projections 73 and 74 additionally
 *     bind their selective affine-empty-range and literal-attention markers in both streams;
 *     {@code 52} for unchanged families, {@code 54} for MATMUL, {@code 55} for Pool2d,
 *     {@code 56} for Pool3d, {@code 74} for attention, {@code 58} for direct loss bodies,
 *     {@code 59} for scalar BFLOAT16 pointwise bodies without cross-type CAST, and {@code 60}
 *     exactly for pointwise bodies containing cross-type CAST, and {@code 61} only for vector
 *     pointwise bodies with a dense FLOAT32/FLOAT64 BOOL mask boundary, and {@code 62} only for
 *     same-typed contiguous FLOAT32/FLOAT64 MSE {@code NONE} vector bodies, and {@code 63} only
 *     for the proved direct dense FLOAT32/FLOAT64 Conv2d/Conv3d output-width vector realization;
 *     width stride and dilation are one, while non-width stride and dilation remain eligible;
 *     {@code 71} is the typed segment-only general-scan family; {@code 72} is exclusively an
 *     ordinary F32/F64 row-map/multiple-terminal full-body region, scalar v1 or selected exec-v1;
 *     {@code 73} is exclusively a dense heap-array affine copy with an empty-range entry guard
 * @param matmulIr exact typed MATMUL code-shaping facts, or empty for every unchanged family
 * @param generalScanIr exact homogeneous FLOAT32/FLOAT64 body facts with optional separate
 *     INT64 length control, present only for family
 *     projection 71; represented dtype distinguishes both identities, and staged payloads bind
 *     every baked affine/resource magnitude; staged v4 advances both streams while the
 *     established unstaged pointwise keys and bytes remain unchanged
 * @param ordinaryBatch whether this is a bounded scalar B-only numeric recipe path;
 *     both identity streams mark its changed baked meaning, never actual invocation B
 * @param typedOrdinaryBatch whether the ordinary recipe uses the additive typed/control/ROW(1)
 *     namespace; this code-shaping fact is independent of actual B and retains old F32 bytes
 * @param rowMapReductionIr immutable ordinary full-body row region, present exactly for
 *     projection 72 and exclusive with MATMUL/Scan facts; Scan embeds its region in generalScanIr
 * @param rowFusionEmission selected exec-v1 resource/helper layout, absent for literal scalar v1
 * @param aggregateFusionIr optional ordinary exact SUM/MEAN composition; exclusive with Scan/row/MATMUL
 * @param softmaxBackwardIr optional fixed static two-pass derivative composition; exclusive
 *     with every other physical family, scalar with exact scratch and projection 52
 * @param compositionUnits ordered ordinary helpers for a composite Scan, empty on old routes
 */
public record CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
        CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
        List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
        int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
        boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
        Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
        boolean ordinaryBatch, boolean typedOrdinaryBatch,
        Optional<CpuRowMapReductionIr> rowMapReductionIr, Optional<RowFusionEmission> rowFusionEmission,
        Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr> aggregateFusionIr,
        Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr> softmaxBackwardIr,
        List<CompositionUnit> compositionUnits) {
    /** Affected-only affine identity projection; the generator envelope remains unchanged. */
    public static final int DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA = 73;
    /** Literal attention working sites and nonempty contractions; obsolete shifted classes reject. */
    public static final int ATTENTION_LITERAL_EXP_SCHEMA = 74;

    /**
     * Retains the standalone specialization constructor and its unchanged identity stream.
     *
     * @param loweringFingerprint canonical lowering fingerprint
     * @param executionStrategy generated compute strategy
     * @param boundaryDataTypes ordered represented boundary types
     * @param carrierPattern ordered carrier representations
     * @param vectorSpeciesBitSize vector width, or zero for scalar
     * @param materializedSourcePosition materialized input, or minus one
     * @param scalarPowerRealizations ordered power realization facts
     * @param scratchParameter whether the old entry accepts scratch
     * @param classIdentitySchema selected identity projection
     * @param matmulIr optional ordinary matrix facts
     * @param generalScanIr optional Scan facts
     * @param ordinaryBatch whether ordinary symbolic recipes apply
     * @param typedOrdinaryBatch whether the typed ordinary namespace applies
     * @param rowMapReductionIr optional row region
     * @param rowFusionEmission optional selected row execution
     * @param aggregateFusionIr optional aggregate composition
     * @param softmaxBackwardIr optional softmax backward composition
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch, boolean typedOrdinaryBatch,
            Optional<CpuRowMapReductionIr> rowMapReductionIr, Optional<RowFusionEmission> rowFusionEmission,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr> aggregateFusionIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr> softmaxBackwardIr) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes, carrierPattern, vectorSpeciesBitSize,
                materializedSourcePosition, scalarPowerRealizations, scratchParameter, classIdentitySchema,
                matmulIr, generalScanIr, ordinaryBatch, typedOrdinaryBatch, rowMapReductionIr,
                rowFusionEmission, aggregateFusionIr, softmaxBackwardIr, List.of());
    }

    /**
     * A fixed helper's existing ordinary specialization and numeric pack initialization facts.
     *
     * @param specialization non-null standalone scalar ordinary specialization
     * @param resets ordered fixed cursor seeds emitted before every call, including retry
     * @param scratchOffset fixed or symbolic aligned helper scratch origin in declared workspace
     * @param iterationCount affine iteration capacity, in logical iterations; the exact bound
     *     range end, including piecewise empty domains, is stored in the invocation pack trailer
     */
    public record CompositionUnit(CpuKernelSpecialization specialization, List<PackReset> resets,
            BatchByteSize scratchOffset, BatchByteSize iterationCount) {
        /**
         * Snapshots one generation-only helper declaration.
         * @param specialization scalar standalone ordinary specialization
         * @param resets non-null cursor seed facts
         * @param scratchOffset non-null workspace origin recipe
         * @param iterationCount non-null affine logical iteration-capacity recipe
         * @throws IllegalArgumentException if a nested Scan or non-scalar helper is supplied
         */
        public CompositionUnit {
            Objects.requireNonNull(specialization, "specialization");
            resets = List.copyOf(resets);
            Objects.requireNonNull(scratchOffset, "scratchOffset");
            Objects.requireNonNull(iterationCount, "iterationCount");
            if (specialization.generalScanIr().isPresent() || !specialization.compositionUnits().isEmpty()
                    || specialization.executionStrategy() != CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR) {
                throw new IllegalArgumentException("Scan helpers must be scalar ordinary methods");
            }
        }
    }

    /**
     * One existing geometry owner's mutable pack cursor seed.
     *
     * @param cell non-negative pack cell index
     * @param value exact seed restored before entering the ordinary helper
     */
    public record PackReset(int cell, long value) {
        /**
         * Checks a cursor-cell index without allocating invocation state.
         * @param cell non-negative pack index
         * @param value exact numeric seed
         * @throws IllegalArgumentException if the index is negative
         */
        public PackReset {
            if (cell < 0) throw new IllegalArgumentException("negative pack reset cell");
        }
    }
    /**
     * Preserves pre-backward callers and their literal identity streams.
     * @param loweringFingerprint canonical structural identity
     * @param executionStrategy selected computation strategy
     * @param boundaryDataTypes ordered represented types
     * @param carrierPattern ordered physical carrier forms
     * @param vectorSpeciesBitSize selected vector width, or zero
     * @param materializedSourcePosition copied input position, or minus one
     * @param scalarPowerRealizations ordered power sites
     * @param scratchParameter whether scratch belongs to the entry ABI
     * @param classIdentitySchema retained family projection
     * @param matmulIr optional matrix facts
     * @param generalScanIr optional Scan facts
     * @param ordinaryBatch whether B-only recipes apply
     * @param typedOrdinaryBatch whether the typed namespace applies
     * @param rowMapReductionIr optional row region
     * @param rowFusionEmission optional selected row execution
     * @param aggregateFusionIr optional ordinary aggregate composition
     * @throws NullPointerException if a required reference or collection element is null
     * @throws IllegalArgumentException if the supplied physical, carrier, strategy or identity
     *     facts disagree; this overload never selects the softmax-backward family
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch, boolean typedOrdinaryBatch, Optional<CpuRowMapReductionIr> rowMapReductionIr,
            Optional<RowFusionEmission> rowFusionEmission,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr> aggregateFusionIr) {
        this(loweringFingerprint,executionStrategy,boundaryDataTypes,carrierPattern,vectorSpeciesBitSize,
                materializedSourcePosition,scalarPowerRealizations,scratchParameter,classIdentitySchema,
                matmulIr,generalScanIr,ordinaryBatch,typedOrdinaryBatch,rowMapReductionIr,rowFusionEmission,
                aggregateFusionIr,Optional.empty());
    }
    /**
     * Retains every pre-fusion caller and its literal identity streams.
     * @param loweringFingerprint canonical fingerprint
     * @param executionStrategy selected compute strategy
     * @param boundaryDataTypes ordered represented types
     * @param carrierPattern ordered carrier forms
     * @param vectorSpeciesBitSize selected vector width or zero
     * @param materializedSourcePosition copied input or minus one
     * @param scalarPowerRealizations ordered power realizations
     * @param scratchParameter whether scratch is present
     * @param classIdentitySchema established family projection
     * @param matmulIr optional matrix facts
     * @param generalScanIr optional Scan facts
     * @param ordinaryBatch whether B-only recipes apply
     * @param typedOrdinaryBatch whether typed/control namespace applies
     * @param rowMapReductionIr optional row region
     * @param rowFusionEmission optional row execution layout
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch, boolean typedOrdinaryBatch, Optional<CpuRowMapReductionIr> rowMapReductionIr,
            Optional<RowFusionEmission> rowFusionEmission) {
        this(loweringFingerprint,executionStrategy,boundaryDataTypes,carrierPattern,vectorSpeciesBitSize,
                materializedSourcePosition,scalarPowerRealizations,scratchParameter,classIdentitySchema,
                matmulIr,generalScanIr,ordinaryBatch,typedOrdinaryBatch,rowMapReductionIr,rowFusionEmission,Optional.empty());
    }
    /**
     * Fixed emitted row-region layout, never worker policy or invocation geometry. Ordinal lists
     * identify proved transpose operands and their distinct lifetimes; they are immutable,
     * duplicate-free and disjoint. Containing specializations validate their source bindings.
     * @param maximumPartialCount positive bounded number of private partial slots
     * @param invariantTransposeValues nonnull source ordinals stable throughout the invocation
     * @param temporalTransposeValues nonnull source ordinals staged at their active time
     * @param stepOperandCount selected Scan operand-array length, {@code 2*V+2+P} for V body
     *     values and P demanded ports; zero for ordinary regions
     * @param stepFrameLongCount selected Scan numeric-frame length, {@code 4+4*V+P} LONG cells;
     *     contains time/row counts and byte bases/strides, never worker policy; zero for ordinary regions
     */
    public record RowFusionEmission(int maximumPartialCount, List<Integer> invariantTransposeValues,
            List<Integer> temporalTransposeValues, int stepOperandCount, int stepFrameLongCount) {
        /**
         * Snapshots ordinal lists and rejects overlapping lifetimes and invalid layout counts.
         * @param maximumPartialCount positive bounded number of partial slots
         * @param invariantTransposeValues immutable-after-copy source ordinals stable across time
         * @param temporalTransposeValues immutable-after-copy source ordinals valid at one step
         * @param stepOperandCount nonnegative selected operand-array length
         * @param stepFrameLongCount nonnegative selected numeric-frame length
         * @throws NullPointerException if either list or an element is null
         * @throws IllegalArgumentException if counts or ordinals are invalid or duplicated
         */
        public RowFusionEmission {
            invariantTransposeValues = List.copyOf(invariantTransposeValues);
            temporalTransposeValues = List.copyOf(temporalTransposeValues);
            var all = new java.util.HashSet<Integer>();
            if (maximumPartialCount <= 0 || stepOperandCount < 0 || stepFrameLongCount < 0)
                throw new IllegalArgumentException("invalid row execution layout");
            for (var list : List.of(invariantTransposeValues,temporalTransposeValues))
                for (int ordinal : list) if (ordinal < 0 || !all.add(ordinal))
                    throw new IllegalArgumentException("invalid row transpose lifetime");
        }
    }

    /**
     * Retains the CPU0073 canonical signature and its exact scalar-v1 identity and ABI.
     * @param loweringFingerprint canonical lowering fingerprint
     * @param executionStrategy selected compute strategy
     * @param boundaryDataTypes ordered represented types
     * @param carrierPattern ordered carrier forms
     * @param vectorSpeciesBitSize matching preferred width, or zero
     * @param materializedSourcePosition copied source, or minus one
     * @param scalarPowerRealizations ordered power realizations
     * @param scratchParameter whether scratch is an entry argument
     * @param classIdentitySchema existing family projection
     * @param matmulIr optional ordinary matrix facts
     * @param generalScanIr optional Scan facts
     * @param ordinaryBatch whether ordinary numeric B recipes apply
     * @param typedOrdinaryBatch whether the typed/control namespace applies
     * @param rowMapReductionIr optional ordinary row region
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch, boolean typedOrdinaryBatch, Optional<CpuRowMapReductionIr> rowMapReductionIr) {
        this(loweringFingerprint,executionStrategy,boundaryDataTypes,carrierPattern,vectorSpeciesBitSize,
                materializedSourcePosition,scalarPowerRealizations,scratchParameter,classIdentitySchema,
                matmulIr,generalScanIr,ordinaryBatch,typedOrdinaryBatch,rowMapReductionIr,Optional.empty());
    }
    /**
     * Preserves the previous canonical constructor and all unselected-family identity bytes.
     * @param loweringFingerprint nonnull canonical fingerprint
     * @param executionStrategy nonnull selected scalar/vector strategy
     * @param boundaryDataTypes nonnull ordered represented types
     * @param carrierPattern nonnull ordered typed carriers
     * @param vectorSpeciesBitSize selected vector width or zero
     * @param materializedSourcePosition copied input or minus one
     * @param scalarPowerRealizations nonnull ordered power facts
     * @param scratchParameter whether one scratch argument is present
     * @param classIdentitySchema established family projection, excluding 72
     * @param matmulIr nonnull optional ordinary matrix facts
     * @param generalScanIr nonnull optional Scan facts
     * @param ordinaryBatch whether ordinary symbolic recipes are used
     * @param typedOrdinaryBatch whether the additional typed/control namespace is used
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch, boolean typedOrdinaryBatch) {
        this(loweringFingerprint,executionStrategy,boundaryDataTypes,carrierPattern,vectorSpeciesBitSize,
                materializedSourcePosition,scalarPowerRealizations,scratchParameter,classIdentitySchema,
                matmulIr,generalScanIr,ordinaryBatch,typedOrdinaryBatch,Optional.empty());
    }
    /**
     * Retains the original ordinary-batch specialization and its exact identity streams.
     * @param loweringFingerprint canonical fingerprint
     * @param executionStrategy prepared scalar/vector strategy
     * @param boundaryDataTypes exact boundary types
     * @param carrierPattern selected typed carriers
     * @param vectorSpeciesBitSize vector width or zero
     * @param materializedSourcePosition copied source or minus one
     * @param scalarPowerRealizations ordered power facts
     * @param scratchParameter whether scratch is an entry argument
     * @param classIdentitySchema retained family projection
     * @param matmulIr optional typed matrix facts
     * @param generalScanIr optional typed Scan facts
     * @param ordinaryBatch whether the retained ordinary F32 numeric path is selected
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr,
            boolean ordinaryBatch) {
        this(loweringFingerprint,executionStrategy,boundaryDataTypes,carrierPattern,vectorSpeciesBitSize,
                materializedSourcePosition,scalarPowerRealizations,scratchParameter,classIdentitySchema,
                matmulIr,generalScanIr,ordinaryBatch,false);
    }
    /**
     * Retains the static/general-scan specialization constructor and all established identities.
     * @param loweringFingerprint non-null canonical fingerprint
     * @param executionStrategy non-null selected strategy
     * @param boundaryDataTypes non-null ordered represented types
     * @param carrierPattern non-null ordered typed carriers
     * @param vectorSpeciesBitSize vector bits, zero for scalar
     * @param materializedSourcePosition copied input or minus one
     * @param scalarPowerRealizations non-null ordered power realizations
     * @param scratchParameter whether scratch is an entry argument
     * @param classIdentitySchema accepted class projection
     * @param matmulIr non-null optional typed matrix facts
     * @param generalScanIr non-null optional typed scan facts
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr> generalScanIr) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes, carrierPattern,
                vectorSpeciesBitSize, materializedSourcePosition, scalarPowerRealizations,
                scratchParameter, classIdentitySchema, matmulIr, generalScanIr, false);
    }

    /**
     * Preserves the established non-scan specialization shape.
     * @param loweringFingerprint exact canonical fingerprint
     * @param executionStrategy selected generated strategy
     * @param boundaryDataTypes ordered represented boundary types
     * @param carrierPattern ordered direct carrier forms
     * @param vectorSpeciesBitSize vector width, zero for scalar
     * @param materializedSourcePosition copied input or minus one
     * @param scalarPowerRealizations ordered scalar power choices
     * @param scratchParameter whether scratch is an entry parameter
     * @param classIdentitySchema accepted family projection
     * @param matmulIr typed matrix facts or empty
     * @throws NullPointerException if a required component or list element is null
     * @throws IllegalArgumentException if specialization facts disagree or a scan projection
     *     is supplied without its required typed scan facts
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition, List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema, Optional<CpuMatmulIr> matmulIr) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes, carrierPattern,
                vectorSpeciesBitSize, materializedSourcePosition, scalarPowerRealizations,
                scratchParameter, classIdentitySchema, matmulIr, Optional.empty());
    }
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
    /**
     * Retains the established scalar/vector constructor shape and the schema-52 family
     * code-shaping projection. The generator-71 prefix deliberately changes class identity.
     *
     * @param loweringFingerprint canonical lowering fingerprint
     * @param executionStrategy generated compute strategy
     * @param boundaryDataTypes ordered boundary types
     * @param carrierPattern ordered carriers
     * @param vectorSpeciesBitSize vector species bits, or zero
     * @param materializedSourcePosition copied source, or minus one
     * @param scalarPowerRealizations scalar-power facts
     * @param scratchParameter whether scratch is present
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, scratchParameter, 52, Optional.empty());
    }

    /**
     * Creates a specialization with an explicit class-identity projection and no MATMUL facts.
     *
     * @param loweringFingerprint canonical lowering fingerprint
     * @param executionStrategy generated compute strategy
     * @param boundaryDataTypes ordered boundary types
     * @param carrierPattern ordered carriers
     * @param vectorSpeciesBitSize vector species bits, or zero
     * @param materializedSourcePosition copied source, or minus one
     * @param scalarPowerRealizations scalar-power facts
     * @param scratchParameter whether scratch is present
     * @param classIdentitySchema class projection schema; must be one of the accepted current
     *     family projections, including {@code 60} for matching cross-type CAST IR and
     *     {@code 61} for matching dense floating-mask vector IR, and {@code 62} for matching
     *     vector MSE {@code NONE} IR, and {@code 63} for matching direct dense FLOAT32/FLOAT64
     *     Conv2d/Conv3d output-width vector IR with unit width stride and dilation
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations,
            boolean scratchParameter, int classIdentitySchema) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, scratchParameter, classIdentitySchema, Optional.empty());
    }
    /**
     * Creates a direct, non-materialized compatibility specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param carrierPattern non-null ordered FLOAT64 compatibility carrier pattern
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species size for vector compute, or
     *     zero for scalar compute
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the components are outside the compatibility
     *     constructor's FLOAT64 boundary contract
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize) {
        this(loweringFingerprint, executionStrategy,
                java.util.Collections.nCopies(carrierPattern.size(), DataType.FLOAT64), carrierPattern,
                vectorSpeciesBitSize, -1, List.of(), false);
    }

    /**
     * Creates a FLOAT64 compatibility specialization with optional one-input materialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
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
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<CarrierAccess> carrierPattern, int vectorSpeciesBitSize,
            int materializedSourcePosition) {
        this(loweringFingerprint, executionStrategy,
                java.util.Collections.nCopies(carrierPattern.size(), DataType.FLOAT64),
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition, List.of(), false);
    }

    /**
     * Creates a specialization without scalar-power realization facts.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param executionStrategy non-null generated scalar or vector compute strategy
     * @param boundaryDataTypes non-null ordered boundary data types
     * @param carrierPattern non-null ordered boundary carrier forms
     * @param vectorSpeciesBitSize positive preferred species size for vector compute, or zero
     * @param materializedSourcePosition copied input position, or {@code -1}
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if specialization facts disagree
     */
    public CpuKernelSpecialization(CpuLoweringFingerprint loweringFingerprint,
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition, List.of(), false);
    }

    /**
     * Creates a specialization with scalar-power facts and no generated scratch parameter.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
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
            CpuPartitionPreparationPlan.ExecutionStrategy executionStrategy,
            List<DataType> boundaryDataTypes, List<CarrierAccess> carrierPattern,
            int vectorSpeciesBitSize, int materializedSourcePosition,
            List<CpuKernelIr.PowerRealization> scalarPowerRealizations) {
        this(loweringFingerprint, executionStrategy, boundaryDataTypes,
                carrierPattern, vectorSpeciesBitSize, materializedSourcePosition,
                scalarPowerRealizations, false);
    }
    /**
     * Validates the exact/default scalar-or-vector generated specialization.
     *
     * @param loweringFingerprint non-null canonical lowering fingerprint
     * @param executionStrategy non-null compute strategy; selected exec-v1 additionally binds
     *     its synchronous helper orchestration
     * @param boundaryDataTypes non-null ordered boundary types; copied defensively
     * @param carrierPattern non-null ordered boundary carrier pattern; copied defensively
     * @param vectorSpeciesBitSize positive preferred typed species bit size for vector
     *     compute, or zero for scalar compute
     * @param materializedSourcePosition copied input position, or {@code -1} for direct access
     * @param scalarPowerRealizations non-null ordered scalar-power realization facts; copied
     * @param scratchParameter whether the direct generated entry accepts one scratch segment
     * @param classIdentitySchema exact accepted family code-shaping projection
     * @param matmulIr optional typed matrix facts, present only for projection 54
     * @param generalScanIr optional typed general-scan facts, present only for projection 71
     * @param ordinaryBatch whether B-only scalar recipes and versioned batch identity are required
     * @param typedOrdinaryBatch whether the additive checked-segment typed/control/ROW(1)
     *     namespace is selected; requires ordinaryBatch and excludes ordinary MATMUL
     * @param rowMapReductionIr optional ordinary full-body region, present exactly for projection 72
     * @param rowFusionEmission optional selected exec-v1 helper/resource layout; absent for scalar v1
     * @param aggregateFusionIr optional static exact SUM/MEAN composition with rounded typed sites
     * @param softmaxBackwardIr optional static two-pass derivative with exact SUM and full-gradient write
     * @throws NullPointerException if a component is {@code null}
     * @throws IllegalArgumentException if the mode, compute/species relationship, boundary/type
     *     mapping, or materialized position is inconsistent
     */
    public CpuKernelSpecialization {
        Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        boundaryDataTypes = List.copyOf(boundaryDataTypes);
        carrierPattern = List.copyOf(carrierPattern);
        scalarPowerRealizations = List.copyOf(scalarPowerRealizations);
        matmulIr = Objects.requireNonNull(matmulIr, "matmulIr");
        generalScanIr = Objects.requireNonNull(generalScanIr, "generalScanIr");
        rowMapReductionIr = Objects.requireNonNull(rowMapReductionIr,"rowMapReductionIr");
        rowFusionEmission = Objects.requireNonNull(rowFusionEmission,"rowFusionEmission");
        aggregateFusionIr = Objects.requireNonNull(aggregateFusionIr,"aggregateFusionIr");
        softmaxBackwardIr = Objects.requireNonNull(softmaxBackwardIr,"softmaxBackwardIr");
        compositionUnits = List.copyOf(compositionUnits);
        if (!compositionUnits.isEmpty()
                && (generalScanIr.isEmpty() || generalScanIr.orElseThrow().composition().isEmpty())) {
            throw new IllegalArgumentException("ordinary helper inventory requires composite Scan facts");
        }
        if (softmaxBackwardIr.isPresent()) {
            var backward = softmaxBackwardIr.orElseThrow();
            if (classIdentitySchema != 52 || ordinaryBatch || typedOrdinaryBatch || matmulIr.isPresent()
                    || generalScanIr.isPresent() || rowMapReductionIr.isPresent() || rowFusionEmission.isPresent()
                    || aggregateFusionIr.isPresent()
                    || !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    || vectorSpeciesBitSize != 0 || materializedSourcePosition != -1 || !scratchParameter
                    || !scalarPowerRealizations.isEmpty() || !loweringFingerprint.hex().equals(backward.structuralKey())
                    || boundaryDataTypes.size() != backward.boundaries().size()
                    || boundaryDataTypes.stream().anyMatch(t -> t != backward.aggregate().dataType()))
                throw new IllegalArgumentException("softmax backward specialization disagrees");
        }
        if (aggregateFusionIr.isPresent()) {
            var fusion = aggregateFusionIr.orElseThrow();
            if (classIdentitySchema != 52 || ordinaryBatch || typedOrdinaryBatch || matmulIr.isPresent()
                    || generalScanIr.isPresent() || rowMapReductionIr.isPresent() || rowFusionEmission.isPresent()
                    || !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    || vectorSpeciesBitSize != 0 || materializedSourcePosition != -1 || !scratchParameter
                    || !scalarPowerRealizations.isEmpty() || !loweringFingerprint.hex().equals(fusion.structuralKey())
                    || boundaryDataTypes.size() != fusion.boundaries().size()
                    || boundaryDataTypes.stream().anyMatch(t -> t != fusion.aggregate().dataType()))
                throw new IllegalArgumentException("ordinary exact fusion specialization disagrees");
        }
        boolean selectedRow = rowFusionEmission.isPresent();
        if ((classIdentitySchema == 72) != rowMapReductionIr.isPresent())
            throw new IllegalArgumentException("row-region projection requires exclusive region facts");
        if (rowMapReductionIr.isPresent()) {
            var region = rowMapReductionIr.orElseThrow();
            var boundaries = java.util.stream.Stream.concat(region.arithmetic().values().stream()
                    .filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT),region.arithmetic().values().stream()
                    .filter(v -> v.kind() == CpuKernelIr.Value.Kind.OUTPUT)).toList();
            if (matmulIr.isPresent() || generalScanIr.isPresent() || !selectedRow && scratchParameter
                    || region.arithmetic().values().stream().anyMatch(v ->
                            v.dataType() != region.arithmetic().values().getFirst().dataType()
                                || v.dataType() != DataType.FLOAT32 && v.dataType() != DataType.FLOAT64)
                    || materializedSourcePosition != -1 || !selectedRow && vectorSpeciesBitSize != 0
                    || !scalarPowerRealizations.isEmpty() || typedOrdinaryBatch
                    || !selectedRow && !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    || !loweringFingerprint.hex().equals(region.structuralKey())
                    || !boundaryDataTypes.equals(boundaries.stream().map(CpuKernelIr.Value::dataType).toList())
                    || ordinaryBatch && boundaryDataTypes.stream().anyMatch(t -> t != DataType.FLOAT32))
                throw new IllegalArgumentException("ordinary row-region specialization disagrees");
        }
        if (typedOrdinaryBatch && (!ordinaryBatch || matmulIr.isPresent()
                || carrierPattern.stream().anyMatch(c -> c != CarrierAccess.MEMORY_SEGMENT)))
            throw new IllegalArgumentException("typed ordinary batch requires checked segments and no ordinary matrix route");
        if (ordinaryBatch && (!selectedRow && !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                || generalScanIr.isPresent() || materializedSourcePosition != -1
                || boundaryDataTypes.stream().anyMatch(type -> type != DataType.FLOAT32
                    && (!typedOrdinaryBatch || type != DataType.FLOAT64 && type != DataType.INT64)))) {
            throw new IllegalArgumentException("ordinary batch requires qualified typed scalar specialization");
        }
        if (matmulIr.isPresent() && ordinaryBatch != (matmulIr.orElseThrow().batchGeometry() != null)) {
            throw new IllegalArgumentException("matrix batch specialization meaning disagrees");
        }
        if (classIdentitySchema != 52 && classIdentitySchema != 54 && classIdentitySchema != 55
                && classIdentitySchema != 56 && classIdentitySchema != ATTENTION_LITERAL_EXP_SCHEMA
                && classIdentitySchema != 58 && classIdentitySchema != 59
                && classIdentitySchema != 60 && classIdentitySchema != 61
                && classIdentitySchema != 62 && classIdentitySchema != 63
                && classIdentitySchema != 68 && classIdentitySchema != 71 && classIdentitySchema != 72
                && classIdentitySchema != DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA) {
            throw new IllegalArgumentException("unsupported CPU generated class identity projection");
        }
        if ((classIdentitySchema == 71) != generalScanIr.isPresent())
            throw new IllegalArgumentException("scan family projection requires typed scan facts");
        if (generalScanIr.isPresent()) {
            var scan = generalScanIr.orElseThrow();
            if (!scan.structuralKey().equals(loweringFingerprint.hex())
                    || !selectedRow && !executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    || carrierPattern.stream().anyMatch(c -> c != CarrierAccess.MEMORY_SEGMENT)
                    || !boundaryDataTypes.equals(scan.boundaryTypes())
                    || boundaryDataTypes.size() != scan.boundaryAccesses().size()
                    || scan.composition().isEmpty() && !selectedRow && scratchParameter != (scan.scratchBanks() > 0)
                    || scan.composition().isEmpty() && boundaryDataTypes.size() > 248
                    || scan.composition().isEmpty() && boundaryDataTypes.size() + 5 + (scratchParameter ? 1 : 0) > 254)
                throw new IllegalArgumentException("scan typed facts/entry budget disagree");
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
        int expectedSpeciesBitSize = vector ? selectedRow && generalScanIr.isPresent()
                ? preferredSpeciesBitSize(List.of(generalScanIr.orElseThrow().dataType()))
                : preferredSpeciesBitSize(boundaryDataTypes) : 0;
        if (vector != (vectorSpeciesBitSize > 0)
                || (vector && vectorSpeciesBitSize != expectedSpeciesBitSize)
                || materializedSourcePosition < -1
                || materializedSourcePosition >= carrierPattern.size() - 1
                || (materializedSourcePosition >= 0
                    && carrierPattern.get(materializedSourcePosition) != CarrierAccess.MEMORY_SEGMENT)
                || scratchParameter && materializedSourcePosition >= 0) {
            throw new IllegalArgumentException("current CPU route requires exact strategy/species facts");
        }
        if (selectedRow) {
            var region = rowMapReductionIr.isPresent() ? rowMapReductionIr.orElseThrow() : generalScanIr.flatMap(s -> s.stages())
                    .flatMap(s -> s.rowMapReductionIr()).orElseThrow(() ->
                            new IllegalArgumentException("selected execution requires a proved row region"));
            var emission = rowFusionEmission.orElseThrow();
            if (executionStrategy.equals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    || !scratchParameter || materializedSourcePosition != -1
                    || emission.maximumPartialCount() > 1 != (executionStrategy.orchestration()
                        == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL)
                    || generalScanIr.isEmpty() && (emission.stepOperandCount() != 0 || emission.stepFrameLongCount() != 0))
                throw new IllegalArgumentException("selected row execution facts disagree");
            for (var ordinals : List.of(emission.invariantTransposeValues(),emission.temporalTransposeValues()))
                for (int v : ordinals) if (v >= region.arithmetic().values().size()
                        || region.rowAxes().get(v) >= 0 || region.accesses().get(v).plan().iterationRank() != 2)
                    throw new IllegalArgumentException("unproved row transpose source");
            if (generalScanIr.isPresent()) {
                int count = region.arithmetic().values().size(), ports = generalScanIr.orElseThrow().demandedPorts().size();
                if (emission.stepOperandCount() != Math.addExact(2*count+2,ports)
                        || emission.stepFrameLongCount() != Math.addExact(4+4*count,ports))
                    throw new IllegalArgumentException("selected Scan frame coverage disagrees");
            }
            var invariant = new java.util.ArrayList<Integer>(); var temporal = new java.util.ArrayList<Integer>();
            if (vector) for (var contraction : region.contractions()) if (region.rowSites().contains(contraction.output())) {
                int v = contraction.right(); var access = region.accesses().get(v);
                if (access.strideConstants().get(1) == 1) continue;
                if (invariant.contains(v) || temporal.contains(v)) continue;
                if (access.strideConstants().get(0) != 1 || access.strideBatchFactors().stream().anyMatch(f -> f != 0))
                    throw new IllegalArgumentException("unproved transpose feature mapping");
                int origin = region.viewOrigins().get(v);
                int formal = generalScanIr.map(s -> s.formalOrdinals().indexOf(origin)).orElse(-1);
                if (generalScanIr.isEmpty() || formal >= generalScanIr.orElseThrow().carryCount()+generalScanIr.orElseThrow().sequenceCount()) invariant.add(v);
                else temporal.add(v);
            }
            if (!emission.invariantTransposeValues().equals(invariant) || !emission.temporalTransposeValues().equals(temporal))
                throw new IllegalArgumentException("transpose source coverage/lifetime disagrees");
        }
    }
    /**
     * Returns the exact generated entry ABI for the authenticated specialization. Boundary
     * carriers precede optional segment scratch and the run-owned geometry array, followed by
     * start/end LONG bounds. Only selected parallel Scan inserts its run-owned StepDispatch
     * after geometry and before those bounds; ordinary and nonparallel entries retain their ABI.
     * @return nonnull immutable method type returning void with the selected ordered parameters
     */
    public MethodType entryType() {
        if (generalScanIr.isPresent() && generalScanIr.orElseThrow().composition().isPresent()) {
            return MethodType.methodType(void.class, MemorySegment[].class, long[][].class,
                    MemorySegment.class, MemorySegment.class, long.class, long.class);
        }
        boolean step = generalScanIr.isPresent() && rowFusionEmission.isPresent()
                && executionStrategy.orchestration() == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL;
        Class<?>[] parameters = new Class<?>[carrierPattern.size() + 3 + (scratchParameter ? 1 : 0) + (step ? 1 : 0)];
        for (int i = 0; i < carrierPattern.size(); i++) parameters[i] = carrierClass(carrierPattern.get(i));
        int next = carrierPattern.size();
        if (scratchParameter) parameters[next++] = MemorySegment.class;
        parameters[next++] = long[].class;
        if (step) parameters[next++] = io.github.pho001.synaptik.backend.cpu.internal.executable.CpuGeneralScanPreparedExecutable.StepDispatch.class;
        parameters[next++] = long.class;
        parameters[next] = long.class;
        return MethodType.methodType(void.class, parameters);
    }

    /**
     * Resolves the typed loop-address proof category encoded by this specialization and IR.
     * Compatible extents do not affect the result because heap-array binding independently
     * validates every concrete address span before invocation.
     * Selected compact folds retain GENERAL_LONG as their structural category but emit a
     * complete invocation-entry int proof and an int body before their long fallback. Ordinary
     * fold addressing and both identity streams remain unchanged outside the selected suffix.
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
        // Compact fold inverse numerators can exceed int even when both arrays fit int.
        // Its selected body performs a complete invocation-entry int proof, with long fallback.
        if (kernelIr.familyIdentity().endsWith(":kernel-broadcast-input-v1")) return LoopAddressing.GENERAL_LONG;
        boolean denseArrays = !carrierPattern.contains(CarrierAccess.MEMORY_SEGMENT)
                && kernelIr.values().stream().filter(value -> value.kind()
                    != CpuKernelIr.Value.Kind.VIRTUAL).allMatch(value -> value.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR
                    || value.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
        return denseArrays ? LoopAddressing.DENSE_HEAP_ARRAY_INT : LoopAddressing.GENERAL_LONG;
    }
    /** Returns compatibility metadata, including actually baked staged-scan geometry only when
     * present. Unchanged families retain their existing geometry projections; ordinary exact
     * fusion binds its baked static access geometry and typed sites. Unchanged pointwise entries
     * append nothing and retain their original bytes. Staged scan v4 advances only the existing
     * baked payload for top-tested loop control, rejecting staged v1/v2/v3 artifacts within schema 71.
     * CPU0071 advances only retained static MATMUL to all-valid v5 or length-aware-matmul v2
     * for invocation-bound strides; fixed length without MATMUL retains v1 and shaped retains v4.
     * Combined symbolic-length mode retains its baked v9 payload
     * in both streams for nominal-stride binding and cell-scoped read/write offset sharing
     * after raw initialization of a stable own-final-port carry without a bank; only proved
     * zero bases are omitted. A nonbanked scratch cut also omits its stable base when both
     * coefficients of its complete {@code offset + offsetPerBatch * B} recipe are zero;
     * derived views retain their own relative strides and exact origin's proof.
     * V7 adds flat initialization, initialized empty-length publication, nested affine prefixes,
     * direct active-row contractions and proved raw singleton reductions.
     * Combined v1 through v8 code is rejected while canonical v1 and
     * the numeric ABI remain unchanged. The
     * separate INT64 boundary dtype includes explicit
     * activity roles and fixed geometry but excluding runtime length values, slots and addresses.
     * Selected row exec-v1 appends its ordinary/Scan payload with strategy, preferred species,
     * helper ABI, transpose lifetimes and partial/frame layout. Actual B/L and assigned slots
     * remain excluded. Scalar row replacements append their scoped load/address-reuse marker
     * even without exec-v1 metadata; unaffected SIMD and nonregional identities remain literal.
     * CPU0076 appends its separate staging selection only when the pure shared query proves
     * actual scalar placement changes, with identical source scopes and typed phase bounds.
     * @return a new deterministic schema byte array */
    public byte[] compatibilityBytes() {
        return (CpuGeneratorSchema.CURRENT_VERSION + "|" + loweringFingerprint.hex() + "|"
                + executionStrategy.compute()
                + "|" + boundaryDataTypes
                + "|" + carrierPattern
                + "|" + vectorSpeciesBitSize + "|materialized=" + materializedSourcePosition
                + "|power=" + scalarPowerRealizations + "|scratch=" + scratchParameter
                + generalScanIr.map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr::bakedSpecializationIdentity).orElse("") + matmulIr.map(CpuMatmulIr::batchSpecializationIdentity).orElse("") + (ordinaryBatch ? typedOrdinaryBatch ? ";ordinary-batch-typed-v1" : ";ordinary-batch-v1" : "")
                + rowMapReductionIr.map(r -> "|ordinary-row-map-multi-reduction-v1="+r.bakedSpecializationIdentity()).orElse("")
                + aggregateFusionIr.map(f -> "|ordinary-aggregate-fusion-v1="+f.bakedSpecializationIdentity()).orElse("")
                + softmaxBackwardIr.map(f -> "|ordinary-softmax-backward-v1="+f.bakedSpecializationIdentity()).orElse("")
                + compositionIdentity()
                + rowExecutionIdentity() + scalarRowReuseIdentity() + invariantRowStagingIdentity()
                + (classIdentitySchema == DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA
                    ? "|affine-dense-array-empty-range-v1" : "")
                + (classIdentitySchema == ATTENTION_LITERAL_EXP_SCHEMA
                    ? "|attention-literal-working-sites-v1" : ""))
                .getBytes(StandardCharsets.US_ASCII);
    }
    /**
     * Returns the stable generated-class identity projection.
     *
     * <p>The current generator schema prefixes the family-specific projection so a pre-cutover
     * class cannot share a binary name with a profile-free class. Family-specific code-shaping
     * facts remain selective and all boundary data types remain distinct, including BFLOAT16 and
     * FLOAT16 even when both use a short carrier.</p>
     * <p>Shaped scan entries additionally bind every baked local extent, stride and resource
     * magnitude through their typed versioned specialization payload. That optional suffix never
     * enters canonical CpuKernelIr and is absent for the established pointwise scan family.
     * Staged v4 therefore advances binary names and keys together with compatibility metadata,
     * while the geometry-free canonical stage projections and global schema remain unchanged.</p>
     * <p>CPU0071 binds the same static MATMUL v5/length-aware-matmul v2 suffix as compatibility
     * metadata; non-MATMUL fixed length retains v1. Combined mode shares its retained v9 payload
     * with compatibility metadata, rejecting combined v1 through v8 artifacts. V7 introduced
     * the combined Java-loop-shape changes described by {@link #compatibilityBytes()}. Each stable
     * own-final-port/no-bank carry view retains its own nominal stride recipe after raw
     * initialization; cell-scoped read/write offset sharing omits only proved zero bases.
     * Nonbanked scratch cuts also omit their stable base when both coefficients of the complete
     * {@code offset + offsetPerBatch * B} recipe are zero. Derived views keep their own relative
     * strides and consult only their exact origin's proof.
     * V9 additionally retains direct carrier/temporal-base and explicit-B affine proofs, indexed
     * contractions and dependency/physical-alias/inclusive-lifetime checked bank-free scheduling.
     * Canonical v1 and the numeric ABI remain unchanged; unrelated identities retain their bytes. Its
     * explicit control/role/storage facts cannot collide with retained all-valid modes; runtime
     * length values never affect generated names or cache identity.</p>
     * <p>Selected row exec-v1 includes the same selective payload as compatibility metadata,
     * including orchestration and helper/resource layout. It changes neither envelope 71 nor
     * ordinary/Scan projections 72/71. Scalar row bodies additionally carry the selective
     * load/address-reuse marker in both streams, including bodies without exec-v1 metadata.
     * CPU0076 appends the same affected-only staging selection as compatibility metadata;
     * no-stage, SIMD and nonregional classes retain their literal identities.</p>
     *
     * @return a new deterministic generator- and family-schema identity byte array
     */
    public byte[] classIdentityBytes() {
        return (CpuGeneratorSchema.CURRENT_VERSION + "|" + classIdentitySchema + "|"
                + loweringFingerprint.hex() + "|" + executionStrategy.compute()
                + "|" + boundaryDataTypes
                + "|" + carrierPattern
                + "|" + vectorSpeciesBitSize + "|materialized=" + materializedSourcePosition
                + "|power=" + scalarPowerRealizations + "|scratch=" + scratchParameter
                + generalScanIr.map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr::bakedSpecializationIdentity).orElse("") + matmulIr.map(CpuMatmulIr::batchSpecializationIdentity).orElse("") + (ordinaryBatch ? typedOrdinaryBatch ? ";ordinary-batch-typed-v1" : ";ordinary-batch-v1" : "")
                + rowMapReductionIr.map(r -> "|ordinary-row-map-multi-reduction-v1="+r.bakedSpecializationIdentity()).orElse("")
                + aggregateFusionIr.map(f -> "|ordinary-aggregate-fusion-v1="+f.bakedSpecializationIdentity()).orElse("")
                + softmaxBackwardIr.map(f -> "|ordinary-softmax-backward-v1="+f.bakedSpecializationIdentity()).orElse("")
                + compositionIdentity()
                + rowExecutionIdentity() + scalarRowReuseIdentity() + invariantRowStagingIdentity()
                + (classIdentitySchema == DENSE_ARRAY_AFFINE_EMPTY_RANGE_SCHEMA
                    ? "|affine-dense-array-empty-range-v1" : "")
                + (classIdentitySchema == ATTENTION_LITERAL_EXP_SCHEMA
                    ? "|attention-literal-working-sites-v1" : ""))
                .getBytes(StandardCharsets.US_ASCII);
    }

    /**
     * Serializes only genuine composite helper code-shaping facts.
     * Composite-only v2 accompanies the corrected interface-static memory-copy calls in both
     * compatibility and class-identity streams; every noncomposite branch retains its suffix.
     *
     * @return selective identity suffix, empty for every retained standalone/Scan recipe
     */
    private String compositionIdentity() {
        if (generalScanIr.isEmpty() || generalScanIr.orElseThrow().composition().isEmpty()) return "";
        var identity = new StringBuilder("|ordinary-scan-helpers-v2=");
        for (CompositionUnit unit : compositionUnits) {
            identity.append(unit.specialization().structuralKey()).append(':')
                    .append(unit.resets()).append(':').append(unit.scratchOffset()).append(':')
                    .append(unit.iterationCount()).append(';');
        }
        return identity.toString();
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
            case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
    }

    /**
     * Binds the exact selected helper algorithm/layout in both identity streams, never actual B/L.
     * @return empty for literal scalar v1, otherwise the selected exec-v1 payload
     */
    private String rowExecutionIdentity() {
        return rowFusionEmission.map(e -> "|"+(rowMapReductionIr.isPresent() ? "ordinary" : "scan")
                +"-row-map-multi-reduction-exec-v1="+executionStrategy+";species="+vectorSpeciesBitSize
                +";layout="+e+";masked-feature;ordered-feature-k;quotient-remainder-rows;valid-range-index-merge;"
                +"typed-mul-add;separate-time-carry-add;helpers="
                +(rowMapReductionIr.isPresent() ? "ordinaryRowRange,ordinaryRowMerge:"+entryType().descriptorString()
                    : "scanRowRange:([Ljava/lang/foreign/MemorySegment;[JJJJ)V")
                +";transpose-raw-active-lifetime-v1;scan-tail-private-base-R;frame-time-R-private-base-N-read-write-strides-v1")
                .orElse("");
    }

    /**
     * Advances only replaced scalar row bodies, including those without exec-v1 metadata.
     * Existing baked payloads supply topology, coefficients, bindings and lifetime facts; this
     * deterministic algorithm marker supplies last-use recycling and conservative epoch policy.
     * @return scoped suffix in both identity streams, or empty for SIMD/nonregional controls
     */
    private String scalarRowReuseIdentity() {
        if (executionStrategy.compute() != CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) return "";
        boolean ordinary = rowMapReductionIr.isPresent();
        boolean scan = generalScanIr.flatMap(s -> s.stages().flatMap(st -> st.rowMapReductionIr())).isPresent();
        if (!ordinary && !scan) return "";
        return "|"+(ordinary ? "ordinary" : "scan")+"-row-scalar-load-address-reuse-v1="
                +"root-affine-c-B-r-rB;immutable-row-epoch;last-use-typed-pool;single-recycled-affine-offset;"
                +"checked-unproved-load;no-sum-branch-cache;no-carry-or-scratch-cache;live128;method48000";
    }

    /**
     * Binds only actual changed scalar placements to both identity streams. Source scopes are
     * exact formal bindings, not shape guesses; all-B recipes and retained lifetimes are already
     * bound by the existing regional payload. The same pure query runs before cuts and emission.
     * @return append-only scoped selection/fixed-budget identity, or empty for literal controls
     */
    private String invariantRowStagingIdentity() {
        if (executionStrategy.compute() != CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR) return "";
        var scan = generalScanIr.orElse(null);
        var region = rowMapReductionIr.orElseGet(() -> scan == null ? null
                : scan.stages().flatMap(s -> s.rowMapReductionIr()).orElse(null));
        if (region == null) return "";
        var invariant = new java.util.ArrayList<Integer>(); var temporal = new java.util.ArrayList<Integer>();
        if (scan == null) {
            for (var v : region.arithmetic().values()) if (v.kind() == CpuKernelIr.Value.Kind.INPUT
                    && region.viewOrigins().get(v.ordinal()) == v.ordinal()) invariant.add(v.ordinal());
        } else for (int f = scan.carryCount(); f < scan.formalOrdinals().size(); f++) {
            int root = scan.formalOrdinals().get(f);
            if (scan.scalarReads().get(scan.actualReads().get(f))) continue;
            if (f >= scan.carryCount()+scan.sequenceCount()) invariant.add(root);
            else if (region.rowAxes().get(root) < 0) temporal.add(root);
        }
        long duplicateBindings = 0;
        if (scan != null) for (int v : region.rowSites()) if (region.retainedStores().contains(v)) {
            long wrappers = 0;
            for (int p : scan.demandedPorts()) if (p < scan.carryCount() ? scan.nextOrdinals().get(p) == v
                    : region.viewOrigins().get(scan.portOrdinals().get(p)) == v) wrappers++;
            duplicateBindings += Math.max(0,wrappers-1);
        }
        long histories = scan == null ? 0 : scan.demandedPorts().stream().filter(p -> p >= scan.carryCount()).count();
        if (scan != null) {
            var finish = scan.stages().orElseThrow().order().stream().filter(s -> s.values().containsAll(region.rowSites())
                    && s.values().containsAll(region.terminalOutputs())).findFirst().orElseThrow().epilogue();
            for (int v = scan.formalOrdinals().size(); v < region.viewOrigins().size(); v++) if (region.viewOrigins().get(v) == v
                    && !region.rowSites().contains(v) && !region.terminalOutputs().contains(v) && !finish.contains(v)) {
                invariant.clear(); temporal.clear(); break;
            }
        }
        long callerBase = 112+3*duplicateBindings+2*histories, helperBase = 116+3*duplicateBindings;
        var selected = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRowMapReductionLowering
                .invariantLoadStaging(region,invariant,temporal,callerBase,helperBase);
        if (selected.isEmpty()) return "";
        return "|"+(scan == null ? "ordinary" : "scan")+"-row-invariant-load-staging-v1="
                +selected+";invocation="+invariant+";original-time="+temporal
                +";caller-base="+callerBase+";helper-base="+helperBase+";typed128;whole-consumed-domain;origin-affine-dedup;"
                +"readonly-full-lifetime;nonempty-active-consumer;helper-first-active;remaining-row-pool";
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
            case BFLOAT16, FLOAT16 -> 0;
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
