package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAdvancedReductionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLowering;

/**
 * Route-neutral immutable selected CPU partition plan.
 *
 * @param units non-null computation-oriented units; copied defensively
 * @param route non-null route selected after common lowering
 * @param executionStrategy non-null selected compute/orchestration strategy
 * @param bufferDeclarations non-null exact post-fusion declarations; copied defensively
 * @param boundaryValues non-null materialized value identities in declaration order; copied
 *     defensively
 * @param accessBindings non-null normalized cold geometry in boundary order; copied defensively
 * @param carrierPattern non-null direct carrier forms in boundary order; copied defensively
 * @param generatedCarrierPattern non-null generated consumer carrier forms in boundary order;
 *     differs from {@code carrierPattern} only when the selected copy replaces one input with the
 *     contiguous workspace segment
 * @param extents non-null cold-bound compatible extents; copied defensively
 * @param elementCount checked logical element count represented by {@code extents}
 * @param affineAddressPairs alternating cold-composed source/result element addresses for an
 *     affine copy, or an empty array for pointwise execution; copied defensively
 * @param selectedRangeCount positive maximum range count selected during cold analysis; one for
 *     single-thread strategies and at least two for parallel strategies
 * @param minimumElementsPerWorker positive minimum logical elements per submitted worker chunk
 * @param vectorSpeciesBitSize exact positive preferred typed species size in bits for
 *     vector strategies, or zero for scalar strategies
 * @param loweringManifest non-null optional cold diagnostic text, empty when disabled
 * @param materialization non-null optional selected one-input copy fact
 * @param workspaceDeclaration non-null optional exact workspace declaration; present exactly when
 *     materialization, floating scatter multiplication, stable ordering, or a nonempty floating
 *     numerical aggregate requires it
 * @param workspaceUse non-null explicit meaning of the optional workspace
 * @param specializationBudget non-null enforced candidate/artifact/shape/unroll ceiling
 * @param movementGeometry non-null optional compact cold non-affine movement geometry
 * @param indexingGeometry non-null optional compact cold indexing geometry; present exactly for
 *     a gather or one-hot plan and mutually exclusive with materialization and workspace
 * @param scatterGeometry non-null optional functional-scatter geometry; floating multiplication
 *     may pair it with the exact declared product workspace
 * @param foldGeometry non-null optional zero-workspace overlap-fold geometry
 * @param orderingGeometry non-null optional stable ordering geometry; present exactly for one
 *     SORT, ARGSORT, or TOP_K plan and paired with exact per-range merge scratch
 * @param randomGeometry non-null optional zero-workspace INITIAL_STATE or DROPOUT geometry
 * @param scanGeometry non-null optional zero-workspace CUM_SUM or CUM_PROD slice geometry
 * @param aggregateGeometry non-null optional ordinary numerical/extrema/Boolean output-cell
 *     geometry; floating numerical rows carry an exact per-range state shape
 * @param argExtremaGeometry non-null optional one-axis logical-index output-cell geometry
 * @param maskedReductionGeometry non-null optional directional masked SUM/MEAN geometry
 * @param advancedReductionGeometry non-null optional logarithmic, statistical, or norm
 *     complete-output-cell geometry
 * @param softmaxGeometry non-null optional zero-workspace shape-preserving normalization geometry
 * @param trailingNormalizationGeometry non-null optional trailing Layer/RMS geometry; Layer may
 *     pair with exact-state workspace while RMS requires none
 * @param batchNormInferenceGeometry non-null optional zero-workspace arbitrary-axis inference
 *     geometry, mutually exclusive with every other specialized-family geometry
 */
public record CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
        ExecutionStrategy executionStrategy,
        List<PreparationResourceRequirement.Buffer> bufferDeclarations,
        List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
        List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
        long[] extents, long elementCount, long[] affineAddressPairs,
        int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
        String loweringManifest, Optional<CpuMaterializationPlan> materialization,
        Optional<PreparationResourceRequirement.Workspace> workspaceDeclaration,
        WorkspaceUse workspaceUse,
        CpuSpecializationBudget specializationBudget,
        Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
        Optional<CpuIndexingLowering.Geometry> indexingGeometry,
        Optional<CpuScatterLowering.Geometry> scatterGeometry,
        Optional<CpuFoldLowering.Geometry> foldGeometry,
        Optional<CpuOrderingLowering.Geometry> orderingGeometry,
        Optional<CpuRandomLowering.Geometry> randomGeometry,
        Optional<CpuScanLowering.Geometry> scanGeometry,
        Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
        Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
        Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
        Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
        Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
        Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
        Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry)
        implements BackendPreparationPlan {

    /**
     * Creates an existing-family plan without batch-normalization inference geometry.
     *
     * <p>Parameters have the ownership, nullability, range, and exclusivity contracts of the
     * corresponding record components.</p>
     *
     * @param units computation-oriented units
     * @param route selected route
     * @param executionStrategy selected compute/orchestration strategy
     * @param bufferDeclarations exact declarations
     * @param boundaryValues ordered boundary identities
     * @param accessBindings ordered cold access geometry
     * @param carrierPattern Runtime carrier forms
     * @param generatedCarrierPattern generated-entry carrier forms
     * @param extents cold-bound iteration extents
     * @param elementCount logical element count
     * @param affineAddressPairs affine addresses or an empty array
     * @param selectedRangeCount selected maximum range count
     * @param minimumElementsPerWorker minimum work per worker chunk
     * @param vectorSpeciesBitSize selected vector width, or zero
     * @param loweringManifest cold diagnostic text
     * @param materialization optional input materialization
     * @param workspaceDeclaration optional workspace declaration
     * @param workspaceUse workspace meaning
     * @param specializationBudget enforced specialization budget
     * @param movementGeometry optional movement geometry
     * @param indexingGeometry optional indexing geometry
     * @param scatterGeometry optional scatter geometry
     * @param foldGeometry optional fold geometry
     * @param orderingGeometry optional ordering geometry
     * @param randomGeometry optional explicit-state random geometry
     * @param scanGeometry optional cumulative-scan geometry
     * @param aggregateGeometry optional ordinary-aggregate geometry
     * @param argExtremaGeometry optional arg-extrema geometry
     * @param maskedReductionGeometry optional masked-reduction geometry
     * @param advancedReductionGeometry optional advanced-reduction geometry
     * @param softmaxGeometry optional softmax geometry
     * @param trailingNormalizationGeometry optional trailing-normalization geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if route, boundary, range, workspace, carrier, or family
     *     geometry facts disagree
     * @throws ArithmeticException if exact geometry or resource validation overflows
     */
    public CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
            ExecutionStrategy executionStrategy,
            List<PreparationResourceRequirement.Buffer> bufferDeclarations,
            List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long[] extents, long elementCount, long[] affineAddressPairs,
            int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
            String loweringManifest, Optional<CpuMaterializationPlan> materialization,
            Optional<PreparationResourceRequirement.Workspace> workspaceDeclaration,
            WorkspaceUse workspaceUse, CpuSpecializationBudget specializationBudget,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry) {
        this(units, route, executionStrategy, bufferDeclarations, boundaryValues, accessBindings,
                carrierPattern, generatedCarrierPattern, extents, elementCount, affineAddressPairs,
                selectedRangeCount, minimumElementsPerWorker, vectorSpeciesBitSize, loweringManifest,
                materialization, workspaceDeclaration, workspaceUse, specializationBudget,
                movementGeometry, indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry,
                randomGeometry, scanGeometry, aggregateGeometry, argExtremaGeometry,
                maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry,
                trailingNormalizationGeometry, Optional.empty());
    }

    /**
     * Creates an existing-family plan without trailing-normalization geometry.
     *
     * @param units non-null computation-oriented units; copied defensively
     * @param route non-null selected route
     * @param executionStrategy non-null selected compute/orchestration strategy
     * @param bufferDeclarations non-null exact declarations; copied defensively
     * @param boundaryValues non-null boundary identities in declaration order; copied defensively
     * @param accessBindings non-null access geometry in boundary order; copied defensively
     * @param carrierPattern non-null Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern non-null generated-entry carrier forms; copied defensively
     * @param extents non-null iteration extents; copied defensively
     * @param elementCount non-negative logical work-item count
     * @param affineAddressPairs non-null affine address pairs, or empty; copied defensively
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param vectorSpeciesBitSize selected vector size in bits, or zero for scalar compute
     * @param loweringManifest non-null deterministic cold diagnostic summary
     * @param materialization non-null optional input materialization
     * @param workspaceDeclaration non-null optional exact workspace declaration
     * @param workspaceUse non-null workspace purpose
     * @param specializationBudget non-null immutable specialization budget
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @param randomGeometry non-null optional explicit-state random geometry
     * @param scanGeometry non-null optional cumulative-scan geometry
     * @param aggregateGeometry non-null optional ordinary-aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional stable-softmax geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if route, boundary, range, workspace, carrier, or family
     *     geometry facts disagree
     * @throws ArithmeticException if exact geometry or resource validation overflows
     */
    public CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
            ExecutionStrategy executionStrategy,
            List<PreparationResourceRequirement.Buffer> bufferDeclarations,
            List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long[] extents, long elementCount, long[] affineAddressPairs,
            int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
            String loweringManifest, Optional<CpuMaterializationPlan> materialization,
            Optional<PreparationResourceRequirement.Workspace> workspaceDeclaration,
            WorkspaceUse workspaceUse, CpuSpecializationBudget specializationBudget,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry) {
        this(units, route, executionStrategy, bufferDeclarations, boundaryValues, accessBindings,
                carrierPattern, generatedCarrierPattern, extents, elementCount, affineAddressPairs,
                selectedRangeCount, minimumElementsPerWorker, vectorSpeciesBitSize, loweringManifest,
                materialization, workspaceDeclaration, workspaceUse, specializationBudget,
                movementGeometry, indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry,
                randomGeometry, scanGeometry, aggregateGeometry, argExtremaGeometry,
                maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry,
                Optional.empty(), Optional.empty());
    }

    /**
     * Creates an existing-family plan without explicit-state random or cumulative-scan geometry.
     *
     * @param units non-null computation-oriented units; copied defensively
     * @param route non-null selected route
     * @param executionStrategy non-null selected compute and orchestration strategy
     * @param bufferDeclarations non-null exact buffer declarations; copied defensively
     * @param boundaryValues non-null ordered boundary values; copied defensively
     * @param accessBindings non-null ordered normalized accesses; copied defensively
     * @param carrierPattern non-null Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern non-null generated-entry carrier forms; copied defensively
     * @param extents non-null iteration extents; copied defensively
     * @param elementCount non-negative iteration element count
     * @param affineAddressPairs non-null affine address pairs; copied defensively
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum work items per worker chunk
     * @param vectorSpeciesBitSize selected vector size in bits, or zero for scalar compute
     * @param loweringManifest non-null deterministic lowering summary
     * @param materialization non-null optional input materialization
     * @param workspaceDeclaration non-null optional workspace declaration
     * @param workspaceUse non-null workspace purpose
     * @param specializationBudget non-null immutable specialization budget
     * @param movementGeometry non-null optional movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional scatter geometry
     * @param foldGeometry non-null optional fold geometry
     * @param orderingGeometry non-null optional ordering geometry
     * @throws NullPointerException if a required reference or list element is null
     * @throws IllegalArgumentException if route, boundary, range, workspace, carrier, or family
     *     geometry facts disagree
     * @throws ArithmeticException if exact geometry or resource validation overflows
     */
    public CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
            ExecutionStrategy executionStrategy,
            List<PreparationResourceRequirement.Buffer> bufferDeclarations,
            List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long[] extents, long elementCount, long[] affineAddressPairs,
            int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
            String loweringManifest, Optional<CpuMaterializationPlan> materialization,
            Optional<PreparationResourceRequirement.Workspace> workspaceDeclaration,
            WorkspaceUse workspaceUse, CpuSpecializationBudget specializationBudget,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry) {
        this(units, route, executionStrategy, bufferDeclarations, boundaryValues, accessBindings,
                carrierPattern, generatedCarrierPattern, extents, elementCount, affineAddressPairs,
                selectedRangeCount, minimumElementsPerWorker, vectorSpeciesBitSize,
                loweringManifest, materialization, workspaceDeclaration, workspaceUse,
                specializationBudget, movementGeometry, indexingGeometry, scatterGeometry,
                foldGeometry, orderingGeometry, Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }
    /**
     * Creates a pointwise or affine plan without non-affine movement geometry.
     *
     * @param units non-null computation units; copied defensively
     * @param route non-null selected route
     * @param executionStrategy non-null selected compute and orchestration strategy
     * @param bufferDeclarations non-null exact post-fusion declarations; copied defensively
     * @param boundaryValues non-null values in declaration order; copied defensively
     * @param accessBindings non-null boundary access geometry; copied defensively
     * @param carrierPattern non-null Runtime carrier forms; copied defensively
     * @param generatedCarrierPattern non-null generated-entry carrier forms; copied defensively
     * @param extents non-null compatible extents; copied defensively
     * @param elementCount checked logical element count
     * @param affineAddressPairs alternating affine source/result addresses, or an empty array for
     *     pointwise execution; copied defensively
     * @param selectedRangeCount positive selected maximum range count
     * @param minimumElementsPerWorker positive minimum elements per worker chunk
     * @param vectorSpeciesBitSize positive selected species size, or zero for scalar compute
     * @param loweringManifest non-null optional cold diagnostic text
     * @param materialization non-null optional contiguous-input copy fact
     * @param workspaceDeclaration non-null optional workspace, present exactly with a selected
     *     materialization
     * @param specializationBudget non-null enforced specialization ceiling
     * @throws NullPointerException if a required component or element is {@code null}
     * @throws IllegalArgumentException if strategy, declarations, boundaries, carrier forms,
     *     ranges, materialization, or specialization facts disagree
     */
    public CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
            ExecutionStrategy executionStrategy,
            List<PreparationResourceRequirement.Buffer> bufferDeclarations,
            List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
            List<CarrierAccess> carrierPattern, List<CarrierAccess> generatedCarrierPattern,
            long[] extents, long elementCount, long[] affineAddressPairs,
            int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
            String loweringManifest, Optional<CpuMaterializationPlan> materialization,
            Optional<PreparationResourceRequirement.Workspace> workspaceDeclaration,
            CpuSpecializationBudget specializationBudget) {
        this(units, route, executionStrategy, bufferDeclarations, boundaryValues, accessBindings,
                carrierPattern, generatedCarrierPattern, extents, elementCount, affineAddressPairs,
                selectedRangeCount, minimumElementsPerWorker, vectorSpeciesBitSize,
                loweringManifest, materialization, workspaceDeclaration,
                materialization.isPresent() ? WorkspaceUse.MATERIALIZATION : WorkspaceUse.NONE,
                specializationBudget, Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }
    /** Meaning of the plan's sole optional CPU workspace. */
    public enum WorkspaceUse {
        /** No workspace is declared. */ NONE,
        /** Contiguous pointwise input materialization. */ MATERIALIZATION,
        /** Per-range exact floating scatter-product accumulator slices. */ SCATTER_PRODUCT,
        /** Per-range two-region stable ordering indices. */ ORDERING_INDICES,
        /** Per-range exact floating ordinary-aggregate or masked-reduction state. */
        AGGREGATE_EXACT_STATE
    }
    /**
     * One computation-oriented execution unit.
     *
     * @param portablePlan non-null already-lowered portable realization plan
     * @param fusionReason non-null cold diagnostic explanation of the selected fusion
     */
    public record ExecutionUnitPlan(CpuPortableRoutePlan portablePlan, String fusionReason) {
        /**
         * Validates one selected unit and its diagnostic explanation.
         *
         * @param portablePlan non-null already-lowered portable realization plan
         * @param fusionReason non-null cold diagnostic explanation
         * @throws NullPointerException if either component is {@code null}
         */
        public ExecutionUnitPlan {
            Objects.requireNonNull(portablePlan, "portablePlan");
            Objects.requireNonNull(fusionReason, "fusionReason");
        }
    }
    /** Route selected after common lowering. */
    public enum Route {
        /** Java 26 Class-File portable route selected after common lowering. */ PORTABLE
    }
    /**
     * Orthogonal compute/orchestration vocabulary.
     *
     * @param compute non-null compute axis
     * @param orchestration non-null orchestration axis
     */
    public record ExecutionStrategy(Compute compute, Orchestration orchestration) {
        /** Compute axis. */
        public enum Compute {
            /** Scalar element computation. */ SCALAR,
            /** Preferred-species Java 26 Vector API element computation. */ VECTOR
        }
        /** Orchestration axis. */
        public enum Orchestration {
            /** Invocation on one orchestrating thread. */ SINGLE_THREAD,
            /** CPU-private external deterministic chunk dispatch. */ PARALLEL
        }
        /** Scalar compute on the invoking thread. */
        public static final ExecutionStrategy SCALAR =
                new ExecutionStrategy(Compute.SCALAR, Orchestration.SINGLE_THREAD);
        public static final ExecutionStrategy VECTOR =
                new ExecutionStrategy(Compute.VECTOR, Orchestration.SINGLE_THREAD);
        public static final ExecutionStrategy PARALLEL_SCALAR =
                new ExecutionStrategy(Compute.SCALAR, Orchestration.PARALLEL);
        public static final ExecutionStrategy PARALLEL_VECTOR =
                new ExecutionStrategy(Compute.VECTOR, Orchestration.PARALLEL);
        /**
         * Validates both execution-strategy axes.
         *
         * @param compute non-null selected scalar or vector compute axis
         * @param orchestration non-null selected single-thread or parallel orchestration axis
         * @throws NullPointerException if either axis is {@code null}
         */
        public ExecutionStrategy {
            Objects.requireNonNull(compute, "compute");
            Objects.requireNonNull(orchestration, "orchestration");
        }
        /** @return the stable non-null strategy name formed from both axes */
        @Override public String toString() {
            if (compute == Compute.SCALAR && orchestration == Orchestration.SINGLE_THREAD) return "scalar";
            if (compute == Compute.VECTOR && orchestration == Orchestration.SINGLE_THREAD) return "vector";
            return compute == Compute.SCALAR ? "parallel-scalar" : "parallel-vector";
        }
    }
    /**
     * Validates and snapshots one complete selected plan.
     *
     * @param units non-null one-entry computation-unit list; copied defensively
     * @param route non-null selected portable route
     * @param executionStrategy non-null selected compute/orchestration strategy
     * @param bufferDeclarations non-null derived-boundary declarations; copied defensively
     * @param boundaryValues non-null materialized values in declaration order; copied
     * @param accessBindings non-null normalized cold bindings in boundary order; copied
     * @param carrierPattern non-null direct carrier forms in boundary order; copied
     * @param generatedCarrierPattern non-null generated-consumer carrier forms; copied
     * @param extents non-null compatible iteration extents; copied defensively
     * @param elementCount checked logical element count represented by {@code extents}
     * @param affineAddressPairs alternating affine source/result addresses, or empty otherwise;
     *     copied defensively
     * @param selectedRangeCount positive maximum selected range count
     * @param minimumElementsPerWorker positive minimum elements per submitted worker chunk
     * @param vectorSpeciesBitSize positive preferred typed species bit size for vector
     *     compute, or zero for scalar compute
     * @param loweringManifest non-null optional cold diagnostic text
     * @param materialization non-null optional selected copy
     * @param workspaceDeclaration non-null optional exact workspace declaration
     * @param workspaceUse non-null purpose of the optional workspace
     * @param specializationBudget non-null current hard specialization ceiling
     * @param movementGeometry non-null optional static movement geometry
     * @param indexingGeometry non-null optional indexing geometry
     * @param scatterGeometry non-null optional functional-scatter geometry
     * @param foldGeometry non-null optional overlap-fold geometry
     * @param orderingGeometry non-null optional stable ordering geometry
     * @param randomGeometry non-null optional explicit-state random geometry
     * @param scanGeometry non-null optional cumulative-scan geometry
     * @param aggregateGeometry non-null optional ordinary-aggregate geometry
     * @param argExtremaGeometry non-null optional arg-extrema geometry
     * @param maskedReductionGeometry non-null optional masked-reduction geometry
     * @param advancedReductionGeometry non-null optional advanced-reduction geometry
     * @param softmaxGeometry non-null optional stable-softmax geometry
     * @param trailingNormalizationGeometry non-null optional trailing Layer/RMS geometry
     * @param batchNormInferenceGeometry non-null optional zero-workspace batch-normalization
     *     inference geometry
     * @throws NullPointerException if a required component is {@code null}
     * @throws IllegalArgumentException if the plan is not one portable unit with matching derived
     *     boundary facts, or if strategy, range, materialization, workspace, species, or budget
     *     facts disagree
     */
    public CpuPartitionPreparationPlan {
        units = List.copyOf(units);
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        bufferDeclarations = List.copyOf(bufferDeclarations);
        boundaryValues = List.copyOf(boundaryValues);
        accessBindings = List.copyOf(accessBindings);
        carrierPattern = List.copyOf(carrierPattern);
        generatedCarrierPattern = List.copyOf(generatedCarrierPattern);
        extents = extents.clone();
        affineAddressPairs = affineAddressPairs.clone();
        Objects.requireNonNull(loweringManifest, "loweringManifest");
        materialization = Objects.requireNonNull(materialization, "materialization");
        workspaceDeclaration = Objects.requireNonNull(workspaceDeclaration, "workspaceDeclaration");
        Objects.requireNonNull(workspaceUse, "workspaceUse");
        Objects.requireNonNull(specializationBudget, "specializationBudget");
        movementGeometry = Objects.requireNonNull(movementGeometry, "movementGeometry");
        indexingGeometry = Objects.requireNonNull(indexingGeometry, "indexingGeometry");
        scatterGeometry = Objects.requireNonNull(scatterGeometry, "scatterGeometry");
        foldGeometry = Objects.requireNonNull(foldGeometry, "foldGeometry");
        orderingGeometry = Objects.requireNonNull(orderingGeometry, "orderingGeometry");
        randomGeometry = Objects.requireNonNull(randomGeometry, "randomGeometry");
        scanGeometry = Objects.requireNonNull(scanGeometry, "scanGeometry");
        aggregateGeometry = Objects.requireNonNull(aggregateGeometry, "aggregateGeometry");
        argExtremaGeometry = Objects.requireNonNull(argExtremaGeometry, "argExtremaGeometry");
        maskedReductionGeometry = Objects.requireNonNull(maskedReductionGeometry,
                "maskedReductionGeometry");
        advancedReductionGeometry = Objects.requireNonNull(advancedReductionGeometry,
                "advancedReductionGeometry");
        softmaxGeometry = Objects.requireNonNull(softmaxGeometry, "softmaxGeometry");
        trailingNormalizationGeometry = Objects.requireNonNull(trailingNormalizationGeometry,
                "trailingNormalizationGeometry");
        batchNormInferenceGeometry = Objects.requireNonNull(batchNormInferenceGeometry,
                "batchNormInferenceGeometry");
        if (units.size() != 1 || route != Route.PORTABLE
                || bufferDeclarations.isEmpty()
                || boundaryValues.size() != bufferDeclarations.size()
                || accessBindings.size() != bufferDeclarations.size()
                || carrierPattern.size() != bufferDeclarations.size()
                || generatedCarrierPattern.size() != bufferDeclarations.size()) {
                throw new IllegalArgumentException("CPU plan must contain one portable unit and matching boundaries");
        }
        boolean affine = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
        boolean movement = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
        boolean indexing = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
        boolean scatter = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
        boolean fold = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
        boolean ordering = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
        boolean random = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
        boolean scan = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
        boolean aggregate = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
        boolean argExtrema = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
        boolean maskedReduction = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
        boolean advancedReduction = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
        boolean softmax = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
        boolean trailingNormalization = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
        boolean batchNormalization = units.getFirst().portablePlan().portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
        if (affine ? affineAddressPairs.length != Math.multiplyExact(elementCount, 2)
                : affineAddressPairs.length != 0) {
            throw new IllegalArgumentException("affine address geometry must match the copy domain");
        }
        if (movement != movementGeometry.isPresent()) {
            throw new IllegalArgumentException("movement IR and cold geometry must agree");
        }
        if (indexing != indexingGeometry.isPresent() || indexing && (materialization.isPresent()
                || workspaceDeclaration.isPresent())) {
            throw new IllegalArgumentException("indexing IR and cold geometry must agree");
        }
        if (scatter != scatterGeometry.isPresent() || scatter && materialization.isPresent()) {
            throw new IllegalArgumentException("scatter IR and cold geometry must agree");
        }
        if (fold != foldGeometry.isPresent() || fold && (bufferDeclarations.size() != 2
                || materialization.isPresent() || workspaceDeclaration.isPresent())) {
            throw new IllegalArgumentException("fold IR and zero-resource geometry must agree");
        }
        if (ordering != orderingGeometry.isPresent() || ordering && materialization.isPresent())
            throw new IllegalArgumentException("ordering IR and geometry must agree");
        if (random != randomGeometry.isPresent() || random && (materialization.isPresent()
                || workspaceDeclaration.isPresent()))
            throw new IllegalArgumentException("random IR and zero-workspace geometry must agree");
        if (scan != scanGeometry.isPresent() || scan && (bufferDeclarations.size() != 2
                || materialization.isPresent() || workspaceDeclaration.isPresent()))
            throw new IllegalArgumentException("scan IR and zero-workspace geometry must agree");
        if (scan) {
            var scanIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = scanGeometry.orElseThrow();
            if (scanIr.kind() != geometry.kind() || scanIr.dataType() != geometry.dataType()
                    || scanIr.axis() != geometry.axis()
                    || scanIr.exclusive() != geometry.exclusive()
                    || scanIr.reverse() != geometry.reverse()
                    || !scanIr.inputAccess().equals(accessBindings.getFirst().plan())
                    || !scanIr.outputAccess().equals(accessBindings.getLast().plan())
                    || elementCount != geometry.sliceCount()
                    || extents.length != 1 || extents[0] != geometry.sliceCount()) {
                throw new IllegalArgumentException("scan structural IR and geometry disagree");
            }
        }
        if (aggregate != aggregateGeometry.isPresent() || aggregate && (bufferDeclarations.size() != 2
                || materialization.isPresent()))
            throw new IllegalArgumentException("aggregate IR and workspace geometry must agree");
        if (aggregate) {
            var aggregateIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = aggregateGeometry.orElseThrow();
            if (aggregateIr.kind() != geometry.kind() || aggregateIr.dataType() != geometry.dataType()
                    || aggregateIr.form() != geometry.form()
                    || aggregateIr.keepDimensions() != geometry.keepDimensions()
                    || !java.util.Arrays.equals(aggregateIr.selectedAxes(), geometry.selectedAxes())
                    || !aggregateIr.inputAccess().equals(accessBindings.getFirst().plan())
                    || !aggregateIr.outputAccess().equals(accessBindings.getLast().plan())
                    || elementCount != geometry.outputCount() || extents.length != 1
                    || extents[0] != geometry.outputCount())
                throw new IllegalArgumentException("aggregate structural IR and geometry disagree");
        }
        if (argExtrema != argExtremaGeometry.isPresent() || argExtrema
                && (bufferDeclarations.size() != 2 || materialization.isPresent()
                    || workspaceDeclaration.isPresent())) {
            throw new IllegalArgumentException("arg-extrema IR and zero-resource geometry must agree");
        }
        if (argExtrema) {
            var argIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = argExtremaGeometry.orElseThrow();
            if (argIr.kind() != geometry.kind() || argIr.inputType() != geometry.inputType()
                    || argIr.axis() != geometry.axis()
                    || argIr.keepDimensions() != geometry.keepDimensions()
                    || argIr.tiePolicy() != geometry.tiePolicy()
                    || !argIr.inputAccess().equals(accessBindings.getFirst().plan())
                    || !argIr.outputAccess().equals(accessBindings.getLast().plan())
                    || elementCount != geometry.outputCount() || extents.length != 1
                    || extents[0] != geometry.outputCount()) {
                throw new IllegalArgumentException("arg-extrema structural IR and geometry disagree");
            }
        }
        if (maskedReduction != maskedReductionGeometry.isPresent() || maskedReduction
                && (bufferDeclarations.size() != 3 || materialization.isPresent())) {
            throw new IllegalArgumentException(
                    "masked-reduction IR and exact-state geometry must agree");
        }
        if (maskedReduction) {
            var maskedIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = maskedReductionGeometry.orElseThrow();
            if (maskedIr.kind() != geometry.kind()
                    || maskedIr.dataType() != geometry.dataType()
                    || maskedIr.axis() != geometry.axis()
                    || !maskedIr.dataAccess().equals(accessBindings.get(0).plan())
                    || !maskedIr.maskAccess().equals(accessBindings.get(1).plan())
                    || !maskedIr.outputAccess().equals(accessBindings.get(2).plan())
                    || elementCount != geometry.outputCount() || extents.length != 1
                    || extents[0] != geometry.outputCount()) {
                throw new IllegalArgumentException(
                        "masked-reduction structural IR and geometry disagree");
            }
        }
        if (advancedReduction != advancedReductionGeometry.isPresent()
                || advancedReduction && (bufferDeclarations.size() != 2
                    || materialization.isPresent())) {
            throw new IllegalArgumentException(
                    "advanced-reduction IR and cold geometry must agree");
        }
        if (advancedReduction) {
            var advancedIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = advancedReductionGeometry.orElseThrow();
            if (advancedIr.kind() != geometry.kind() || advancedIr.dataType() != geometry.dataType()
                    || advancedIr.keepDimensions() != geometry.keepDimensions()
                    || advancedIr.correction() != geometry.correction()
                    || !java.util.Arrays.equals(advancedIr.orderedAxes(), geometry.orderedAxes())
                    || !java.util.Arrays.equals(advancedIr.selectedAxes(), geometry.selectedAxes())
                    || elementCount != geometry.outputCount() || extents.length != 1
                    || extents[0] != geometry.outputCount()) {
                throw new IllegalArgumentException(
                        "advanced-reduction structural IR and geometry disagree");
            }
        }
        if (softmax != softmaxGeometry.isPresent() || softmax && (bufferDeclarations.size() != 2
                || materialization.isPresent() || workspaceDeclaration.isPresent()))
            throw new IllegalArgumentException("softmax IR and zero-resource geometry must agree");
        if (softmax) {
            var softmaxIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = softmaxGeometry.orElseThrow();
            if (softmaxIr.kind() != geometry.kind() || softmaxIr.dataType() != geometry.dataType()
                    || softmaxIr.axis() != geometry.axis()
                    || !softmaxIr.inputAccess().equals(accessBindings.getFirst().plan())
                    || !softmaxIr.outputAccess().equals(accessBindings.getLast().plan())
                    || elementCount != geometry.sliceCount() || extents.length != 1
                    || extents[0] != geometry.sliceCount())
                throw new IllegalArgumentException("softmax structural IR and geometry disagree");
        }
        if (trailingNormalization != trailingNormalizationGeometry.isPresent()
                || trailingNormalization && materialization.isPresent())
            throw new IllegalArgumentException(
                    "trailing-normalization IR and cold geometry must agree");
        if (trailingNormalization) {
            var normalizationIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = trailingNormalizationGeometry.orElseThrow();
            if (normalizationIr.kind() != geometry.kind()
                    || normalizationIr.form() != geometry.form()
                    || normalizationIr.resultType() != geometry.resultType()
                    || normalizationIr.epsilonBits() != geometry.epsilonBits()
                    || normalizationIr.normalizedRank() != geometry.normalizedRank()
                    || normalizationIr.normalizedCount() != geometry.normalizedCount()
                    || !normalizationIr.positionToBoundary().equals(geometry.positionToBoundary())
                    || elementCount != (geometry.normalizedCount() == 0 ? 0 : geometry.leadingCount())
                    || extents.length != 1 || extents[0] != elementCount)
                throw new IllegalArgumentException(
                        "trailing-normalization structural IR and geometry disagree");
        }
        if (batchNormalization != batchNormInferenceGeometry.isPresent()
                || batchNormalization && (materialization.isPresent()
                    || workspaceDeclaration.isPresent())) {
            throw new IllegalArgumentException(
                    "batch-normalization IR and zero-resource geometry must agree");
        }
        if (batchNormalization) {
            var batchIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = batchNormInferenceGeometry.orElseThrow();
            if (batchIr.resultType() != geometry.resultType()
                    || batchIr.epsilonBits() != geometry.epsilonBits()
                    || batchIr.inputRank() != geometry.output().extents().length
                    || batchIr.channelAxis() != geometry.channelAxis()
                    || batchIr.rangeForm() != geometry.rangeForm()
                    || !batchIr.positionToBoundary().equals(geometry.positionToBoundary())
                    || elementCount != geometry.rangeItemCount()
                    || extents.length != 1 || extents[0] != elementCount) {
                throw new IllegalArgumentException(
                        "batch-normalization structural IR and geometry disagree");
            }
        }
        if (fold) {
            var foldIr = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr)
                    units.getFirst().portablePlan().portableKernelIr();
            var geometry = foldGeometry.orElseThrow();
            if (foldIr.family() != geometry.family() || foldIr.dataType() != geometry.dataType()
                    || !foldIr.inputAccess().equals(accessBindings.getFirst().plan())
                    || !foldIr.outputAccess().equals(accessBindings.getLast().plan())
                    || !java.util.Arrays.equals(extents, geometry.outputExtents())) {
                throw new IllegalArgumentException("fold structural IR and geometry disagree");
            }
        }
        boolean vector = executionStrategy.compute() == ExecutionStrategy.Compute.VECTOR;
        boolean parallel = executionStrategy.orchestration() == ExecutionStrategy.Orchestration.PARALLEL;
        if (selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                || parallel != (selectedRangeCount >= 2)
                || vector != (vectorSpeciesBitSize > 0)
                || units.getFirst().portablePlan().specialization().vectorSpeciesBitSize()
                    != vectorSpeciesBitSize) {
            throw new IllegalArgumentException("portable strategy facts are inconsistent");
        }
        WorkspaceUse expectedUse = materialization.isPresent() ? WorkspaceUse.MATERIALIZATION
                : scatterGeometry.filter(g -> g.scratchSliceBytes() > 0).isPresent()
                    ? WorkspaceUse.SCATTER_PRODUCT
                    : orderingGeometry.isPresent()
                        ? WorkspaceUse.ORDERING_INDICES
                        : aggregateGeometry.filter(g -> g.scratchSliceBytes() > 0
                                && g.outputCount() > 0).isPresent()
                            ? WorkspaceUse.AGGREGATE_EXACT_STATE
                            : maskedReductionGeometry.filter(g -> g.outputCount() > 0).isPresent()
                                ? WorkspaceUse.AGGREGATE_EXACT_STATE
                                : advancedReductionGeometry.filter(g -> g.scratchSliceBytes() > 0
                                    && g.outputCount() > 0).isPresent()
                                    ? WorkspaceUse.AGGREGATE_EXACT_STATE
                                    : trailingNormalizationGeometry.filter(g -> g.scratchSliceBytes() > 0
                                        && g.normalizedCount() > 0).isPresent()
                                        ? WorkspaceUse.AGGREGATE_EXACT_STATE : WorkspaceUse.NONE;
        if (workspaceUse != expectedUse
                || workspaceDeclaration.isPresent() != (workspaceUse != WorkspaceUse.NONE)) {
            throw new IllegalArgumentException("workspace purpose and declaration must agree");
        }
        if (workspaceUse == WorkspaceUse.SCATTER_PRODUCT) {
            var geometry = scatterGeometry.orElseThrow();
            var workspace = workspaceDeclaration.orElseThrow();
            if (workspace.requirementId() != 0 || workspace.byteAlignment() != Long.BYTES
                    || workspace.byteSize() != geometry.workspaceBytes(selectedRangeCount)) {
                throw new IllegalArgumentException("scatter product workspace facts disagree");
            }
        }
        if (workspaceUse == WorkspaceUse.ORDERING_INDICES) {
            var geometry = orderingGeometry.orElseThrow();
            var workspace = workspaceDeclaration.orElseThrow();
            if (workspace.requirementId() != 0 || workspace.byteAlignment() != Long.BYTES
                    || workspace.byteSize() != geometry.workspaceBytes(selectedRangeCount))
                throw new IllegalArgumentException("ordering workspace facts disagree");
        }
        if (workspaceUse == WorkspaceUse.AGGREGATE_EXACT_STATE) {
            var workspace = workspaceDeclaration.orElseThrow();
            long expected = aggregateGeometry.isPresent()
                    ? aggregateGeometry.orElseThrow().workspaceBytes(selectedRangeCount)
                    : maskedReductionGeometry.isPresent()
                        ? maskedReductionGeometry.orElseThrow().workspaceBytes(selectedRangeCount)
                        : advancedReductionGeometry.isPresent()
                            ? advancedReductionGeometry.orElseThrow().workspaceBytes(selectedRangeCount)
                            : trailingNormalizationGeometry.orElseThrow()
                                .workspaceBytes(selectedRangeCount);
            if (workspace.requirementId() != 0 || workspace.byteAlignment() != Long.BYTES
                    || workspace.byteSize() != expected)
                throw new IllegalArgumentException("aggregate exact-state workspace facts disagree");
        }
        if (materialization.isPresent()) {
            var copy = materialization.orElseThrow();
            var workspace = workspaceDeclaration.orElseThrow();
            if (workspace.requirementId() != copy.workspaceRequirementId()
                    || workspace.byteSize() != copy.byteCount()
                    || workspace.byteAlignment() != copy.byteAlignment()
                    || generatedCarrierPattern.get(copy.sourceBoundaryIndex())
                            != CarrierAccess.MEMORY_SEGMENT) {
                throw new IllegalArgumentException("materialization workspace facts disagree");
            }
        }
    }
    /** Returns instance geometry.
     * @return a new defensive copy of compatible extents */
    @Override public long[] extents() { return extents.clone(); }
    /**
     * Returns cold-composed affine source/result element addresses.
     *
     * @return a defensive copy of alternating source/result addresses, or an empty array for a
     *     pointwise plan
     */
    @Override public long[] affineAddressPairs() { return affineAddressPairs.clone(); }
}
