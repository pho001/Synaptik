package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.List;
import java.util.Objects;

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
 * @param extents non-null cold-bound compatible extents; copied defensively
 * @param elementCount checked logical element count represented by {@code extents}
 * @param selectedRangeCount positive maximum range count selected during cold analysis; one for
 *     single-thread strategies and at least two for parallel strategies
 * @param minimumElementsPerWorker positive minimum logical elements per submitted worker chunk
 * @param vectorSpeciesBitSize exact positive preferred FLOAT64 species size in bits for vector
 *     strategies, or zero for scalar strategies
 * @param loweringManifest non-null optional cold diagnostic text, empty when disabled
 */
public record CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
        ExecutionStrategy executionStrategy,
        List<PreparationResourceRequirement.Buffer> bufferDeclarations,
        List<ValueId> boundaryValues, List<CpuAccessPlan.Binding> accessBindings,
        List<CarrierAccess> carrierPattern, long[] extents, long elementCount,
        int selectedRangeCount, long minimumElementsPerWorker, int vectorSpeciesBitSize,
        String loweringManifest)
        implements BackendPreparationPlan {
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
     * @param bufferDeclarations non-null four-boundary declarations; copied defensively
     * @param boundaryValues non-null four materialized values in declaration order; copied
     * @param accessBindings non-null four normalized cold bindings in boundary order; copied
     * @param carrierPattern non-null four direct carrier forms in boundary order; copied
     * @param extents non-null compatible iteration extents; copied defensively
     * @param elementCount checked logical element count represented by {@code extents}
     * @param selectedRangeCount positive maximum selected range count
     * @param minimumElementsPerWorker positive minimum elements per submitted worker chunk
     * @param vectorSpeciesBitSize positive preferred FLOAT64 species bit size for vector compute,
     *     or zero for scalar compute
     * @param loweringManifest non-null optional cold diagnostic text
     * @throws NullPointerException if a required component is {@code null}
     * @throws IllegalArgumentException if the plan is outside the current one-unit portable,
     *     four-buffer proving slice or its strategy/range/species facts disagree
     */
    public CpuPartitionPreparationPlan {
        units = List.copyOf(units);
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        bufferDeclarations = List.copyOf(bufferDeclarations);
        boundaryValues = List.copyOf(boundaryValues);
        accessBindings = List.copyOf(accessBindings);
        carrierPattern = List.copyOf(carrierPattern);
        extents = extents.clone();
        Objects.requireNonNull(loweringManifest, "loweringManifest");
        if (units.size() != 1 || route != Route.PORTABLE
                || bufferDeclarations.size() != 4 || boundaryValues.size() != 4
                || accessBindings.size() != 4 || carrierPattern.size() != 4) {
            throw new IllegalArgumentException("current CPU plan must contain one portable unit and four buffers");
        }
        boolean vector = executionStrategy.compute() == ExecutionStrategy.Compute.VECTOR;
        boolean parallel = executionStrategy.orchestration() == ExecutionStrategy.Orchestration.PARALLEL;
        if (selectedRangeCount <= 0 || minimumElementsPerWorker <= 0
                || parallel != (selectedRangeCount >= 2)
                || vector != (vectorSpeciesBitSize > 0)) {
            throw new IllegalArgumentException("portable strategy facts are inconsistent");
        }
    }
    /** Returns instance geometry.
     * @return a new defensive copy of compatible extents */
    @Override public long[] extents() { return extents.clone(); }
}
