package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
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
 * @param extents non-null cold-bound compatible extents; copied defensively
 * @param elementCount checked logical element count represented by {@code extents}
 * @param loweringManifest non-null optional cold diagnostic text, empty when disabled
 */
public record CpuPartitionPreparationPlan(List<ExecutionUnitPlan> units, Route route,
        ExecutionStrategy executionStrategy,
        List<PreparationResourceRequirement.Buffer> bufferDeclarations,
        List<ValueId> boundaryValues, long[] extents, long elementCount, String loweringManifest)
        implements BackendPreparationPlan {
    /**
     * One computation-oriented execution unit.
     *
     * @param portablePlan non-null already-lowered portable realization plan
     * @param fusionReason non-null cold diagnostic explanation of the selected fusion
     */
    public record ExecutionUnitPlan(CpuPortableRoutePlan portablePlan, String fusionReason) {
        /** @throws NullPointerException if either component is {@code null} */
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
            /** Vector API element computation, reserved for a later task. */ VECTOR
        }
        /** Orchestration axis. */
        public enum Orchestration {
            /** Invocation on one orchestrating thread. */ SINGLE_THREAD,
            /** External chunk dispatch, reserved for a later task. */ PARALLEL
        }
        /** The sole strategy implemented by CPU 0005A. */
        public static final ExecutionStrategy SCALAR =
                new ExecutionStrategy(Compute.SCALAR, Orchestration.SINGLE_THREAD);
        /** Validates both axes. @throws NullPointerException if either axis is {@code null} */
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
     * @throws NullPointerException if a required component is {@code null}
     * @throws IllegalArgumentException if the plan is outside the one-unit, scalar, portable,
     *     four-buffer CPU 0005A slice
     */
    public CpuPartitionPreparationPlan {
        units = List.copyOf(units);
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(executionStrategy, "executionStrategy");
        bufferDeclarations = List.copyOf(bufferDeclarations);
        boundaryValues = List.copyOf(boundaryValues);
        extents = extents.clone();
        Objects.requireNonNull(loweringManifest, "loweringManifest");
        if (units.size() != 1 || route != Route.PORTABLE
                || !executionStrategy.equals(ExecutionStrategy.SCALAR)
                || bufferDeclarations.size() != 4 || boundaryValues.size() != 4) {
            throw new IllegalArgumentException("CPU 0005A plan must contain one scalar portable unit and four buffers");
        }
    }
    /** Returns instance geometry.
     * @return a new defensive copy of compatible extents */
    @Override public long[] extents() { return extents.clone(); }
}
