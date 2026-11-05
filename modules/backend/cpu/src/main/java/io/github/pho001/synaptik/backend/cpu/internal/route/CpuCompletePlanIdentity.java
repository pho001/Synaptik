package io.github.pho001.synaptik.backend.cpu.internal.route;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasRoutePlan;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * CPU-owned complete topology/representation identity and exact optional implementation choices.
 * This value carries immutable declarations only: no executable, native handle or physical storage.
 * Unit choices are exact non-default settings; omission retains the existing selected ordinary
 * configuration. An OpenBLAS assignment is whole-unit and cannot coexist with other overrides.
 * A present Phase-1 decision forbids new complete local overrides; absent Phase 1 permits complete
 * realizations of the already-retained topology without repeating any measured local search.
 *
 * @param topology exact existing retained topology and representation identity
 * @param units ordered non-default ordinary MATMUL realization/range overrides
 * @param nativeAssignment optional exact qualified oneDNN assignment, session-bound
 * @param openBlasAssignment optional exact qualified whole-unit OpenBLAS assignment
 */
public record CpuCompletePlanIdentity(CpuRepresentationDecision.VariantIdentity topology,
        List<UnitRealization> units, Optional<NativeAssignment> nativeAssignment,
        Optional<OpenBlasAssignment> openBlasAssignment) {
    /** Validates ordered unique unit positions and retains immutable cold facts. */
    public CpuCompletePlanIdentity {
        Objects.requireNonNull(topology, "topology");
        units = List.copyOf(units);
        nativeAssignment = Objects.requireNonNull(nativeAssignment, "nativeAssignment");
        openBlasAssignment = Objects.requireNonNull(openBlasAssignment, "openBlasAssignment");
        if (openBlasAssignment.isPresent() && (nativeAssignment.isPresent() || !units.isEmpty()))
            throw new IllegalArgumentException("OpenBLAS is one complete whole-unit alternative");
        int previous = -1;
        for (var unit : units) {
            if (unit.unitPosition() <= previous || unit.unitPosition() >= topology.topology().units().size())
                throw new IllegalArgumentException("complete unit realization positions disagree");
            previous = unit.unitPosition();
            if (nativeAssignment.stream().flatMap(assignment -> assignment.plan().units().stream())
                    .anyMatch(nativeUnit -> nativeUnit.unitPosition() == unit.unitPosition()))
                throw new IllegalArgumentException("native-replaced units cannot carry ordinary overrides");
        }
    }

    /**
     * One exact ordinary MATMUL configuration in the complete unit schedule.
     * @param unitPosition zero-based retained unit position
     * @param realization implemented fixed full-K realization
     * @param rangeCount positive exact maximum range count
     * @param vectorSpeciesBits preferred typed species in bits, zero for scalar
     */
    public record UnitRealization(int unitPosition, CpuMatmulIr.Realization realization,
            int rangeCount, int vectorSpeciesBits) {
        /** Rejects malformed positions, ranges or scalar/vector disagreement. */
        public UnitRealization {
            Objects.requireNonNull(realization, "realization");
            boolean vector = realization == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                    || realization == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
            if (unitPosition < 0 || rangeCount <= 0 || vectorSpeciesBits < 0
                    || vector != (vectorSpeciesBits > 0))
                throw new IllegalArgumentException("invalid complete MATMUL realization");
        }
    }

    /**
     * Exact whole-unit OpenBLAS route and its resource/configuration identity.
     * @param identity checked copy mask, thread/permit configuration and workspace geometry
     * @param plan immutable qualified route containing no provider handle
     */
    public record OpenBlasAssignment(CpuLocalTuningBatch.OpenBlasIdentity identity, CpuOpenBlasRoutePlan plan) {
        /** Validates resource and route identity agreement without acquiring provider resources. */
        public OpenBlasAssignment {
            new CpuLocalTuningBatch.Candidate(CpuLocalTuningBatch.RouteKind.OPENBLAS,
                    identity, Optional.of(plan), Optional.empty());
        }
    }
    /**
     * Exact native APIs, descriptors, implementation identities and declared workspace geometry.
     * @param fingerprint canonical native owner fingerprint, including session and API policy
     * @param plan immutable native assignment containing no native handle
     */
    public record NativeAssignment(String fingerprint, CpuOneDnnRoutePlan plan) {
        /** Requires a nonblank canonical fingerprint and immutable native facts. */
        public NativeAssignment {
            Objects.requireNonNull(fingerprint, "fingerprint");
            Objects.requireNonNull(plan, "plan");
            if (fingerprint.isBlank()) throw new IllegalArgumentException("blank native fingerprint");
        }
    }
}
