package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable selected OpenBLAS SGEMM/DGEMM plan produced after common CPU lowering.
 * The plan contains exact logical geometry, stable boundary positions, one externally coordinated
 * thread configuration, at most one existing affine-copy materialization, and the checked
 * whole-plan comparison. It owns no provider, segment, slot, graph object, or mutable resource.
 *
 * @param dataType exact same FLOAT32 or FLOAT64 operand and result type
 * @param m positive provider-representable output row count
 * @param n positive provider-representable output column count
 * @param k positive provider-representable contraction count
 * @param leftBoundaryPosition exact left-input position in the common boundary list
 * @param rightBoundaryPosition exact right-input position in the common boundary list
 * @param outputBoundaryPosition exact output position in the common boundary list
 * @param threadConfiguration exact selected single-thread provider configuration
 * @param representation direct or one-input-copy representation choice
 * @param materialization exact existing affine-copy plan when one input is copied
 * @param workspaceRequirement exact run-owned copy workspace declaration when one input is copied
 * @param portableCost checked complete portable cost across expected runs
 * @param openBlasCost checked complete native plan cost across expected runs
 * @param netBenefit checked {@code portableCost - openBlasCost}
 * @param benefitBasisPoints checked non-negative benefit relative to portable cost
 */
public record CpuOpenBlasRoutePlan(DataType dataType, int m, int n, int k,
        int leftBoundaryPosition, int rightBoundaryPosition, int outputBoundaryPosition,
        CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadConfiguration threadConfiguration,
        Representation representation, Optional<CpuMaterializationPlan> materialization,
        Optional<PreparationResourceRequirement.Workspace> workspaceRequirement,
        long portableCost, long openBlasCost, long netBenefit, int benefitBasisPoints) {

    /** Closed representation choice for the first native route. */
    public enum Representation {
        /** All three matrices use their directly bound native segments. */ DIRECT,
        /** The left input is copied once into the declared native workspace. */ COPY_LEFT,
        /** The right input is copied once into the declared native workspace. */ COPY_RIGHT
    }

    /**
     * Validates all route, representation, resource, and cost invariants.
     *
     * @throws NullPointerException if a required component is {@code null}
     * @throws IllegalArgumentException if type, geometry, boundary, thread, representation,
     *     resource, or cost facts disagree
     * @throws ArithmeticException if supplied cost facts overflow their checked relationship
     */
    public CpuOpenBlasRoutePlan {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(threadConfiguration, "threadConfiguration");
        Objects.requireNonNull(representation, "representation");
        materialization = Objects.requireNonNull(materialization, "materialization");
        workspaceRequirement = Objects.requireNonNull(workspaceRequirement,
                "workspaceRequirement");
        int copiedBoundary = representation == Representation.COPY_LEFT
                ? leftBoundaryPosition : representation == Representation.COPY_RIGHT
                    ? rightBoundaryPosition : -1;
        boolean copied = representation != Representation.DIRECT;
        if ((dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64)
                || m <= 0 || n <= 0 || k <= 0
                || leftBoundaryPosition != 0 || rightBoundaryPosition != 1
                || outputBoundaryPosition != 2
                || threadConfiguration
                    != CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadConfiguration
                            .SINGLE_THREAD
                || materialization.isPresent() != copied
                || workspaceRequirement.isPresent() != copied
                || portableCost < 0 || openBlasCost < 0 || openBlasCost >= portableCost
                || netBenefit <= 0 || netBenefit != Math.subtractExact(portableCost, openBlasCost)
                || benefitBasisPoints < 0 || benefitBasisPoints > 10_000
                || benefitBasisPoints != Math.toIntExact(Math.floorDiv(
                        Math.multiplyExact(10_000L, netBenefit), portableCost))) {
            throw new IllegalArgumentException("OpenBLAS route-plan facts disagree");
        }
        if (copied) {
            CpuMaterializationPlan copy = materialization.orElseThrow();
            PreparationResourceRequirement.Workspace workspace = workspaceRequirement.orElseThrow();
            if (copy.sourceBoundaryIndex() != copiedBoundary || copy.dataType() != dataType
                    || copy.consumers().size() != 1
                    || copy.consumers().getFirst().unitPosition() != 0
                    || copy.consumers().getFirst().boundaryPosition() != copiedBoundary
                    || workspace.requirementId() != copy.workspaceRequirementId()
                    || workspace.byteSize() != copy.byteCount()
                    || workspace.byteAlignment() != copy.byteAlignment()) {
                throw new IllegalArgumentException("OpenBLAS copy and workspace facts disagree");
            }
        }
    }

    /** Returns the boundary position whose effective input comes from the copy workspace.
     * @return zero or one when copied, or {@code -1} for direct execution */
    public int copiedBoundaryPosition() {
        return switch (representation) {
            case DIRECT -> -1;
            case COPY_LEFT -> leftBoundaryPosition;
            case COPY_RIGHT -> rightBoundaryPosition;
        };
    }
}
