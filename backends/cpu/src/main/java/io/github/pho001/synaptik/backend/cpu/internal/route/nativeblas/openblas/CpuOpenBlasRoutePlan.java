package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable selected OpenBLAS SGEMM/DGEMM plan with the complete bounded representation
 * transition and checked whole-plan comparison. It owns no provider, segment, slot, or mutable
 * resource.
 *
 * @param dataType exact common FLOAT32 or FLOAT64 matrix type
 * @param m positive output row count
 * @param n positive output column count
 * @param k positive contraction extent
 * @param leftBoundaryPosition stable left-input boundary position, exactly zero
 * @param rightBoundaryPosition stable right-input boundary position, exactly one
 * @param outputBoundaryPosition stable output boundary position, exactly two
 * @param threadCandidate complete selected provider-thread cost candidate retained by identity
 * @param threadCount fixed positive provider thread count
 * @param permitDemand fixed positive shared-budget demand, identical to {@code threadCount}
 * @param analysisCapacity positive analysis-time CPU capacity snapshot
 * @param candidateOrder stable zero-based position in the configured candidate list, used only
 *     as the final tie-break after cost and lower thread count
 * @param representation exact three-boundary copy mask
 * @param leftMaterialization optional external-read copy for the left input
 * @param rightMaterialization optional external-read copy for the right input
 * @param outputCopy optional route-local copy from the canonical result workspace
 * @param workspaceRequirements immutable left, right, then output workspace declarations for the
 *     selected copies
 * @param expectedRunCount positive run count used by the complete cost calculation
 * @param workspaceBytes exact sum of selected workspace bytes
 * @param inputCopiedElements exact sum of copied left and right elements
 * @param outputCopiedElements exact copied output elements, or zero
 * @param portableCost checked total portable cost across expected runs
 * @param openBlasCost checked total OpenBLAS and representation cost across expected runs
 * @param netBenefit exact positive {@code portableCost - openBlasCost}
 * @param benefitBasisPoints floored relative benefit in {@code [0, 10_000]}
 * @param qualification exact successful session credential retained for live finalization
 */
public record CpuOpenBlasRoutePlan(DataType dataType, int m, int n, int k,
        int leftBoundaryPosition, int rightBoundaryPosition, int outputBoundaryPosition,
        CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate threadCandidate,
        int threadCount, int permitDemand, int analysisCapacity, int candidateOrder,
        Representation representation, Optional<CpuMaterializationPlan> leftMaterialization,
        Optional<CpuMaterializationPlan> rightMaterialization,
        Optional<CpuOpenBlasOutputCopyPlan> outputCopy,
        List<PreparationResourceRequirement.Workspace> workspaceRequirements,
        long expectedRunCount, long workspaceBytes, long inputCopiedElements,
        long outputCopiedElements, long portableCost, long openBlasCost, long netBenefit,
        int benefitBasisPoints, Optional<CpuOpenBlasQualification> qualification) {

    /**
     * Compatibility constructor for direct executable tests; an unqualified plan cannot pass
     * coordinated finalization.
     *
     * @param dataType exact common FLOAT32 or FLOAT64 matrix type
     * @param m positive output row count
     * @param n positive output column count
     * @param k positive contraction extent
     * @param leftBoundaryPosition stable left-input boundary position, exactly zero
     * @param rightBoundaryPosition stable right-input boundary position, exactly one
     * @param outputBoundaryPosition stable output boundary position, exactly two
     * @param threadCandidate complete selected provider-thread cost candidate
     * @param threadCount fixed positive provider thread count
     * @param permitDemand fixed positive shared-budget demand equal to {@code threadCount}
     * @param analysisCapacity positive analysis-time CPU capacity snapshot
     * @param candidateOrder stable non-negative configured-candidate position
     * @param representation exact three-boundary copy mask
     * @param leftMaterialization optional left-input copy
     * @param rightMaterialization optional right-input copy
     * @param outputCopy optional route-local output copy
     * @param workspaceRequirements ordered declarations for every selected copy
     * @param expectedRunCount positive run count used by cost calculation
     * @param workspaceBytes exact total selected workspace bytes
     * @param inputCopiedElements exact total copied input elements
     * @param outputCopiedElements exact copied output elements
     * @param portableCost checked total portable cost
     * @param openBlasCost checked lower OpenBLAS cost
     * @param netBenefit exact positive portable-minus-OpenBLAS cost
     * @param benefitBasisPoints floored relative benefit in {@code [0, 10_000]}
     * @throws NullPointerException if a required reference or workspace entry is {@code null}
     * @throws IllegalArgumentException if semantic, geometry, representation, resource, or cost
     *     facts disagree
     * @throws ArithmeticException if exact validation arithmetic overflows
     */
    public CpuOpenBlasRoutePlan(DataType dataType, int m, int n, int k,
            int leftBoundaryPosition, int rightBoundaryPosition, int outputBoundaryPosition,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate threadCandidate,
            int threadCount, int permitDemand, int analysisCapacity, int candidateOrder,
            Representation representation, Optional<CpuMaterializationPlan> leftMaterialization,
            Optional<CpuMaterializationPlan> rightMaterialization,
            Optional<CpuOpenBlasOutputCopyPlan> outputCopy,
            List<PreparationResourceRequirement.Workspace> workspaceRequirements,
            long expectedRunCount, long workspaceBytes, long inputCopiedElements,
            long outputCopiedElements, long portableCost, long openBlasCost, long netBenefit,
            int benefitBasisPoints) {
        this(dataType, m, n, k, leftBoundaryPosition, rightBoundaryPosition,
                outputBoundaryPosition, threadCandidate, threadCount, permitDemand,
                analysisCapacity, candidateOrder, representation, leftMaterialization,
                rightMaterialization, outputCopy, workspaceRequirements, expectedRunCount,
                workspaceBytes, inputCopiedElements, outputCopiedElements, portableCost,
                openBlasCost, netBenefit, benefitBasisPoints, Optional.empty());
    }

    /**
     * Preserves the CPU 0010 direct/one-input construction surface for focused tests.
     *
     * @param dataType exact common FLOAT32 or FLOAT64 type
     * @param m positive output row count
     * @param n positive output column count
     * @param k positive contraction extent
     * @param leftBoundaryPosition left boundary position, exactly zero
     * @param rightBoundaryPosition right boundary position, exactly one
     * @param outputBoundaryPosition output boundary position, exactly two
     * @param threadCandidate externally coordinated count-one candidate
     * @param representation direct, copy-left, or copy-right representation
     * @param materialization selected sole input copy, if any
     * @param workspaceRequirement selected sole workspace, if any
     * @param portableCost checked positive portable total
     * @param openBlasCost checked lower OpenBLAS total
     * @param netBenefit exact positive cost difference
     * @param benefitBasisPoints floored relative benefit
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if geometry, representation, resource, or cost facts
     *     disagree
     * @throws ArithmeticException if derived totals overflow
     */
    public CpuOpenBlasRoutePlan(DataType dataType, int m, int n, int k,
            int leftBoundaryPosition, int rightBoundaryPosition, int outputBoundaryPosition,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate threadCandidate,
            Representation representation, Optional<CpuMaterializationPlan> materialization,
            Optional<PreparationResourceRequirement.Workspace> workspaceRequirement,
            long portableCost, long openBlasCost, long netBenefit, int benefitBasisPoints) {
        this(dataType, m, n, k, leftBoundaryPosition, rightBoundaryPosition,
                outputBoundaryPosition, threadCandidate, 1, 1, 1, 0, representation,
                representation == Representation.COPY_LEFT ? materialization : Optional.empty(),
                representation == Representation.COPY_RIGHT ? materialization : Optional.empty(),
                Optional.empty(), workspaceRequirement.stream().toList(),
                materialization.map(CpuMaterializationPlan::expectedRunCount).orElse(1L),
                workspaceRequirement.map(PreparationResourceRequirement.Workspace::byteSize)
                        .orElse(0L),
                materialization.map(CpuMaterializationPlan::elementCount).orElse(0L), 0,
                portableCost, openBlasCost, netBenefit, benefitBasisPoints, Optional.empty());
    }

    /** Stable closed copy-mask order for the bounded route. */
    public enum Representation {
        /** No copies. */ DIRECT(false, false, false),
        /** Copy left input. */ COPY_LEFT(true, false, false),
        /** Copy right input. */ COPY_RIGHT(false, true, false),
        /** Copy provider result to the logical output. */ COPY_OUTPUT(false, false, true),
        /** Copy both inputs. */ COPY_LEFT_RIGHT(true, true, false),
        /** Copy left input and output. */ COPY_LEFT_OUTPUT(true, false, true),
        /** Copy right input and output. */ COPY_RIGHT_OUTPUT(false, true, true),
        /** Copy both inputs and output. */ COPY_LEFT_RIGHT_OUTPUT(true, true, true);

        private final boolean left;
        private final boolean right;
        private final boolean output;
        Representation(boolean left, boolean right, boolean output) {
            this.left = left; this.right = right; this.output = output;
        }
        /** Reports whether the left boundary needs canonicalization.
         * @return whether the left input is copied */
        public boolean copiesLeft() { return left; }
        /** Reports whether the right boundary needs canonicalization.
         * @return whether the right input is copied */
        public boolean copiesRight() { return right; }
        /** Reports whether GEMM writes a route-owned output workspace.
         * @return whether the output is copied */
        public boolean copiesOutput() { return output; }
        /** Counts the distinct workspaces required by this mask.
         * @return the total number of copied boundaries */
        public int copyCount() { return (left ? 1 : 0) + (right ? 1 : 0) + (output ? 1 : 0); }
    }

    /**
     * Validates and snapshots route, copy-mask, resource, geometry, expected-run, and cost facts.
     * Optional copy plans and the workspace list are never {@code null}; the list is copied and
     * every selected copy must own its correspondingly ordered, distinct workspace declaration.
     *
     * @throws NullPointerException if a required reference or workspace element is {@code null}
     * @throws IllegalArgumentException if semantic, copy-mask, resource, geometry, expected-run,
     *     or cost invariants disagree
     * @throws ArithmeticException if exact aggregate or relative-benefit arithmetic overflows
     */
    public CpuOpenBlasRoutePlan {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(threadCandidate, "threadCandidate");
        Objects.requireNonNull(representation, "representation");
        leftMaterialization = Objects.requireNonNull(leftMaterialization,
                "leftMaterialization");
        rightMaterialization = Objects.requireNonNull(rightMaterialization,
                "rightMaterialization");
        outputCopy = Objects.requireNonNull(outputCopy, "outputCopy");
        qualification = Objects.requireNonNull(qualification, "qualification");
        workspaceRequirements = List.copyOf(workspaceRequirements);
        if ((dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64)
                || m <= 0 || n <= 0 || k <= 0
                || leftBoundaryPosition != 0 || rightBoundaryPosition != 1
                || outputBoundaryPosition != 2
                || threadCount <= 0 || permitDemand != threadCount
                || threadCount > analysisCapacity || analysisCapacity <= 0 || candidateOrder < 0
                || threadCandidate.threadCount() != threadCount
                || leftMaterialization.isPresent() != representation.copiesLeft()
                || rightMaterialization.isPresent() != representation.copiesRight()
                || outputCopy.isPresent() != representation.copiesOutput()
                || workspaceRequirements.size() != representation.copyCount()
                || expectedRunCount <= 0 || workspaceBytes < 0 || inputCopiedElements < 0
                || outputCopiedElements < 0 || portableCost < 0 || openBlasCost < 0
                || openBlasCost >= portableCost || netBenefit <= 0
                || netBenefit != Math.subtractExact(portableCost, openBlasCost)
                || benefitBasisPoints < 0 || benefitBasisPoints > 10_000
                || portableCost == 0
                || benefitBasisPoints != Math.toIntExact(Math.floorDiv(
                        Math.multiplyExact(10_000L, netBenefit), portableCost))) {
            throw new IllegalArgumentException("OpenBLAS route-plan facts disagree");
        }
        var copies = new ArrayList<CpuMaterializationPlan>(2);
        leftMaterialization.ifPresent(copies::add);
        rightMaterialization.ifPresent(copies::add);
        long actualInputElements = 0;
        long actualBytes = 0;
        int workspaceIndex = 0;
        for (CpuMaterializationPlan copy : copies) {
            int boundary = workspaceIndex == 0 && representation.copiesLeft()
                    ? leftBoundaryPosition : rightBoundaryPosition;
            var workspace = workspaceRequirements.get(workspaceIndex++);
            if (copy.sourceBoundaryIndex() != boundary || copy.dataType() != dataType
                    || copy.expectedRunCount() != expectedRunCount
                    || copy.consumers().size() != 1
                    || copy.consumers().getFirst().unitPosition() != 0
                    || copy.consumers().getFirst().boundaryPosition() != boundary
                    || workspace.requirementId() != copy.workspaceRequirementId()
                    || workspace.byteSize() != copy.byteCount()
                    || workspace.byteAlignment() != copy.byteAlignment()) {
                throw new IllegalArgumentException("OpenBLAS input-copy facts disagree");
            }
            actualInputElements = Math.addExact(actualInputElements, copy.elementCount());
            actualBytes = Math.addExact(actualBytes, copy.byteCount());
        }
        if (outputCopy.isPresent()) {
            var copy = outputCopy.orElseThrow();
            var workspace = workspaceRequirements.get(workspaceIndex);
            if (copy.outputBoundaryPosition() != outputBoundaryPosition
                    || copy.dataType() != dataType
                    || workspace.requirementId() != copy.workspaceRequirementId()
                    || workspace.byteSize() != copy.byteCount()
                    || workspace.byteAlignment() != copy.byteAlignment()) {
                throw new IllegalArgumentException("OpenBLAS output-copy facts disagree");
            }
            actualBytes = Math.addExact(actualBytes, copy.byteCount());
        }
        long actualOutputElements = outputCopy.map(CpuOpenBlasOutputCopyPlan::elementCount)
                .orElse(0L);
        if (actualInputElements != inputCopiedElements
                || actualOutputElements != outputCopiedElements
                || actualBytes != workspaceBytes
                || workspaceRequirements.stream()
                    .map(PreparationResourceRequirement.Workspace::requirementId)
                    .distinct().count() != workspaceRequirements.size()) {
            throw new IllegalArgumentException("OpenBLAS representation totals disagree");
        }
    }

    /**
     * Collects selected external-read transitions in execution order.
     *
     * @return an immutable ordered left-then-right list of selected input materializations
     */
    public List<CpuMaterializationPlan> inputMaterializations() {
        var result = new ArrayList<CpuMaterializationPlan>(2);
        leftMaterialization.ifPresent(result::add);
        rightMaterialization.ifPresent(result::add);
        return List.copyOf(result);
    }

    /**
     * Returns the legacy single-input-copy view without reinterpreting output copy-out.
     *
     * @return the sole selected input materialization, or empty for zero or two input copies
     */
    public Optional<CpuMaterializationPlan> materialization() {
        List<CpuMaterializationPlan> copies = inputMaterializations();
        return copies.size() == 1 ? Optional.of(copies.getFirst()) : Optional.empty();
    }

    /**
     * Returns the legacy single-workspace view across both input and output transitions.
     *
     * @return the sole workspace declaration, or empty when the route declares zero or multiple
     */
    public Optional<PreparationResourceRequirement.Workspace> workspaceRequirement() {
        return workspaceRequirements.size() == 1 ? Optional.of(workspaceRequirements.getFirst())
                : Optional.empty();
    }

    /**
     * Reports the legacy single copied-input position without including output copy-out.
     *
     * @return the sole copied input position, or {@code -1} for zero or two copied inputs
     */
    public int copiedBoundaryPosition() {
        return representation.copiesLeft() == representation.copiesRight() ? -1
                : representation.copiesLeft() ? leftBoundaryPosition : rightBoundaryPosition;
    }
}
