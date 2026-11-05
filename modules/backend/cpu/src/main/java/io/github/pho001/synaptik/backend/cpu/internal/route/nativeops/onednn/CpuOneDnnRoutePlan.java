package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Selected realizations of retained common units, owning no native handles or run data.
 * Explicit finite tuning alternatives carry zero for both heuristic costs because enumeration
 * does not rank or cost-prune them; ordinary deterministic selection retains its calculated costs.
 *
 * @param sessionIdentity non-null non-blank exact library-load identity
 * @param units non-null non-empty native selections in unique topological order; copied
 * @param portableCost non-negative dimensionless complete portable/OpenBLAS peer heuristic
 * @param selectedCost non-negative dimensionless complete selected-plan heuristic, not elapsed time
 */
public record CpuOneDnnRoutePlan(String sessionIdentity, List<Unit> units,
        long portableCost, long selectedCost) {
    /** Immutable API preference; each selected Unit must resolve AUTO to one concrete API. */
    public enum Api { AUTO, PRIMITIVE, GRAPH }

    private record CopySource(io.github.pho001.synaptik.model.graph.ValueId value,
            long offset, long byteSize) { }

    /**
     * One represented source span shared by input roles and later native units. Only its first
     * topological consumer refreshes the run-owned copy; every invocation refreshes it again.
     *
     * @param inputPositions non-null non-empty unique spec-input positions sharing this span; copied
     * @param workspace non-null 64-byte-aligned declaration shared by exact matching sources
     * @param refresh whether this is the first consumer responsible for copying each run's data
     */
    public record InputCopy(List<Integer> inputPositions,
            PreparationResourceRequirement.Workspace workspace, boolean refresh) {
        /**
         * Snapshots shared operand roles without allocating a representation.
         *
         * @param inputPositions non-null non-empty unique non-negative input roles; copied
         * @param workspace non-null shared ingress declaration with 64-byte alignment
         * @param refresh first-consumer refresh ownership within each run
         * @throws NullPointerException if a required object or input position is null
         * @throws IllegalArgumentException if roles or workspace alignment are invalid
         */
        public InputCopy {
            inputPositions = List.copyOf(inputPositions);
            Objects.requireNonNull(workspace, "workspace");
            if (inputPositions.isEmpty() || inputPositions.stream().anyMatch(input -> input < 0)
                    || inputPositions.stream().distinct().count() != inputPositions.size()
                    || workspace.byteAlignment() != 64) {
                throw new IllegalArgumentException("invalid oneDNN ingress copy");
            }
        }
    }

    /**
     * One native substitution preserving a common unit's external memory obligations.
     *
     * @param unitPosition non-negative common-unit topological index
     * @param projection non-null exact numeric/boundary projection
     * @param api non-null resolved native API, never AUTO
     * @param scratchpadBytes non-negative complete queried invocation bytes, including fixed literal
     *     intermediates/constants and native per-step scratch, charged once
     * @param implementation non-null non-blank copied native implementation identity
     * @param inputCopies non-null immutable ingress role declarations; copied
     * @param outputStagingWorkspace declared F32 native destination exactly when final CAST is needed
     * @param invocationWorkspace non-null private scratch declaration, aligned to 64 bytes
     */
    public record Unit(int unitPosition, CpuOneDnnProjection projection, Api api,
            long scratchpadBytes, String implementation, List<InputCopy> inputCopies,
            PreparationResourceRequirement.Workspace invocationWorkspace,
            Optional<PreparationResourceRequirement.Workspace> outputStagingWorkspace) {
        /**
         * Authenticates descriptor, shared ingress and invocation-workspace geometry.
         *
         * @param unitPosition non-negative common-unit index
         * @param projection non-null common descriptor/boundary facts
         * @param api non-null resolved PRIMITIVE or GRAPH
         * @param scratchpadBytes non-negative complete fixed invocation-storage bytes
         * @param implementation non-null non-blank implementation identity
         * @param inputCopies non-null copy roles, snapshotted without retaining run data
         * @param invocationWorkspace non-null private 64-byte-aligned scratch declaration
         * @param outputStagingWorkspace private F32 destination declaration, absent for direct output
         * @throws NullPointerException if a required object or list element is null
         * @throws IllegalArgumentException if selection, scratch, copy roles or source spans disagree
         */
        public Unit {
            Objects.requireNonNull(projection, "projection");
            Objects.requireNonNull(api, "api");
            Objects.requireNonNull(implementation, "implementation");
            inputCopies = List.copyOf(inputCopies);
            Objects.requireNonNull(invocationWorkspace, "invocationWorkspace");
            Objects.requireNonNull(outputStagingWorkspace, "outputStagingWorkspace");
            if (projection.requiresOutputNarrowing() != outputStagingWorkspace.isPresent()
                    || outputStagingWorkspace.filter(workspace ->
                        workspace.byteSize() != projection.spec().output().byteSize()
                            || workspace.byteAlignment() != 64).isPresent()) {
                throw new IllegalArgumentException("oneDNN output staging disagrees with logical result");
            }
            if (unitPosition < 0 || api == Api.AUTO || scratchpadBytes < 0
                    || implementation.isBlank() || invocationWorkspace.byteSize() != scratchpadBytes
                    || invocationWorkspace.byteAlignment() != 64) {
                throw new IllegalArgumentException("invalid selected oneDNN unit");
            }
            var inputs = new HashSet<Integer>();
            for (InputCopy copy : inputCopies) {
                int first = copy.inputPositions().getFirst();
                if (first >= projection.spec().inputs().size()) {
                    throw new IllegalArgumentException("oneDNN copy input is outside the projection");
                }
                long origin = projection.byteOffsets().get(first);
                for (int input : copy.inputPositions()) {
                    if (input >= projection.spec().inputs().size() || !inputs.add(input)
                            || projection.byteOffsets().get(input) != origin
                            || copy.workspace().byteSize() != projection.spec().inputs().get(input).byteSize()) {
                        throw new IllegalArgumentException("oneDNN input copy disagrees with descriptor");
                    }
                }
            }
        }
        /**
         * Creates a direct-output native selection with no final destination conversion.
         * @param unitPosition common-unit index
         * @param projection direct-output numeric projection
         * @param api resolved native API
         * @param scratchpadBytes exact queried native invocation bytes
         * @param implementation retained selected implementation
         * @param inputCopies declared ingress copies
         * @param invocationWorkspace private invocation declaration
         */
        public Unit(int unitPosition, CpuOneDnnProjection projection, Api api,
                long scratchpadBytes, String implementation, List<InputCopy> inputCopies,
                PreparationResourceRequirement.Workspace invocationWorkspace) {
            this(unitPosition, projection, api, scratchpadBytes, implementation, inputCopies,
                    invocationWorkspace, Optional.empty());
        }
        /**
         * Lists ingress declarations, optional F32 result staging, then the private invocation.
         *
         * @return non-null immutable ordered declarations; shared ingress may recur across units
         */
        public List<PreparationResourceRequirement.Workspace> workspaceRequirements() {
            var result = new ArrayList<PreparationResourceRequirement.Workspace>(inputCopies.size() + 1);
            for (InputCopy copy : inputCopies) result.add(copy.workspace());
            outputStagingWorkspace.ifPresent(result::add);
            result.add(invocationWorkspace);
            return List.copyOf(result);
        }
    }

    /**
     * Validates topological order and one-refresh-owner sharing before shared assignment.
     *
     * @param sessionIdentity non-null non-blank exact native load session
     * @param units non-null non-empty selected units; copied
     * @param portableCost non-negative complete peer heuristic
     * @param selectedCost non-negative complete selected-plan heuristic
     * @throws NullPointerException if a required object or unit is null
     * @throws IllegalArgumentException if identity, costs, ordering or workspace sharing is invalid
     */
    public CpuOneDnnRoutePlan {
        Objects.requireNonNull(sessionIdentity, "sessionIdentity");
        units = List.copyOf(units);
        if (sessionIdentity.isBlank() || units.isEmpty() || portableCost < 0 || selectedCost < 0) {
            throw new IllegalArgumentException("invalid oneDNN route facts");
        }
        int previous = -1;
        var declarations = new java.util.LinkedHashMap<Long, PreparationResourceRequirement.Workspace>();
        var initialized = new HashSet<Long>();
        for (Unit unit : units) {
            if (unit.unitPosition() <= previous) {
                throw new IllegalArgumentException("oneDNN units must be unique and topological");
            }
            previous = unit.unitPosition();
            for (InputCopy copy : unit.inputCopies()) {
                long id = copy.workspace().requirementId();
                var previousDeclaration = declarations.putIfAbsent(id, copy.workspace());
                if (previousDeclaration != null && previousDeclaration != copy.workspace()
                        || copy.refresh() && !initialized.add(id)
                        || !copy.refresh() && !initialized.contains(id)) {
                    throw new IllegalArgumentException("oneDNN ingress must have one preceding refresh owner");
                }
            }
            unit.outputStagingWorkspace().ifPresent(staging -> {
                if (declarations.putIfAbsent(staging.requirementId(), staging) != null) {
                    throw new IllegalArgumentException("oneDNN output staging must be private");
                }
            });
            var invocation = unit.invocationWorkspace();
            if (declarations.putIfAbsent(invocation.requirementId(), invocation) != null) {
                throw new IllegalArgumentException("oneDNN invocation workspace must be private");
            }
        }
    }

    /**
     * Authenticates selected units against the retained common plan without relowering it.
     *
     * @param commonUnits non-null ordered original common units, borrowed without mutation
     * @throws NullPointerException if commonUnits is null
     * @throws IllegalArgumentException if a unit projection or shared copy source differs
     */
    public void validate(List<ExecutionUnitPlan> commonUnits) {
        var copies = new java.util.HashMap<Long, CopySource>();
        for (Unit unit : units) {
            if (unit.unitPosition() >= commonUnits.size()
                    || !CpuOneDnnProjection.from(commonUnits.get(unit.unitPosition()))
                            .filter(unit.projection()::equals).isPresent()) {
                throw new IllegalArgumentException("oneDNN unit is not the retained common semantic unit");
            }
            var common = commonUnits.get(unit.unitPosition());
            for (InputCopy copy : unit.inputCopies()) {
                for (int input : copy.inputPositions()) {
                    var source = new CopySource(common.boundaryValues().get(
                            unit.projection().inputBoundaryPositions().get(input)),
                            unit.projection().byteOffsets().get(input), copy.workspace().byteSize());
                    var previousSource = copies.putIfAbsent(copy.workspace().requirementId(), source);
                    if (previousSource != null && !previousSource.equals(source)) {
                        throw new IllegalArgumentException("shared oneDNN ingress refers to different source spans");
                    }
                }
            }
        }
    }

    /**
     * Deduplicates declarations by requirement identity in first-consumer order.
     *
     * @return non-null immutable shared ingress and private invocation requirements;
     *     no workspace is allocated and no ownership transfers
     */
    public List<PreparationResourceRequirement.Workspace> workspaceRequirements() {
        var declarations = new java.util.LinkedHashMap<Long, PreparationResourceRequirement.Workspace>();
        for (Unit unit : units) for (var workspace : unit.workspaceRequirements()) {
            declarations.putIfAbsent(workspace.requirementId(), workspace);
        }
        return List.copyOf(declarations.values());
    }
}
