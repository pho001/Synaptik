package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable complete portable specialization candidate selected before shared slot assignment.
 *
 * <p>The candidate binds one generated specialization to its exact family emitter, ordered shared
 * declarations, identity-bound uses, and cold invocation binder. List structure is snapshotted;
 * collaborator references are retained exactly. Construction performs no allocation, slot
 * assignment, artifact access, generation, binding, or execution.</p>
 */
final class CpuPortableKernelCandidate {
    /**
     * Aligns one generated argument with an exact declared buffer and representation position.
     *
     * @param requirement exact declared buffer requirement retained by identity
     * @param representationIndex non-negative per-run representation position
     */
    record BufferUse(
            PreparationResourceRequirement.Buffer requirement, int representationIndex) {
        /**
         * Validates one identity-bound buffer use.
         *
         * @param requirement exact non-null declared buffer requirement retained by reference
         * @param representationIndex non-negative per-run representation position
         * @throws NullPointerException if {@code requirement} is {@code null}
         * @throws IllegalArgumentException if {@code representationIndex} is negative
         */
        BufferUse {
            Objects.requireNonNull(requirement, "requirement");
            if (representationIndex < 0) {
                throw new IllegalArgumentException("representationIndex must be non-negative");
            }
        }

        /**
         * Returns the declaration used by this argument.
         *
         * @return exact non-null declared buffer requirement retained by reference
         */
        @Override public PreparationResourceRequirement.Buffer requirement() { return requirement; }

        /**
         * Returns the selected representation position within the eventual run-state buffer.
         *
         * @return non-negative representation position selected before slot assignment
         */
        @Override public int representationIndex() { return representationIndex; }
    }

    /**
     * Selects an exact declared workspace for one binder role.
     *
     * @param requirement exact declared workspace requirement retained by identity
     */
    record WorkspaceUse(PreparationResourceRequirement.Workspace requirement) {
        /**
         * Validates one identity-bound workspace use.
         *
         * @param requirement exact non-null declared workspace requirement retained by reference
         * @throws NullPointerException if {@code requirement} is {@code null}
         */
        WorkspaceUse { Objects.requireNonNull(requirement, "requirement"); }

        /**
         * Returns the declaration used by this binder role.
         *
         * @return exact non-null declared workspace requirement retained by reference
         */
        @Override public PreparationResourceRequirement.Workspace requirement() { return requirement; }
    }

    private final CpuKernelSpecialization specialization;
    private final CpuFamilyKernelEmitter familyEmitter;
    private final List<PreparationResourceRequirement> requirements;
    private final List<BufferUse> bufferUses;
    private final List<WorkspaceUse> workspaceUses;
    private final CpuPortableInvocationBinder invocationBinder;

    /**
     * Validates and snapshots one complete candidate without assigning slots or generating code.
     *
     * @param specialization non-null complete generated specialization
     * @param familyEmitter non-null emitter with the exact lowering fingerprint
     * @param requirements non-null ordered exact shared declarations
     * @param bufferUses non-null uses aligned one-for-one with specialization arguments
     * @param workspaceUses non-null ordered workspace roles
     * @param invocationBinder non-null family-owned typed cold binder
     * @throws NullPointerException if an input or indexed entry is null
     * @throws IllegalArgumentException if fingerprints, use alignment, declaration identity, or
     *     complete declaration use is invalid
     */
    CpuPortableKernelCandidate(
            CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter,
            List<PreparationResourceRequirement> requirements,
            List<BufferUse> bufferUses,
            List<WorkspaceUse> workspaceUses,
            CpuPortableInvocationBinder invocationBinder) {
        this.specialization = Objects.requireNonNull(specialization, "specialization");
        this.familyEmitter = Objects.requireNonNull(familyEmitter, "familyEmitter");
        CpuLoweringFingerprint emitterFingerprint = Objects.requireNonNull(
                familyEmitter.loweringFingerprint(), "familyEmitter.loweringFingerprint()");
        if (!emitterFingerprint.equals(specialization.loweringFingerprint())) {
            throw new IllegalArgumentException(
                    "familyEmitter lowering fingerprint does not match specialization");
        }
        this.requirements = checkedCopy(requirements, "requirements");
        var bufferIds = new HashSet<io.github.pho001.synaptik.model.graph.ValueId>();
        var workspaceIds = new HashSet<Long>();
        for (int index = 0; index < this.requirements.size(); index++) {
            switch (this.requirements.get(index)) {
                case PreparationResourceRequirement.Buffer buffer -> {
                    if (!bufferIds.add(buffer.valueId())) {
                        throw new IllegalArgumentException(
                                "requirements[" + index + "] duplicates buffer " + buffer.valueId());
                    }
                }
                case PreparationResourceRequirement.Workspace workspace -> {
                    if (!workspaceIds.add(workspace.requirementId())) {
                        throw new IllegalArgumentException(
                                "requirements[" + index + "] duplicates workspace requirementId "
                                        + workspace.requirementId());
                    }
                }
            }
        }
        this.bufferUses = checkedCopy(bufferUses, "bufferUses");
        this.workspaceUses = checkedCopy(workspaceUses, "workspaceUses");
        this.invocationBinder = Objects.requireNonNull(invocationBinder, "invocationBinder");
        if (this.bufferUses.size() != specialization.arguments().size()) {
            throw new IllegalArgumentException(
                    "bufferUses size must equal specialization argument count "
                            + specialization.arguments().size());
        }

        Set<PreparationResourceRequirement> declared =
                Collections.newSetFromMap(new IdentityHashMap<>());
        declared.addAll(this.requirements);
        Set<PreparationResourceRequirement> used =
                Collections.newSetFromMap(new IdentityHashMap<>());
        for (int index = 0; index < this.bufferUses.size(); index++) {
            var requirement = this.bufferUses.get(index).requirement();
            if (!declared.contains(requirement)) {
                throw new IllegalArgumentException(
                        "bufferUses[" + index + "].requirement is not declared");
            }
            used.add(requirement);
        }
        for (int index = 0; index < this.workspaceUses.size(); index++) {
            var requirement = this.workspaceUses.get(index).requirement();
            if (!declared.contains(requirement)) {
                throw new IllegalArgumentException(
                        "workspaceUses[" + index + "].requirement is not declared");
            }
            used.add(requirement);
        }
        for (int index = 0; index < this.requirements.size(); index++) {
            if (!used.contains(this.requirements.get(index))) {
                throw new IllegalArgumentException("requirements[" + index + "] is unused");
            }
        }
    }

    private static <T> List<T> checkedCopy(List<T> values, String name) {
        Objects.requireNonNull(values, name);
        for (int index = 0; index < values.size(); index++) {
            Objects.requireNonNull(values.get(index), name + "[" + index + "]");
        }
        return List.copyOf(values);
    }

    /**
     * Returns the complete selected generated-code facts.
     *
     * @return exact non-null immutable specialization retained at construction
     */
    CpuKernelSpecialization specialization() { return specialization; }
    /**
     * Returns the family emitter matched to the specialization fingerprint.
     *
     * @return exact non-null family emitter retained at construction
     */
    CpuFamilyKernelEmitter familyEmitter() { return familyEmitter; }
    /**
     * Returns the shared resource declarations in assignment order.
     *
     * @return non-null immutable ordered requirement snapshot with no null entries
     */
    List<PreparationResourceRequirement> requirements() { return requirements; }
    /**
     * Returns uses aligned one-for-one with specialization arguments.
     *
     * @return non-null immutable ordered buffer-use snapshot with no null entries
     */
    List<BufferUse> bufferUses() { return bufferUses; }
    /**
     * Returns the ordered workspace roles supplied to the family binder.
     *
     * @return non-null immutable ordered workspace-use snapshot with no null entries
     */
    List<WorkspaceUse> workspaceUses() { return workspaceUses; }
    /**
     * Returns the cold signature-specific invocation constructor.
     *
     * @return exact non-null family-owned binder retained at construction
     */
    CpuPortableInvocationBinder invocationBinder() { return invocationBinder; }
}
