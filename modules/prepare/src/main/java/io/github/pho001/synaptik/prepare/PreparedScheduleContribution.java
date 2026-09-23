package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Declares one backend owner's exact physical representation-creation contribution.
 *
 * <p>A contribution contains no schedule step, executable, transfer, physical resource, or
 * mutable state. Buffer entries address owner positions already assigned by shared Prepare;
 * workspace entries address exact partition-owned plan positions. The complete schedule assembler
 * combines validated contributions without asking a backend to own global ordering.</p>
 *
 * @param owner exact non-null backend owner of every contribution entry
 * @param buffers non-null sparse assigned-buffer contribution list
 * @param workspaces non-null sparse assigned-workspace contribution list
 */
public record PreparedScheduleContribution(
        BackendId owner,
        List<PreparedScheduleContribution.Buffer> buffers,
        List<PreparedScheduleContribution.Workspace> workspaces) {
    /**
     * Validates and snapshots one owner contribution.
     *
     * @param owner exact non-null backend owner
     * @param buffers non-null sparse buffer entries to snapshot
     * @param workspaces non-null sparse workspace entries to snapshot
     * @throws NullPointerException if a top-level input or indexed entry is null
     * @throws IllegalArgumentException if an entry does not belong to {@code owner} or a physical
     *     coordinate repeats
     */
    public PreparedScheduleContribution {
        Objects.requireNonNull(owner, "owner");
        Objects.requireNonNull(buffers, "buffers");
        Objects.requireNonNull(workspaces, "workspaces");
        var bufferCoordinates = new HashSet<BufferCoordinate>();
        for (int index = 0; index < buffers.size(); index++) {
            Buffer buffer = Objects.requireNonNull(buffers.get(index), "buffers[" + index + "]");
            List<BackendId> representationOwners = buffer.assignment().representationOwners();
            if (buffer.representationIndex() >= representationOwners.size()
                    || !representationOwners.get(buffer.representationIndex()).equals(owner)) {
                throw new IllegalArgumentException(
                        "buffers[" + index + "] does not belong to contribution owner");
            }
            if (!bufferCoordinates.add(new BufferCoordinate(
                    buffer.assignment().planIndex(), buffer.representationIndex()))) {
                throw new IllegalArgumentException(
                        "buffers[" + index + "] duplicates a buffer representation coordinate");
            }
        }
        var workspaceIndices = new HashSet<Integer>();
        for (int index = 0; index < workspaces.size(); index++) {
            Workspace workspace = Objects.requireNonNull(
                    workspaces.get(index), "workspaces[" + index + "]");
            if (!workspace.assignment().partition().owner().equals(owner)) {
                throw new IllegalArgumentException(
                        "workspaces[" + index + "] does not belong to contribution owner");
            }
            if (!workspaceIndices.add(workspace.assignment().planIndex())) {
                throw new IllegalArgumentException(
                        "workspaces[" + index + "] duplicates a workspace coordinate");
            }
        }
        buffers = List.copyOf(buffers);
        workspaces = List.copyOf(workspaces);
    }

    /**
     * Supplies one exact buffer representation origin for an assigned owner position.
     *
     * @param assignment exact non-null plan-ordered logical buffer assignment
     * @param representationIndex non-negative dense owner representation position
     * @param preparation exact non-null immutable caller-input or creation recipe
     */
    public record Buffer(
            PreparedBufferAssignment assignment,
            int representationIndex,
            PreparedRepresentationPlan.BufferPreparation preparation) {
        /**
         * Validates one exact representation contribution.
         *
         * @param assignment exact non-null assignment to retain
         * @param representationIndex non-negative representation position
         * @param preparation exact non-null physical origin recipe to retain
         * @throws NullPointerException if a reference is null
         * @throws IllegalArgumentException if {@code representationIndex} is negative
         */
        public Buffer {
            Objects.requireNonNull(assignment, "assignment");
            Objects.requireNonNull(preparation, "preparation");
            if (representationIndex < 0) {
                throw new IllegalArgumentException(
                        "representationIndex must be non-negative");
            }
        }
    }

    /**
     * Supplies one exact partition-owned workspace creator.
     *
     * @param assignment exact non-null plan-ordered workspace assignment
     * @param creator exact non-null immutable backend creator
     */
    public record Workspace(
            PreparedWorkspaceAssignment assignment,
            PreparedRepresentationPlan.WorkspaceCreator creator) {
        /**
         * Validates one workspace contribution.
         *
         * @param assignment exact non-null assignment to retain
         * @param creator exact non-null creator to retain
         * @throws NullPointerException if a component is null
         */
        public Workspace {
            Objects.requireNonNull(assignment, "assignment");
            Objects.requireNonNull(creator, "creator");
        }
    }

    private record BufferCoordinate(int bufferIndex, int representationIndex) {}
}
