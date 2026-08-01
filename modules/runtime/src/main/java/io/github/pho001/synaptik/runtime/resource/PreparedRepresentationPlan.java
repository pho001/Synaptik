package io.github.pho001.synaptik.runtime.resource;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.List;
import java.util.Objects;

/**
 * Describes how every physical representation required by one prepared memory plan originates.
 *
 * <p>The plan is immutable and reusable. Caller-input occurrences identify dense borrowed input
 * positions, while created-buffer and workspace callbacks are implemented by concrete backends
 * and produce fresh run-owned representations for each run. Buffer positions and their
 * representation positions use the dense encounter order of {@link PreparedMemoryPlan}, not slot
 * numeric components. Construction snapshots both list levels, retains the exact preparation and
 * callback references, invokes no callback, and creates no physical resource.
 *
 * <p>Instances and all callback implementations must be immutable and thread-safe so one plan
 * can be reused concurrently to create isolated runs. A later Runtime runner may reach this plan
 * through a schedule's first representation-creation occurrence; the plan itself performs no
 * orchestration, binding, transfer, validity change, execution, or cleanup.
 *
 * @param memoryPlan the exact non-null prepared memory plan described by this plan
 * @param bufferPreparations the non-null ordered non-empty preparation list for every prepared
 *     buffer; both list levels are snapshotted and their elements are retained exactly
 * @param workspaceCreators the non-null ordered creator for every prepared workspace; the list is
 *     snapshotted and creator references are retained exactly
 */
public record PreparedRepresentationPlan(
        PreparedMemoryPlan memoryPlan,
        List<List<PreparedRepresentationPlan.BufferPreparation>> bufferPreparations,
        List<PreparedRepresentationPlan.WorkspaceCreator> workspaceCreators) {
    /**
     * Creates an immutable representation-creation description.
     *
     * <p>Validation completes without invoking any creator. The plan accepts any non-empty mix of
     * caller-input and created-buffer preparations at each prepared buffer position.
     *
     * @param memoryPlan the exact non-null prepared memory plan to retain; never copied
     * @param bufferPreparations the non-null plan-ordered buffer preparations to snapshot at both
     *     list levels; every inner list must be non-empty and contain no null element
     * @param workspaceCreators the non-null plan-ordered workspace creators to snapshot; elements
     *     must be non-null
     * @throws NullPointerException if a top-level input, inner list, preparation, or creator is
     *     {@code null}
     * @throws IllegalArgumentException if a list count differs from the plan or an inner buffer
     *     list is empty
     */
    public PreparedRepresentationPlan {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(bufferPreparations, "bufferPreparations");
        Objects.requireNonNull(workspaceCreators, "workspaceCreators");

        int bufferCount = memoryPlan.buffers().size();
        if (bufferPreparations.size() != bufferCount) {
            throw new IllegalArgumentException(
                    "bufferPreparations size must equal prepared buffer count " + bufferCount);
        }
        int workspaceCount = memoryPlan.workspaces().size();
        if (workspaceCreators.size() != workspaceCount) {
            throw new IllegalArgumentException(
                    "workspaceCreators size must equal prepared workspace count "
                            + workspaceCount);
        }

        var copiedBufferPreparations =
                new java.util.ArrayList<List<BufferPreparation>>(bufferCount);
        for (int bufferIndex = 0; bufferIndex < bufferCount; bufferIndex++) {
            List<BufferPreparation> preparations =
                    Objects.requireNonNull(
                            bufferPreparations.get(bufferIndex),
                            "bufferPreparations[" + bufferIndex + "]");
            if (preparations.isEmpty()) {
                throw new IllegalArgumentException(
                        "bufferPreparations[" + bufferIndex + "] must not be empty");
            }
            for (int representationIndex = 0;
                    representationIndex < preparations.size();
                    representationIndex++) {
                Objects.requireNonNull(
                        preparations.get(representationIndex),
                        "bufferPreparations["
                                + bufferIndex
                                + "]["
                                + representationIndex
                                + "]");
            }
            copiedBufferPreparations.add(List.copyOf(preparations));
        }
        bufferPreparations = List.copyOf(copiedBufferPreparations);

        for (int workspaceIndex = 0; workspaceIndex < workspaceCount; workspaceIndex++) {
            Objects.requireNonNull(
                    workspaceCreators.get(workspaceIndex),
                    "workspaceCreators[" + workspaceIndex + "]");
        }
        workspaceCreators = List.copyOf(workspaceCreators);
    }

    /**
     * Identifies one resident buffer representation's prepared origin.
     *
     * <p>The closed family distinguishes a borrowed caller input from a fresh backend-created
     * run-owned representation. It carries no physical representation, validity bit, transfer
     * route, graph identity, backend lookup, or creation method shared by both variants.
     */
    public sealed interface BufferPreparation permits CallerInput, CreatedBuffer {}

    /**
     * Identifies one borrowed caller-input occurrence in dense encounter order.
     *
     * <p>Each occurrence consumes one entry from the later caller-input list. The occurrence has
     * no name or global identity and creates no resource. Cold setup validates all supplied caller
     * inputs before invoking a creator; the resulting binding is borrowed and initially valid.
     */
    public record CallerInput() implements BufferPreparation {
        /**
         * Creates one dense caller-input occurrence without invoking or retaining a resource.
         */
        public CallerInput {}
    }

    /**
     * Identifies one buffer representation created for each run.
     *
     * <p>The retained creator is invoked during cold run setup, after complete caller-input
     * validation. Its successful result is run-owned, structurally resident until state closure,
     * and initially invalid because newly created storage does not yet contain the logical buffer
     * value.
     *
     * @param creator the non-null immutable thread-safe backend creator retained exactly
     */
    public record CreatedBuffer(BufferCreator creator) implements BufferPreparation {
        /**
         * Retains one non-null backend creator.
         *
         * @param creator the creator to retain exactly
         * @throws NullPointerException if {@code creator} is {@code null}
         */
        public CreatedBuffer {
            Objects.requireNonNull(creator, "creator");
        }
    }

    /**
     * Creates one fresh non-null run-owned physical buffer representation.
     *
     * <p>A concrete backend owns allocation and physical cleanup. Every successful call must
     * return a new exact object not reused by another position or concurrent run. Runtime calls
     * this only during cold setup and closes a successful result on later setup failure or when
     * the completed run state closes.
     */
    @FunctionalInterface
    public interface BufferCreator {
        /**
         * Creates one backend-owned physical representation for the current run.
         *
         * @return a fresh non-null representation conditionally owned by cold setup until the
         *     complete run state succeeds, then owned by that run
         * @throws RuntimeException if backend creation fails; cold setup preserves it and may add
         *     cleanup failures as suppressed exceptions
         * @throws Error if backend creation reports an error; cold setup preserves it and may add
         *     cleanup failures as suppressed exceptions
         */
        BufferRepresentation create();
    }

    /**
     * Creates one fresh non-null run-owned physical workspace representation.
     *
     * <p>A concrete backend owns allocation and physical cleanup. Every successful call must
     * return a new exact object not reused by any caller input, buffer result, workspace result,
     * or concurrent run. Workspace scratch is structurally resident until state closure and has
     * no logical buffer-validity state.
     */
    @FunctionalInterface
    public interface WorkspaceCreator {
        /**
         * Creates one backend-local scratch representation for the current run.
         *
         * @return a fresh non-null workspace conditionally owned by cold setup until the complete
         *     run state succeeds, then owned by that run
         * @throws RuntimeException if backend creation fails; cold setup preserves it and may add
         *     cleanup failures as suppressed exceptions
         * @throws Error if backend creation reports an error; cold setup preserves it and may add
         *     cleanup failures as suppressed exceptions
         */
        WorkspaceRepresentation create();
    }
}
