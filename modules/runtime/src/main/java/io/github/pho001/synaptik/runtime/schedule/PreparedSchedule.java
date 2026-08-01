package io.github.pho001.synaptik.runtime.schedule;

import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import java.util.List;
import java.util.Objects;

/**
 * Orders immutable prepared work associated with one exact prepared memory plan.
 *
 * <p>A schedule is a reusable prepared recipe. It owns only an immutable snapshot of the supplied
 * step-list structure and retains every exact step reference in encounter order. It permits an
 * empty list and repeated executable or transfer occurrences. One optional
 * representation-creation occurrence may appear only at index zero, keeping cold setup reachable
 * through the unchanged two-component prepared-execution aggregate. The schedule does not itself
 * create a state, invoke a creator, bind or execute work, allocate or close resources, or define
 * publication policy. A transfer occurrence names explicit work between already-created
 * representations; transfer to an equivalent destination is materialization rather than a second
 * operation kind.
 *
 * <p>The schedule and its current step values are immutable and may be traversed concurrently
 * while distinct logical runs are prepared. This does not make mutable per-run objects safe for
 * concurrent use.
 *
 * @param memoryPlan the exact non-null immutable prepared memory plan shared by every step
 * @param steps the non-null ordered step occurrences to validate and snapshot; elements must be
 *     non-null and report exactly {@code memoryPlan} by reference identity
 */
public record PreparedSchedule(
        PreparedMemoryPlan memoryPlan,
        List<PreparedSchedule.Step> steps) {
    /**
     * Validates and snapshots one immutable schedule recipe.
     *
     * <p>Top-level references are validated in component order. Steps are then validated in
     * supplied encounter order, and the list is copied only after the complete scan succeeds.
     * Construction performs no binding, execution, physical resource action, or ownership
     * transfer.
     *
     * @param memoryPlan the exact non-null prepared memory plan to retain
     * @param steps the non-null ordered step occurrences to snapshot; every element must be
     *     non-null and associated with the exact supplied plan
     * @throws NullPointerException if {@code memoryPlan}, {@code steps}, or a step is {@code null};
     *     an element failure identifies its zero-based supplied position
     * @throws IllegalArgumentException if a step reports a different plan reference or a
     *     representation-creation step occurs anywhere except index zero; the failure identifies
     *     the first rejected supplied position
     */
    public PreparedSchedule {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(steps, "steps");
        int publicationCount = 0;
        boolean publicationSuffix = false;
        for (int index = 0; index < steps.size(); index++) {
            Step step = Objects.requireNonNull(steps.get(index), "steps[" + index + "]");
            if (step.memoryPlan() != memoryPlan) {
                throw new IllegalArgumentException(
                        "steps["
                                + index
                                + "] memory plan does not match schedule memory plan");
            }
            if (step instanceof RepresentationCreationStep && index != 0) {
                throw new IllegalArgumentException(
                        "steps["
                                + index
                                + "] representation creation must be the first schedule occurrence");
            }
            if (step instanceof PublicationStep publicationStep) {
                if (publicationStep.publication().resultIndex() != publicationCount) {
                    throw new IllegalArgumentException(
                            "steps["
                                    + index
                                    + "] publication resultIndex must equal publication encounter index "
                                    + publicationCount);
                }
                publicationCount++;
                publicationSuffix = true;
            } else if (publicationSuffix) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] non-publication occurrence follows publication suffix");
            }
        }
        steps = List.copyOf(steps);
    }

    /**
     * Returns this schedule's exact prepared memory plan.
     *
     * @return the retained non-null immutable plan reference; never a structural replacement
     */
    @Override
    public PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the deterministic ordered schedule occurrences.
     *
     * @return a non-null immutable list snapshot retaining every exact step reference, including
     *     repeated occurrences
     */
    @Override
    public List<PreparedSchedule.Step> steps() {
        return steps;
    }

    /**
     * Returns the number of publication occurrences in the dense schedule suffix.
     *
     * @return the non-negative publication count; zero when the suffix is empty
     */
    public int publicationCount() {
        int count = 0;
        for (Step step : steps) {
            if (step instanceof PublicationStep) {
                count++;
            }
        }
        return count;
    }

    /**
     * Identifies one immutable prepared work occurrence associated with a prepared memory plan.
     *
     * <p>The sealed family currently contains the optional cold representation-creation prefix
     * and executable or buffer-transfer occurrences. The plan association lets a schedule
     * validate one exact reusable prepared context without a generic payload, registry,
     * identifier, or execution method.
     */
    public sealed interface Step
            permits ExecutionStep, RepresentationCreationStep, BufferTransferStep, PublicationStep {
        /**
         * Returns the exact prepared memory plan required by this step.
         *
         * @return a non-null immutable prepared memory plan reference
         */
        PreparedMemoryPlan memoryPlan();
    }

    /**
     * Represents one ordered publication occurrence in the schedule's dense final suffix.
     *
     * @param publication the exact non-null immutable publication recipe
     */
    public record PublicationStep(PreparedPublication publication) implements Step {
        /**
         * Retains one prepared publication occurrence without binding or publishing it.
         *
         * @param publication the non-null publication recipe to retain exactly
         * @throws NullPointerException if {@code publication} is {@code null}
         */
        public PublicationStep {
            Objects.requireNonNull(publication, "publication");
        }

        /**
         * Returns the publication recipe's exact prepared memory plan.
         *
         * @return exactly {@code publication().memoryPlan()}
         */
        @Override
        public PreparedMemoryPlan memoryPlan() {
            return publication.memoryPlan();
        }
    }

    /**
     * Represents one ordered occurrence of an already-prepared buffer transfer recipe.
     *
     * <p>The step retains the recipe exactly and derives its plan association directly from that
     * recipe. Construction performs no binding, physical work, or validity transition. Repeated
     * occurrences are explicit repeated work. Materializing an equivalent already-created
     * destination uses this same transfer step and no separate schedule kind.
     *
     * @param transfer the exact non-null immutable reusable transfer recipe
     */
    public record BufferTransferStep(PreparedBufferTransfer transfer) implements Step {
        /**
         * Retains one prepared buffer-transfer occurrence.
         *
         * @param transfer the non-null transfer recipe to retain exactly
         * @throws NullPointerException if {@code transfer} is {@code null}
         */
        public BufferTransferStep {
            Objects.requireNonNull(transfer, "transfer");
        }

        /**
         * Returns the transfer's exact prepared memory plan.
         *
         * @return exactly {@code transfer().memoryPlan()}; never cached or copied
         */
        @Override
        public PreparedMemoryPlan memoryPlan() {
            return transfer.memoryPlan();
        }
    }

    /**
     * Retains the sole optional prepared representation-creation prefix for a schedule.
     *
     * <p>When present, schedule validation requires this occurrence at index zero. The step
     * retains an immutable reusable description only: constructing it or a containing schedule
     * invokes no creator, creates no {@code RunState}, changes no validity, and owns no resource.
     * Empty and executable-only schedules remain valid for compatibility; a later Prepare
     * validator may require the prefix for a runnable result.
     *
     * @param representationPlan the exact non-null immutable reusable creation plan
     */
    public record RepresentationCreationStep(
            PreparedRepresentationPlan representationPlan) implements Step {
        /**
         * Retains one representation plan.
         *
         * @param representationPlan the non-null plan to retain exactly
         * @throws NullPointerException if {@code representationPlan} is {@code null}
         */
        public RepresentationCreationStep {
            Objects.requireNonNull(representationPlan, "representationPlan");
        }

        /**
         * Returns the representation plan's exact prepared memory plan.
         *
         * @return exactly {@code representationPlan().memoryPlan()}
         */
        @Override
        public PreparedMemoryPlan memoryPlan() {
            return representationPlan.memoryPlan();
        }
    }

    /**
     * Represents one ordered occurrence of an already-prepared executable recipe.
     *
     * <p>The step retains the executable exactly and derives its plan association directly from
     * that executable. Repeating a step or executable in a schedule represents repeated work; it
     * does not duplicate ownership of either prepared or per-run resources.
     *
     * @param executable the exact non-null immutable reusable executable recipe
     */
    public record ExecutionStep(PreparedExecutable executable) implements Step {
        /**
         * Retains one prepared executable occurrence.
         *
         * @param executable the exact non-null executable reference to retain
         * @throws NullPointerException if {@code executable} is {@code null}
         */
        public ExecutionStep {
            Objects.requireNonNull(executable, "executable");
        }

        /**
         * Returns this occurrence's exact executable recipe.
         *
         * @return the retained non-null immutable executable reference
         */
        @Override
        public PreparedExecutable executable() {
            return executable;
        }

        /**
         * Returns the executable's exact prepared memory plan.
         *
         * @return exactly {@code executable().memoryPlan()}; never cached or copied
         */
        @Override
        public PreparedMemoryPlan memoryPlan() {
            return executable.memoryPlan();
        }
    }
}
