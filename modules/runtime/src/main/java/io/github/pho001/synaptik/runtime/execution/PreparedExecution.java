package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.Objects;

/**
 * Retains the complete current immutable Runtime recipe for one prepared execution.
 *
 * <p>The aggregate keeps the exact {@link PreparedMemoryPlan} and {@link PreparedSchedule}
 * references supplied at construction and requires the schedule to report that same plan by
 * reference identity. Both components are immutable reusable contracts, so one prepared
 * execution may be shared safely while distinct logical runs use isolated mutable run state.
 *
 * <p>This record owns no closeable, persistent, or per-run resource and transfers no ownership.
 * It does not create run state, bind or execute the schedule, allocate representations, or
 * provide publication or result behavior. Construction and component access are constant-time.
 * Ordinary record equality and hashing remain structural over the two components; the stricter
 * plan-reference rule applies when the aggregate is constructed.
 *
 * @param memoryPlan the exact non-null immutable prepared memory plan to retain
 * @param schedule the exact non-null immutable prepared schedule to retain; it must report
 *     {@code memoryPlan} by reference identity
 */
public record PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule) {
    /**
     * Validates and retains one immutable reusable prepared recipe.
     *
     * <p>Components are checked in declaration order. The schedule-plan association is checked
     * only after both components are known to be non-null. Neither component is copied or
     * wrapped, and construction performs no lifecycle or physical-resource action.
     *
     * @param memoryPlan the exact non-null immutable prepared memory plan to retain
     * @param schedule the exact non-null immutable prepared schedule to retain; it must report
     *     {@code memoryPlan} by reference identity
     * @throws NullPointerException if {@code memoryPlan} or {@code schedule} is {@code null}
     * @throws IllegalArgumentException if {@code schedule.memoryPlan()} is not the exact supplied
     *     {@code memoryPlan} reference
     */
    public PreparedExecution {
        Objects.requireNonNull(memoryPlan, "memoryPlan");
        Objects.requireNonNull(schedule, "schedule");
        if (schedule.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "schedule memory plan does not match prepared execution memory plan");
        }
    }

    /**
     * Returns the exact prepared memory plan supplied at construction.
     *
     * @return the retained non-null immutable plan reference; never a copy or structural
     *     replacement
     */
    @Override
    public PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the exact prepared schedule supplied at construction.
     *
     * @return the retained non-null immutable schedule reference whose plan is exactly
     *     {@link #memoryPlan()} by reference identity
     */
    @Override
    public PreparedSchedule schedule() {
        return schedule;
    }
}
