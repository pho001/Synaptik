package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;

/**
 * Assembles one complete immutable Runtime schedule from finalized Prepare facts.
 *
 * <p>An implementation constructs a recipe only. It performs no execution, allocation, search,
 * mutation, resource acquisition, or backend discovery.</p>
 */
@FunctionalInterface
public interface PreparedScheduleAssembler {
    /**
     * Assembles one complete schedule after every backend partition has finalized.
     *
     * @param context non-null immutable complete schedule context
     * @return a non-null immutable schedule, subsequently validated by Prepare
     * @throws NullPointerException if {@code context} is null
     * @throws IllegalArgumentException if the supplied facts cannot form a supported recipe
     */
    PreparedSchedule assemble(PreparedScheduleContext context);
}
