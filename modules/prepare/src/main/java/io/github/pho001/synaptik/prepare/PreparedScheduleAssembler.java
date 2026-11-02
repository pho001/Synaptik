package io.github.pho001.synaptik.prepare;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.Objects;
import java.util.Optional;

/**
 * Assembles one complete immutable Runtime schedule from finalized Prepare facts.
 *
 * <p>An implementation constructs a recipe only. It performs no execution, allocation, search,
 * mutation, resource acquisition, or backend discovery.</p>
 */
@FunctionalInterface
public interface PreparedScheduleAssembler {
    /**
     * Contributes geometry for an authenticated producerless published caller or constant.
     *
     * @param value exact non-null stable graph value
     * @param logicalRequirement exact non-null matching producerless logical requirement
     * @param scalar exact constant scalar when authenticated as a constant, otherwise empty
     *     for an authenticated caller; initialization remains a per-run responsibility
     * @return a non-null immutable geometry contribution retaining both exact references
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if this assembler cannot represent the source
     */
    default ProducerlessPublishedValueResource producerlessPublishedValue(
            GraphValue value,
            LogicalMemoryRequirement logicalRequirement,
            Optional<ScalarValue> scalar) {
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(logicalRequirement, "logicalRequirement");
        Objects.requireNonNull(scalar, "scalar");
        throw new IllegalArgumentException(
                "schedule assembler does not support producerless published values");
    }

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
