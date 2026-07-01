package planning.partition.execution;

import java.util.List;

/**
 * Diagnostics emitted by partition planning.
 *
 * @param events ordered textual events
 */
public record PartitionExecutionTrace(
        List<String> events
) {
    public PartitionExecutionTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    /**
     * Returns an empty partition planning trace.
     *
     * @return empty trace
     */
    public static PartitionExecutionTrace empty() {
        return new PartitionExecutionTrace(List.of());
    }
}
