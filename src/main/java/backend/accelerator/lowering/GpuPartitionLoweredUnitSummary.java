package backend.accelerator.lowering;

import planning.partition.execution.ExecutionUnit;
import planning.partition.execution.ExecutionUnitKind;

import java.util.List;

/**
 * Traceable summary of one partition-internal GPU lowering unit.
 */
public record GpuPartitionLoweredUnitSummary(
        String unitId,
        ExecutionUnitKind kind,
        List<Integer> orderedNodeIds,
        List<String> traceEvents
) {
    public GpuPartitionLoweredUnitSummary {
        unitId = unitId == null ? "" : unitId;
        kind = kind == null ? ExecutionUnitKind.UNIT_KERNEL : kind;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        traceEvents = List.copyOf(traceEvents == null ? List.of() : traceEvents);
    }

    public static GpuPartitionLoweredUnitSummary fromExecutionUnit(ExecutionUnit unit) {
        return new GpuPartitionLoweredUnitSummary(
                unit.unitId(),
                unit.kind(),
                unit.orderedNodeIds(),
                unit.trace().events()
        );
    }
}
