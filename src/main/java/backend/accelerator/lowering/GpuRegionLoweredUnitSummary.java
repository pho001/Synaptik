package backend.accelerator.lowering;

import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.ExecutionUnitKind;

import java.util.List;

/**
 * Traceable summary of one region-internal GPU lowering unit.
 */
public record GpuRegionLoweredUnitSummary(
        String unitId,
        ExecutionUnitKind kind,
        List<Integer> orderedNodeIds,
        List<String> traceEvents
) {
    public GpuRegionLoweredUnitSummary {
        unitId = unitId == null ? "" : unitId;
        kind = kind == null ? ExecutionUnitKind.UNIT_KERNEL : kind;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        traceEvents = List.copyOf(traceEvents == null ? List.of() : traceEvents);
    }

    public static GpuRegionLoweredUnitSummary fromExecutionUnit(ExecutionUnit unit) {
        return new GpuRegionLoweredUnitSummary(
                unit.unitId(),
                unit.kind(),
                unit.orderedNodeIds(),
                unit.trace().events()
        );
    }
}
