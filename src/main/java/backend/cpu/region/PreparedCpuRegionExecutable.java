package backend.cpu.region;

import backend.lowering.region.RegionExecutionPlan;
import backend.runtime.ExecutionContext;
import graph.execution.PreparedNodeExecution;

import java.util.List;

/**
 * Prepared CPU executable for a lowered execution region anchored on a single compiled node.
 */
public interface PreparedCpuRegionExecutable {
    void execute(ExecutionContext context);

    RegionExecutionPlan regionExecutionPlan();

    default List<PreparedNodeExecution> nativeSteps() {
        return List.of();
    }

    default List<PreparedNodeExecution> fallbackSteps() {
        return List.of();
    }

    default String lastRoute() {
        return "";
    }

    default String lastFallbackReason() {
        return "";
    }

    default int lastRegionLocalKernelCount() {
        return 0;
    }

    default int lastRegionLocalViewCount() {
        return 0;
    }

    default int lastExecutedGroupCount() {
        return 0;
    }
}
