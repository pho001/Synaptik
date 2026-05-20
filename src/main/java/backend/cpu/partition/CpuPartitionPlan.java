package backend.cpu.partition;

import backend.ComputeBackend;
import graph.compile.planning.partition.PartitionPlan;

import java.util.List;

public record CpuPartitionPlan(
        int anchorNodeId,
        List<Integer> nodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> producedOutputNodeIds,
        long estimatedWork
) implements PartitionPlan {
    public CpuPartitionPlan {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        producedOutputNodeIds = List.copyOf(producedOutputNodeIds == null ? List.of() : producedOutputNodeIds);
        estimatedWork = Math.max(0L, estimatedWork);
    }

    @Override
    public ComputeBackend backend() {
        return ComputeBackend.CPU;
    }
}
