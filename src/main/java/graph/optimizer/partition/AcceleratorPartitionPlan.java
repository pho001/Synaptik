package graph.optimizer.partition;

import backend.ComputeBackend;

import java.util.List;

public interface AcceleratorPartitionPlan {
    ComputeBackend backend();

    int anchorNodeId();

    List<Integer> nodeIds();

    List<Integer> externalInputNodeIds();

    List<Integer> producedOutputNodeIds();

    long estimatedWork();
}
