package backend.lowering.partition;

import java.util.List;

public record PartitionExecutionGroup(
        String groupId,
        List<Integer> orderedNodeIds,
        PartitionExecutionKind executionKind,
        String physicalKernel,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        List<String> tempValueIds,
        PartitionStorageContract storageContract,
        String reason
) {
    public PartitionExecutionGroup {
        groupId = groupId == null ? "" : groupId;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        executionKind = executionKind == null ? PartitionExecutionKind.UNKNOWN : executionKind;
        physicalKernel = physicalKernel == null ? "" : physicalKernel;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        tempValueIds = List.copyOf(tempValueIds == null ? List.of() : tempValueIds);
        storageContract = storageContract == null ? PartitionStorageContract.UNKNOWN : storageContract;
        reason = reason == null ? "" : reason;
    }
}
