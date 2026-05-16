package backend.lowering.region;

import java.util.List;

public record RegionExecutionGroup(
        String groupId,
        List<Integer> orderedNodeIds,
        RegionExecutionKind executionKind,
        String physicalKernel,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        List<String> tempValueIds,
        RegionStorageContract storageContract,
        String reason
) {
    public RegionExecutionGroup {
        groupId = groupId == null ? "" : groupId;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        executionKind = executionKind == null ? RegionExecutionKind.UNKNOWN : executionKind;
        physicalKernel = physicalKernel == null ? "" : physicalKernel;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        tempValueIds = List.copyOf(tempValueIds == null ? List.of() : tempValueIds);
        storageContract = storageContract == null ? RegionStorageContract.UNKNOWN : storageContract;
        reason = reason == null ? "" : reason;
    }
}
