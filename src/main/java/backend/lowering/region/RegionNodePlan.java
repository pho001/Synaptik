package backend.lowering.region;

import operations.Operation;
import tensor.DataType;

import java.util.List;

public record RegionNodePlan(
        int nodeId,
        Operation.OpType opType,
        DataType dataType,
        RegionRole regionRole,
        RegionExecutionKind executionKind,
        String physicalKernel,
        RegionStorageContract storageContract,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        RegionLegalityStatus legalityStatus,
        String reason
) {
    public RegionNodePlan {
        opType = opType == null ? Operation.OpType.UNKNOWN : opType;
        dataType = dataType == null ? DataType.FLOAT64 : dataType;
        regionRole = regionRole == null ? RegionRole.UNKNOWN : regionRole;
        executionKind = executionKind == null ? RegionExecutionKind.UNKNOWN : executionKind;
        physicalKernel = physicalKernel == null ? "" : physicalKernel;
        storageContract = storageContract == null ? RegionStorageContract.UNKNOWN : storageContract;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        legalityStatus = legalityStatus == null ? RegionLegalityStatus.UNKNOWN : legalityStatus;
        reason = reason == null ? "" : reason;
    }
}
