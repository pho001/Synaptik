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
        String segmentKernelFamily,
        String layoutClass,
        List<String> inputLayoutClasses,
        String outputLayoutClass,
        String materializationReason,
        RegionStorageContract storageContract,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        RegionLegalityStatus legalityStatus,
        String reason
) {
    public RegionNodePlan(
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
        this(
                nodeId,
                opType,
                dataType,
                regionRole,
                executionKind,
                physicalKernel,
                "",
                "",
                List.of(),
                "",
                "",
                storageContract,
                inputNodeIds,
                outputNodeIds,
                legalityStatus,
                reason
        );
    }

    public RegionNodePlan {
        opType = opType == null ? Operation.OpType.UNKNOWN : opType;
        dataType = dataType == null ? DataType.FLOAT64 : dataType;
        regionRole = regionRole == null ? RegionRole.UNKNOWN : regionRole;
        executionKind = executionKind == null ? RegionExecutionKind.UNKNOWN : executionKind;
        physicalKernel = physicalKernel == null ? "" : physicalKernel;
        segmentKernelFamily = segmentKernelFamily == null ? "" : segmentKernelFamily;
        layoutClass = layoutClass == null ? "" : layoutClass;
        inputLayoutClasses = List.copyOf(inputLayoutClasses == null ? List.of() : inputLayoutClasses);
        outputLayoutClass = outputLayoutClass == null ? "" : outputLayoutClass;
        materializationReason = materializationReason == null ? "" : materializationReason;
        storageContract = storageContract == null ? RegionStorageContract.UNKNOWN : storageContract;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        legalityStatus = legalityStatus == null ? RegionLegalityStatus.UNKNOWN : legalityStatus;
        reason = reason == null ? "" : reason;
    }
}
