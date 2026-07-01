package backend.lowering.partition;

import operations.Operation;
import tensor.DataType;

import java.util.List;

public record PartitionNodePlan(
        int nodeId,
        Operation.OpType opType,
        DataType dataType,
        PartitionRole partitionRole,
        PartitionExecutionKind executionKind,
        String physicalKernel,
        String segmentKernelFamily,
        String layoutClass,
        List<String> inputLayoutClasses,
        String outputLayoutClass,
        String materializationReason,
        PartitionStorageContract storageContract,
        List<Integer> inputNodeIds,
        List<Integer> outputNodeIds,
        PartitionLegalityStatus legalityStatus,
        String reason
) {
    public PartitionNodePlan(
            int nodeId,
            Operation.OpType opType,
            DataType dataType,
            PartitionRole partitionRole,
            PartitionExecutionKind executionKind,
            String physicalKernel,
            PartitionStorageContract storageContract,
            List<Integer> inputNodeIds,
            List<Integer> outputNodeIds,
            PartitionLegalityStatus legalityStatus,
            String reason
    ) {
        this(
                nodeId,
                opType,
                dataType,
                partitionRole,
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

    public PartitionNodePlan {
        opType = opType == null ? Operation.OpType.UNKNOWN : opType;
        dataType = dataType == null ? DataType.FLOAT64 : dataType;
        partitionRole = partitionRole == null ? PartitionRole.UNKNOWN : partitionRole;
        executionKind = executionKind == null ? PartitionExecutionKind.UNKNOWN : executionKind;
        physicalKernel = physicalKernel == null ? "" : physicalKernel;
        segmentKernelFamily = segmentKernelFamily == null ? "" : segmentKernelFamily;
        layoutClass = layoutClass == null ? "" : layoutClass;
        inputLayoutClasses = List.copyOf(inputLayoutClasses == null ? List.of() : inputLayoutClasses);
        outputLayoutClass = outputLayoutClass == null ? "" : outputLayoutClass;
        materializationReason = materializationReason == null ? "" : materializationReason;
        storageContract = storageContract == null ? PartitionStorageContract.UNKNOWN : storageContract;
        inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        legalityStatus = legalityStatus == null ? PartitionLegalityStatus.UNKNOWN : legalityStatus;
        reason = reason == null ? "" : reason;
    }
}
