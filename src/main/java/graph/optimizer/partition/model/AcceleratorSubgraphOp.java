package graph.optimizer.partition.model;

import operations.Operation;

public record AcceleratorSubgraphOp(
        int nodeId,
        Operation.OpType opType
) {
}
