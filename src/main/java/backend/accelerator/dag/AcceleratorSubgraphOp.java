package backend.accelerator.dag;

import operations.Operation;

public record AcceleratorSubgraphOp(
        int nodeId,
        Operation.OpType opType
) {
}
