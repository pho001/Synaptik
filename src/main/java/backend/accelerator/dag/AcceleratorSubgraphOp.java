package backend.accelerator.dag;

import operations.Operation;

/**
 * Operation summary for a node inside an accelerator subgraph candidate.
 *
 * @param nodeId compiled-node id
 * @param opType operation kind to lower
 */
public record AcceleratorSubgraphOp(
        int nodeId,
        Operation.OpType opType
) {
}
