package backend.accelerator.lowering;

import backend.accelerator.dag.AcceleratorPostOp;
import java.util.List;

/**
 * Legacy matmul-focused accelerator descriptor retained beside the general DAG ABI.
 *
 * @param leftInputNodeId compiled-node id of the left matrix input
 * @param rightInputNodeId compiled-node id of the right matrix input
 * @param biasInputNodeId optional bias input node id, or {@code -1}
 * @param outputNodeId compiled-node id produced by the matmul chain
 * @param m row dimension
 * @param n column dimension
 * @param k reduction dimension
 * @param biasVector whether the bias is a vector rather than a full output tensor
 * @param postOps ordered fused post operations
 */
public record AcceleratorMatMulSpec(
        int leftInputNodeId,
        int rightInputNodeId,
        int biasInputNodeId,
        int outputNodeId,
        int m,
        int n,
        int k,
        boolean biasVector,
        List<AcceleratorPostOp> postOps
) {
    public AcceleratorMatMulSpec {
        postOps = List.copyOf(postOps == null ? List.of() : postOps);
    }
}
