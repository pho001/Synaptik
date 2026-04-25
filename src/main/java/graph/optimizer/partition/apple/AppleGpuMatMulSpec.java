package graph.optimizer.partition.apple;

import graph.optimizer.partition.model.AcceleratorPostOp;
import java.util.List;

public record AppleGpuMatMulSpec(
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
    public AppleGpuMatMulSpec {
        postOps = List.copyOf(postOps == null ? List.of() : postOps);
    }
}
