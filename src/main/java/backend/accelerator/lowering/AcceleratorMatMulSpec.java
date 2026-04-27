package backend.accelerator.lowering;

import backend.accelerator.dag.AcceleratorPostOp;
import java.util.List;

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
