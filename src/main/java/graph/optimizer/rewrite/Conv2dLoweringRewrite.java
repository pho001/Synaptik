package graph.optimizer.rewrite;

import operations.Operation;
import operations.conv2d;
import operations.conv2dGemm;
import tensor.Tensor;

import java.util.List;

public class Conv2dLoweringRewrite extends AbstractRewriteRule {
    private static final boolean DISABLE_CONV2D_GEMM_LOWERING =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.rewrite.disableConv2dGemmLowering", "false"));

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        if (DISABLE_CONV2D_GEMM_LOWERING) {
            return tensor;
        }
        Operation op = tensor.getOperation();
        if (!(op instanceof conv2d conv)) {
            return tensor;
        }
        if (tensor.isBackward()) {
            return tensor;
        }

        Tensor lowered = new Tensor(
                tensor.getShape().clone(),
                List.copyOf(tensor.getPrevTensors()),
                new conv2dGemm(conv.getOptions(), conv.hasBias()),
                tensor.getLabel(),
                tensor.getDataType()
        );
        lowered.setRequiresGrad(tensor.getRequiresGrad());
        return lowered;
    }
}
