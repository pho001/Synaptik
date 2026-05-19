package graph.optimizer.rewrite.lowering;

import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import graph.optimizer.rewrite.LocalTensorRewriteRule;
import operations.Operation;
import operations.nn.conv.conv2d;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardInputGemm;
import operations.nn.conv.conv2dBackwardWeight;
import operations.nn.conv.conv2dBackwardWeightGemm;
import operations.nn.conv.conv2dGemm;
import tensor.Tensor;

import java.util.List;

/**
 * Lowers convolution operations to GEMM-backed convolution operations according to configured heuristics.
 */
public final class Conv2dGemmLoweringRule extends LocalTensorRewriteRule {
    private final Conv2dLoweringConfig config;

    /**
     * Creates a convolution lowering rewrite with default configuration.
     */
    public Conv2dGemmLoweringRule() {
        this(Conv2dLoweringConfig.defaults());
    }

    /**
     * Creates a convolution lowering rewrite.
     *
     * @param config lowering configuration, or {@code null} for defaults
     */
    public Conv2dGemmLoweringRule(Conv2dLoweringConfig config) {
        this.config = config == null ? Conv2dLoweringConfig.defaults() : config;
    }

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (op instanceof conv2d conv) {
            if (!shouldLower(tensor, conv)) {
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
        if (op instanceof conv2dBackwardInput backwardInput) {
            if (!shouldLower(tensor, backwardInput)) {
                return tensor;
            }
            Tensor lowered = new Tensor(
                    tensor.getShape().clone(),
                    List.copyOf(tensor.getPrevTensors()),
                    new conv2dBackwardInputGemm(backwardInput.getOptions(), backwardInput.getInputShape()),
                    tensor.getLabel(),
                    tensor.getDataType()
            );
            lowered.setRequiresGrad(tensor.getRequiresGrad());
            return lowered;
        }
        if (op instanceof conv2dBackwardWeight backwardWeight) {
            if (!shouldLower(tensor, backwardWeight)) {
                return tensor;
            }
            Tensor lowered = new Tensor(
                    tensor.getShape().clone(),
                    List.copyOf(tensor.getPrevTensors()),
                    new conv2dBackwardWeightGemm(backwardWeight.getOptions(), backwardWeight.getWeightShape()),
                    tensor.getLabel(),
                    tensor.getDataType()
            );
            lowered.setRequiresGrad(tensor.getRequiresGrad());
            return lowered;
        }
        return tensor;
    }

    private boolean shouldLower(Tensor tensor, conv2d conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dGemmLoweringHeuristics.shouldLower(tensor, conv);
        };
    }

    private boolean shouldLower(Tensor tensor, conv2dBackwardInput conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dGemmLoweringHeuristics.shouldLower(tensor, conv);
        };
    }

    private boolean shouldLower(Tensor tensor, conv2dBackwardWeight conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dGemmLoweringHeuristics.shouldLower(tensor, conv);
        };
    }
}
