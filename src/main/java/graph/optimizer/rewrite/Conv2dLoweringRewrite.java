package graph.optimizer.rewrite;

import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import operations.Operation;
import operations.conv2d;
import operations.conv2dBackwardInput;
import operations.conv2dBackwardInputGemm;
import operations.conv2dBackwardWeight;
import operations.conv2dBackwardWeightGemm;
import operations.conv2dGemm;
import tensor.Tensor;

import java.util.List;

public class Conv2dLoweringRewrite extends AbstractRewriteRule {
    private final Conv2dLoweringConfig config;

    public Conv2dLoweringRewrite() {
        this(Conv2dLoweringConfig.defaults());
    }

    public Conv2dLoweringRewrite(Conv2dLoweringConfig config) {
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
            case HEURISTIC -> Conv2dLoweringHeuristics.shouldLower(tensor, conv);
        };
    }

    private boolean shouldLower(Tensor tensor, conv2dBackwardInput conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dLoweringHeuristics.shouldLower(tensor, conv);
        };
    }

    private boolean shouldLower(Tensor tensor, conv2dBackwardWeight conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dLoweringHeuristics.shouldLower(tensor, conv);
        };
    }
}
