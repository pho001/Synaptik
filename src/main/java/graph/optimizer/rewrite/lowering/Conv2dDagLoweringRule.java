package graph.optimizer.rewrite.lowering;

import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import graph.optimizer.rewrite.LocalTensorRewriteRule;
import operations.Operation;
import operations.nn.conv.conv2d;
import tensor.Tensor;
import tensor.TensorOps;
import tensor.options.Conv2dOptions;
import tensor.options.Window2dOptions;

import java.util.ArrayList;
import java.util.List;

/**
 * Lowers semantic conv2d operations to canonical Tensor primitives.
 */
public final class Conv2dDagLoweringRule extends LocalTensorRewriteRule {
    private final Conv2dLoweringConfig config;

    public Conv2dDagLoweringRule() {
        this(Conv2dLoweringConfig.defaults());
    }

    public Conv2dDagLoweringRule(Conv2dLoweringConfig config) {
        this.config = config == null ? Conv2dLoweringConfig.defaults() : config;
    }

    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (!(op instanceof conv2d conv) || !shouldLower(tensor, conv)) {
            return tensor;
        }
        Tensor lowered = lower(tensor, conv);
        lowered.setLabel(tensor.getLabel());
        lowered.setRequiresGrad(tensor.getRequiresGrad());
        return lowered;
    }

    private boolean shouldLower(Tensor tensor, conv2d conv) {
        return switch (config.mode()) {
            case OFF -> false;
            case ALWAYS -> true;
            case HEURISTIC -> Conv2dDagLoweringHeuristics.shouldLower(tensor, conv, config.profile());
        };
    }

    private static Tensor lower(Tensor tensor, conv2d conv) {
        List<Tensor> inputs = tensor.getPrevTensors();
        Tensor input = inputs.get(0);
        Tensor weight = inputs.get(1);
        Tensor bias = conv.hasBias() ? inputs.get(2) : null;
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        int[] outputShape = tensor.getShapeUnsafe();
        Conv2dOptions options = conv.getOptions();
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outputShape[2];
        int outW = outputShape[3];
        int groups = options.groups();
        int outChannelsPerGroup = outChannels / groups;
        int flatChannelsPerGroup = channelsPerGroup * kernelH * kernelW;
        Tensor columns = input.unfold2d(windowOptions(options, kernelH, kernelW));
        Tensor flattened;
        if (groups == 1) {
            Tensor weight2d = weight.reshape(outChannels, flatChannelsPerGroup);
            flattened = weight2d.matmul(columns);
        } else {
            List<Tensor> groupOutputs = new ArrayList<>(groups);
            for (int group = 0; group < groups; group++) {
                int outStart = group * outChannelsPerGroup;
                int outEnd = outStart + outChannelsPerGroup;
                int columnStart = group * flatChannelsPerGroup;
                int columnEnd = columnStart + flatChannelsPerGroup;
                Tensor weightGroup = weight.sliceAxis(0, outStart, outEnd)
                        .reshape(outChannelsPerGroup, flatChannelsPerGroup);
                Tensor columnsGroup = columns.sliceAxis(1, columnStart, columnEnd);
                groupOutputs.add(weightGroup.matmul(columnsGroup));
            }
            flattened = TensorOps.concat(1, groupOutputs);
        }
        Tensor out = flattened.reshape(batch, outChannels, outH, outW);
        if (bias != null) {
            out = out.add(bias.reshape(1, outChannels, 1, 1));
        }
        return out;
    }

    private static Window2dOptions windowOptions(Conv2dOptions options, int kernelH, int kernelW) {
        return new Window2dOptions(
                kernelH,
                kernelW,
                options.strideH(),
                options.strideW(),
                options.padH(),
                options.padW(),
                options.dilationH(),
                options.dilationW()
        );
    }
}
