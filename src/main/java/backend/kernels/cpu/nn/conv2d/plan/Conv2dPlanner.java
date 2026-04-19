package backend.kernels.cpu.nn.conv2d.plan;

import backend.blas.BlasProvider;
import backend.kernels.cpu.nn.conv2d.plan.ResolvedConv2dHints;
import config.runtime.Conv2dConfig;
import operations.Operation;
import operations.nn.conv.conv2dBackwardInputGemm;
import operations.nn.conv.conv2dBackwardWeightGemm;
import operations.nn.conv.conv2dGemm;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class Conv2dPlanner {
    public ResolvedConv2dHints resolve(Operation op, List<Tensor> inputs, Tensor node, Conv2dConfig conv2dConfig) {
        if (op == null || node == null || conv2dConfig == null) {
            return null;
        }
        return switch (op.opType()) {
            case CONV2D_GEMM -> resolveForward(inputs, node, conv2dConfig, (conv2dGemm) op);
            case CONV2D_BACKWARD_WEIGHT_GEMM -> resolveBackwardWeight(inputs, conv2dConfig, (conv2dBackwardWeightGemm) op);
            case CONV2D_BACKWARD_INPUT_GEMM -> resolveBackwardInput(inputs, conv2dConfig, (conv2dBackwardInputGemm) op);
            default -> null;
        };
    }

    private ResolvedConv2dHints resolveForward(
            List<Tensor> inputs,
            Tensor node,
            Conv2dConfig conv2dConfig,
            conv2dGemm op
    ) {
        if (inputs == null || inputs.size() < 2) {
            return null;
        }
        Tensor weight = inputs.get(1);
        if (weight == null) {
            return null;
        }
        int[] weightShape = weight.getShapeUnsafe();
        int[] outShape = node.getShapeUnsafe();
        if (weightShape.length != 4 || outShape.length != 4) {
            return null;
        }
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / op.getOptions().groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        return resolveGemmHint(node.getDataType(), outH * outW, outChannelsPerGroup, kSize, conv2dConfig);
    }

    private ResolvedConv2dHints resolveBackwardWeight(
            List<Tensor> inputs,
            Conv2dConfig conv2dConfig,
            conv2dBackwardWeightGemm op
    ) {
        if (inputs == null || inputs.size() < 2) {
            return null;
        }
        Tensor outGrad = inputs.get(1);
        if (outGrad == null) {
            return null;
        }
        int[] weightShape = op.getWeightShape();
        int[] outGradShape = outGrad.getShapeUnsafe();
        if (weightShape.length != 4 || outGradShape.length != 4) {
            return null;
        }
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / op.getOptions().groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        return resolveGemmHint(outGrad.getDataType(), kSize, outChannelsPerGroup, outH * outW, conv2dConfig);
    }

    private ResolvedConv2dHints resolveBackwardInput(
            List<Tensor> inputs,
            Conv2dConfig conv2dConfig,
            conv2dBackwardInputGemm op
    ) {
        if (inputs == null || inputs.size() < 2) {
            return null;
        }
        Tensor weight = inputs.get(0);
        Tensor outGrad = inputs.get(1);
        if (weight == null || outGrad == null) {
            return null;
        }
        int[] weightShape = weight.getShapeUnsafe();
        int[] outGradShape = outGrad.getShapeUnsafe();
        if (weightShape.length != 4 || outGradShape.length != 4) {
            return null;
        }
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / op.getOptions().groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        return resolveGemmHint(outGrad.getDataType(), outH * outW, kSize, outChannelsPerGroup, conv2dConfig);
    }

    private ResolvedConv2dHints resolveGemmHint(
            DataType dataType,
            int m,
            int n,
            int k,
            Conv2dConfig conv2dConfig
    ) {
        if (dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32 && dataType != DataType.BFLOAT16) {
            return new ResolvedConv2dHints(false, BlasProvider.NONE, m, n, k, 0L);
        }
        long work = (long) m * n * k;
        BlasProvider provider = conv2dConfig.provider();
        boolean useBlas = shouldUseBlas(dataType, m, n, k, work, conv2dConfig);
        return new ResolvedConv2dHints(useBlas, useBlas ? provider : BlasProvider.NONE, m, n, k, work);
    }

    private boolean shouldUseBlas(
            DataType dataType,
            int m,
            int n,
            int k,
            long work,
            Conv2dConfig conv2dConfig
    ) {
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            return false;
        }
        if (!backend.blas.OpenBlasFfmBridge.isAvailable()) {
            return false;
        }
        if (conv2dConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        if (work < conv2dConfig.minWork(dataType)) {
            return false;
        }
        if (dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16) {
            if (conv2dConfig.requireMgeK(dataType) && m < k) {
                return false;
            }
            if (((double) n / Math.max(1, k)) > conv2dConfig.maxNOverK(dataType)) {
                return false;
            }
        }
        return true;
    }
}
