package backend.cpu.prepare.linalg.attention;

import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.linalg.attention.ResolvedAttentionHints;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.prepare.linalg.matmul.MatMulPlanner;
import backend.cpu.prepare.reduction.ReductionPlanner;
import config.runtime.BlasConfig;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import tensor.DataType;

import java.util.Arrays;
import java.util.List;

public final class ScaledDotProductAttentionPlanner {
    private final ReductionPlanner reductionPlanner;
    private final MatMulPlanner matMulPlanner;

    public ScaledDotProductAttentionPlanner(ReductionPlanner reductionPlanner, MatMulPlanner matMulPlanner) {
        this.reductionPlanner = reductionPlanner;
        this.matMulPlanner = matMulPlanner;
    }

    public ResolvedScaledDotProductAttentionPlan resolve(
            Operation op,
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node,
            CompiledTensorDescriptorIndex descriptorIndex,
            BlasConfig blasConfig
    ) {
        if (op == null || node == null || blasConfig == null) {
            return null;
        }
        return switch (op.opType()) {
            case SCALED_DOT_PRODUCT_ATTENTION -> resolveForward(inputs, node);
            case SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> resolveBackward(inputs, node, descriptorIndex, blasConfig);
            default -> null;
        };
    }

    private ResolvedScaledDotProductAttentionPlan resolveForward(List<CompiledTensorDescriptor> inputs, CompiledTensorDescriptor node) {
        if (inputs == null || inputs.size() < 3) {
            return null;
        }
        CompiledTensorDescriptor query = inputs.get(0);
        CompiledTensorDescriptor key = inputs.get(1);
        if (query == null || key == null) {
            return null;
        }
        int[] outShape = node.shape();
        int batchCount = attentionBatchCount(outShape);
        int queryLen = outShape[outShape.length - 2];
        int[] keyShape = key.shape();
        int[] queryShape = query.shape();
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        int valueDim = outShape[outShape.length - 1];
        ResolvedAttentionHints directHints = reductionPlanner.resolveAttentionHints(
                batchCount * queryLen,
                Math.max(1, keyLen * (depth + valueDim)),
                Math.max(depth, valueDim),
                attentionComputeContract(node.dataType())
        );
        return new ResolvedScaledDotProductAttentionPlan(directHints, null, null, null, null, null);
    }

    private ResolvedScaledDotProductAttentionPlan resolveBackward(
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node,
            CompiledTensorDescriptorIndex descriptorIndex,
            BlasConfig blasConfig
    ) {
        if (inputs == null || inputs.size() < 2) {
            return null;
        }
        CompiledTensorDescriptor attentionOut = inputs.get(0);
        CompiledTensorDescriptor outGrad = inputs.get(1);
        if (attentionOut == null || outGrad == null || descriptorIndex == null) {
            return null;
        }
        if (attentionOut.opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || attentionOut.inputIds().size() < 3) {
            return null;
        }
        CompiledTensorDescriptor query = descriptorIndex.byNodeId(attentionOut.inputIds().get(0));
        CompiledTensorDescriptor key = descriptorIndex.byNodeId(attentionOut.inputIds().get(1));
        CompiledTensorDescriptor value = descriptorIndex.byNodeId(attentionOut.inputIds().get(2));
        if (query == null || key == null || value == null) {
            return null;
        }

        DataType dataType = node.dataType();
        int[] attentionOutShape = attentionOut.shape();
        int[] queryShape = query.shape();
        int[] keyShape = key.shape();
        int[] valueShape = value.shape();
        int[] weightsShape = attentionScoreShape(queryShape, keyShape);
        int[] queryGradShape = attentionRawQueryGradShape(attentionOutShape, queryShape);
        int[] keyGradShape = attentionRawKeyGradShape(attentionOutShape, keyShape);
        int[] valueGradShape = attentionRawValueGradShape(attentionOutShape, valueShape);

        int rowLength = weightsShape[weightsShape.length - 1];
        int rowCount = attentionProduct(weightsShape) / Math.max(1, rowLength);
        ResolvedAttentionHints softmaxGradHints = reductionPlanner.resolveAttentionHints(
                rowCount,
                rowLength,
                rowLength,
                attentionComputeContract(dataType)
        );

        ResolvedMatMulHints queryGradHints = matMulPlanner.resolveAttention(
                weightsShape,
                true,
                keyShape,
                key.contiguous(),
                queryGradShape,
                dataType,
                true,
                blasConfig
        );
        ResolvedMatMulHints dWeightsHints = matMulPlanner.resolveAttentionJava(
                outGrad.shape(),
                attentionTransposedShape(valueShape),
                weightsShape,
                dataType
        );
        ResolvedMatMulHints valueGradHints = matMulPlanner.resolveAttentionJava(
                attentionTransposedShape(weightsShape),
                outGrad.shape(),
                valueGradShape,
                dataType
        );
        ResolvedMatMulHints keyGradHints = matMulPlanner.resolveAttentionJava(
                attentionTransposedShape(weightsShape),
                queryShape,
                keyGradShape,
                dataType
        );

        return new ResolvedScaledDotProductAttentionPlan(
                null,
                softmaxGradHints,
                queryGradHints,
                dWeightsHints,
                valueGradHints,
                keyGradHints
        );
    }

    private ResolvedCpuComputeContract attentionComputeContract(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case FLOAT32, BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Attention compute contract requires floating dtype.");
        };
    }

    private static int[] attentionTransposedShape(int[] shape) {
        int[] out = shape.clone();
        int tmp = out[out.length - 1];
        out[out.length - 1] = out[out.length - 2];
        out[out.length - 2] = tmp;
        return out;
    }

    private static int[] attentionScoreShape(int[] queryShape, int[] keyShape) {
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] outBatch = broadcastLeadingShape(qBatch, kBatch);
        int[] out = Arrays.copyOf(outBatch, outBatch.length + 2);
        out[outBatch.length] = queryShape[queryShape.length - 2];
        out[outBatch.length + 1] = keyShape[keyShape.length - 2];
        return out;
    }

    private static int[] attentionRawQueryGradShape(int[] outShape, int[] queryShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = queryShape[queryShape.length - 2];
        out[out.length - 1] = queryShape[queryShape.length - 1];
        return out;
    }

    private static int[] attentionRawKeyGradShape(int[] outShape, int[] keyShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = keyShape[keyShape.length - 2];
        out[out.length - 1] = keyShape[keyShape.length - 1];
        return out;
    }

    private static int[] attentionRawValueGradShape(int[] outShape, int[] valueShape) {
        int[] out = outShape.clone();
        out[out.length - 2] = valueShape[valueShape.length - 2];
        out[out.length - 1] = valueShape[valueShape.length - 1];
        return out;
    }

    private static int[] broadcastLeadingShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int a = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int b = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (a != b && a != 1 && b != 1) {
                throw new IllegalArgumentException("attention batch dimensions are not broadcast-compatible.");
            }
            out[i] = Math.max(a, b);
        }
        return out;
    }

    private static int attentionBatchCount(int[] shape) {
        int count = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            count *= shape[i];
        }
        return count;
    }

    private static int attentionProduct(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
}
