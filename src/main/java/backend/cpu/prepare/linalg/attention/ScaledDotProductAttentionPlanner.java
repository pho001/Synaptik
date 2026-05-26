package backend.cpu.prepare.linalg.attention;

import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.linalg.attention.ResolvedAttentionHints;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.prepare.reduction.ReductionPlanner;
import config.runtime.BlasConfig;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import tensor.DataType;

import java.util.List;

public final class ScaledDotProductAttentionPlanner {
    private final ReductionPlanner reductionPlanner;

    public ScaledDotProductAttentionPlanner(ReductionPlanner reductionPlanner) {
        this.reductionPlanner = reductionPlanner;
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

    private ResolvedCpuComputeContract attentionComputeContract(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case FLOAT32, BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Attention compute contract requires floating dtype.");
        };
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
