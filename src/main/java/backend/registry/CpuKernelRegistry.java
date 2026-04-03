package backend.registry;

import backend.kernels.cpu.*;
import operations.Operation;

import java.util.EnumMap;
import java.util.Map;

public final class CpuKernelRegistry {
    private static final Map<Operation.OpType, CpuKernel> KERNELS = new EnumMap<>(Operation.OpType.class);

    static {
        KERNELS.put(Operation.OpType.ADD, new CpuAddKernel());
        KERNELS.put(Operation.OpType.SUB, new CpuSubKernel());
        KERNELS.put(Operation.OpType.MUL, new CpuMulKernel());
        KERNELS.put(Operation.OpType.DIV, new CpuDivKernel());
        KERNELS.put(Operation.OpType.MIN, new CpuMinKernel());
        KERNELS.put(Operation.OpType.MAX, new CpuMaxKernel());
        KERNELS.put(Operation.OpType.GT, new CpuGreaterThanKernel());
        KERNELS.put(Operation.OpType.GE, new CpuGreaterOrEqualKernel());
        KERNELS.put(Operation.OpType.LT, new CpuLessThanKernel());
        KERNELS.put(Operation.OpType.LE, new CpuLessOrEqualKernel());
        KERNELS.put(Operation.OpType.EQ, new CpuEqualToKernel());
        KERNELS.put(Operation.OpType.NE, new CpuNotEqualToKernel());
        KERNELS.put(Operation.OpType.LOGICAL_AND, new CpuLogicalAndKernel());
        KERNELS.put(Operation.OpType.LOGICAL_OR, new CpuLogicalOrKernel());
        KERNELS.put(Operation.OpType.LOGICAL_NOT, new CpuLogicalNotKernel());
        KERNELS.put(Operation.OpType.MIN_GRAD, new CpuMinGradKernel());
        KERNELS.put(Operation.OpType.MAX_GRAD, new CpuMaxGradKernel());
        KERNELS.put(Operation.OpType.REDUCE_MIN, new CpuReduceMinKernel());
        KERNELS.put(Operation.OpType.REDUCE_MAX, new CpuReduceMaxKernel());
        KERNELS.put(Operation.OpType.REDUCE_ALL, new CpuReduceAllKernel());
        KERNELS.put(Operation.OpType.REDUCE_ANY, new CpuReduceAnyKernel());
        KERNELS.put(Operation.OpType.REDUCE_MIN_GRAD, new CpuReduceMinGradKernel());
        KERNELS.put(Operation.OpType.REDUCE_MAX_GRAD, new CpuReduceMaxGradKernel());
        KERNELS.put(Operation.OpType.MATMUL, new CpuMatMulKernel());
        KERNELS.put(Operation.OpType.NEG, new CpuNegKernel());
        KERNELS.put(Operation.OpType.INV, new CpuInvKernel());
        KERNELS.put(Operation.OpType.LOG, new CpuLogKernel());
        KERNELS.put(Operation.OpType.EXP, new CpuExpKernel());
        KERNELS.put(Operation.OpType.FAST_EXP, new CpuFastExpKernel());
        KERNELS.put(Operation.OpType.TANH, new CpuTanhKernel());
        KERNELS.put(Operation.OpType.FAST_TANH, new CpuFastTanhKernel());
        KERNELS.put(Operation.OpType.POW, new CpuPowKernel());
        KERNELS.put(Operation.OpType.SQRT, new CpuSqrtKernel());
        KERNELS.put(Operation.OpType.MUL_SCALAR, new CpuMulScalarKernel());
        KERNELS.put(Operation.OpType.SUM, new CpuSumKernel());
        KERNELS.put(Operation.OpType.RELU, new CpuReluKernel());
        KERNELS.put(Operation.OpType.SIGMOID, new CpuSigmoidKernel());
        KERNELS.put(Operation.OpType.WHERE, new CpuWhereKernel());
        KERNELS.put(Operation.OpType.CONTIGUOUS, new CpuContiguousKernel());
        KERNELS.put(Operation.OpType.RESHAPE, new CpuReshapeLikeKernel());
        KERNELS.put(Operation.OpType.EXPAND, new CpuExpandKernel());
        KERNELS.put(Operation.OpType.EXPAND_DIMS, new CpuAliasViewKernel());
        KERNELS.put(Operation.OpType.SQUEEZE, new CpuAliasViewKernel());
        KERNELS.put(Operation.OpType.PERMUTE, new CpuPermuteKernel());
        KERNELS.put(Operation.OpType.NOOP, new CpuNoopKernel());
        KERNELS.put(Operation.OpType.FUSED, new CpuFusedKernel());
    }

    private CpuKernelRegistry() {}

    public static CpuKernel resolve(Operation.OpType type) {
        return KERNELS.get(type);
    }
}
