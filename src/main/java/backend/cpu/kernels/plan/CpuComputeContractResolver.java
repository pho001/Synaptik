package backend.cpu.kernels.plan;

import backend.cpu.kernels.CpuAccumulateDType;
import backend.cpu.kernels.CpuComputeDType;
import backend.cpu.kernels.CpuExecutionBackend;
import backend.cpu.kernels.nn.conv2d.plan.ResolvedConv2dHints;
import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

public final class CpuComputeContractResolver {
    public ResolvedCpuComputeContract resolve(
            Operation op,
            Tensor node,
            ResolvedMatMulHints matMulHints,
            ResolvedConv2dHints conv2dHints
    ) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        DataType dataType = node.getDataType() == null ? DataType.FLOAT64 : node.getDataType();
        if (op == null) {
            return defaultContractFor(dataType, CpuExecutionBackend.CPU_GENERIC);
        }
        return switch (op.opType()) {
            case MATMUL, LINEAR -> resolveMatMulContract(dataType, matMulHints);
            case CONV2D_GEMM, CONV2D_BACKWARD_INPUT_GEMM, CONV2D_BACKWARD_WEIGHT_GEMM ->
                    resolveConv2dContract(dataType, conv2dHints);
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD ->
                    resolveReductionContract(dataType);
            case FUSED -> defaultContractFor(dataType, CpuExecutionBackend.CPU_FUSED);
            default -> (op.opType().category() == Operation.OpArityClass.ELEMENT_WISE)
                    ? defaultContractFor(dataType, CpuExecutionBackend.CPU_ELEMENTWISE)
                    : defaultContractFor(dataType, CpuExecutionBackend.CPU_GENERIC);
        };
    }

    private ResolvedCpuComputeContract resolveMatMulContract(DataType dataType, ResolvedMatMulHints matMulHints) {
        CpuExecutionBackend backend = (matMulHints != null && (matMulHints.useBlas() || matMulHints.useBatchedBlas()))
                ? CpuExecutionBackend.CPU_MATMUL_BLAS
                : CpuExecutionBackend.CPU_MATMUL_JAVA;
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract resolveReductionContract(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract resolveConv2dContract(DataType dataType, ResolvedConv2dHints conv2dHints) {
        CpuExecutionBackend backend = (conv2dHints != null && conv2dHints.useBlas())
                ? CpuExecutionBackend.CPU_MATMUL_BLAS
                : CpuExecutionBackend.CPU_MATMUL_JAVA;
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract defaultContractFor(DataType dataType, CpuExecutionBackend backend) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }
}
