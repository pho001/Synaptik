package backend.cpu.prepare;

import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import planning.descriptor.CompiledTensorDescriptor;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

public final class CpuComputeContractResolver {
    public ResolvedCpuComputeContract resolve(
            Operation op,
            Tensor node,
            ResolvedMatMulHints matMulHints
    ) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        DataType dataType = node.getDataType() == null ? DataType.FLOAT64 : node.getDataType();
        return resolve(op, dataType, matMulHints);
    }

    public ResolvedCpuComputeContract resolve(
            Operation op,
            CompiledTensorDescriptor descriptor,
            ResolvedMatMulHints matMulHints
    ) {
        if (descriptor == null) {
            throw new IllegalArgumentException("descriptor cannot be null");
        }
        return resolve(op, descriptor.dataType(), matMulHints);
    }

    private ResolvedCpuComputeContract resolve(
            Operation op,
            DataType dataType,
            ResolvedMatMulHints matMulHints
    ) {
        if (op == null) {
            return defaultContractFor(dataType, CpuExecutionBackend.CPU_GENERIC);
        }
        return switch (op.opType()) {
            case MATMUL, LINEAR -> resolveMatMulContract(dataType, matMulHints);
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, LOG_SOFTMAX,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES ->
                    resolveReductionContract(dataType);
            case FUSED -> defaultContractFor(dataType, CpuExecutionBackend.CPU_FUSED);
            default -> (op.arityClass() == Operation.OpArityClass.ELEMENT_WISE)
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
            case INT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT64, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract resolveReductionContract(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
            case INT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract defaultContractFor(DataType dataType, CpuExecutionBackend backend) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case INT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT64, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }
}
