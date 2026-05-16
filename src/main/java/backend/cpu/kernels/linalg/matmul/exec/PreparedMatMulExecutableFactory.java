package backend.cpu.kernels.linalg.matmul.exec;

import backend.cpu.kernels.linalg.matmul.bf16.BF16BatchedBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.bf16.BF16BlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.bf16.BF16JavaMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.bf16.BF16NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32BatchedBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32BlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32JavaMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f64.F64BatchedBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f64.F64BlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f64.F64JavaMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f64.F64NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import operations.Operation;
import tensor.Tensor;

public final class PreparedMatMulExecutableFactory {
    private PreparedMatMulExecutableFactory() {
    }

    public static PreparedMatMulExecutable create(
            Operation op,
            Tensor node,
            ResolvedMatMulHints hints,
            boolean publishFloatContinuation
    ) {
        if (op == null || node == null || hints == null) {
            return null;
        }
        if (op.opType() != Operation.OpType.MATMUL && op.opType() != Operation.OpType.LINEAR) {
            return null;
        }
        return switch (node.getDataType()) {
            case FLOAT64 -> createF64(hints);
            case FLOAT32 -> createF32(hints);
            case BFLOAT16 -> createBF16(hints, publishFloatContinuation);
            default -> null;
        };
    }

    private static PreparedMatMulExecutable createF64(ResolvedMatMulHints hints) {
        if (hints.route() == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT) {
            return new F64NativeBlasMatMulExecutable(hints);
        }
        if (hints.useBatchedBlas()) {
            return new F64BatchedBlasMatMulExecutable(hints);
        }
        if (hints.useBlas()) {
            return new F64BlasMatMulExecutable(hints);
        }
        return new F64JavaMatMulExecutable(hints);
    }

    private static PreparedMatMulExecutable createF32(ResolvedMatMulHints hints) {
        if (hints.route() == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT) {
            return new F32NativeBlasMatMulExecutable(hints);
        }
        if (hints.useBatchedBlas()) {
            return new F32BatchedBlasMatMulExecutable(hints);
        }
        if (hints.useBlas()) {
            return new F32BlasMatMulExecutable(hints);
        }
        return new F32JavaMatMulExecutable(hints);
    }

    private static PreparedMatMulExecutable createBF16(ResolvedMatMulHints hints, boolean publishFloatContinuation) {
        if (hints.route() == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT) {
            return new BF16NativeBlasMatMulExecutable(hints);
        }
        if (hints.useBatchedBlas()) {
            return new BF16BatchedBlasMatMulExecutable(hints, publishFloatContinuation);
        }
        if (hints.useBlas()) {
            return new BF16BlasMatMulExecutable(hints, publishFloatContinuation);
        }
        return new BF16JavaMatMulExecutable(hints, publishFloatContinuation);
    }
}
