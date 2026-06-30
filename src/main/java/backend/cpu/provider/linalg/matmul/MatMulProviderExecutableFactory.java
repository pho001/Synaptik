package backend.cpu.provider.linalg.matmul;

import backend.cpu.provider.linalg.matmul.bf16.BF16BatchedBlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.bf16.BF16BlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.bf16.BF16JavaMatMulExecutable;
import backend.cpu.provider.linalg.matmul.bf16.BF16NativeBlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f32.F32BatchedBlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f32.F32BlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f32.F32JavaMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f32.F32NativeBlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f64.F64BatchedBlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f64.F64BlasMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f64.F64JavaMatMulExecutable;
import backend.cpu.provider.linalg.matmul.f64.F64NativeBlasMatMulExecutable;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import planning.descriptor.CompiledTensorDescriptor;
import operations.Operation;
import tensor.DataType;

public final class MatMulProviderExecutableFactory {
    private MatMulProviderExecutableFactory() {
    }

    public static PreparedMatMulExecutable create(
            Operation op,
            CompiledTensorDescriptor node,
            ResolvedMatMulHints hints,
            boolean publishFloatContinuation
    ) {
        if (op == null || node == null || hints == null) {
            return null;
        }
        if (op.opType() != Operation.OpType.MATMUL && op.opType() != Operation.OpType.LINEAR) {
            return null;
        }
        return switch (hints.route()) {
            case OPENBLAS_NATIVE_SEGMENT -> createOpenBlasNativeSegment(node.dataType(), hints);
            case OPENBLAS_ARRAY_COPYING -> createOpenBlasArrayCopying(node.dataType(), hints, publishFloatContinuation);
            case JAVA_DIRECT -> createJavaDirect(node.dataType(), hints, publishFloatContinuation);
        };
    }

    private static PreparedMatMulExecutable createOpenBlasNativeSegment(
            DataType dataType,
            ResolvedMatMulHints hints
    ) {
        return switch (dataType) {
            case FLOAT64 -> new F64NativeBlasMatMulExecutable(hints);
            case FLOAT32 -> new F32NativeBlasMatMulExecutable(hints);
            case BFLOAT16 -> new BF16NativeBlasMatMulExecutable(hints);
            default -> null;
        };
    }

    private static PreparedMatMulExecutable createOpenBlasArrayCopying(
            DataType dataType,
            ResolvedMatMulHints hints,
            boolean publishFloatContinuation
    ) {
        return switch (dataType) {
            case FLOAT64 -> hints.useBatchedBlas()
                    ? new F64BatchedBlasMatMulExecutable(hints)
                    : new F64BlasMatMulExecutable(hints);
            case FLOAT32 -> hints.useBatchedBlas()
                    ? new F32BatchedBlasMatMulExecutable(hints)
                    : new F32BlasMatMulExecutable(hints);
            case BFLOAT16 -> hints.useBatchedBlas()
                    ? new BF16BatchedBlasMatMulExecutable(hints, publishFloatContinuation)
                    : new BF16BlasMatMulExecutable(hints, publishFloatContinuation);
            default -> null;
        };
    }

    private static PreparedMatMulExecutable createJavaDirect(
            DataType dataType,
            ResolvedMatMulHints hints,
            boolean publishFloatContinuation
    ) {
        return switch (dataType) {
            case FLOAT64 -> new F64JavaMatMulExecutable(hints);
            case FLOAT32 -> new F32JavaMatMulExecutable(hints);
            case BFLOAT16 -> new BF16JavaMatMulExecutable(hints, publishFloatContinuation);
            default -> null;
        };
    }
}
