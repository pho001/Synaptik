package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import tensor.Tensor;

import java.util.List;

public final class ElementwiseUnaryExecutor {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private ElementwiseUnaryExecutor() {}

    public static void execute(UnaryOp op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (node.getDataType()) {
            case FLOAT64 -> runF64(op, inputs.getFirst().getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
            case FLOAT32 -> runF32(op, inputs.getFirst().getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
            case BFLOAT16 -> runBF16(op, inputs.getFirst().getBFloat16Data(), context.inputFloatContinuation(0, node.getFlatDataSize()), node.getBFloat16Data(), context.dispatchHints());
            default -> throw new UnsupportedOperationException("Unsupported dtype for unary elementwise op: " + node.getDataType());
        }
    }

    public static void execute(ScalarUnaryOp op, double parameter, float parameterF32, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (node.getDataType()) {
            case FLOAT64 -> runF64(op, parameter, inputs.getFirst().getFloat64Data(), node.getFloat64Data(), context.dispatchHints());
            case FLOAT32 -> runF32(op, parameterF32, inputs.getFirst().getFloat32Data(), node.getFloat32Data(), context.dispatchHints());
            case BFLOAT16 -> runBF16(op, parameterF32, inputs.getFirst().getBFloat16Data(), context.inputFloatContinuation(0, node.getFlatDataSize()), node.getBFloat16Data(), context.dispatchHints());
            default -> throw new UnsupportedOperationException("Unsupported dtype for scalar unary elementwise op: " + node.getDataType());
        }
    }

    private static void runF64(UnaryOp op, double[] in, double[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            parallelF64((start, end) -> scalarF64(op, in, out, start, end), (start, end) -> vectorF64(op, in, out, start, end), hints, op.supportsVectorF64());
            return;
        }
        if (preferVector(hints) && op.supportsVectorF64()) {
            vectorF64(op, in, out, 0, out.length);
        } else {
            scalarF64(op, in, out, 0, out.length);
        }
    }

    private static void runF32(UnaryOp op, float[] in, float[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            parallelF32((start, end) -> scalarF32(op, in, out, start, end), (start, end) -> vectorF32(op, in, out, start, end), hints, op.supportsVectorF32());
            return;
        }
        if (preferVector(hints) && op.supportsVectorF32()) {
            vectorF32(op, in, out, 0, out.length);
        } else {
            scalarF32(op, in, out, 0, out.length);
        }
    }

    private static void runBF16(UnaryOp op, short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBF16(op, in, continuation, out, start, end);
            });
            return;
        }
        scalarBF16(op, in, continuation, out, 0, out.length);
    }

    private static void runF64(ScalarUnaryOp op, double parameter, double[] in, double[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            parallelF64((start, end) -> scalarF64(op, parameter, in, out, start, end), (start, end) -> vectorF64(op, parameter, in, out, start, end), hints, op.supportsVectorF64());
            return;
        }
        if (preferVector(hints) && op.supportsVectorF64()) {
            vectorF64(op, parameter, in, out, 0, out.length);
        } else {
            scalarF64(op, parameter, in, out, 0, out.length);
        }
    }

    private static void runF32(ScalarUnaryOp op, float parameter, float[] in, float[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            parallelF32((start, end) -> scalarF32(op, parameter, in, out, start, end), (start, end) -> vectorF32(op, parameter, in, out, start, end), hints, op.supportsVectorF32());
            return;
        }
        if (preferVector(hints) && op.supportsVectorF32()) {
            vectorF32(op, parameter, in, out, 0, out.length);
        } else {
            scalarF32(op, parameter, in, out, 0, out.length);
        }
    }

    private static void runBF16(ScalarUnaryOp op, float parameter, short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        if (preferParallel(hints)) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBF16(op, parameter, in, continuation, out, start, end);
            });
            return;
        }
        scalarBF16(op, parameter, in, continuation, out, 0, out.length);
    }

    private static boolean preferParallel(ResolvedDispatchHints hints) {
        return hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR;
    }

    private static boolean preferVector(ResolvedDispatchHints hints) {
        return hints.mode() == CpuExecutionMode.VECTOR || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR;
    }

    private static void scalarF64(UnaryOp op, double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF64(in[i]);
        }
    }

    private static void scalarF32(UnaryOp op, float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF32(in[i]);
        }
    }

    private static void scalarBF16(UnaryOp op, short[] in, float[] continuation, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float value = continuation != null ? continuation[i] : CpuDTypeOps.fromBFloat16Bits(in[i]);
            out[i] = CpuDTypeOps.toBFloat16Bits(op.applyBF16(value));
        }
    }

    private static void scalarF64(ScalarUnaryOp op, double parameter, double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF64(in[i], parameter);
        }
    }

    private static void scalarF32(ScalarUnaryOp op, float parameter, float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF32(in[i], parameter);
        }
    }

    private static void scalarBF16(ScalarUnaryOp op, float parameter, short[] in, float[] continuation, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float value = continuation != null ? continuation[i] : CpuDTypeOps.fromBFloat16Bits(in[i]);
            out[i] = CpuDTypeOps.toBFloat16Bits(op.applyBF16(value, parameter));
        }
    }

    private static void vectorF64(UnaryOp op, double[] in, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            op.applyVectorF64(DoubleVector.fromArray(F64, in, i)).intoArray(out, i);
        }
        scalarF64(op, in, out, i, end);
    }

    private static void vectorF32(UnaryOp op, float[] in, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            op.applyVectorF32(FloatVector.fromArray(F32, in, i)).intoArray(out, i);
        }
        scalarF32(op, in, out, i, end);
    }

    private static void vectorF64(ScalarUnaryOp op, double parameter, double[] in, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            op.applyVectorF64(DoubleVector.fromArray(F64, in, i), parameter).intoArray(out, i);
        }
        scalarF64(op, parameter, in, out, i, end);
    }

    private static void vectorF32(ScalarUnaryOp op, float parameter, float[] in, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            op.applyVectorF32(FloatVector.fromArray(F32, in, i), parameter).intoArray(out, i);
        }
        scalarF32(op, parameter, in, out, i, end);
    }

    private static void parallelF64(RangeConsumer scalar, RangeConsumer vector, ResolvedDispatchHints hints, boolean vectorSupported) {
        int chunkSize = vectorSupported && hints.mode() == CpuExecutionMode.PARALLEL_VECTOR ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (hints.totalLength() + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, hints.totalLength());
            if (vectorSupported && hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
                vector.accept(start, end);
            } else {
                scalar.accept(start, end);
            }
        });
    }

    private static void parallelF32(RangeConsumer scalar, RangeConsumer vector, ResolvedDispatchHints hints, boolean vectorSupported) {
        int chunkSize = vectorSupported && hints.mode() == CpuExecutionMode.PARALLEL_VECTOR ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (hints.totalLength() + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, hints.totalLength());
            if (vectorSupported && hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
                vector.accept(start, end);
            } else {
                scalar.accept(start, end);
            }
        });
    }

    @FunctionalInterface
    private interface RangeConsumer {
        void accept(int start, int end);
    }
}
