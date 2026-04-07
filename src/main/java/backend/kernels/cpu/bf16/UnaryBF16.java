package backend.kernels.cpu.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import utils.FastExp;

public final class UnaryBF16 {
    private UnaryBF16() {}

    public static void inv(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.INV); }
    public static void relu(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.RELU); }
    public static void abs(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.ABS); }
    public static void clampMin(short[] in, float minValue, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.CLAMP_MIN, minValue); }
    public static void clampMax(short[] in, float maxValue, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.CLAMP_MAX, maxValue); }
    public static void exp(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.EXP); }
    public static void fastExp(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.FAST_EXP); }
    public static void log(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.LOG); }
    public static void tanh(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.TANH); }
    public static void fastTanh(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.FAST_TANH); }
    public static void sqrt(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.SQRT); }
    public static void sigmoid(short[] in, short[] out, ResolvedDispatchHints hints) { run(in, out, hints, Op.SIGMOID); }
    public static void inv(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.INV, 0.0f); }
    public static void relu(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.RELU, 0.0f); }
    public static void abs(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.ABS, 0.0f); }
    public static void clampMin(float[] in, float minValue, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.CLAMP_MIN, minValue); }
    public static void clampMax(float[] in, float maxValue, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.CLAMP_MAX, maxValue); }
    public static void exp(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.EXP, 0.0f); }
    public static void fastExp(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.FAST_EXP, 0.0f); }
    public static void log(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.LOG, 0.0f); }
    public static void tanh(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.TANH, 0.0f); }
    public static void fastTanh(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.FAST_TANH, 0.0f); }
    public static void sqrt(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.SQRT, 0.0f); }
    public static void sigmoid(float[] in, short[] out, ResolvedDispatchHints hints) { runFloat(in, out, hints, Op.SIGMOID, 0.0f); }

    private static void run(short[] in, short[] out, ResolvedDispatchHints hints, Op op) {
        run(in, out, hints, op, 0.0f);
    }

    private static void run(short[] in, short[] out, ResolvedDispatchHints hints, Op op, float scalar) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length, op, scalar);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, hints, op, scalar);
        }
    }

    private static void runFloat(float[] in, short[] out, ResolvedDispatchHints hints, Op op, float scalar) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalarFloat(in, out, 0, out.length, op, scalar);
            case PARALLEL, PARALLEL_VECTOR -> parallelFloat(in, out, hints, op, scalar);
        }
    }

    private static void scalar(short[] in, short[] out, int start, int end, Op op, float scalar) {
        for (int i = start; i < end; i++) {
            float x = CpuDTypeOps.fromBFloat16Bits(in[i]);
            float y = switch (op) {
                case INV -> 1.0f / x;
                case RELU -> Math.max(0.0f, x);
                case ABS -> Math.abs(x);
                case CLAMP_MIN -> Math.max(scalar, x);
                case CLAMP_MAX -> Math.min(scalar, x);
                case EXP -> (float) Math.exp(x);
                case FAST_EXP -> FastExp.fastExpF32(x);
                case LOG -> (float) Math.log(x);
                case TANH -> (float) Math.tanh(x);
                case FAST_TANH -> FastExp.fastTanhF32(x);
                case SQRT -> (float) Math.sqrt(x);
                case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-x));
            };
            out[i] = CpuDTypeOps.toBFloat16Bits(y);
        }
    }

    private static void parallel(short[] in, short[] out, ResolvedDispatchHints hints, Op op, float scalar) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end, op, scalar);
        });
    }

    private static void scalarFloat(float[] in, short[] out, int start, int end, Op op, float scalar) {
        for (int i = start; i < end; i++) {
            float x = in[i];
            float y = switch (op) {
                case INV -> 1.0f / x;
                case RELU -> Math.max(0.0f, x);
                case ABS -> Math.abs(x);
                case CLAMP_MIN -> Math.max(scalar, x);
                case CLAMP_MAX -> Math.min(scalar, x);
                case EXP -> (float) Math.exp(x);
                case FAST_EXP -> FastExp.fastExpF32(x);
                case LOG -> (float) Math.log(x);
                case TANH -> (float) Math.tanh(x);
                case FAST_TANH -> FastExp.fastTanhF32(x);
                case SQRT -> (float) Math.sqrt(x);
                case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-x));
            };
            out[i] = CpuDTypeOps.toBFloat16Bits(y);
        }
    }

    private static void parallelFloat(float[] in, short[] out, ResolvedDispatchHints hints, Op op, float scalar) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarFloat(in, out, start, end, op, scalar);
        });
    }

    private enum Op { INV, RELU, ABS, CLAMP_MIN, CLAMP_MAX, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SQRT, SIGMOID }
}
