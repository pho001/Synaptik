package Backend.kernels.cpu.f16;

import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuThreadPool;
import Utils.FastExp;

public final class UnaryF16 {
    private UnaryF16() {}

    public static void inv(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.INV); }
    public static void relu(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.RELU); }
    public static void exp(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.EXP); }
    public static void fastExp(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.FAST_EXP); }
    public static void log(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.LOG); }
    public static void tanh(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.TANH); }
    public static void fastTanh(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.FAST_TANH); }
    public static void sqrt(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.SQRT); }
    public static void sigmoid(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) { run(in, out, mode, config, Op.SIGMOID); }

    private static void run(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config, Op op) {
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length, op);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, config, op);
        }
    }

    private static void scalar(short[] in, short[] out, int start, int end, Op op) {
        for (int i = start; i < end; i++) {
            float x = CpuDTypeOps.fromHalfBits(in[i]);
            float y = switch (op) {
                case INV -> 1.0f / x;
                case RELU -> Math.max(0.0f, x);
                case EXP -> (float) Math.exp(x);
                case FAST_EXP -> FastExp.fastExpF32(x);
                case LOG -> (float) Math.log(x);
                case TANH -> (float) Math.tanh(x);
                case FAST_TANH -> FastExp.fastTanhF32(x);
                case SQRT -> (float) Math.sqrt(x);
                case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-x));
            };
            out[i] = CpuDTypeOps.toHalfBits(y);
        }
    }

    private static void parallel(short[] in, short[] out, CpuExecutionConfig config, Op op) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end, op);
        });
    }

    private enum Op { INV, RELU, EXP, FAST_EXP, LOG, TANH, FAST_TANH, SQRT, SIGMOID }
}
