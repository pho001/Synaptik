package Backend.kernels.cpu.f64;

import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuThreadPool;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class UnaryF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private UnaryF64() {}

    public static void inv(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorInv(in, out);
            case PARALLEL -> parallelInv(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorInv(in, out, config);
            case SCALAR -> scalarInv(in, out, 0, out.length);
        }
    }

    public static void relu(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorRelu(in, out);
            case PARALLEL -> parallelRelu(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorRelu(in, out, config);
            case SCALAR -> scalarRelu(in, out, 0, out.length);
        }
    }

    public static void exp(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorExp(in, out);
            case PARALLEL -> parallelExp(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorExp(in, out, config);
            case SCALAR -> scalarExp(in, out, 0, out.length);
        }
    }

    public static void log(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorLog(in, out);
            case PARALLEL -> parallelLog(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorLog(in, out, config);
            case SCALAR -> scalarLog(in, out, 0, out.length);
        }
    }

    public static void tanh(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorTanh(in, out);
            case PARALLEL -> parallelTanh(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorTanh(in, out, config);
            case SCALAR -> scalarTanh(in, out, 0, out.length);
        }
    }

    public static void sqrt(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorSqrt(in, out);
            case PARALLEL -> parallelSqrt(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorSqrt(in, out, config);
            case SCALAR -> scalarSqrt(in, out, 0, out.length);
        }
    }

    public static void sigmoid(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vectorSigmoid(in, out);
            case PARALLEL -> parallelSigmoid(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorSigmoid(in, out, config);
            case SCALAR -> scalarSigmoid(in, out, 0, out.length);
        }
    }

    private static void scalarInv(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0 / in[i]; }
    private static void scalarRelu(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.max(0.0, in[i]); }
    private static void scalarExp(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.exp(in[i]); }
    private static void scalarLog(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.log(in[i]); }
    private static void scalarTanh(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.tanh(in[i]); }
    private static void scalarSqrt(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.sqrt(in[i]); }
    private static void scalarSigmoid(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0 / (1.0 + Math.exp(-in[i])); }

    private static void vectorInv(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
        for (; i < upper; i += SPECIES.length()) ones.div(DoubleVector.fromArray(SPECIES, in, i)).intoArray(out, i);
        scalarInv(in, out, i, out.length);
    }

    private static void vectorRelu(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        DoubleVector zero = DoubleVector.zero(SPECIES);
        for (; i < upper; i += SPECIES.length()) DoubleVector.fromArray(SPECIES, in, i).max(zero).intoArray(out, i);
        scalarRelu(in, out, i, out.length);
    }

    private static void vectorExp(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.EXP).intoArray(out, i);
        scalarExp(in, out, i, out.length);
    }

    private static void vectorLog(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.LOG).intoArray(out, i);
        scalarLog(in, out, i, out.length);
    }

    private static void vectorTanh(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.TANH).intoArray(out, i);
        scalarTanh(in, out, i, out.length);
    }

    private static void vectorSqrt(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.SQRT).intoArray(out, i);
        scalarSqrt(in, out, i, out.length);
    }

    private static void vectorSigmoid(double[] in, double[] out) {
        int i = 0, upper = SPECIES.loopBound(out.length);
        DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
        DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).mul(half).lanewise(VectorOperators.TANH).add(one).mul(half).intoArray(out, i);
        }
        scalarSigmoid(in, out, i, out.length);
    }

    private static void parallelInv(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarInv); }
    private static void parallelRelu(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarRelu); }
    private static void parallelExp(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarExp); }
    private static void parallelLog(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarLog); }
    private static void parallelTanh(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarTanh); }
    private static void parallelSqrt(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarSqrt); }
    private static void parallelSigmoid(double[] in, double[] out, CpuExecutionConfig config) { parallelScalar(in, out, config, UnaryF64::scalarSigmoid); }

    private static void parallelVectorInv(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorInvChunk, UnaryF64::scalarInv); }
    private static void parallelVectorRelu(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorReluChunk, UnaryF64::scalarRelu); }
    private static void parallelVectorExp(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorExpChunk, UnaryF64::scalarExp); }
    private static void parallelVectorLog(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorLogChunk, UnaryF64::scalarLog); }
    private static void parallelVectorTanh(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorTanhChunk, UnaryF64::scalarTanh); }
    private static void parallelVectorSqrt(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorSqrtChunk, UnaryF64::scalarSqrt); }
    private static void parallelVectorSigmoid(double[] in, double[] out, CpuExecutionConfig config) { parallelVector(in, out, config, UnaryF64::vectorSigmoidChunk, UnaryF64::scalarSigmoid); }

    private static void parallelScalar(double[] in, double[] out, CpuExecutionConfig config, ScalarOp op) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            op.apply(in, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double[] out, CpuExecutionConfig config, ChunkVectorOp vecOp, ScalarOp scalarOp) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            vecOp.apply(in, out, start, end, width);
            int i = end - ((end - start) % width);
            scalarOp.apply(in, out, i, end);
        });
    }

    private static void vectorReluChunk(double[] in, double[] out, int start, int end, int width) {
        DoubleVector zero = DoubleVector.zero(SPECIES);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).max(zero).intoArray(out, i);
        }
    }

    private static void vectorExpChunk(double[] in, double[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.EXP).intoArray(out, i);
        }
    }

    private static void vectorLogChunk(double[] in, double[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.LOG).intoArray(out, i);
        }
    }

    private static void vectorTanhChunk(double[] in, double[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.TANH).intoArray(out, i);
        }
    }

    private static void vectorSqrtChunk(double[] in, double[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.SQRT).intoArray(out, i);
        }
    }

    private static void vectorSigmoidChunk(double[] in, double[] out, int start, int end, int width) {
        DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
        DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).mul(half).lanewise(VectorOperators.TANH).add(one).mul(half).intoArray(out, i);
        }
    }

    private static void vectorInvChunk(double[] in, double[] out, int start, int end, int width) {
        DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            ones.div(DoubleVector.fromArray(SPECIES, in, i)).intoArray(out, i);
        }
    }

    @FunctionalInterface
    private interface ScalarOp { void apply(double[] in, double[] out, int start, int end); }

    @FunctionalInterface
    private interface ChunkVectorOp { void apply(double[] in, double[] out, int start, int end, int width); }
}
