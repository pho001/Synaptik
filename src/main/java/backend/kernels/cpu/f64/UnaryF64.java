package backend.kernels.cpu.f64;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import utils.FastExp;

public final class UnaryF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private UnaryF64() {}

    public static void inv(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorInv(in, out);
            case PARALLEL -> parallelInv(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorInv(in, out, hints);
            case SCALAR -> scalarInv(in, out, 0, out.length);
        }
    }

    public static void relu(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorRelu(in, out);
            case PARALLEL -> parallelRelu(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorRelu(in, out, hints);
            case SCALAR -> scalarRelu(in, out, 0, out.length);
        }
    }

    public static void clampMin(double[] in, double minValue, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorClampMin(in, minValue, out);
            case PARALLEL -> parallelClampMin(in, minValue, out, hints);
            case PARALLEL_VECTOR -> parallelVectorClampMin(in, minValue, out, hints);
            case SCALAR -> scalarClampMin(in, minValue, out, 0, out.length);
        }
    }

    public static void clampMax(double[] in, double maxValue, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorClampMax(in, maxValue, out);
            case PARALLEL -> parallelClampMax(in, maxValue, out, hints);
            case PARALLEL_VECTOR -> parallelVectorClampMax(in, maxValue, out, hints);
            case SCALAR -> scalarClampMax(in, maxValue, out, 0, out.length);
        }
    }

    public static void exp(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorExp(in, out);
            case PARALLEL -> parallelExp(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorExp(in, out, hints);
            case SCALAR -> scalarExp(in, out, 0, out.length);
        }
    }

    public static void fastExp(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR, SCALAR -> scalarFastExp(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelFastExp(in, out, hints);
        }
    }

    public static void log(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorLog(in, out);
            case PARALLEL -> parallelLog(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorLog(in, out, hints);
            case SCALAR -> scalarLog(in, out, 0, out.length);
        }
    }

    public static void tanh(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorTanh(in, out);
            case PARALLEL -> parallelTanh(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorTanh(in, out, hints);
            case SCALAR -> scalarTanh(in, out, 0, out.length);
        }
    }

    public static void fastTanh(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR, SCALAR -> scalarFastTanh(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelFastTanh(in, out, hints);
        }
    }

    public static void sqrt(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorSqrt(in, out);
            case PARALLEL -> parallelSqrt(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorSqrt(in, out, hints);
            case SCALAR -> scalarSqrt(in, out, 0, out.length);
        }
    }

    public static void abs(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorAbs(in, out);
            case PARALLEL -> parallelAbs(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorAbs(in, out, hints);
            case SCALAR -> scalarAbs(in, out, 0, out.length);
        }
    }

    public static void sigmoid(double[] in, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorSigmoid(in, out);
            case PARALLEL -> parallelSigmoid(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorSigmoid(in, out, hints);
            case SCALAR -> scalarSigmoid(in, out, 0, out.length);
        }
    }

    private static void scalarInv(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0 / in[i]; }
    private static void scalarRelu(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.max(0.0, in[i]); }
    private static void scalarClampMin(double[] in, double minValue, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.max(minValue, in[i]); }
    private static void scalarClampMax(double[] in, double maxValue, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.min(maxValue, in[i]); }
    private static void scalarExp(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.exp(in[i]); }
    private static void scalarFastExp(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = FastExp.fastExpF64(in[i]); }
    private static void scalarLog(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.log(in[i]); }
    private static void scalarTanh(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.tanh(in[i]); }
    private static void scalarFastTanh(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = FastExp.fastTanhF64(in[i]); }
    private static void scalarSqrt(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.sqrt(in[i]); }
    private static void scalarAbs(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.abs(in[i]); }
    private static void scalarSigmoid(double[] in, double[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0 / (1.0 + Math.exp(-in[i])); }

    private static void vectorInv(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
        for (; i < upper; i += SPECIES.length()) {
            ones.div(DoubleVector.fromArray(SPECIES, in, i)).intoArray(out, i);
        }
        scalarInv(in, out, i, out.length);
    }

    private static void vectorRelu(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector zero = DoubleVector.zero(SPECIES);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).max(zero).intoArray(out, i);
        }
        scalarRelu(in, out, i, out.length);
    }

    private static void vectorClampMin(double[] in, double minValue, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector floor = DoubleVector.broadcast(SPECIES, minValue);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).max(floor).intoArray(out, i);
        }
        scalarClampMin(in, minValue, out, i, out.length);
    }

    private static void vectorClampMax(double[] in, double maxValue, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector ceil = DoubleVector.broadcast(SPECIES, maxValue);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).min(ceil).intoArray(out, i);
        }
        scalarClampMax(in, maxValue, out, i, out.length);
    }

    private static void vectorExp(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.EXP).intoArray(out, i);
        }
        scalarExp(in, out, i, out.length);
    }

    private static void vectorLog(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.LOG).intoArray(out, i);
        }
        scalarLog(in, out, i, out.length);
    }

    private static void vectorTanh(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.TANH).intoArray(out, i);
        }
        scalarTanh(in, out, i, out.length);
    }

    private static void vectorSqrt(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.SQRT).intoArray(out, i);
        }
        scalarSqrt(in, out, i, out.length);
    }

    private static void vectorAbs(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.ABS).intoArray(out, i);
        }
        scalarAbs(in, out, i, out.length);
    }

    private static void vectorSigmoid(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
        DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).mul(half).lanewise(VectorOperators.TANH).add(one).mul(half).intoArray(out, i);
        }
        scalarSigmoid(in, out, i, out.length);
    }

    private static void parallelInv(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarInv); }
    private static void parallelRelu(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarRelu); }
    private static void parallelClampMin(double[] in, double minValue, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, (src, dst, start, end) -> scalarClampMin(src, minValue, dst, start, end)); }
    private static void parallelClampMax(double[] in, double maxValue, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, (src, dst, start, end) -> scalarClampMax(src, maxValue, dst, start, end)); }
    private static void parallelExp(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarExp); }
    private static void parallelFastExp(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarFastExp); }
    private static void parallelLog(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarLog); }
    private static void parallelTanh(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarTanh); }
    private static void parallelFastTanh(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarFastTanh); }
    private static void parallelSqrt(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarSqrt); }
    private static void parallelAbs(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarAbs); }
    private static void parallelSigmoid(double[] in, double[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF64::scalarSigmoid); }

    private static void parallelVectorInv(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorInvChunk, UnaryF64::scalarInv); }
    private static void parallelVectorRelu(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorReluChunk, UnaryF64::scalarRelu); }
    private static void parallelVectorClampMin(double[] in, double minValue, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, (src, dst, start, end, width) -> vectorClampMinChunk(src, minValue, dst, start, end, width), (src, dst, start, end) -> scalarClampMin(src, minValue, dst, start, end)); }
    private static void parallelVectorClampMax(double[] in, double maxValue, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, (src, dst, start, end, width) -> vectorClampMaxChunk(src, maxValue, dst, start, end, width), (src, dst, start, end) -> scalarClampMax(src, maxValue, dst, start, end)); }
    private static void parallelVectorExp(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorExpChunk, UnaryF64::scalarExp); }
    private static void parallelVectorLog(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorLogChunk, UnaryF64::scalarLog); }
    private static void parallelVectorTanh(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorTanhChunk, UnaryF64::scalarTanh); }
    private static void parallelVectorSqrt(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorSqrtChunk, UnaryF64::scalarSqrt); }
    private static void parallelVectorAbs(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorAbsChunk, UnaryF64::scalarAbs); }
    private static void parallelVectorSigmoid(double[] in, double[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF64::vectorSigmoidChunk, UnaryF64::scalarSigmoid); }

    private static void parallelScalar(double[] in, double[] out, ResolvedDispatchHints hints, ScalarOp op) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            op.apply(in, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double[] out, ResolvedDispatchHints hints, ChunkVectorOp vecOp, ScalarOp scalarOp) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
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

    private static void vectorClampMinChunk(double[] in, double minValue, double[] out, int start, int end, int width) {
        DoubleVector floor = DoubleVector.broadcast(SPECIES, minValue);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).max(floor).intoArray(out, i);
        }
    }

    private static void vectorClampMaxChunk(double[] in, double maxValue, double[] out, int start, int end, int width) {
        DoubleVector ceil = DoubleVector.broadcast(SPECIES, maxValue);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).min(ceil).intoArray(out, i);
        }
    }

    private static void vectorAbsChunk(double[] in, double[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            DoubleVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.ABS).intoArray(out, i);
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
