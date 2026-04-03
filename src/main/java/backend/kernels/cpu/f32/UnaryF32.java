package backend.kernels.cpu.f32;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import utils.FastExp;

public final class UnaryF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private UnaryF32() {}

    public static void inv(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorInv(in, out);
            case PARALLEL -> parallelInv(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorInv(in, out, hints);
            case SCALAR -> scalarInv(in, out, 0, out.length);
        }
    }

    public static void relu(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorRelu(in, out);
            case PARALLEL -> parallelRelu(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorRelu(in, out, hints);
            case SCALAR -> scalarRelu(in, out, 0, out.length);
        }
    }

    public static void clampMin(float[] in, float minValue, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorClampMin(in, minValue, out);
            case PARALLEL -> parallelClampMin(in, minValue, out, hints);
            case PARALLEL_VECTOR -> parallelVectorClampMin(in, minValue, out, hints);
            case SCALAR -> scalarClampMin(in, minValue, out, 0, out.length);
        }
    }

    public static void clampMax(float[] in, float maxValue, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorClampMax(in, maxValue, out);
            case PARALLEL -> parallelClampMax(in, maxValue, out, hints);
            case PARALLEL_VECTOR -> parallelVectorClampMax(in, maxValue, out, hints);
            case SCALAR -> scalarClampMax(in, maxValue, out, 0, out.length);
        }
    }

    public static void exp(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorExp(in, out);
            case PARALLEL -> parallelExp(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorExp(in, out, hints);
            case SCALAR -> scalarExp(in, out, 0, out.length);
        }
    }

    public static void fastExp(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR, SCALAR -> scalarFastExp(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelFastExp(in, out, hints);
        }
    }

    public static void log(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorLog(in, out);
            case PARALLEL -> parallelLog(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorLog(in, out, hints);
            case SCALAR -> scalarLog(in, out, 0, out.length);
        }
    }

    public static void tanh(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorTanh(in, out);
            case PARALLEL -> parallelTanh(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorTanh(in, out, hints);
            case SCALAR -> scalarTanh(in, out, 0, out.length);
        }
    }

    public static void fastTanh(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR, SCALAR -> scalarFastTanh(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelFastTanh(in, out, hints);
        }
    }

    public static void sqrt(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorSqrt(in, out);
            case PARALLEL -> parallelSqrt(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorSqrt(in, out, hints);
            case SCALAR -> scalarSqrt(in, out, 0, out.length);
        }
    }

    public static void abs(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorAbs(in, out);
            case PARALLEL -> parallelAbs(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorAbs(in, out, hints);
            case SCALAR -> scalarAbs(in, out, 0, out.length);
        }
    }

    public static void sigmoid(float[] in, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case VECTOR -> vectorSigmoid(in, out);
            case PARALLEL -> parallelSigmoid(in, out, hints);
            case PARALLEL_VECTOR -> parallelVectorSigmoid(in, out, hints);
            case SCALAR -> scalarSigmoid(in, out, 0, out.length);
        }
    }

    private static void scalarInv(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0f / in[i]; }
    private static void scalarRelu(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.max(0.0f, in[i]); }
    private static void scalarClampMin(float[] in, float minValue, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.max(minValue, in[i]); }
    private static void scalarClampMax(float[] in, float maxValue, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.min(maxValue, in[i]); }
    private static void scalarExp(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = (float) Math.exp(in[i]); }
    private static void scalarFastExp(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = FastExp.fastExpF32(in[i]); }
    private static void scalarLog(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = (float) Math.log(in[i]); }
    private static void scalarTanh(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = (float) Math.tanh(in[i]); }
    private static void scalarFastTanh(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = FastExp.fastTanhF32(in[i]); }
    private static void scalarSqrt(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = (float) Math.sqrt(in[i]); }
    private static void scalarAbs(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = Math.abs(in[i]); }
    private static void scalarSigmoid(float[] in, float[] out, int start, int end) { for (int i = start; i < end; i++) out[i] = 1.0f / (1.0f + (float) Math.exp(-in[i])); }

    private static void vectorInv(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector ones = FloatVector.broadcast(SPECIES, 1.0f);
        for (; i < upper; i += SPECIES.length()) {
            ones.div(FloatVector.fromArray(SPECIES, in, i)).intoArray(out, i);
        }
        scalarInv(in, out, i, out.length);
    }

    private static void vectorRelu(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).max(zero).intoArray(out, i);
        }
        scalarRelu(in, out, i, out.length);
    }

    private static void vectorClampMin(float[] in, float minValue, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector floor = FloatVector.broadcast(SPECIES, minValue);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).max(floor).intoArray(out, i);
        }
        scalarClampMin(in, minValue, out, i, out.length);
    }

    private static void vectorClampMax(float[] in, float maxValue, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector ceil = FloatVector.broadcast(SPECIES, maxValue);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).min(ceil).intoArray(out, i);
        }
        scalarClampMax(in, maxValue, out, i, out.length);
    }

    private static void vectorExp(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.EXP).intoArray(out, i);
        }
        scalarExp(in, out, i, out.length);
    }

    private static void vectorLog(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.LOG).intoArray(out, i);
        }
        scalarLog(in, out, i, out.length);
    }

    private static void vectorTanh(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.TANH).intoArray(out, i);
        }
        scalarTanh(in, out, i, out.length);
    }

    private static void vectorSqrt(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.SQRT).intoArray(out, i);
        }
        scalarSqrt(in, out, i, out.length);
    }

    private static void vectorAbs(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.ABS).intoArray(out, i);
        }
        scalarAbs(in, out, i, out.length);
    }

    private static void vectorSigmoid(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
        FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).mul(half).lanewise(VectorOperators.TANH).add(one).mul(half).intoArray(out, i);
        }
        scalarSigmoid(in, out, i, out.length);
    }

    private static void parallelInv(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarInv); }
    private static void parallelRelu(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarRelu); }
    private static void parallelClampMin(float[] in, float minValue, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, (src, dst, start, end) -> scalarClampMin(src, minValue, dst, start, end)); }
    private static void parallelClampMax(float[] in, float maxValue, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, (src, dst, start, end) -> scalarClampMax(src, maxValue, dst, start, end)); }
    private static void parallelExp(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarExp); }
    private static void parallelFastExp(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarFastExp); }
    private static void parallelLog(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarLog); }
    private static void parallelTanh(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarTanh); }
    private static void parallelFastTanh(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarFastTanh); }
    private static void parallelSqrt(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarSqrt); }
    private static void parallelAbs(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarAbs); }
    private static void parallelSigmoid(float[] in, float[] out, ResolvedDispatchHints hints) { parallelScalar(in, out, hints, UnaryF32::scalarSigmoid); }

    private static void parallelVectorInv(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorInvChunk, UnaryF32::scalarInv); }
    private static void parallelVectorRelu(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorReluChunk, UnaryF32::scalarRelu); }
    private static void parallelVectorClampMin(float[] in, float minValue, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, (src, dst, start, end, width) -> vectorClampMinChunk(src, minValue, dst, start, end, width), (src, dst, start, end) -> scalarClampMin(src, minValue, dst, start, end)); }
    private static void parallelVectorClampMax(float[] in, float maxValue, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, (src, dst, start, end, width) -> vectorClampMaxChunk(src, maxValue, dst, start, end, width), (src, dst, start, end) -> scalarClampMax(src, maxValue, dst, start, end)); }
    private static void parallelVectorExp(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorExpChunk, UnaryF32::scalarExp); }
    private static void parallelVectorLog(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorLogChunk, UnaryF32::scalarLog); }
    private static void parallelVectorTanh(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorTanhChunk, UnaryF32::scalarTanh); }
    private static void parallelVectorSqrt(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorSqrtChunk, UnaryF32::scalarSqrt); }
    private static void parallelVectorAbs(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorAbsChunk, UnaryF32::scalarAbs); }
    private static void parallelVectorSigmoid(float[] in, float[] out, ResolvedDispatchHints hints) { parallelVector(in, out, hints, UnaryF32::vectorSigmoidChunk, UnaryF32::scalarSigmoid); }

    private static void parallelScalar(float[] in, float[] out, ResolvedDispatchHints hints, ScalarOp op) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            op.apply(in, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float[] out, ResolvedDispatchHints hints, ChunkVectorOp vecOp, ScalarOp scalarOp) {
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

    private static void vectorInvChunk(float[] in, float[] out, int start, int end, int width) {
        FloatVector ones = FloatVector.broadcast(SPECIES, 1.0f);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            ones.div(FloatVector.fromArray(SPECIES, in, i)).intoArray(out, i);
        }
    }

    private static void vectorReluChunk(float[] in, float[] out, int start, int end, int width) {
        FloatVector zero = FloatVector.zero(SPECIES);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).max(zero).intoArray(out, i);
        }
    }

    private static void vectorClampMinChunk(float[] in, float minValue, float[] out, int start, int end, int width) {
        FloatVector floor = FloatVector.broadcast(SPECIES, minValue);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).max(floor).intoArray(out, i);
        }
    }

    private static void vectorClampMaxChunk(float[] in, float maxValue, float[] out, int start, int end, int width) {
        FloatVector ceil = FloatVector.broadcast(SPECIES, maxValue);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).min(ceil).intoArray(out, i);
        }
    }

    private static void vectorAbsChunk(float[] in, float[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.ABS).intoArray(out, i);
        }
    }

    private static void vectorExpChunk(float[] in, float[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.EXP).intoArray(out, i);
        }
    }

    private static void vectorLogChunk(float[] in, float[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.LOG).intoArray(out, i);
        }
    }

    private static void vectorTanhChunk(float[] in, float[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.TANH).intoArray(out, i);
        }
    }

    private static void vectorSqrtChunk(float[] in, float[] out, int start, int end, int width) {
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).lanewise(VectorOperators.SQRT).intoArray(out, i);
        }
    }

    private static void vectorSigmoidChunk(float[] in, float[] out, int start, int end, int width) {
        FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
        FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
        for (int i = start, upper = end - ((end - start) % width); i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i).mul(half).lanewise(VectorOperators.TANH).add(one).mul(half).intoArray(out, i);
        }
    }

    @FunctionalInterface
    private interface ScalarOp { void apply(float[] in, float[] out, int start, int end); }

    @FunctionalInterface
    private interface ChunkVectorOp { void apply(float[] in, float[] out, int start, int end, int width); }
}
