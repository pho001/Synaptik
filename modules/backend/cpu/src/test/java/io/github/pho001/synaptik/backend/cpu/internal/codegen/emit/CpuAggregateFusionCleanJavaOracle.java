package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Independently written clean Java for the specialized [64,256] rounded MUL/exact axis-1 SUM.
 * Arrays and a domain-sized native limb slice are prebound; no invocation or cell allocates.
 * Signed coefficient addition, flags, magnitude conversion and guard/sticky rounding match
 * the existing exact-state algorithm, rather than a sequential sum or a dot-product algorithm.
 * Helpers express the exact same limb loops; generated code embeds them directly.
 */
final class CpuAggregateFusionCleanJavaOracle {
    private CpuAggregateFusionCleanJavaOracle() { }

    /**
     * Explicitly launched ordinary-only diagnostic; never invoked by the test task.
     * Requires MAIN release after untimed/source review. One JVM covers both dtypes with six
     * warmup and nine alternating paired samples of 25 calls; a whole-invocation cap is 60s.
     * @param arguments must be exactly the explicit release token --ordinary-diagnostic
     * @throws Throwable if generation, output verification, an invocation or the cap fails
     */
    public static void main(String[] arguments) throws Throwable {
        if (arguments.length != 1 || !arguments[0].equals("--ordinary-diagnostic"))
            throw new IllegalArgumentException("explicit ordinary diagnostic release token required");
        long invocation = System.nanoTime();
        for (var type : java.util.List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT64)) {
            long setupStart = System.nanoTime();
            var carrier = type == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                    ? io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                    : io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            var plan = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(
                    CpuAggregateFusionGeneratedKernelTest.context(type,io.github.pho001.synaptik.model.shape.Shape.of(64,256),
                        new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(1,false),
                        io.github.pho001.synaptik.model.shape.Shape.of(64),true,0,false,java.util.Set.of(),java.util.Collections.nCopies(3,carrier))).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            var generated = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Object left, right, actual, direct;
            if (type == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32) {
                float[] a = new float[64*256], b = new float[a.length];
                for (int i = 0; i < a.length; i++) { a[i] = (i%17-8)*.25f; b[i] = (i%7-3)*.5f; }
                left = a; right = b; actual = new float[64]; direct = new float[64];
            } else {
                double[] a = new double[64*256], b = new double[a.length];
                for (int i = 0; i < a.length; i++) { a[i] = (i%17-8)*.25; b[i] = (i%7-3)*.5; }
                left = a; right = b; actual = new double[64]; direct = new double[64];
            }
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                long bytes = plan.aggregateGeometry().orElseThrow().workspaceBytes(1);
                var generatedState = arena.allocate(bytes,8); var directState = arena.allocate(bytes,8);
                var geometry = plan.aggregateGeometry().orElseThrow().packFusion(new long[3]);
                long setup = System.nanoTime()-setupStart;
                // One empty-entry observation separates setup and call/entry costs. It is not
                // subtracted from kernel samples: their nonstationary JIT costs are reported.
                long generatedEntry = time(generated,left,right,actual,generatedState,geometry,0);
                long directEntry = time(null,left,right,direct,directState,geometry,0);
                for (int warmup = 0; warmup < 6; warmup++) {
                    time(generated,left,right,actual,generatedState,geometry,64);
                    time(null,left,right,direct,directState,geometry,64); cap(invocation);
                }
                long[] g = new long[9], d = new long[9];
                for (int sample = 0; sample < 9; sample++) {
                    if ((sample&1) == 0) {
                        g[sample] = time(generated,left,right,actual,generatedState,geometry,64);
                        d[sample] = time(null,left,right,direct,directState,geometry,64);
                    } else {
                        d[sample] = time(null,left,right,direct,directState,geometry,64);
                        g[sample] = time(generated,left,right,actual,generatedState,geometry,64);
                    }
                    if (actual instanceof float[] a) {
                        float[] b = (float[])direct;
                        for (int i = 0; i < 64; i++) if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) throw new AssertionError("F32 output mismatch");
                    } else {
                        double[] a = (double[])actual, b = (double[])direct;
                        for (int i = 0; i < 64; i++) if (Double.doubleToRawLongBits(a[i]) != Double.doubleToRawLongBits(b[i])) throw new AssertionError("F64 output mismatch");
                    }
                    cap(invocation);
                }
                System.out.println(type+" setup_ns="+setup+" empty_entry_generated_25_ns="+generatedEntry
                        +" empty_entry_direct_25_ns="+directEntry+" workspace="+bytes);
                System.out.println("generated_25_ns="+java.util.Arrays.toString(g));
                System.out.println("direct_25_ns="+java.util.Arrays.toString(d));
            }
        }
        System.out.println("verified=true elapsed_ns="+(System.nanoTime()-invocation));
    }

    /** Enforces the one-invocation diagnostic budget without retrying.
     * @param start monotonic invocation start in nanoseconds
     * @throws AssertionError if the 60-second cap has elapsed */
    private static void cap(long start) {
        if (System.nanoTime()-start > 60_000_000_000L) throw new AssertionError("ordinary diagnostic 60s cap exceeded; no retry");
    }

    /** Measures one 25-call sample with type/path dispatch outside the call loop.
     * @param generated typed generated entry, or null for the direct Java oracle
     * @param left prebound typed left array
     * @param right prebound same-typed right array
     * @param output preallocated same-typed output array
     * @param state preallocated native exact state slice
     * @param geometry prebound unchanged exact header and physical bases
     * @param end output-cell limit, zero for entry-only observation or 64 for kernel samples
     * @return elapsed sample nanoseconds without setup, allocation or verification
     * @throws Throwable if the exact typed generated entry fails */
    private static long time(java.lang.invoke.MethodHandle generated,Object left,Object right,Object output,
            MemorySegment state,long[] geometry,int end) throws Throwable {
        long start = System.nanoTime();
        if (left instanceof float[] a) {
            float[] b = (float[])right, out = (float[])output;
            if (generated != null) for (int call = 0; call < 25; call++) generated.invokeExact(a,b,out,state,geometry,0L,(long)end);
            else for (int call = 0; call < 25; call++) f32(a,0,b,0,out,0,state,0,end);
        } else {
            double[] a = (double[])left, b = (double[])right, out = (double[])output;
            if (generated != null) for (int call = 0; call < 25; call++) generated.invokeExact(a,b,out,state,geometry,0L,(long)end);
            else for (int call = 0; call < 25; call++) f64(a,0,b,0,out,0,state,0,end);
        }
        return System.nanoTime()-start;
    }

    static void f32(float[] left, int leftBase, float[] right, int rightBase,
            float[] output, int outputBase, MemorySegment state, int start, int end) {
        for (int cell = start; cell < end; cell++) {
            reset(state,5);
            long flags = 0;
            int a = leftBase+cell*256, b = rightBase+cell*256;
            for (int term = 0; term < 256; term++) {
                float product = left[a+term]*right[b+term];
                flags = add(state,5,Float.floatToRawIntBits(product)&0xffffffffL,23,255,1L<<31,flags);
            }
            float sum = Float.intBitsToFloat((int)finish(state,5,23,127,-149,1L<<31,flags));
            output[outputBase+cell] = sum;
        }
    }

    static void f64(double[] left, int leftBase, double[] right, int rightBase,
            double[] output, int outputBase, MemorySegment state, int start, int end) {
        for (int cell = start; cell < end; cell++) {
            reset(state,33);
            long flags = 0;
            int a = leftBase+cell*256, b = rightBase+cell*256;
            for (int term = 0; term < 256; term++) {
                double product = left[a+term]*right[b+term];
                flags = add(state,33,Double.doubleToRawLongBits(product),52,2047,Long.MIN_VALUE,flags);
            }
            double sum = Double.longBitsToDouble(finish(state,33,52,1023,-1074,Long.MIN_VALUE,flags));
            output[outputBase+cell] = sum;
        }
    }

    private static void reset(MemorySegment state, int limbs) {
        state.set(ValueLayout.JAVA_LONG,0,0);
        for (int i = 0; i < limbs; i++) state.set(ValueLayout.JAVA_LONG,8L+8L*i,0);
    }

    private static long add(MemorySegment state, int limbs, long bits, int fractionBits,
            long maximumExponent, long signMask, long flags) {
        boolean negative = (bits&signMask) != 0;
        long fraction = bits&((1L<<fractionBits)-1);
        long exponent = (bits>>>fractionBits)&maximumExponent;
        if (exponent == maximumExponent) {
            return flags|(fraction != 0 ? 1 : negative ? 4 : 2);
        }
        if (exponent == 0 && fraction == 0) {
            return flags|(negative ? 16 : 8);
        }
        long significand = exponent == 0 ? fraction : fraction|(1L<<fractionBits);
        int shift = exponent == 0 ? 0 : (int)exponent-1;
        int word = shift>>>6, offset = shift&63;
        long carry = negative ? 1 : 0;
        for (int i = 0; i < limbs; i++) {
            long term = i == word ? significand<<offset
                    : offset != 0 && i == word+1 ? significand>>>(64-offset) : 0;
            if (negative) term = ~term;
            long withCarry = term+carry;
            carry = Long.compareUnsigned(withCarry,term) < 0 ? 1 : 0;
            long old = state.get(ValueLayout.JAVA_LONG,8L+8L*i);
            long sum = old+withCarry;
            if (Long.compareUnsigned(sum,old) < 0) carry |= 1;
            state.set(ValueLayout.JAVA_LONG,8L+8L*i,sum);
        }
        return flags|32;
    }

    private static long finish(MemorySegment state, int limbs, int fractionBits, int bias,
            int unitExponent, long signMask, long flags) {
        state.set(ValueLayout.JAVA_LONG,0,flags);
        long infinity = (2L*bias+1)<<fractionBits;
        if ((flags&1) != 0 || (flags&6) == 6) return infinity|(1L<<(fractionBits-1));
        if ((flags&6) != 0) return infinity|((flags&4) != 0 ? signMask : 0);
        boolean negative = state.get(ValueLayout.JAVA_LONG,8L*limbs) < 0;
        if (negative) {
            long carry = 1;
            for (int i = 0; i < limbs; i++) {
                long complemented = ~state.get(ValueLayout.JAVA_LONG,8L+8L*i);
                long magnitude = complemented+carry;
                if (Long.compareUnsigned(magnitude,complemented) >= 0) carry = 0;
                state.set(ValueLayout.JAVA_LONG,8L+8L*i,magnitude);
            }
        }
        int topWord = limbs-1;
        while (topWord > 0 && state.get(ValueLayout.JAVA_LONG,8L+8L*topWord) == 0) topWord--;
        long top = state.get(ValueLayout.JAVA_LONG,8L+8L*topWord);
        if (top == 0) return flags == 16 ? signMask : 0;
        int bitLength = 64*topWord+64-Long.numberOfLeadingZeros(top);
        int exponent = unitExponent+bitLength-1;
        int shift = exponent < 1-bias ? 0 : bitLength-fractionBits-1;
        int word = shift>>>6, offset = shift&63;
        long quotient = state.get(ValueLayout.JAVA_LONG,8L+8L*word)>>>offset;
        if (offset != 0 && word+1 < limbs)
            quotient |= state.get(ValueLayout.JAVA_LONG,16L+8L*word)<<(64-offset);
        if (shift != 0) {
            int guardWord = (shift-1)>>>6, guardOffset = (shift-1)&63;
            long guardLimb = state.get(ValueLayout.JAVA_LONG,8L+8L*guardWord);
            boolean guard = ((guardLimb>>>guardOffset)&1) != 0, sticky = false;
            for (int i = 0; i < guardWord; i++) if (state.get(ValueLayout.JAVA_LONG,8L+8L*i) != 0) { sticky = true; break; }
            sticky |= guardOffset != 0 && (guardLimb&((1L<<guardOffset)-1)) != 0;
            if (guard && (sticky || (quotient&1) != 0)) quotient++;
        }
        long sign = negative ? signMask : 0;
        if (exponent < 1-bias) return sign|quotient;
        if (quotient == 1L<<(fractionBits+1)) { quotient >>>= 1; exponent++; }
        if (exponent > bias) return sign|infinity;
        return sign|((long)(exponent+bias)<<fractionBits)|(quotient&((1L<<fractionBits)-1));
    }
}
