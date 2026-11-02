package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Independent specialized clean Java algorithm oracle: typed arrays, direct two-pass slice
 * cursors and fixed-width signed exact limbs. No production arithmetic evaluator is used.
 * Helpers below express the limb loops embedded by generated code; neither element nor row
 * allocates. Flags and state are per invocation and reused only after a complete slice.
 * Inputs must satisfy the prepared nonoverlap and range proofs; these test-only routines do
 * not replace cold binding validation. The opt-in diagnostic uses synthetic primitive-pattern
 * values, not normalized saved probabilities or representative end-to-end training inputs.
 */
final class CpuSoftmaxBackwardCleanJavaOracle {
    /** Prevents instances of this test-only collection of specialized static algorithms. */
    private CpuSoftmaxBackwardCleanJavaOracle() { }

    /**
     * Explicitly released short diagnostic; ordinary tests never invoke this method.
     * One JVM covers only [32,128] heap softmax/log-softmax in F32/F64, with five warmups
     * and five alternating paired samples of five calls. Setup and empty entry are separate.
     * Each warmup is also a five-call group. The saved-value pattern is synthetic, including
     * negative softmax values; timings cannot establish representative training throughput,
     * universal speedup or steady-state parity. The caller must also enforce the wall cap if
     * an individual invocation does not return to the cooperative cap check.
     * @param args exactly --softmax-backward-diagnostic after independent review and MAIN release
     * @throws Throwable if generation, verification, invocation or the whole-invocation 60s cap fails
     */
    public static void main(String[] args) throws Throwable {
        if (args.length != 1 || !args[0].equals("--softmax-backward-diagnostic"))
            throw new IllegalArgumentException("explicit post-review diagnostic release token required");
        long start = System.nanoTime();
        System.out.println("JDK="+Runtime.version()+" JVM="+java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments());
        for (var type : java.util.List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT64)) for (boolean log : java.util.List.of(false,true)) {
            long begin = System.nanoTime();
            var carrier = type == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                    ? io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                    : io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            var plan = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(
                    io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxBackwardLoweringTest.context(type,
                        io.github.pho001.synaptik.model.shape.Shape.of(32,128),log,java.util.Collections.nCopies(3,carrier))).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            var entry = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Object y, g, actual, expected;
            if (type == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32) {
                float[] a = new float[4096], b = new float[4096];
                for (int i = 0; i < 4096; i++) { a[i] = (i%11-5)*.125f; b[i] = (i%7-3)*.5f; }
                y = a; g = b; actual = new float[4096]; expected = new float[4096];
            } else {
                double[] a = new double[4096], b = new double[4096];
                for (int i = 0; i < 4096; i++) { a[i] = (i%11-5)*.125; b[i] = (i%7-3)*.5; }
                y = a; g = b; actual = new double[4096]; expected = new double[4096];
            }
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                long bytes = plan.aggregateGeometry().orElseThrow().workspaceBytes(1);
                var generatedState = arena.allocate(bytes,8); var directState = arena.allocate(bytes,8);
                var geometry = plan.aggregateGeometry().orElseThrow().packFusion(new long[3]);
                long setup = System.nanoTime()-begin;
                long generatedEntry = sample(entry,log,y,g,actual,generatedState,geometry,0);
                long directEntry = sample(null,log,y,g,expected,directState,geometry,0);
                for (int warmup = 0; warmup < 5; warmup++) {
                    sample(entry,log,y,g,actual,generatedState,geometry,32);
                    sample(null,log,y,g,expected,directState,geometry,32); cap(start);
                }
                long[] generated = new long[5], direct = new long[5];
                for (int i = 0; i < 5; i++) {
                    if ((i&1) == 0) {
                        generated[i] = sample(entry,log,y,g,actual,generatedState,geometry,32);
                        direct[i] = sample(null,log,y,g,expected,directState,geometry,32);
                    } else {
                        direct[i] = sample(null,log,y,g,expected,directState,geometry,32);
                        generated[i] = sample(entry,log,y,g,actual,generatedState,geometry,32);
                    }
                    if (actual instanceof float[] a) {
                        float[] e = (float[])expected;
                        for (int j = 0; j < a.length; j++) if (Float.floatToRawIntBits(a[j]) != Float.floatToRawIntBits(e[j])) throw new AssertionError("F32 mismatch");
                    } else {
                        double[] a = (double[])actual, e = (double[])expected;
                        for (int j = 0; j < a.length; j++) if (Double.doubleToRawLongBits(a[j]) != Double.doubleToRawLongBits(e[j])) throw new AssertionError("F64 mismatch");
                    }
                    cap(start);
                }
                System.out.println(type+" log="+log+" shape=[32,128] setup_ns="+setup+" workspace_bytes="+bytes
                        +" empty_generated_5_ns="+generatedEntry+" empty_direct_5_ns="+directEntry);
                System.out.println("generated_5_ns="+java.util.Arrays.toString(generated));
                System.out.println("direct_5_ns="+java.util.Arrays.toString(direct));
            }
        }
        System.out.println("verified=true elapsed_ns="+(System.nanoTime()-start)+" diagnostic_only=true steady_state_parity_unproved=true");
    }

    /** Checks the single short invocation's cap, with no retry or additional campaign.
     * @param start monotonic nanosecond invocation origin
     * @throws AssertionError if the permitted 60 seconds have elapsed */
    private static void cap(long start) {
        if (System.nanoTime()-start > 60_000_000_000L) throw new AssertionError("60s cap exceeded; no retry");
    }

    /** Measures exactly five matched calls with type/kind/path selection outside call loops.
     * @param entry generated entry or null for direct Java
     * @param log selected fixed kind
     * @param y prebound saved-output array
     * @param g prebound cotangent array
     * @param out preallocated gradient
     * @param state exact run-private limb slice
     * @param geometry unchanged aggregate header with physical bases
     * @param end zero for entry-only observation, otherwise 32
     * @return elapsed sample nanoseconds, excluding setup and verification
     * @throws Throwable if a generated call fails */
    private static long sample(java.lang.invoke.MethodHandle entry,boolean log,Object y,Object g,Object out,
            MemorySegment state,long[] geometry,int end) throws Throwable {
        int limbs = (int)geometry[8]; long start = System.nanoTime();
        if (y instanceof float[] a) {
            float[] b = (float[])g, o = (float[])out;
            if (entry != null) for (int call = 0; call < 5; call++) entry.invokeExact(a,b,o,state,geometry,0L,(long)end);
            else if (log) for (int call = 0; call < 5; call++) f32Log(a,0,b,0,128,1,o,0,128,0,end,state,limbs);
            else for (int call = 0; call < 5; call++) f32Soft(a,0,b,0,128,1,o,0,128,0,end,state,limbs);
        } else {
            double[] a = (double[])y, b = (double[])g, o = (double[])out;
            if (entry != null) for (int call = 0; call < 5; call++) entry.invokeExact(a,b,o,state,geometry,0L,(long)end);
            else if (log) for (int call = 0; call < 5; call++) f64Log(a,0,b,0,128,1,o,0,128,0,end,state,limbs);
            else for (int call = 0; call < 5; call++) f64Soft(a,0,b,0,128,1,o,0,128,0,end,state,limbs);
        }
        return System.nanoTime()-start;
    }

    /**
     * Evaluates represented FLOAT32 products, exact SUM and typed SUB/MUL over complete rows.
     * Broadcast/shared-input selection precedes feature loops; each contributor remains present.
     * @param y nonnull stored saved values, read without mutation
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull cotangents, read without mutation; may be the exact same mapping as y
     * @param gb nonnegative cotangent origin in elements
     * @param gr cotangent row stride in elements, zero for scalar broadcast
     * @param gf cotangent feature stride, zero for broadcast or one for canonical reads
     * @param out nonnull writable gradient array, disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive nonnegative row bound
     * @param end exclusive row bound, not less than start and within all selected spans
     * @param state nonnull live writable exact-state segment, disjoint from arrays and reused per row
     * @param limbs positive prepared limb count; state holds flags plus this many 64-bit limbs
     */
    static void f32Soft(float[] y,int yb,float[] g,int gb,int gr,int gf,float[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (gf == 0) { f32Broadcast(false,y,yb,g,gb,gr,out,ob,width,start,end,state,limbs); return; }
        if (y == g && yb == gb && gr == width && gf == 1) { f32Shared(false,y,yb,out,ob,width,start,end,state,limbs); return; }
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0;
            int a = yb+row*width, b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf) {
                float product = g[b]*y[a+f];
                flags = accumulate(state,limbs,Float.floatToRawIntBits(product)&0xffffffffL,23,255,1L<<31,flags);
            }
            float sum = Float.intBitsToFloat((int)round(state,limbs,23,127,-149,1L<<31,flags));
            b = gb+row*gr; int o = ob+row*width;
            for (int f = 0; f < width; f++,b+=gf) { float delta = g[b]-sum; out[o+f] = y[a+f]*delta; }
        }
    }

    /**
     * Evaluates exact FLOAT32 cotangent SUM followed by typed EXP conversion, MUL and SUB.
     * The stored log-output is not reconstructed from probabilities or checked for finiteness.
     * @param y nonnull stored log-output values, read without mutation
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull cotangents, read without mutation; may be the exact same mapping as y
     * @param gb nonnegative cotangent origin in elements
     * @param gr cotangent row stride in elements, zero for scalar broadcast
     * @param gf cotangent feature stride, zero for broadcast or one for canonical reads
     * @param out nonnull writable gradient array, disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive nonnegative row bound
     * @param end exclusive row bound, not less than start and within all selected spans
     * @param state nonnull live writable exact-state segment, disjoint from arrays and reused per row
     * @param limbs positive prepared limb count; state holds flags plus this many 64-bit limbs
     */
    static void f32Log(float[] y,int yb,float[] g,int gb,int gr,int gf,float[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (gf == 0) { f32Broadcast(true,y,yb,g,gb,gr,out,ob,width,start,end,state,limbs); return; }
        if (y == g && yb == gb && gr == width && gf == 1) { f32Shared(true,y,yb,out,ob,width,start,end,state,limbs); return; }
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0;
            int b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf)
                flags = accumulate(state,limbs,Float.floatToRawIntBits(g[b])&0xffffffffL,23,255,1L<<31,flags);
            float sum = Float.intBitsToFloat((int)round(state,limbs,23,127,-149,1L<<31,flags));
            int a = yb+row*width, o = ob+row*width; b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf) {
                float exp = (float)StrictMath.exp(y[a+f]); float product = exp*sum; out[o+f] = g[b]-product;
            }
        }
    }

    /**
     * Evaluates represented FLOAT64 products, exact SUM and typed SUB/MUL without DAZ/FTZ.
     * Broadcast/shared-input selection precedes feature loops; each contributor remains present.
     * @param y nonnull stored saved values, read without mutation
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull cotangents, read without mutation; may be the exact same mapping as y
     * @param gb nonnegative cotangent origin in elements
     * @param gr cotangent row stride in elements, zero for scalar broadcast
     * @param gf cotangent feature stride, zero for broadcast or one for canonical reads
     * @param out nonnull writable gradient array, disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive nonnegative row bound
     * @param end exclusive row bound, not less than start and within all selected spans
     * @param state nonnull live writable exact-state segment, disjoint from arrays and reused per row
     * @param limbs positive prepared limb count; state holds flags plus this many 64-bit limbs
     */
    static void f64Soft(double[] y,int yb,double[] g,int gb,int gr,int gf,double[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (gf == 0) { f64Broadcast(false,y,yb,g,gb,gr,out,ob,width,start,end,state,limbs); return; }
        if (y == g && yb == gb && gr == width && gf == 1) { f64Shared(false,y,yb,out,ob,width,start,end,state,limbs); return; }
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0;
            int a = yb+row*width, b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf) {
                double product = g[b]*y[a+f];
                flags = accumulate(state,limbs,Double.doubleToRawLongBits(product),52,2047,Long.MIN_VALUE,flags);
            }
            double sum = Double.longBitsToDouble(round(state,limbs,52,1023,-1074,Long.MIN_VALUE,flags));
            b = gb+row*gr; int o = ob+row*width;
            for (int f = 0; f < width; f++,b+=gf) { double delta = g[b]-sum; out[o+f] = y[a+f]*delta; }
        }
    }

    /**
     * Evaluates exact FLOAT64 cotangent SUM, then EXP/MUL/SUB without DAZ/FTZ or narrowing.
     * The stored log-output is not reconstructed from probabilities or checked for finiteness.
     * @param y nonnull stored log-output values, read without mutation
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull cotangents, read without mutation; may be the exact same mapping as y
     * @param gb nonnegative cotangent origin in elements
     * @param gr cotangent row stride in elements, zero for scalar broadcast
     * @param gf cotangent feature stride, zero for broadcast or one for canonical reads
     * @param out nonnull writable gradient array, disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive nonnegative row bound
     * @param end exclusive row bound, not less than start and within all selected spans
     * @param state nonnull live writable exact-state segment, disjoint from arrays and reused per row
     * @param limbs positive prepared limb count; state holds flags plus this many 64-bit limbs
     */
    static void f64Log(double[] y,int yb,double[] g,int gb,int gr,int gf,double[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (gf == 0) { f64Broadcast(true,y,yb,g,gb,gr,out,ob,width,start,end,state,limbs); return; }
        if (y == g && yb == gb && gr == width && gf == 1) { f64Shared(true,y,yb,out,ob,width,start,end,state,limbs); return; }
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0;
            int b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf)
                flags = accumulate(state,limbs,Double.doubleToRawLongBits(g[b]),52,2047,Long.MIN_VALUE,flags);
            double sum = Double.longBitsToDouble(round(state,limbs,52,1023,-1074,Long.MIN_VALUE,flags));
            int a = yb+row*width, o = ob+row*width; b = gb+row*gr;
            for (int f = 0; f < width; f++,b+=gf) {
                double exp = StrictMath.exp(y[a+f]); double product = exp*sum; out[o+f] = g[b]-product;
            }
        }
    }

    /**
     * Uses one FLOAT32 cotangent row load for both passes, guarded against zero width.
     * @param log true for log-softmax, false for softmax; selected outside feature loops
     * @param y nonnull canonical saved-input array, read only
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull scalar/column cotangent array, read only
     * @param gb nonnegative cotangent origin in elements
     * @param gr zero or one cotangent row stride in elements
     * @param out nonnull gradient array disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative canonical slice width
     * @param start inclusive row bound
     * @param end exclusive row bound within all spans and not less than start
     * @param state live writable flags-plus-limbs segment, exclusively used by this invocation
     * @param limbs positive prepared exact-state limb count
     */
    private static void f32Broadcast(boolean log,float[] y,int yb,float[] g,int gb,int gr,float[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (width == 0) return;
        for (int row = start; row < end; row++) {
            float seed = g[gb+row*gr]; int a = yb+row*width, o = ob+row*width;
            clear(state,limbs); long flags = 0;
            if (log) for (int f = 0; f < width; f++) flags = accumulate(state,limbs,Float.floatToRawIntBits(seed)&0xffffffffL,23,255,1L<<31,flags);
            else for (int f = 0; f < width; f++) { float product = seed*y[a+f]; flags = accumulate(state,limbs,Float.floatToRawIntBits(product)&0xffffffffL,23,255,1L<<31,flags); }
            float sum = Float.intBitsToFloat((int)round(state,limbs,23,127,-149,1L<<31,flags));
            if (log) for (int f = 0; f < width; f++) { float exp = (float)StrictMath.exp(y[a+f]); float product = exp*sum; out[o+f] = seed-product; }
            else for (int f = 0; f < width; f++) { float delta = seed-sum; out[o+f] = y[a+f]*delta; }
        }
    }

    /**
     * Uses one FLOAT64 cotangent row load for both passes, guarded against zero width.
     * @param log true for log-softmax, false for softmax; selected outside feature loops
     * @param y nonnull canonical saved-input array, read only
     * @param yb nonnegative saved-input origin in elements
     * @param g nonnull scalar/column cotangent array, read only
     * @param gb nonnegative cotangent origin in elements
     * @param gr zero or one cotangent row stride in elements
     * @param out nonnull gradient array disjoint from reads
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative canonical slice width
     * @param start inclusive row bound
     * @param end exclusive row bound within all spans and not less than start
     * @param state live writable flags-plus-limbs segment, exclusively used by this invocation
     * @param limbs positive prepared exact-state limb count
     */
    private static void f64Broadcast(boolean log,double[] y,int yb,double[] g,int gb,int gr,double[] out,int ob,
            int width,int start,int end,MemorySegment state,int limbs) {
        if (width == 0) return;
        for (int row = start; row < end; row++) {
            double seed = g[gb+row*gr]; int a = yb+row*width, o = ob+row*width;
            clear(state,limbs); long flags = 0;
            if (log) for (int f = 0; f < width; f++) flags = accumulate(state,limbs,Double.doubleToRawLongBits(seed),52,2047,Long.MIN_VALUE,flags);
            else for (int f = 0; f < width; f++) { double product = seed*y[a+f]; flags = accumulate(state,limbs,Double.doubleToRawLongBits(product),52,2047,Long.MIN_VALUE,flags); }
            double sum = Double.longBitsToDouble(round(state,limbs,52,1023,-1074,Long.MIN_VALUE,flags));
            if (log) for (int f = 0; f < width; f++) { double exp = StrictMath.exp(y[a+f]); double product = exp*sum; out[o+f] = seed-product; }
            else for (int f = 0; f < width; f++) { double delta = seed-sum; out[o+f] = y[a+f]*delta; }
        }
    }

    /**
     * Shares each FLOAT32 array load between identical saved/cotangent roles in that pass.
     * @param log true for log-softmax, false for softmax; selected outside feature loops
     * @param y nonnull canonical array supplying both read roles
     * @param yb common nonnegative read origin in elements
     * @param out nonnull writable gradient array disjoint from y
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive row bound
     * @param end exclusive row bound within all spans and not less than start
     * @param state live writable flags-plus-limbs segment, exclusively used by this invocation
     * @param limbs positive prepared exact-state limb count
     */
    private static void f32Shared(boolean log,float[] y,int yb,float[] out,int ob,int width,int start,int end,MemorySegment state,int limbs) {
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0; int a = yb+row*width, o = ob+row*width;
            if (log) for (int f = 0; f < width; f++) { float v = y[a+f]; flags = accumulate(state,limbs,Float.floatToRawIntBits(v)&0xffffffffL,23,255,1L<<31,flags); }
            else for (int f = 0; f < width; f++) { float v = y[a+f]; float product = v*v; flags = accumulate(state,limbs,Float.floatToRawIntBits(product)&0xffffffffL,23,255,1L<<31,flags); }
            float sum = Float.intBitsToFloat((int)round(state,limbs,23,127,-149,1L<<31,flags));
            if (log) for (int f = 0; f < width; f++) { float v = y[a+f]; float exp = (float)StrictMath.exp(v); float product = exp*sum; out[o+f] = v-product; }
            else for (int f = 0; f < width; f++) { float v = y[a+f]; float delta = v-sum; out[o+f] = v*delta; }
        }
    }

    /**
     * Shares each FLOAT64 array load between identical saved/cotangent roles in that pass.
     * @param log true for log-softmax, false for softmax; selected outside feature loops
     * @param y nonnull canonical array supplying both read roles
     * @param yb common nonnegative read origin in elements
     * @param out nonnull writable gradient array disjoint from y
     * @param ob nonnegative gradient origin in elements
     * @param width nonnegative common slice width
     * @param start inclusive row bound
     * @param end exclusive row bound within all spans and not less than start
     * @param state live writable flags-plus-limbs segment, exclusively used by this invocation
     * @param limbs positive prepared exact-state limb count
     */
    private static void f64Shared(boolean log,double[] y,int yb,double[] out,int ob,int width,int start,int end,MemorySegment state,int limbs) {
        for (int row = start; row < end; row++) {
            clear(state,limbs); long flags = 0; int a = yb+row*width, o = ob+row*width;
            if (log) for (int f = 0; f < width; f++) { double v = y[a+f]; flags = accumulate(state,limbs,Double.doubleToRawLongBits(v),52,2047,Long.MIN_VALUE,flags); }
            else for (int f = 0; f < width; f++) { double v = y[a+f]; double product = v*v; flags = accumulate(state,limbs,Double.doubleToRawLongBits(product),52,2047,Long.MIN_VALUE,flags); }
            double sum = Double.longBitsToDouble(round(state,limbs,52,1023,-1074,Long.MIN_VALUE,flags));
            if (log) for (int f = 0; f < width; f++) { double v = y[a+f]; double exp = StrictMath.exp(v); double product = exp*sum; out[o+f] = v-product; }
            else for (int f = 0; f < width; f++) { double v = y[a+f]; double delta = v-sum; out[o+f] = v*delta; }
        }
    }

    /**
     * Resets one exact state before its next complete slice, without allocating.
     * @param state live writable segment with flags and at least limbs 64-bit state words
     * @param limbs positive prepared state width, excluding the flag word
     */
    private static void clear(MemorySegment state,int limbs) {
        for (int i = 0; i <= limbs; i++) state.set(ValueLayout.JAVA_LONG,8L*i,0L);
    }

    /**
     * Adds one represented contributor to signed exact limbs and classifies special values.
     * @param state live writable flags-plus-limbs segment mutated by finite nonzero contributors
     * @param limbs positive prepared state width, excluding flags
     * @param bits raw FLOAT32 bits zero-extended to long, or raw FLOAT64 bits
     * @param precision fraction width, 23 for FLOAT32 or 52 for FLOAT64
     * @param exponentMask all-one represented exponent field, 255 or 2047
     * @param sign represented sign mask, bit 31 or bit 63
     * @param flags prior accumulated class/zero flags for this slice
     * @return updated class/zero flags; a finite nonzero contributor enters the limbs once,
     *     while zeros and special classes update flags without an arithmetic replacement
     */
    private static long accumulate(MemorySegment state,int limbs,long bits,int precision,long exponentMask,long sign,long flags) {
        long exponent = (bits>>>precision)&exponentMask, fraction = bits&((1L<<precision)-1);
        boolean negative = (bits&sign) != 0;
        if (exponent == exponentMask) return flags|(fraction != 0 ? 1 : negative ? 4 : 2);
        if ((exponent|fraction) == 0) return flags|(negative ? 16 : 8);
        long significant = fraction|(exponent == 0 ? 0 : 1L<<precision);
        int shift = exponent == 0 ? 0 : (int)exponent-1;
        int word = shift/64, offset = shift%64;
        long carry = negative ? 1 : 0;
        for (int i = 0; i < limbs; i++) {
            long v = i == word ? significant<<offset : offset != 0 && i == word+1 ? significant>>>(64-offset) : 0;
            if (negative) v = ~v;
            long term = v+carry;
            carry = Long.compareUnsigned(term,v) < 0 ? 1 : 0;
            long previous = state.get(ValueLayout.JAVA_LONG,8L*(i+1)), next = previous+term;
            if (Long.compareUnsigned(next,previous) < 0) carry = 1;
            state.set(ValueLayout.JAVA_LONG,8L*(i+1),next);
        }
        return flags|32;
    }

    /**
     * Finishes one exact sum with special classification and nearest-even represented rounding.
     * The sign-to-magnitude conversion mutates state, which must be cleared before another row.
     * @param state live writable flags-plus-limbs segment containing the complete exact sum
     * @param limbs positive prepared state width, excluding flags
     * @param precision fraction width, 23 for FLOAT32 or 52 for FLOAT64
     * @param bias exponent bias, 127 or 1023
     * @param unit exponent of the least positive subnormal, -149 or -1074
     * @param signMask represented sign bit, bit 31 or bit 63
     * @param flags complete class/zero flags for this slice, written to the state flag word
     * @return represented raw result bits, zero-extended for FLOAT32; NaN payload is unspecified
     */
    private static long round(MemorySegment state,int limbs,int precision,int bias,int unit,long signMask,long flags) {
        state.set(ValueLayout.JAVA_LONG,0,flags);
        long infinity = (2L*bias+1)<<precision;
        if ((flags&1) != 0 || (flags&6) == 6) return infinity|(1L<<(precision-1));
        if ((flags&6) != 0) return infinity|((flags&4) != 0 ? signMask : 0);
        boolean negative = state.get(ValueLayout.JAVA_LONG,8L*limbs) < 0;
        if (negative) {
            long carry = 1;
            for (int i = 1; i <= limbs; i++) {
                long complement = ~state.get(ValueLayout.JAVA_LONG,8L*i), next = complement+carry;
                carry = Long.compareUnsigned(next,complement) < 0 ? 1 : 0;
                state.set(ValueLayout.JAVA_LONG,8L*i,next);
            }
        }
        int top = limbs;
        while (top > 1 && state.get(ValueLayout.JAVA_LONG,8L*top) == 0) top--;
        long high = state.get(ValueLayout.JAVA_LONG,8L*top);
        if (high == 0) return flags == 16 ? signMask : 0;
        int length = (top-1)*64+64-Long.numberOfLeadingZeros(high), exponent = unit+length-1;
        int shift = exponent < 1-bias ? 0 : length-precision-1;
        int word = shift/64, offset = shift%64;
        long significant = state.get(ValueLayout.JAVA_LONG,8L*(word+1))>>>offset;
        if (offset != 0 && word+1 < limbs) significant |= state.get(ValueLayout.JAVA_LONG,8L*(word+2))<<(64-offset);
        if (shift > 0) {
            int guardWord = (shift-1)/64, guardBit = (shift-1)%64;
            long guardLimb = state.get(ValueLayout.JAVA_LONG,8L*(guardWord+1));
            boolean guard = ((guardLimb>>>guardBit)&1) != 0;
            boolean sticky = guardBit != 0 && (guardLimb&((1L<<guardBit)-1)) != 0;
            for (int i = 0; !sticky && i < guardWord; i++) sticky = state.get(ValueLayout.JAVA_LONG,8L*(i+1)) != 0;
            if (guard && (sticky || (significant&1) != 0)) significant++;
        }
        long sign = negative ? signMask : 0;
        if (exponent < 1-bias) return sign|significant;
        if (significant == 1L<<(precision+1)) { significant >>>= 1; exponent++; }
        if (exponent > bias) return sign|infinity;
        return sign|((long)(exponent+bias)<<precision)|(significant&((1L<<precision)-1));
    }
}
