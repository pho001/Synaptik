package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/** Ordinary MATMUL TANH numerical semantics and bounded opt-in diagnostic. */
public final class CpuMatmulTanhEpilogueTest {
    private static final Path EVIDENCE = Path.of("/private/tmp/synaptik-cpu0065-evidence");
    private static volatile double sink;

    /**
     * Compares matched heap FLOAT32 vector kernels in one short JVM, not a performance gate.
     * Generation/binding is outside timing; both actions enter through a bound MethodHandle.
     * Samples include the whole vector kernel with ordered bias, TANH, tails and one checksum load.
     */
    @Test
    @EnabledIfEnvironmentVariable(named = "SYNAPTIK_CPU0065_DIAGNOSTIC", matches = "true")
    void shortGeneratedVersusEquivalentVectorJavaDiagnostic() throws Throwable {
        var report = new StringBuilder("java=" + System.getProperty("java.runtime.version")
                + " arch=" + System.getProperty("os.arch") + " species=" + bits(DataType.FLOAT32)
                + " carrier=heap-f32 bias=MATMUL_RIGHT\n");
        for (boolean tiled : List.of(false, true)) {
            int lanes = FloatVector.SPECIES_PREFERRED.length();
            int m = tiled ? 33 : 2, k = 63, n = tiled ? 4 * lanes + 1 : 32 * lanes + 1;
            float[] a = new float[m*k], b = new float[k*n], x = new float[n];
            float[] generatedOutput = new float[m*n], directOutput = new float[m*n];
            for (int i=0;i<a.length;i++) a[i]=(i%11-5)*.0625f;
            for (int i=0;i<b.length;i++) b[i]=(i%13-6)*.03125f;
            for (int i=0;i<x.length;i++) x[i]=(i%5-2)*.125f;
            var form = tiled ? Realization.TILED_N_VECTOR_2X2 : Realization.DIRECT_N_VECTOR;
            var spec = specialization(DataType.FLOAT32, form, false,
                    Epilogue.AddInputOrder.MATMUL_RIGHT, Epilogue.Terminal.TANH);
            var generator = new CpuClassFileKernelGenerator();
            var generated = generator.defineClassBytes(spec, spec.matmulIr().orElseThrow().encodedKernelIr(), generator.generateClassBytes(spec, spec.matmulIr().orElseThrow().encodedKernelIr()));
            long[] geometry = {0,0,0,1,m,k,n,k,1,n,1,n,1,0,0,1,(long)m*n,1};
            long work = tiled ? ((m+1L)/2)*((n+2L*lanes-1)/(2L*lanes)) : m;
            MethodHandle gh = java.lang.invoke.MethodHandles.insertArguments(generated.entryPoint(),
                    0,a,b,x,generatedOutput,geometry,0L,work);
            MethodHandle dh = java.lang.invoke.MethodHandles.lookup().findStatic(
                    CpuMatmulTanhEpilogueTest.class, tiled ? "directTiledTanh" : "directTanh",
                    java.lang.invoke.MethodType.methodType(void.class, float[].class,float[].class,
                            float[].class,float[].class,int.class,int.class,int.class));
            dh = java.lang.invoke.MethodHandles.insertArguments(dh,0,a,b,x,directOutput,m,k,n);
            gh.invokeExact(); dh.invokeExact(); assertArrayEquals(directOutput,generatedOutput);
            // Independent scalar check before timing; integer-exact inputs avoid cancellation ambiguity.
            for(int row=0;row<m;row++)for(int col=0;col<n;col++) {
                float sum=0;for(int p=0;p<k;p++)sum+=a[row*k+p]*b[p*n+col];
                float expected=(float)StrictMath.tanh(x[col]+sum);
                assertEquals(expected,generatedOutput[row*n+col],5*Math.ulp(expected));
            }
            for(int i=0;i<3_000;i++){gh.invokeExact();dh.invokeExact();}
            long[] gt=new long[5],dt=new long[5];int repetitions=500;
            for(int sample=0;sample<5;sample++) {
                if((sample&1)==0){gt[sample]=time(gh,generatedOutput,repetitions);dt[sample]=time(dh,directOutput,repetitions);}
                else {dt[sample]=time(dh,directOutput,repetitions);gt[sample]=time(gh,generatedOutput,repetitions);}
            }
            assertArrayEquals(directOutput,generatedOutput);
            long[] gs=gt.clone(),ds=dt.clone();java.util.Arrays.sort(gs);java.util.Arrays.sort(ds);
            report.append(form).append(" M/K/N=").append(m).append('/').append(k).append('/').append(n)
                    .append(" generated-ns=").append(java.util.Arrays.toString(gt))
                    .append(" direct-ns=").append(java.util.Arrays.toString(dt))
                    .append(" median-ratio=").append((double)gs[2]/ds[2]).append('\n');
        }
        report.append("semantic=PASS no-threshold; bounded warmed diagnostic, not a speed guarantee\n");
        Files.createDirectories(EVIDENCE);Files.writeString(EVIDENCE.resolve("diagnostic.txt"),report);
        System.out.print(report);
    }

    private static long time(MethodHandle handle,float[] output,int repetitions)throws Throwable {
        long start=System.nanoTime();
        for(int i=0;i<repetitions;i++){handle.invokeExact();sink=output[0];}
        return (System.nanoTime()-start)/repetitions;
    }

    /**
     * Direct-N oracle: full-K represented sum, bias-left ADD, TANH, sole store.
     * @param a dense borrowed M-by-K matrix
     * @param b dense borrowed K-by-N matrix
     * @param x contiguous borrowed N-element bias
     * @param out dense M-by-N destination overwritten by this invocation
     * @param m positive row count
     * @param k positive contraction count
     * @param n positive column count including the scalar tail
     */
    private static void directTanh(float[] a,float[] b,float[] x,float[] out,int m,int k,int n) {
        int lanes=FloatVector.SPECIES_PREFERRED.length();
        for(int row=0;row<m;row++) {
            int col=0;
            for(;col+lanes<=n;col+=lanes) {
                var sum=FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                for(int p=0;p<k;p++)sum=sum.add(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,
                        b,p*n+col).mul(a[row*k+p]));
                CpuVectorMath.tanh(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,col).add(sum))
                        .intoArray(out,row*n+col);
            }
            for(;col<n;col++) {
                float sum=0;for(int p=0;p<k;p++)sum+=a[row*k+p]*b[p*n+col];
                out[row*n+col]=(float)StrictMath.tanh(x[col]+sum);
            }
        }
    }

    /**
     * Two-row/two-species oracle shares right loads and retains guarded row/species tails.
     * @param a dense borrowed M-by-K matrix
     * @param b dense borrowed K-by-N matrix
     * @param x contiguous borrowed N-element bias
     * @param out dense M-by-N destination overwritten by this invocation
     * @param m positive row count, possibly odd
     * @param k positive contraction count
     * @param n positive column count including the scalar tail
     */
    private static void directTiledTanh(float[] a,float[] b,float[] x,float[] out,int m,int k,int n) {
        int lanes=FloatVector.SPECIES_PREFERRED.length();
        for(int row=0;row<m;row+=2)for(int col=0;col<n;col+=2*lanes) {
            if(col+lanes>n) {
                for(int r=row;r<Math.min(row+2,m);r++)for(int c=col;c<n;c++) {
                    float sum=0;for(int p=0;p<k;p++)sum+=a[r*k+p]*b[p*n+c];
                    out[r*n+c]=(float)StrictMath.tanh(x[c]+sum);
                }
                continue;
            }
            boolean row1=row+1<m, second=col+2*lanes<=n;
            var s00=FloatVector.zero(FloatVector.SPECIES_PREFERRED);var s01=s00;var s10=s00;var s11=s00;
            for(int p=0;p<k;p++) {
                float a0=a[row*k+p],a1=row1?a[(row+1)*k+p]:0;
                var b0=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,b,p*n+col);
                s00=s00.add(b0.mul(a0));if(row1)s10=s10.add(b0.mul(a1));
                if(second) {
                    var b1=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,b,p*n+col+lanes);
                    s01=s01.add(b1.mul(a0));if(row1)s11=s11.add(b1.mul(a1));
                }
            }
            var x0=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,col);
            CpuVectorMath.tanh(x0.add(s00)).intoArray(out,row*n+col);
            if(row1)CpuVectorMath.tanh(x0.add(s10)).intoArray(out,(row+1)*n+col);
            if(second) {
                var x1=FloatVector.fromArray(FloatVector.SPECIES_PREFERRED,x,col+lanes);
                CpuVectorMath.tanh(x1.add(s01)).intoArray(out,row*n+col+lanes);
                if(row1)CpuVectorMath.tanh(x1.add(s11)).intoArray(out,(row+1)*n+col+lanes);
            } else for(int r=row;r<Math.min(row+2,m);r++)for(int c=col+lanes;c<n;c++) {
                float sum=0;for(int p=0;p<k;p++)sum+=a[r*k+p]*b[p*n+c];
                out[r*n+c]=(float)StrictMath.tanh(x[c]+sum);
            }
        }
    }

    @Test void bothVectorFormsCoverFloatingTypesCarriersBiasOrdersAndDisjointRanges() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var form : List.of(Realization.DIRECT_N_VECTOR, Realization.TILED_N_VECTOR_2X2))
                for (boolean segment : List.of(false, true))
                    for (var order : Epilogue.AddInputOrder.values())
                        exercise(type, form, segment, order, false);
    }

    @Test void specialClassesAndZeroSignsAreCheckedSeparatelyFromFiniteTolerance() throws Throwable {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var form : List.of(Realization.DIRECT_N_VECTOR, Realization.TILED_N_VECTOR_2X2))
                for (boolean segment : List.of(false, true))
                    exercise(type, form, segment, Epilogue.AddInputOrder.NONE, true);
        // MATMUL starts at +0, so a -0 product sums to +0. TANH itself must retain -0.
        assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(CpuVectorMath.tanh(
                DoubleVector.broadcast(DoubleVector.SPECIES_PREFERRED, -0.0)).lane(0)));
        assertEquals(Integer.MIN_VALUE, Float.floatToRawIntBits(CpuVectorMath.tanh(
                FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -0.0f)).lane(0)));
    }

    @Test void cacheStreamsDistinguishNewFormsWithoutAdvancingTheExistingSchema() throws Throwable {
        var identities = new HashSet<String>();
        var compatibility = new HashSet<String>();
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var form : List.of(Realization.DIRECT_N_VECTOR, Realization.TILED_N_VECTOR_2X2))
                for (var order : Epilogue.AddInputOrder.values()) {
                    var spec = specialization(type, form, false, order, Epilogue.Terminal.TANH);
                    var none = specialization(type, form, false, order, Epilogue.Terminal.NONE);
                    assertTrue(identities.add(java.util.HexFormat.of().formatHex(spec.classIdentityBytes())));
                    assertTrue(compatibility.add(java.util.HexFormat.of().formatHex(spec.compatibilityBytes())));
                    assertFalse(java.util.Arrays.equals(spec.classIdentityBytes(), none.classIdentityBytes()));
                    assertFalse(java.util.Arrays.equals(spec.compatibilityBytes(), none.compatibilityBytes()));
                    assertEquals(54, spec.classIdentitySchema());
                    var generator = new CpuClassFileKernelGenerator();
                    var ir = spec.matmulIr().orElseThrow();
                    assertArrayEquals(generator.generateClassBytes(spec, ir.encodedKernelIr()),
                            generator.generateClassBytes(spec, ir.encodedKernelIr()));
                }
    }

    private static void exercise(DataType type, Realization form, boolean segment,
            Epilogue.AddInputOrder order, boolean special) throws Throwable {
        int lanes = bits(type) / type.bitWidth(), m = 3, k = special ? 1 : 7, n = 3 * lanes + 1;
        Object a = array(type, m * k + 2), b = array(type, k * n + 2);
        Object bias = array(type, n + 2), out = array(type, m * n + 2);
        for (int i = 0; i < m * k; i++) put(a, i + 1, special ? 1 : (i % 9 - 4) * .125);
        double[] classes = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                -0.0, +0.0, -1e-20, +1e-20};
        for (int i = 0; i < k * n; i++) put(b, i + 1, special ? classes[i % classes.length]
                : (i % 13 - 6) * .0625);
        for (int i = 0; i < n; i++) put(bias, i + 1, (i % 5 - 2) * .25);
        for (int i = 0; i < m * n + 2; i++) put(out, i, 42);
        var spec = specialization(type, form, segment, order, Epilogue.Terminal.TANH);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(spec, spec.matmulIr().orElseThrow().encodedKernelIr());
        var generated = generator.defineClassBytes(spec, spec.matmulIr().orElseThrow().encodedKernelIr(), bytes);
        long[] geometry = {1,1,1,1,m,k,n,k,1,n,1,n,1,0,1,1,(long)m*n,1};
        long work = form == Realization.DIRECT_N_VECTOR ? m
                : ((m + 1L) / 2) * ((n + 2L * lanes - 1) / (2L * lanes));
        try (var arena = Arena.ofConfined()) {
            Object ac = segment ? copy(arena, a) : a, bc = segment ? copy(arena, b) : b;
            Object xc = segment ? copy(arena, bias) : bias, oc = segment ? copy(arena, out) : out;
            var args = new ArrayList<Object>(List.of(ac, bc));
            if (order != Epilogue.AddInputOrder.NONE) args.add(xc);
            args.add(oc); args.add(geometry);
            MethodHandle bound = java.lang.invoke.MethodHandles.insertArguments(generated.entryPoint(),
                    0, args.toArray());
            long split = work / 2;
            bound.invokeExact(split, work);
            // The second range may not touch the first range's first cell.
            assertEquals(42, segment ? getSegment((MemorySegment) oc, type, 1) : get(out, 1));
            bound.invokeExact(0L, split);
            if (segment) MemorySegment.copy((MemorySegment) oc, 0, memory(out), 0, memory(out).byteSize());
        }
        assertEquals(42, get(out, 0)); assertEquals(42, get(out, m * n + 1));
        for (int row = 0; row < m; row++) for (int col = 0; col < n; col++) {
            double sum = 0;
            for (int inner = 0; inner < k; inner++) {
                double product = get(a, 1 + row * k + inner) * get(b, 1 + inner * n + col);
                if (k == 1) sum = type == DataType.FLOAT32 ? (float) product : product;
                else sum = type == DataType.FLOAT32 ? (float) ((float) sum + (float) product) : sum + product;
            }
            if (order != Epilogue.AddInputOrder.NONE) {
                double x = get(bias, 1 + col);
                sum = order == Epilogue.AddInputOrder.MATMUL_LEFT ? sum + x : x + sum;
                if (type == DataType.FLOAT32) sum = (float) sum;
            }
            double expected = type == DataType.FLOAT32 ? (float) StrictMath.tanh(sum) : StrictMath.tanh(sum);
            double actual = get(out, 1 + row * n + col);
            if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
            else if (expected == 0 || Math.abs(sum) == Double.POSITIVE_INFINITY)
                assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
            else {
                // No cancellation-sensitive relative bound: five binary32 / three binary64 ulps
                // against independently accumulated represented scalar arithmetic and StrictMath.
                double ulp = type == DataType.FLOAT32 ? Math.ulp((float) expected) : Math.ulp(expected);
                assertEquals(expected, actual, (type == DataType.FLOAT32 ? 5 : 3) * ulp,
                        type + " " + form + " " + segment + " " + order + " cell=" + (row*n+col));
            }
        }
    }


    private static CpuKernelSpecialization specialization(DataType type, Realization form,
            boolean segment, Epilogue.AddInputOrder order, Epilogue.Terminal terminal) {
        var read = plan(CpuAccessPlan.AccessKind.READ);
        var write = plan(CpuAccessPlan.AccessKind.WRITE);
        var bias = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.LAST_AXIS_BIAS,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var suffix = new Epilogue(order, terminal, null);
        var ir = new CpuMatmulIr(type, type, type, form, suffix, bits(type),
                CpuMatmulIr.NumericalForm.SEQUENTIAL, suffix.hasBias() ? List.of(read,read,bias)
                        : List.of(read,read), write);
        CarrierAccess carrier = segment ? CarrierAccess.MEMORY_SEGMENT : type == DataType.FLOAT32
                ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY;
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                ExecutionStrategy.VECTOR, Collections.nCopies(suffix.hasBias()?4:3,type),
                Collections.nCopies(suffix.hasBias()?4:3,carrier), bits(type), -1, List.of(), false,
                54, Optional.of(ir));
    }

    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
    }
    private static int bits(DataType type) {
        return type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : DoubleVector.SPECIES_PREFERRED.vectorBitSize();
    }
    private static Object array(DataType type, int size) {
        return type == DataType.FLOAT32 ? new float[size] : new double[size];
    }
    private static double get(Object array, int index) {
        return array instanceof float[] f ? f[index] : ((double[])array)[index];
    }
    private static void put(Object array, int index, double value) {
        if (array instanceof float[] f) f[index] = (float)value; else ((double[])array)[index] = value;
    }
    private static MemorySegment memory(Object array) {
        return array instanceof float[] f ? MemorySegment.ofArray(f) : MemorySegment.ofArray((double[])array);
    }
    private static MemorySegment copy(Arena arena, Object array) {
        var source = memory(array); var target = arena.allocate(source.byteSize(), 8);
        target.copyFrom(source); return target;
    }
    private static double getSegment(MemorySegment segment, DataType type, int index) {
        return type == DataType.FLOAT32 ? segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,index)
                : segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,index);
    }
}
