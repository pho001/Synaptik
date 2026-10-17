package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Finite generated-support and clean-Java semantic closure.
 *
 * <p>This test does not claim generated-versus-{@code javac} CFG or full structural equivalence.
 * The matrix structural test retains proportionate direct Class-File hygiene checks, so the
 * coverage inventory correctly keeps its structural rows partial.</p>
 */
class CpuScalarImmediateClampCleanJavaStructuralEquivalenceTest {
    @Test void everyExactFiniteFormHasIndependentTypedCounterpartAndTypedProvenance() {
        var forms = CpuScalarImmediateClampMatrixOracle.forms();
        assertEquals(263, forms.size());
        var compiled = CpuScalarImmediateClampCleanJavaOracle.compile(forms);
        var repeat = CpuScalarImmediateClampCleanJavaOracle.compile(forms);
        assertEquals(CpuScalarImmediateClampMatrixOracle.sha256(compiled.classBytes()),
                CpuScalarImmediateClampMatrixOracle.sha256(repeat.classBytes()));
        Map<List<?>, List<String>> generated = new LinkedHashMap<>(), clean = new LinkedHashMap<>();
        for (var form : forms) {
            var pair = CpuScalarImmediateClampCleanJavaOracle.pair(form, compiled);
            CpuScalarImmediateClampCleanJavaOracle.validatePairBinding(form, pair, repeat);
            var binding = pair.binding();
            // formId is deliberately absent: provenance itself must expose a collision.
            generated.computeIfAbsent(List.of(binding.topology(), binding.generatedClassSha256(),
                    binding.selectedEntryDescriptor(), binding.generatedMemberSchemaHash(),
                    binding.preparedIrStructuralKey(), binding.generatorSchema(), binding.classIdentitySchema(),
                    binding.selectedStrategy(), binding.materializationCandidateCount(),
                    binding.materializationSelectedCount(), binding.materializationPolicyIdentity()), ignored -> new ArrayList<>()).add(form.id());
            clean.computeIfAbsent(List.of(binding.cleanSelectedMethodName(), binding.cleanSelectedMethodDescriptor(),
                    binding.cleanMemberSchemaHash(), binding.cleanSourcePolicyHash()), ignored -> new ArrayList<>()).add(form.id());
        }
        assertEquals(263, generated.values().stream().mapToLong(List::size).sum());
        assertEquals(263, clean.values().stream().mapToLong(List::size).sum());
        assertTrue(generated.values().stream().allMatch(rows -> !rows.isEmpty()));
        assertTrue(clean.values().stream().allMatch(rows -> !rows.isEmpty()));
    }

    /** The generated-coverage checkpoint invokes this exact all-row semantic witness. */
    @Test void everyFormExecutesItsCompiledCleanCounterpartAgainstTheGeneratedOutput() throws Throwable {
        var forms = CpuScalarImmediateClampMatrixOracle.forms();
        Class<?> clean = define(CpuScalarImmediateClampCleanJavaOracle.compile(forms).classBytes());
        for (var form : forms) {
            long end = form.outputShape().knownElementCount().orElseThrow();
            for (var range : ranges(end)) executeRange(form, clean, range[0], range[1]);
        }
    }

    /**
     * Exercises the exact closed invocation allowlist for every generated form and independently
     * javac-compiled typed counterpart.  This is direct hygiene evidence only; it intentionally
     * does not compare their control-flow or claim full structural equivalence.
     */
    @Test void everyFormPassesClosedInvocationInspectionForBothSelectedArtifacts() {
        var forms = CpuScalarImmediateClampMatrixOracle.forms();
        var clean = CpuScalarImmediateClampCleanJavaOracle.compile(forms);
        for (var form : forms) {
            var generated = CpuScalarImmediateClampMatrixOracle.formArtifact(form).artifact();
            CpuScalarImmediateClampCleanJavaOracle.assertClosedInvocationAllowlist(form,
                    generated.bytes(), clean);
        }
    }

    private static void executeRange(CpuScalarImmediateClampMatrixOracle.Form form, Class<?> clean,
            long start, long end) throws Throwable {
        var fixture = form.fixture();
        Object expected = seeded(fixture.type()), actual = seeded(fixture.type());
        Object expectedSentinel = copy(expected);
        var context = CpuScalarImmediateClampMatrixOracle.contextFor(fixture, form.inputShape(),
                form.inputLayout(), form.outputShape(), form.outputLayout(), form.materializationPolicy());
        long[] geometry = CpuScalarImmediateClampMatrixOracle.geometryFor(context, start, end);
        clean.getMethod(CpuScalarImmediateClampCleanJavaOracle.method(form),
                carrierClass(fixture.type(), fixture.carriers().getFirst()),
                carrierClass(fixture.type(), fixture.carriers().getLast()), long[].class, long.class,
                long.class).invoke(null, carrier(input(fixture.type()), fixture.carriers().getFirst()),
                carrier(expected, fixture.carriers().getLast()), geometry, start, end);
        invokeGenerated(CpuScalarImmediateClampMatrixOracle.formArtifact(form).artifact().bytes(),
                carrier(input(fixture.type()), fixture.carriers().getFirst()),
                carrier(actual, fixture.carriers().getLast()), geometry, start, end);
        assertSameBits(form.id() + " [" + start + ',' + end + ')', expected, actual);
        assertOutsideRangeUnchanged(form.id(), expectedSentinel, actual, start, end);
    }

    private static List<long[]> ranges(long count) {
        var result = new ArrayList<long[]>(); result.add(new long[] {0L, count});
        if (count > 0L) result.add(new long[] {0L, 0L});
        if (count > 1L) result.add(new long[] {1L, count});
        if (count > 5L) result.add(new long[] {3L, count - 2L});
        return result;
    }

    @Test void everyCounterpartLoadsAndAcceptsItsExactTypedAbi() throws Exception {
        var forms = CpuScalarImmediateClampMatrixOracle.forms(); Class<?> clean = define(CpuScalarImmediateClampCleanJavaOracle.compile(forms).classBytes());
        for (var form : forms) { var f = form.fixture(); clean.getMethod(CpuScalarImmediateClampCleanJavaOracle.method(form),
                carrierClass(f.type(), f.carriers().getFirst()), carrierClass(f.type(), f.carriers().getLast()), long[].class, long.class, long.class)
                .invoke(null, carrier(seeded(f.type()), f.carriers().getFirst()), carrier(seeded(f.type()), f.carriers().getLast()), new long[32], 0L, 0L); }
    }

    private static Class<?> define(byte[] bytes) { return new ClassLoader(CpuScalarImmediateClampCleanJavaStructuralEquivalenceTest.class.getClassLoader()) {
        Class<?> bytes() { return defineClass("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.generated.ScalarImmediateClampCleanJava", bytes, 0, bytes.length); }
    }.bytes(); }
    private static void invokeGenerated(byte[] bytes, Object in, Object out, long[] geometry, long start, long end) throws Throwable {
        Class<?> type = new ClassLoader(CpuScalarImmediateClampCleanJavaStructuralEquivalenceTest.class.getClassLoader()) {
            Class<?> bytes() { return defineClass(null, bytes, 0, bytes.length); }
        }.bytes(); var methods = Arrays.stream(type.getDeclaredMethods()).filter(m -> java.lang.reflect.Modifier.isStatic(m.getModifiers())).toList();
        assertEquals(1, methods.size()); methods.getFirst().trySetAccessible();
        methods.getFirst().invoke(null, in, out, geometry, start, end);
    }
    private static Object carrier(Object value, CarrierAccess access) { if (access != CarrierAccess.MEMORY_SEGMENT) return value; return switch (value) {
        case short[] a -> MemorySegment.ofArray(a); case float[] a -> MemorySegment.ofArray(a); case double[] a -> MemorySegment.ofArray(a);
        case int[] a -> MemorySegment.ofArray(a); case long[] a -> MemorySegment.ofArray(a); default -> throw new AssertionError(value.getClass()); }; }
    private static Class<?> carrierClass(DataType type, CarrierAccess access) { if (access == CarrierAccess.MEMORY_SEGMENT) return MemorySegment.class; return switch(type) {
        case BFLOAT16 -> short[].class; case FLOAT32 -> float[].class; case FLOAT64 -> double[].class; case INT32 -> int[].class; case INT64 -> long[].class; default -> throw new AssertionError(type); }; }
    private static Object input(DataType type) { return switch(type) {
        case BFLOAT16 -> new short[]{0,Short.MIN_VALUE,(short)0x3fc0,(short)0xc020,(short)0x7fc1,(short)0x7f80,(short)0xff80,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
        case FLOAT32 -> new float[]{0f,-0f,1.5f,-2.5f,Float.NaN,Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY,Float.MIN_VALUE,-Float.MIN_VALUE,Float.MAX_VALUE,-Float.MAX_VALUE,3f,4f,5f,6f,7f,8f,9f,10f,11f,12f,13f,14f,15f,16f,17f,18f,19f,20f,21f,22f,23f};
        case FLOAT64 -> new double[]{0d,-0d,1.5d,-2.5d,Double.NaN,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,Double.MIN_VALUE,-Double.MIN_VALUE,Double.MAX_VALUE,-Double.MAX_VALUE,3d,4d,5d,6d,7d,8d,9d,10d,11d,12d,13d,14d,15d,16d,17d,18d,19d,20d,21d,22d,23d};
        case INT32 -> new int[]{Integer.MIN_VALUE,Integer.MAX_VALUE,Integer.MIN_VALUE+1,Integer.MAX_VALUE-1,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11};
        case INT64 -> new long[]{Long.MIN_VALUE,Long.MAX_VALUE,Long.MIN_VALUE+1,Long.MAX_VALUE-1,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11}; default -> throw new AssertionError(type); }; }
    private static Object seeded(DataType type) { return switch(type) {
        case BFLOAT16 -> { short[] a=new short[32]; Arrays.fill(a,(short)0x55aa); yield a; } case FLOAT32 -> { float[] a=new float[32]; Arrays.fill(a,Float.intBitsToFloat(0x7fc01234)); yield a; }
        case FLOAT64 -> { double[] a=new double[32]; Arrays.fill(a,Double.longBitsToDouble(0x7ff8000000001234L)); yield a; } case INT32 -> { int[] a=new int[32]; Arrays.fill(a,0x55aa55aa); yield a; }
        case INT64 -> { long[] a=new long[32]; Arrays.fill(a,0x55aa55aa55aa55aaL); yield a; } default -> throw new AssertionError(type); }; }
    private static void assertSameBits(String form, Object expected, Object actual) { switch(expected) {
        case short[] a -> assertArrayEquals(a,(short[])actual,form); case float[] a -> assertArrayEquals(a,(float[])actual,form); case double[] a -> assertArrayEquals(a,(double[])actual,form);
        case int[] a -> assertArrayEquals(a,(int[])actual,form); case long[] a -> assertArrayEquals(a,(long[])actual,form); default -> throw new AssertionError(expected.getClass()); } }
    private static Object copy(Object value) { return switch (value) {
        case short[] a -> a.clone(); case float[] a -> a.clone(); case double[] a -> a.clone();
        case int[] a -> a.clone(); case long[] a -> a.clone(); default -> throw new AssertionError(value.getClass()); }; }
    private static void assertOutsideRangeUnchanged(String form, Object sentinel, Object actual,
            long start, long end) {
        switch (sentinel) {
            case short[] expected -> { short[] observed = (short[]) actual; for (int i = 0; i < expected.length; i++) if (i < start || i >= end) assertEquals(expected[i], observed[i], form + " sentinel " + i); }
            case float[] expected -> { float[] observed = (float[]) actual; for (int i = 0; i < expected.length; i++) if (i < start || i >= end) assertEquals(Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(observed[i]), form + " sentinel " + i); }
            case double[] expected -> { double[] observed = (double[]) actual; for (int i = 0; i < expected.length; i++) if (i < start || i >= end) assertEquals(Double.doubleToRawLongBits(expected[i]), Double.doubleToRawLongBits(observed[i]), form + " sentinel " + i); }
            case int[] expected -> { int[] observed = (int[]) actual; for (int i = 0; i < expected.length; i++) if (i < start || i >= end) assertEquals(expected[i], observed[i], form + " sentinel " + i); }
            case long[] expected -> { long[] observed = (long[]) actual; for (int i = 0; i < expected.length; i++) if (i < start || i >= end) assertEquals(expected[i], observed[i], form + " sentinel " + i); }
            default -> throw new AssertionError(sentinel.getClass());
        }
    }
}
