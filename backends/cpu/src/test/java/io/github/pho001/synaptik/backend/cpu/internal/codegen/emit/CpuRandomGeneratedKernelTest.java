package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuRandomGeneratedKernelTest {
    private static final long ORACLE_KEY_BIAS = 0x9e3779b97f4a7c15L;
    private static final long ORACLE_M1 = 0xbf58476d1ce4e5b9L;
    private static final long ORACLE_M2 = 0x94d049bb133111ebL;

    @Test void computesKeyOffsetOnceBeforeTheGeneratedInvocationElementLoop() throws Exception {
        String source = Files.readString(Path.of("src/main/java/io/github/pho001/synaptik/backend/"
                + "cpu/internal/codegen/emit/CpuRandomEmitter.java"));
        int loop = source.indexOf("for (long logical = start; logical < end; logical++)");
        assertTrue(loop > 0);
        String setup = source.substring(source.lastIndexOf("long key =", loop), loop);
        String body = source.substring(loop, source.indexOf("\n        }", loop));
        assertAll(() -> assertTrue(setup.contains("long keyOffset = keyOffset(key);")),
                () -> assertTrue(body.contains(
                        "wordFromOffset(keyOffset, counter, logical)")),
                () -> assertFalse(body.contains("keyOffset(key)")),
                () -> assertFalse(body.contains("word(key, counter, logical)")));
    }

    @Test void exactIndependentCounterVectorsAndUniformHexValues() {
        assertVector(0, 0, 0, 0x48218226ff3cd4bfL, 0x1.2086089bfcf34p-2);
        assertVector(0, 0, 1, 0xea8568d2e45fd6cbL, 0x1.d50ad1a5c8bfap-1);
        assertVector(1, 0, 0, 0xdce423fc82c0d5b8L, 0x1.b9c847f90581ap-1);
        assertVector(-1L, -1L, 0, 0xe8ba9f99ca933538L, 0x1.d1753f3395266p-1);
        assertVector(0x1234, 7, 0, 0x3e4cf5a0c9489779L, 0x1.f267ad064a448p-3);
    }

    @Test void generatedInitializerWritesEveryRawWordPair() throws Throwable {
        long[] output = new long[2];
        var invocation = generated(CpuRandomLoweringTest.initialContext(Long.MIN_VALUE, Long.MAX_VALUE),
                List.of(CarrierAccess.LONG_ARRAY));
        invocation.handle.invokeWithArguments(output, invocation.geometry, 0L, 0L);
        assertArrayEquals(new long[] {Long.MIN_VALUE, Long.MAX_VALUE}, output);
    }

    @Test void generatedFloat64AndFloat32UseExactMaskScaleAndModuloState() throws Throwable {
        var f64 = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64,
                Shape.of(5), .5), carriers(DataType.FLOAT64));
        double[] input = {2, -0.0, Double.NaN, Double.POSITIVE_INFINITY, -4};
        long[] state = {0, -2}, next = new long[2]; double[] output = new double[5]; byte[] mask = new byte[5];
        f64.handle.invokeWithArguments(input, state, output, mask, next, f64.geometry, 0L, 0L);
        f64.handle.invokeWithArguments(input, state, output, mask, next, f64.geometry, 0L, 5L);
        byte[] expectedMask = new byte[5];
        double[] expected = new double[5];
        for (int i = 0; i < 5; i++) {
            boolean keep = oracleUniform(oracleWord(0, -2, i)) >= .5;
            expectedMask[i] = keep ? (byte) 1 : 0;
            expected[i] = keep ? input[i] / .5 : 0.0;
        }
        assertAll(() -> assertArrayEquals(expectedMask, mask),
                () -> assertArrayEquals(java.util.Arrays.stream(expected)
                                .mapToLong(Double::doubleToRawLongBits).toArray(),
                        java.util.Arrays.stream(output).mapToLong(Double::doubleToRawLongBits).toArray()),
                () -> assertArrayEquals(new long[] {0, 3}, next));

        var f32 = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32,
                Shape.of(3), .1), carriers(DataType.FLOAT32));
        float[] fi = {1.0000001f, -3.25f, 7}; float[] fo = new float[3]; byte[] fm = new byte[3];
        long[] fs = {0x1234, 7}, fn = new long[2];
        f32.handle.invokeWithArguments(fi, fs, fo, fm, fn, f32.geometry, 0L, 0L);
        f32.handle.invokeWithArguments(fi, fs, fo, fm, fn, f32.geometry, 0L, 3L);
        for (int i = 0; i < 3; i++) if (fm[i] == 1) assertEquals(
                Float.floatToRawIntBits((float) (((double) fi[i]) / (1.0d - .1d))),
                Float.floatToRawIntBits(fo[i]));
        assertArrayEquals(new long[] {0x1234, 10}, fn);
    }

    @Test void exactRepresentableThresholdKeepsEqualityAndDropsItsSuccessor() throws Throwable {
        double boundary = 0x1.2086089bfcf34p-2;
        assertEquals(Double.doubleToRawLongBits(boundary),
                Double.doubleToRawLongBits(oracleUniform(oracleWord(0, 0, 0))));
        for (double probability : new double[] {boundary, Math.nextUp(boundary)}) {
            var generated = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64,
                    Shape.of(1), probability), carriers(DataType.FLOAT64));
            double[] output = {-1}; byte[] mask = {-1}; long[] next = {-1, -1};
            long[] state = {0, 0}; double[] input = {2};
            generated.handle.invokeWithArguments(input, state, output, mask, next,
                    generated.geometry, 0L, 0L);
            generated.handle.invokeWithArguments(input, state, output, mask, next,
                    generated.geometry, 0L, 1L);
            boolean keep = Double.doubleToRawLongBits(probability)
                    == Double.doubleToRawLongBits(boundary);
            assertAll(() -> assertEquals(keep ? (byte) 1 : (byte) 0, mask[0]),
                    () -> assertEquals(Double.doubleToRawLongBits(
                                    keep ? input[0] / (1.0d - probability) : 0.0d),
                            Double.doubleToRawLongBits(output[0])),
                    () -> assertArrayEquals(new long[] {0, 1}, next));
        }
    }

    @Test void nonDenseGeometryUsesLogicalDrawsAndExactPhysicalAddresses() throws Throwable {
        Shape shape = Shape.of(2, 2); Shape stateShape = Shape.of(2);
        var context = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, shape, .5,
                List.of(LayoutDescriptor.of(shape, new long[] {0, 2}, 1, true),
                        LayoutDescriptor.of(stateShape, new long[] {2}, 1, true),
                        LayoutDescriptor.of(shape, new long[] {7, 2}, 1, true),
                        LayoutDescriptor.of(shape, new long[] {8, 3}, 2, true),
                        LayoutDescriptor.of(stateShape, new long[] {3}, 2, true)));
        var generated = generated(context, carriers(DataType.FLOAT64));
        double[] input = {-50, 2, -50, -6};
        long[] state = {-50, 0x1234, -50, 7};
        double[] output = new double[11]; Arrays.fill(output, -50);
        byte[] mask = new byte[14]; Arrays.fill(mask, (byte) -50);
        long[] next = new long[6]; Arrays.fill(next, -50);
        generated.handle.invokeWithArguments(input, state, output, mask, next,
                generated.geometry, 0L, 0L);
        generated.handle.invokeWithArguments(input, state, output, mask, next,
                generated.geometry, 0L, 4L);
        int[] inputAddresses = {1, 3, 1, 3};
        int[] outputAddresses = {1, 3, 8, 10};
        int[] maskAddresses = {2, 5, 10, 13};
        for (int logical = 0; logical < 4; logical++) {
            boolean keep = oracleUniform(oracleWord(0x1234, 7, logical)) >= .5;
            assertEquals(keep ? (byte) 1 : (byte) 0, mask[maskAddresses[logical]]);
            assertEquals(Double.doubleToRawLongBits(keep
                            ? input[inputAddresses[logical]] / .5 : 0.0d),
                    Double.doubleToRawLongBits(output[outputAddresses[logical]]));
        }
        assertAll(() -> assertEquals(-50, output[0]),
                () -> assertEquals((byte) -50, mask[0]),
                () -> assertEquals(0x1234, next[2]),
                () -> assertEquals(11, next[5]),
                () -> assertEquals(-50, next[0]));
    }

    @Test void scalarShapeWrapsCounterAcrossHeapNativeAndMixedCarriers() throws Throwable {
        var base = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(), 0.0d);
        try (Arena arena = Arena.ofConfined()) {
            List<CarrierScenario> scenarios = List.of(
                    new CarrierScenario("heap", carriers(DataType.FLOAT64),
                            new Object[] {new double[1], new long[2], new double[1],
                                    new byte[1], new long[2]}),
                    new CarrierScenario("native", Collections.nCopies(5,
                            CarrierAccess.MEMORY_SEGMENT), new Object[] {arena.allocate(8, 8),
                                    arena.allocate(16, 8), arena.allocate(8, 8),
                                    arena.allocate(1, 1), arena.allocate(16, 8)}),
                    new CarrierScenario("mixed", List.of(CarrierAccess.DOUBLE_ARRAY,
                            CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                            CarrierAccess.BYTE_ARRAY, CarrierAccess.LONG_ARRAY),
                            new Object[] {new double[1], arena.allocate(16, 8),
                                    arena.allocate(8, 8), new byte[1], new long[2]}));
            for (CarrierScenario scenario : scenarios) {
                writeDouble(scenario.arguments[0], 0, 3.25);
                writeLong(scenario.arguments[1], 0, 9);
                writeLong(scenario.arguments[1], 1, -1L);
                var generated = generated(base, scenario.carriers);
                generated.handle.invokeWithArguments(scenario.arguments[0], scenario.arguments[1],
                        scenario.arguments[2], scenario.arguments[3], scenario.arguments[4],
                        generated.geometry, 0L, 0L);
                generated.handle.invokeWithArguments(scenario.arguments[0], scenario.arguments[1],
                        scenario.arguments[2], scenario.arguments[3], scenario.arguments[4],
                        generated.geometry, 0L, 1L);
                assertAll(scenario.name,
                        () -> assertEquals(Double.doubleToRawLongBits(3.25),
                                Double.doubleToRawLongBits(readDouble(scenario.arguments[2], 0))),
                        () -> assertEquals((byte) 1, readByte(scenario.arguments[3], 0)),
                        () -> assertEquals(9, readLong(scenario.arguments[4], 0)),
                        () -> assertEquals(0, readLong(scenario.arguments[4], 1)));
            }
        }
    }

    @Test void zeroProbabilityConsumesDrawsAndEmptyDropoutStillWritesStateOnce() throws Throwable {
        var zero = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64,
                Shape.of(2), -0.0), carriers(DataType.FLOAT64));
        double[] input = {-0.0, Double.longBitsToDouble(0x7ff8000000000001L)}, output = new double[2];
        byte[] mask = new byte[2]; long[] state = {9, -1}, next = new long[2];
        zero.handle.invokeWithArguments(input, state, output, mask, next, zero.geometry, 0L, 0L);
        zero.handle.invokeWithArguments(input, state, output, mask, next, zero.geometry, 0L, 2L);
        assertAll(() -> assertArrayEquals(new byte[] {1, 1}, mask),
                () -> assertArrayEquals(new long[] {9, 1}, next),
                () -> assertEquals(Double.doubleToRawLongBits(input[0]), Double.doubleToRawLongBits(output[0])));

        var empty = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32,
                Shape.of(0), .75), carriers(DataType.FLOAT32));
        long[] emptyState = {4, 5}, emptyNext = new long[2];
        empty.handle.invokeWithArguments(new float[0], emptyState, new float[0], new byte[0],
                emptyNext, empty.geometry, 0L, 0L);
        assertArrayEquals(emptyState, emptyNext);
    }

    private static void assertVector(long key, long counter, long index, long word, double uniform) {
        long actual = oracleWord(key, counter, index);
        assertAll(() -> assertEquals(word, actual),
                () -> assertEquals(Double.doubleToRawLongBits(uniform),
                        Double.doubleToRawLongBits(oracleUniform(actual))));
    }

    private static long oracleWord(long key, long counter, long logical) {
        return oracleMix(counter + logical + oracleMix(key + ORACLE_KEY_BIAS));
    }

    private static long oracleMix(long value) {
        value = (value ^ (value >>> 30)) * ORACLE_M1;
        value = (value ^ (value >>> 27)) * ORACLE_M2;
        return value ^ (value >>> 31);
    }

    private static double oracleUniform(long word) { return (word >>> 11) * 0x1.0p-53; }

    private static void writeDouble(Object carrier, long index, double value) {
        if (carrier instanceof double[] array) array[Math.toIntExact(index)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE, index * 8, value);
    }

    private static double readDouble(Object carrier, long index) {
        return carrier instanceof double[] array ? array[Math.toIntExact(index)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE, index * 8);
    }

    private static void writeLong(Object carrier, long index, long value) {
        if (carrier instanceof long[] array) array[Math.toIntExact(index)] = value;
        else ((MemorySegment) carrier).set(ValueLayout.JAVA_LONG, index * 8, value);
    }

    private static long readLong(Object carrier, long index) {
        return carrier instanceof long[] array ? array[Math.toIntExact(index)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_LONG, index * 8);
    }

    private static byte readByte(Object carrier, long index) {
        return carrier instanceof byte[] array ? array[Math.toIntExact(index)]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_BYTE, index);
    }

    private static List<CarrierAccess> carriers(DataType type) {
        CarrierAccess value = type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.FLOAT_ARRAY;
        return List.of(value, CarrierAccess.LONG_ARRAY, value, CarrierAccess.BYTE_ARRAY,
                CarrierAccess.LONG_ARRAY);
    }

    private static Generated generated(PrepareContext<CpuPartitionAnalysisInputs> base,
            List<CarrierAccess> carriers) {
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        return new Generated(artifact.entryPoint(), plan.randomGeometry().orElseThrow()
                .pack(new long[carriers.size()]));
    }

    private record Generated(java.lang.invoke.MethodHandle handle, long[] geometry) { }
    private record CarrierScenario(String name, List<CarrierAccess> carriers, Object[] arguments) { }
}
