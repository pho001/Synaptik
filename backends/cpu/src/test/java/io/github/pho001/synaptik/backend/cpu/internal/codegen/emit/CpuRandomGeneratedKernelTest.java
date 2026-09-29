package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.classfile.*;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuRandomGeneratedKernelTest {
    private static final long ORACLE_KEY_BIAS = 0x9e3779b97f4a7c15L;
    private static final long ORACLE_M1 = 0xbf58476d1ce4e5b9L;
    private static final long ORACLE_M2 = 0x94d049bb133111ebL;

    @Test void everyCarrierPatternUsesDirectTypedAllocationFreeClassFileBodies() {
        assertDirectShape(CpuRandomLoweringTest.initialContext(1, 2),
                List.of(CarrierAccess.LONG_ARRAY), DataType.INT64, true);
        assertDirectShape(CpuRandomLoweringTest.initialContext(1, 2),
                List.of(CarrierAccess.MEMORY_SEGMENT), DataType.INT64, false);
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            CarrierAccess value = valueCarrier(type);
            for (int pattern = 0; pattern < 32; pattern++) {
                List<CarrierAccess> carriers = new ArrayList<>();
                for (int role = 0; role < 5; role++) carriers.add((pattern & 1 << role) == 0
                        ? switch (role) {
                            case 0, 2 -> value;
                            case 1, 4 -> CarrierAccess.LONG_ARRAY;
                            default -> CarrierAccess.BYTE_ARRAY;
                        } : CarrierAccess.MEMORY_SEGMENT);
                assertDirectShape(CpuRandomLoweringTest.dropoutContext(type, Shape.of(4), .25d),
                        carriers, type, pattern == 0);
            }
        }
    }

    @Test void exactIndependentCounterVectorsAndUniformHexValues() {
        assertVector(0, 0, 0, 0x48218226ff3cd4bfL, 0x1.2086089bfcf34p-2);
        assertVector(0, 0, 1, 0xea8568d2e45fd6cbL, 0x1.d50ad1a5c8bfap-1);
        assertVector(1, 0, 0, 0xdce423fc82c0d5b8L, 0x1.b9c847f90581ap-1);
        assertVector(-1L, -1L, 0, 0xe8ba9f99ca933538L, 0x1.d1753f3395266p-1);
        assertVector(0x1234, 7, 0, 0x3e4cf5a0c9489779L, 0x1.f267ad064a448p-3);
    }

    @Test void generatedCounterVectorsHitEveryExactThresholdBoundary() throws Throwable {
        long[][] vectors = {{0, 0}, {0, 1}, {1, 0}, {-1L, -1L}, {0x1234, 7}};
        double[] boundaries = {0x1.2086089bfcf34p-2, 0x1.d50ad1a5c8bfap-1,
                0x1.b9c847f90581ap-1, 0x1.d1753f3395266p-1, 0x1.f267ad064a448p-3};
        for (int vector = 0; vector < vectors.length; vector++) {
            for (double probability : new double[]{boundaries[vector],
                    Math.nextUp(boundaries[vector])}) {
                var generated = generated(CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64,
                        Shape.of(1), probability), carriers(DataType.FLOAT64));
                double[] output = {-1};
                byte[] mask = {-1};
                long[] next = {-1, -1};
                long[] state = vectors[vector].clone();
                generated.handle.invokeWithArguments(new double[]{2}, state, output, mask, next,
                        generated.geometry, 0L, 0L);
                generated.handle.invokeWithArguments(new double[]{2}, state, output, mask, next,
                        generated.geometry, 0L, 1L);
                boolean keep = Double.doubleToRawLongBits(probability)
                        == Double.doubleToRawLongBits(boundaries[vector]);
                assertAll(() -> assertEquals(keep ? (byte) 1 : (byte) 0, mask[0]),
                        () -> assertEquals(state[0], next[0]),
                        () -> assertEquals(state[1] + 1, next[1]));
            }
        }
    }

    @Test void generatedInitializerWritesEveryRawWordPair() throws Throwable {
        long[] output = new long[2];
        var invocation = generated(CpuRandomLoweringTest.initialContext(Long.MIN_VALUE, Long.MAX_VALUE),
                List.of(CarrierAccess.LONG_ARRAY));
        invocation.handle.invokeWithArguments(output, invocation.geometry, 0L, 0L);
        assertArrayEquals(new long[] {Long.MIN_VALUE, Long.MAX_VALUE}, output);
    }

    @Test void generatedInitializerHonorsGeneralOffsetAndStride() throws Throwable {
        var base = CpuRandomLoweringTest.initialContext(Long.MIN_VALUE, Long.MAX_VALUE);
        var values = new ArrayList<>(base.values());
        var memory = new ArrayList<>(base.memoryRequirements());
        Shape shape = Shape.of(2);
        var descriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.INT64, shape, Optional.of(LayoutDescriptor.of(
                        shape, new long[]{3}, 1, true)), false);
        values.set(0, new io.github.pho001.synaptik.model.graph.GraphValue(
                values.getFirst().id(), descriptor));
        memory.set(0, new io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement(
                values.getFirst().id(), descriptor, Optional.of(base.partition()), List.of(), true));
        var context = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, base.partition(), base.nodes(), values, memory, base.constants(), base.backendInputs());
        var invocation = generated(context, List.of(CarrierAccess.LONG_ARRAY));
        long[] output = new long[6];
        Arrays.fill(output, 7);
        invocation.handle.invokeWithArguments(output, invocation.geometry, 0L, 0L);
        assertArrayEquals(new long[]{7, Long.MIN_VALUE, 7, 7, Long.MAX_VALUE, 7}, output);
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
                generated.geometry, 2L, 4L);
        generated.handle.invokeWithArguments(input, state, output, mask, next,
                generated.geometry, 0L, 2L);
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

    @Test void boundedMixedDropoutPreservesPartialRangesStateAndTypedFallback() throws Throwable {
        int count = 1 << 20;
        var context = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32, Shape.of(count), .25,
                List.of(LayoutDescriptor.of(Shape.of(count), new long[]{2}, 3, true),
                        LayoutDescriptor.of(Shape.of(2), new long[]{1}, 0, true),
                        LayoutDescriptor.of(Shape.of(count), new long[]{2}, 4, true),
                        LayoutDescriptor.of(Shape.of(count), new long[]{2}, 4, true),
                        LayoutDescriptor.of(Shape.of(2), new long[]{1}, 0, true)));
        List<CarrierAccess> carriers = List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.LONG_ARRAY, CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.LONG_ARRAY);
        var invocation = generated(context, carriers);
        float[] input = new float[count * 2 + 3];
        float[] output = new float[count * 2 + 4];
        byte[] mask = new byte[count * 2 + 4];
        Arrays.fill(output, -19.0f);
        Arrays.fill(mask, (byte) -19);
        for (int logical = 7; logical < 29; logical++) input[3 + logical * 2] = logical * .125f;
        long[] state = {0x1234, 7};
        long[] next = {-1, -1};
        invocation.handle.invokeWithArguments(MemorySegment.ofArray(input), state, output,
                MemorySegment.ofArray(mask), next, invocation.geometry, 0L, 0L);
        for (long[] range : List.of(new long[]{7, 18}, new long[]{18, 29})) {
            invocation.handle.invokeWithArguments(MemorySegment.ofArray(input), state, output,
                    MemorySegment.ofArray(mask), next, invocation.geometry, range[0], range[1]);
        }
        for (int logical = 7; logical < 29; logical++) {
            boolean keep = oracleUniform(oracleWord(state[0], state[1], logical)) >= .25;
            assertEquals(keep ? (byte) 1 : (byte) 0, mask[4 + logical * 2]);
            assertEquals(Float.floatToRawIntBits(keep
                            ? (float) (((double) input[3 + logical * 2]) / .75d) : 0.0f),
                    Float.floatToRawIntBits(output[4 + logical * 2]));
        }
        assertAll(() -> assertArrayEquals(new long[]{state[0], state[1] + count}, next),
                () -> assertEquals(-19.0f, output[3]),
                () -> assertEquals((byte) -19, mask[3]));

        var route = new CpuPartitionPreparer().analyze(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, context.partition(), context.nodes(), context.values(), context.memoryRequirements(), context.constants(), new CpuPartitionAnalysisInputs(false, carriers))).plan().units().getFirst()
                .portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        var instructions = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow()
                .elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).toList();
        assertAll(
                () -> assertTrue(instructions.stream().map(Instruction::opcode)
                        .anyMatch(opcode -> opcode == Opcode.IINC)),
                () -> assertTrue(instructions.stream().map(Instruction::opcode)
                        .anyMatch(opcode -> opcode == Opcode.LREM)),
                () -> assertTrue(instructions.stream().noneMatch(instruction ->
                        instruction instanceof NewObjectInstruction
                                || instruction instanceof NewPrimitiveArrayInstruction
                                || instruction instanceof NewReferenceArrayInstruction
                                || instruction instanceof NewMultiArrayInstruction)));
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

    @Test void eachDropoutBoundaryRoleIndependentlyUsesItsSelectedCarrier() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32,
                    DataType.BFLOAT16)) {
                CarrierAccess value = valueCarrier(type);
                List<CarrierAccess> heap = List.of(value, CarrierAccess.LONG_ARRAY, value,
                        CarrierAccess.BYTE_ARRAY, CarrierAccess.LONG_ARRAY);
                for (int segmentRole = 0; segmentRole < 5; segmentRole++) {
                    List<CarrierAccess> selected = new ArrayList<>(heap);
                    selected.set(segmentRole, CarrierAccess.MEMORY_SEGMENT);
                    Object input = valueArray(type, 1);
                    Object state = new long[]{9, -1};
                    Object output = valueArray(type, 1);
                    Object mask = new byte[1];
                    Object next = new long[2];
                    Object[] arguments = {input, state, output, mask, next};
                    arguments[segmentRole] = switch (segmentRole) {
                        case 0, 2 -> arena.allocate(type.byteWidth(), type.byteWidth());
                        case 1, 4 -> arena.allocate(16, 8);
                        default -> arena.allocate(1, 1);
                    };
                    writeValue(arguments[0], type, 3.25);
                    writeLong(arguments[1], 0, 9);
                    writeLong(arguments[1], 1, -1);
                    var generated = generated(CpuRandomLoweringTest.dropoutContext(type,
                            Shape.of(), 0.0d), selected);
                    generated.handle.invokeWithArguments(arguments[0], arguments[1], arguments[2],
                            arguments[3], arguments[4], generated.geometry, 0L, 0L);
                    generated.handle.invokeWithArguments(arguments[0], arguments[1], arguments[2],
                            arguments[3], arguments[4], generated.geometry, 0L, 1L);
                    assertAll("type=" + type + " role=" + segmentRole,
                            () -> assertEquals(expectedRaw(type, 3.25),
                                    rawValue(arguments[2], type)),
                            () -> assertEquals((byte) 1, readByte(arguments[3], 0)),
                            () -> assertEquals(9, readLong(arguments[4], 0)),
                            () -> assertEquals(0, readLong(arguments[4], 1)));
                }
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

    @Test void bfloat16ZeroProbabilityPreservesSignedZeroCanonicalizesNaNAndReplaysOddTail()
            throws Throwable {
        var invocation = generated(CpuRandomLoweringTest.dropoutContext(DataType.BFLOAT16,
                Shape.of(7), -0.0d), carriers(DataType.BFLOAT16));
        short[] input = {(short) 0x8000, (short) 0x7fa1, (short) 0x3f80,
                (short) 0xbf80, (short) 0x7f80, (short) 0xff80, (short) 0x0001};
        long[] state = {0x1234, 7};
        short[] first = new short[7], replay = new short[7];
        byte[] firstMask = new byte[7], replayMask = new byte[7];
        long[] firstNext = new long[2], replayNext = new long[2];
        invocation.handle.invokeWithArguments(input, state, first, firstMask, firstNext,
                invocation.geometry, 0L, 0L);
        invocation.handle.invokeWithArguments(input, state, first, firstMask, firstNext,
                invocation.geometry, 0L, 3L);
        invocation.handle.invokeWithArguments(input, state, first, firstMask, firstNext,
                invocation.geometry, 3L, 7L);
        invocation.handle.invokeWithArguments(input, state, replay, replayMask, replayNext,
                invocation.geometry, 0L, 0L);
        invocation.handle.invokeWithArguments(input, state, replay, replayMask, replayNext,
                invocation.geometry, 0L, 3L);
        invocation.handle.invokeWithArguments(input, state, replay, replayMask, replayNext,
                invocation.geometry, 3L, 7L);
        short[] expected = input.clone();
        expected[1] = (short) 0x7fc0;
        assertAll(() -> assertArrayEquals(expected, first),
                () -> assertArrayEquals(first, replay),
                () -> assertArrayEquals(new byte[]{1, 1, 1, 1, 1, 1, 1}, firstMask),
                () -> assertArrayEquals(firstMask, replayMask),
                () -> assertArrayEquals(new long[]{state[0], state[1] + 7}, firstNext),
                () -> assertArrayEquals(firstNext, replayNext));
    }

    @Test void bfloat16GeneralBroadcastAndStridedWritesMatchLogicalDrawsAcrossRanges()
            throws Throwable {
        Shape shape = Shape.of(2, 2);
        Shape stateShape = Shape.of(2);
        var context = CpuRandomLoweringTest.dropoutContext(DataType.BFLOAT16, shape, .5d,
                List.of(LayoutDescriptor.of(shape, new long[]{0, 2}, 1, true),
                        LayoutDescriptor.of(stateShape, new long[]{2}, 1, true),
                        LayoutDescriptor.of(shape, new long[]{7, 2}, 1, true),
                        LayoutDescriptor.of(shape, new long[]{8, 3}, 2, true),
                        LayoutDescriptor.of(stateShape, new long[]{3}, 2, true)));
        var invocation = generated(context, carriers(DataType.BFLOAT16));
        short[] input = {(short) 0x55aa, (short) 0x3fc0, (short) 0x55aa, (short) 0xc010};
        long[] state = {-1, 0x1234, -1, 7};
        short[] first = new short[11], replay = new short[11];
        byte[] firstMask = new byte[14], replayMask = new byte[14];
        long[] firstNext = new long[6], replayNext = new long[6];
        Arrays.fill(first, (short) 0x55aa);
        Arrays.fill(replay, (short) 0x55aa);
        Arrays.fill(firstMask, (byte) -1);
        Arrays.fill(replayMask, (byte) -1);
        Arrays.fill(firstNext, -1);
        Arrays.fill(replayNext, -1);
        for (Object[] outputs : List.of(new Object[]{first, firstMask, firstNext},
                new Object[]{replay, replayMask, replayNext})) {
            invocation.handle.invokeWithArguments(input, state, outputs[0], outputs[1], outputs[2],
                    invocation.geometry, 0L, 0L);
            invocation.handle.invokeWithArguments(input, state, outputs[0], outputs[1], outputs[2],
                    invocation.geometry, 2L, 4L);
            invocation.handle.invokeWithArguments(input, state, outputs[0], outputs[1], outputs[2],
                    invocation.geometry, 0L, 2L);
        }
        int[] inputAddresses = {1, 3, 1, 3};
        int[] outputAddresses = {1, 3, 8, 10};
        int[] maskAddresses = {2, 5, 10, 13};
        for (int logical = 0; logical < 4; logical++) {
            boolean keep = oracleUniform(oracleWord(0x1234, 7, logical)) >= .5d;
            assertEquals(keep ? (byte) 1 : (byte) 0, firstMask[maskAddresses[logical]]);
            double value = CastValueConversions.convert(
                    ScalarValue.bfloat16Bits(input[inputAddresses[logical]]),
                    DataType.FLOAT64).float64Value();
            assertEquals(keep ? expectedBfloat(value / .5d) : (short) 0,
                    first[outputAddresses[logical]]);
        }
        assertAll(() -> assertArrayEquals(first, replay),
                () -> assertArrayEquals(firstMask, replayMask),
                () -> assertArrayEquals(firstNext, replayNext),
                () -> assertEquals(0x1234, firstNext[2]),
                () -> assertEquals(11, firstNext[5]),
                () -> assertEquals((short) 0x55aa, first[0]),
                () -> assertEquals((byte) -1, firstMask[0]));
    }

    @Test void preparedBfloat16DropoutBindsAndExecutesTheGeneratedArtifact() {
        var base = CpuRandomLoweringTest.dropoutContext(DataType.BFLOAT16, Shape.of(3), 0.0d);
        List<CarrierAccess> selected = carriers(DataType.BFLOAT16);
        var context = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE,
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, selected));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        short[] input = {(short) 0x8000, (short) 0x7fa1, (short) 0x3f80};
        long[] stateWords = {9, -1};
        short[] output = new short[3];
        byte[] mask = new byte[3];
        long[] next = new long[2];
        var state = new RunState(executable.memoryPlan(), List.of(
                borrowed(DataType.BFLOAT16, input),
                borrowed(DataType.INT64, stateWords),
                borrowed(DataType.BFLOAT16, output),
                borrowed(DataType.BOOL, mask),
                borrowed(DataType.INT64, next)).stream().map(buffer -> List.of(
                        new BufferRepresentationBinding(buffer, RunResourceOwnership.BORROWED)))
                        .toList(), List.of());
        try {
            assertDoesNotThrow(() -> executable.bind(state).execute());
            assertAll(() -> assertArrayEquals(new short[]{(short) 0x8000, (short) 0x7fc0,
                            (short) 0x3f80}, output),
                    () -> assertArrayEquals(new byte[]{1, 1, 1}, mask),
                    () -> assertArrayEquals(new long[]{9, 2}, next));
        } finally {
            state.close();
        }
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

    private static CpuBorrowedBuffer borrowed(DataType type, Object carrier) {
        MemorySegment segment = switch (carrier) {
            case short[] values -> MemorySegment.ofArray(values);
            case long[] values -> MemorySegment.ofArray(values);
            case byte[] values -> MemorySegment.ofArray(values);
            default -> throw new IllegalArgumentException("unsupported prepared dropout carrier");
        };
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,
                java.lang.reflect.Array.getLength(carrier), segment));
    }

    private static void writeValue(Object carrier, DataType type, double value) {
        if (type == DataType.FLOAT64) {
            writeDouble(carrier, 0, value);
        } else if (type == DataType.FLOAT32) {
            if (carrier instanceof float[] array) array[0] = (float) value;
            else ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT, 0, (float) value);
        } else {
            short bits = expectedBfloat(value);
            if (carrier instanceof short[] array) array[0] = bits;
            else ((MemorySegment) carrier).set(ValueLayout.JAVA_SHORT_UNALIGNED, 0, bits);
        }
    }

    private static long rawValue(Object carrier, DataType type) {
        if (type == DataType.FLOAT64) return Double.doubleToRawLongBits(readDouble(carrier, 0));
        if (type == DataType.FLOAT32) {
            float value = carrier instanceof float[] array ? array[0]
                    : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT, 0);
            return Integer.toUnsignedLong(Float.floatToRawIntBits(value));
        }
        short value = carrier instanceof short[] array ? array[0]
                : ((MemorySegment) carrier).get(ValueLayout.JAVA_SHORT_UNALIGNED, 0);
        return Short.toUnsignedLong(value);
    }

    private static long expectedRaw(DataType type, double value) {
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(value);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits((float) value));
            case BFLOAT16 -> Short.toUnsignedLong(expectedBfloat(value));
            default -> throw new IllegalArgumentException("unsupported dropout value type");
        };
    }

    private static short expectedBfloat(double value) {
        return CastValueConversions.convert(ScalarValue.float64(value), DataType.BFLOAT16)
                .bfloat16Bits();
    }

    private static Object valueArray(DataType type, int length) {
        return switch (type) {
            case FLOAT64 -> new double[length];
            case FLOAT32 -> new float[length];
            case BFLOAT16 -> new short[length];
            default -> throw new IllegalArgumentException("unsupported dropout value type");
        };
    }

    private static CarrierAccess valueCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            default -> throw new IllegalArgumentException("unsupported dropout value type");
        };
    }

    private static List<CarrierAccess> carriers(DataType type) {
        CarrierAccess value = valueCarrier(type);
        return List.of(value, CarrierAccess.LONG_ARRAY, value, CarrierAccess.BYTE_ARRAY,
                CarrierAccess.LONG_ARRAY);
    }

    private static void assertDirectShape(PrepareContext<CpuPartitionAnalysisInputs> base,
            List<CarrierAccess> carriers, DataType valueType, boolean dense) {
        var context = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        ClassModel model = ClassFile.of().parse(bytes);
        var method = model.methods().getFirst();
        var instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        var invokes = instructions.stream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        var references = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        Set<String> actualReferences = references.stream().map(reference ->
                reference.owner().asInternalName() + "#"
                        + reference.nameAndType().name().stringValue() + ":"
                        + reference.type().stringValue()).collect(
                                java.util.stream.Collectors.toSet());
        Set<String> expectedReferences = expectedMemberReferences(carriers, valueType);
        assertAll(
                () -> assertFalse(method.methodTypeSymbol().descriptorString()
                        .contains("Ljava/lang/Object;")),
                () -> assertEquals(dense
                                ? io.github.pho001.synaptik.backend.cpu.internal.cache
                                    .CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT
                                : io.github.pho001.synaptik.backend.cpu.internal.cache
                                    .CpuKernelSpecialization.LoopAddressing.GENERAL_LONG,
                        route.specialization().loopAddressing(route.kernelIr())),
                () -> assertTrue(instructions.stream().noneMatch(instruction ->
                        instruction instanceof TypeCheckInstruction
                                || instruction instanceof NewObjectInstruction
                                || instruction instanceof NewPrimitiveArrayInstruction
                                || instruction instanceof NewReferenceArrayInstruction
                                || instruction instanceof NewMultiArrayInstruction),
                        instructions.toString()),
                () -> assertTrue(references.stream().noneMatch(reference ->
                        reference.owner().asInternalName().equals(
                                CpuRandomEmitter.class.getName().replace('.', '/'))
                                || reference.type().stringValue().contains("Ljava/lang/Object;")),
                        references.toString()),
                () -> assertTrue(references.stream().allMatch(reference -> Set.of(
                                "java/lang/foreign/ValueLayout",
                                "java/lang/foreign/MemorySegment",
                                "java/lang/Float",
                                "java/lang/Double").contains(reference.owner().asInternalName())),
                        references.toString()),
                () -> assertEquals(expectedReferences, actualReferences),
                () -> assertEquals(dense && valueType != DataType.BFLOAT16,
                        references.isEmpty(), references.toString()),
                () -> assertTrue(invokes.stream().allMatch(call ->
                        call.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
                                && (call.name().stringValue().equals("get")
                                    || call.name().stringValue().equals("set"))
                        || valueType == DataType.BFLOAT16
                                && call.owner().asInternalName().equals("java/lang/Float")
                                && call.name().stringValue().equals("intBitsToFloat")
                        || valueType == DataType.BFLOAT16
                                && call.owner().asInternalName().equals("java/lang/Double")
                                && call.name().stringValue().equals("doubleToRawLongBits")),
                        invokes.toString()),
                () -> assertTrue(instructions.stream().map(Instruction::opcode)
                        .anyMatch(opcode -> opcode == Opcode.GOTO)));
    }

    private static Set<String> expectedMemberReferences(List<CarrierAccess> carriers,
            DataType valueType) {
        Set<String> expected = new HashSet<>();
        if (carriers.size() == 1) {
            if (carriers.getFirst() == CarrierAccess.MEMORY_SEGMENT) {
                addSegmentReferences(expected, DataType.INT64, false, true);
            }
            return expected;
        }
        if (valueType == DataType.BFLOAT16) {
            expected.add("java/lang/Float#intBitsToFloat:(I)F");
            expected.add("java/lang/Double#doubleToRawLongBits:(D)J");
        }
        if (carriers.get(0) == CarrierAccess.MEMORY_SEGMENT) {
            addSegmentReferences(expected, valueType, true, false);
        }
        if (carriers.get(1) == CarrierAccess.MEMORY_SEGMENT) {
            addSegmentReferences(expected, DataType.INT64, true, false);
        }
        if (carriers.get(2) == CarrierAccess.MEMORY_SEGMENT) {
            addSegmentReferences(expected, valueType, false, true);
        }
        if (carriers.get(3) == CarrierAccess.MEMORY_SEGMENT) {
            addSegmentReferences(expected, DataType.BOOL, false, true);
        }
        if (carriers.get(4) == CarrierAccess.MEMORY_SEGMENT) {
            addSegmentReferences(expected, DataType.INT64, false, true);
        }
        return expected;
    }

    private static void addSegmentReferences(Set<String> references, DataType type,
            boolean read, boolean write) {
        String layout = switch (type) {
            case FLOAT64 -> "DOUBLE";
            case FLOAT32 -> "FLOAT";
            case BFLOAT16 -> "SHORT";
            case INT64 -> "LONG";
            case BOOL -> "BYTE";
            default -> throw new IllegalArgumentException("unsupported random segment type");
        };
        String primitive = switch (type) {
            case FLOAT64 -> "D";
            case FLOAT32 -> "F";
            case BFLOAT16 -> "S";
            case INT64 -> "J";
            case BOOL -> "B";
            default -> throw new IllegalArgumentException("unsupported random segment type");
        };
        String layoutType = "Ljava/lang/foreign/ValueLayout$Of" + switch (type) {
            case FLOAT64 -> "Double";
            case FLOAT32 -> "Float";
            case BFLOAT16 -> "Short";
            case INT64 -> "Long";
            case BOOL -> "Byte";
            default -> throw new IllegalArgumentException("unsupported random segment type");
        } + ";";
        references.add("java/lang/foreign/ValueLayout#JAVA_" + layout
                + (type == DataType.BOOL ? "" : "_UNALIGNED") + ":" + layoutType);
        if (read) {
            references.add("java/lang/foreign/MemorySegment#get:(" + layoutType + "J)"
                    + primitive);
        }
        if (write) {
            references.add("java/lang/foreign/MemorySegment#set:(" + layoutType + "J"
                    + primitive + ")V");
        }
    }

    private static Generated generated(PrepareContext<CpuPartitionAnalysisInputs> base,
            List<CarrierAccess> carriers) {
        var context = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
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
