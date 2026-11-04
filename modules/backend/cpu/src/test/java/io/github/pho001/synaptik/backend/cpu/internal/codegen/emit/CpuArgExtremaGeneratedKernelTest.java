package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.reflect.AccessFlag;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CpuArgExtremaGeneratedKernelTest {
    @Test void everyFp8EncodingPairUsesExactNumericNanZeroAndTieIndicesAcrossCarriers()
            throws Throwable {
        int rows = 256 * 256;
        Shape inputShape = Shape.of(rows, 2), outputShape = Shape.of(rows);
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            byte[] input = new byte[5 * rows + 1];
            java.util.Arrays.fill(input, (byte) 0x42);
            for (int pair = 0; pair < rows; pair++) {
                input[1 + 5 * pair] = (byte) (pair >>> 8);
                input[3 + 5 * pair] = (byte) pair;
            }
            byte[] original = input.clone();
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.ARG_MIN,
                    AggregateReductionKind.ARG_MAX)) for (ArgExtremaTiePolicy tie : ArgExtremaTiePolicy.values()) {
                for (int pattern = 0; pattern < 4; pattern++) {
                    var selected = plan(kind,
                            descriptor(type, inputShape, new long[] {5, 2}, 1),
                            descriptor(DataType.INT64, outputShape, new long[] {2}, 1), 1, false, tie,
                            List.of((pattern & 1) == 0 ? CarrierAccess.BYTE_ARRAY : CarrierAccess.MEMORY_SEGMENT,
                                    (pattern & 2) == 0 ? CarrierAccess.LONG_ARRAY : CarrierAccess.MEMORY_SEGMENT));
                    var route = selected.units().getFirst().portablePlan();
                    var generator = new CpuClassFileKernelGenerator();
                    var handle = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                            generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
                    long[] output = new long[2 * rows + 3];
                    java.util.Arrays.fill(output, -9L);
                    Object source = (pattern & 1) == 0 ? input : MemorySegment.ofArray(input);
                    Object sink = (pattern & 2) == 0 ? output : MemorySegment.ofArray(output);
                    long[] geometry = selected.argExtremaGeometry().orElseThrow().pack(new long[2]);
                    handle.invokeWithArguments(source, sink, geometry, 0L, rows / 3L);
                    handle.invokeWithArguments(source, sink, geometry, rows / 3L, rows / 3L);
                    handle.invokeWithArguments(source, sink, geometry, rows / 3L, (long) rows);
                    for (int pair = 0; pair < rows; pair++) {
                        float left = fp8Value(type, (byte) (pair >>> 8));
                        float right = fp8Value(type, (byte) pair);
                        boolean last = tie == ArgExtremaTiePolicy.LAST_INDEX;
                        boolean rightWins = Float.isNaN(right) ? !Float.isNaN(left) || last
                                : !Float.isNaN(left) && (kind == AggregateReductionKind.ARG_MIN
                                    ? Float.compare(right, left) < 0 || last && Float.compare(right, left) == 0
                                    : Float.compare(right, left) > 0 || last && Float.compare(right, left) == 0);
                        assertEquals(rightWins ? 1L : 0L, output[1 + 2 * pair],
                                type + "/" + kind + "/" + tie + "/pattern=" + pattern + "/bits=" + pair);
                        assertEquals(-9L, output[2 * pair]);
                    }
                    assertEquals(-9L, output[2 * rows]);
                    assertEquals(-9L, output[2 * rows + 1]);
                    assertEquals(-9L, output[2 * rows + 2]);
                }
            }
            assertArrayEquals(original, input);
        }
    }

    private static float fp8Value(DataType type, byte bits) {
        return type == DataType.FLOAT8_E4M3
                ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.toFloat(bits)
                : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.toFloat(bits);
    }


    @Test void controlledArgExtremaUsesNonleadingOriginalRowsBeforeInputLoads() throws Throwable {
        var selected = plan(AggregateReductionKind.ARG_MAX,
                descriptor(DataType.FLOAT32, Shape.of(2, 3, 2), new long[]{6, 2, 1}, 0),
                descriptor(DataType.INT64, Shape.of(2, 3), new long[]{3, 1}, 0), 2,
                false, ArgExtremaTiePolicy.FIRST_INDEX,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY));
        var inputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 2);
        var outputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 1);
        var entry = CpuAffineCopyGeneratedKernelTest.controlledEntry(
                selected.units().getFirst().portablePlan(), List.of(inputRow, outputRow));
        long[] output = {-99, -99, -99, -99, -99, -99};
        entry.invokeWithArguments(new float[]{1, 4, Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                        9, 2}, output, selected.argExtremaGeometry().orElseThrow().pack(new long[2]),
                0L, 6L, CpuAffineCopyGeneratedKernelTest.control(new byte[]{1, 0, 0}));
        assertArrayEquals(new long[]{1, 0, 0, 0, 0, 0}, output);
    }
    @Test void executesExactLogicalIndicesForEveryTypeKindAndTiePolicy() throws Throwable {
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.ARG_MIN,
                AggregateReductionKind.ARG_MAX)) {
            assertArrayEquals(expected(kind, false, new double[] {1, Double.NaN, 4, Double.NaN}),
                    invoke(kind, DataType.FLOAT64, false, new double[] {1, Double.NaN, 4, Double.NaN}));
            assertArrayEquals(expected(kind, true, new float[] {-0.0f, 0.0f, -0.0f, 0.0f}),
                    invoke(kind, DataType.FLOAT32, true,
                            new float[] {-0.0f, 0.0f, -0.0f, 0.0f}));
            assertArrayEquals(expected(kind, true,
                            new short[] {(short) 0x3f80, (short) 0x7f81,
                                    (short) 0xffc1, (short) 0x4000}),
                    invoke(kind, DataType.BFLOAT16, true,
                            new short[] {(short) 0x3f80, (short) 0x7f81,
                                    (short) 0xffc1, (short) 0x4000}));
            assertArrayEquals(expected(kind, false,
                            new int[] {Integer.MIN_VALUE, 7, 7, Integer.MAX_VALUE}),
                    invoke(kind, DataType.INT32, false,
                            new int[] {Integer.MIN_VALUE, 7, 7, Integer.MAX_VALUE}));
            assertArrayEquals(expected(kind, true,
                            new long[] {Long.MIN_VALUE, 9, 9, Long.MAX_VALUE}),
                    invoke(kind, DataType.INT64, true,
                            new long[] {Long.MIN_VALUE, 9, 9, Long.MAX_VALUE}));
        }
    }

    @Test void generatedClassIsTypedFieldFreeAndOwnsDirectLoadCompareStoreLoop() {
        var plan = plan(AggregateReductionKind.ARG_MAX, DataType.FLOAT32, false);
        var route = plan.units().getFirst().portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        var instructions = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        assertAll(() -> assertEquals("([F[J[JJJ)V",
                        model.methods().getFirst().methodTypeSymbol().descriptorString()),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.PUBLIC)),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.STATIC)),
                () -> assertTrue(instructions.stream().anyMatch(i -> i.opcode() == Opcode.FALOAD)),
                () -> assertTrue(instructions.stream().anyMatch(i -> i.opcode() == Opcode.LASTORE)),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()));
    }

    @Test void executesGeneralZeroStrideInputAndStridedOutputOverPartialAndEmptyRanges()
            throws Throwable {
        Shape inputShape = Shape.of(2, 3, 2);
        Shape outputShape = Shape.of(2, 2);
        var inputDescriptor = descriptor(DataType.FLOAT32, inputShape,
                new long[] {9, 0, 2}, 1);
        var outputDescriptor = descriptor(DataType.INT64, outputShape,
                new long[] {3, 1}, 1);
        var plan = plan(AggregateReductionKind.ARG_MAX, inputDescriptor, outputDescriptor, 1,
                false, ArgExtremaTiePolicy.LAST_INDEX,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.LONG_ARRAY));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] storage = new float[13];
        storage[1] = -4; storage[3] = 7; storage[10] = 11; storage[12] = -9;
        long[] output = {-8, -8, -8, -8, -8, -8};
        long[] geometry = plan.argExtremaGeometry().orElseThrow().pack(new long[2]);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(storage), output,
                geometry, 1L, 3L);
        assertArrayEquals(new long[] {-8, -8, 2, -8, 2, -8}, output);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(storage), output,
                geometry, 3L, 3L);
        assertArrayEquals(new long[] {-8, -8, 2, -8, 2, -8}, output);
    }

    @Test void generatedBytesAndKeysAreDeterministicAndBehaviorFactsChangeBoth() {
        var first = plan(AggregateReductionKind.ARG_MIN, DataType.FLOAT64, false);
        var same = plan(AggregateReductionKind.ARG_MIN, DataType.FLOAT64, false);
        var different = plan(AggregateReductionKind.ARG_MAX, DataType.FLOAT64, true);
        var generator = new CpuClassFileKernelGenerator();
        var firstRoute = first.units().getFirst().portablePlan();
        var sameRoute = same.units().getFirst().portablePlan();
        var differentRoute = different.units().getFirst().portablePlan();
        byte[] firstBytes = generator.generateClassBytes(firstRoute.specialization(),
                firstRoute.kernelIr());
        assertAll(
                () -> assertArrayEquals(firstBytes, generator.generateClassBytes(
                        sameRoute.specialization(), sameRoute.kernelIr())),
                () -> assertEquals(firstRoute.specialization().structuralKey(),
                        sameRoute.specialization().structuralKey()),
                () -> assertNotEquals(firstRoute.specialization().structuralKey(),
                        differentRoute.specialization().structuralKey()),
                () -> assertFalse(java.util.Arrays.equals(firstBytes,
                        generator.generateClassBytes(differentRoute.specialization(),
                                differentRoute.kernelIr()))));
    }

    @Test void guardedStrideTwoAndArbitraryStrideFallbackReturnExactIndices() throws Throwable {
        assertArrayEquals(new long[] {1, 3}, invokeStridedInt32(2));
        assertArrayEquals(new long[] {1, 3}, invokeStridedInt32(3));
    }

    private static long[] invokeStridedInt32(long selectedStride) throws Throwable {
        Shape inputShape = Shape.of(2, 4);
        var inputDescriptor = descriptor(DataType.INT32, inputShape,
                new long[] {20, selectedStride}, 1);
        var outputDescriptor = descriptor(DataType.INT64, Shape.of(2), new long[] {1}, 0);
        var plan = plan(AggregateReductionKind.ARG_MAX, inputDescriptor, outputDescriptor, 1,
                false, ArgExtremaTiePolicy.FIRST_INDEX,
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.LONG_ARRAY));
        int[] input = new int[32];
        int[][] values = {{5, 9, 9, 1}, {-2, -5, -5, 7}};
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                input[(int) (1 + row * 20 + column * selectedStride)] = values[row][column];
            }
        }
        long[] output = {-1, -1};
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.argExtremaGeometry().orElseThrow().pack(new long[2]), 0L, 2L);
        return output;
    }

    private static long[] invoke(AggregateReductionKind kind, DataType type, boolean last,
            Object input) throws Throwable {
        var plan = plan(kind, type, last);
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long[] output = {-99};
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.argExtremaGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
        return output;
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan(
            AggregateReductionKind kind, DataType type, boolean last) {
        var base = CpuArgExtremaLoweringTest.context(kind, type, Shape.of(4), 0, false,
                last ? ArgExtremaTiePolicy.LAST_INDEX : ArgExtremaTiePolicy.FIRST_INDEX);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, List.of(carrier(type),
                CarrierAccess.LONG_ARRAY)));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan(
            AggregateReductionKind kind, TensorDescriptor input, TensorDescriptor output, int axis,
            boolean keep, ArgExtremaTiePolicy tie, List<CarrierAccess> carriers) {
        var base = CpuScatterLoweringTest.context(new Operation(kind,
                        new ArgExtremaAttrs(axis, keep, tie)), List.of(0), List.of(input), output);
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, long[] strides,
            long offset) {
        return new TensorDescriptor(type, shape,
                java.util.Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

    private static CarrierAccess carrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case FLOAT8_E4M3, FLOAT8_E5M2 -> CarrierAccess.BYTE_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> throw new AssertionError();
        };
    }

    private static long[] expected(AggregateReductionKind kind, boolean last, Object input) {
        int best = 0;
        for (int i = 1; i < java.lang.reflect.Array.getLength(input); i++) {
            boolean candidateNaN = nan(input, i);
            boolean bestNaN = nan(input, best);
            int comparison = compare(input, i, best);
            if ((candidateNaN && !bestNaN)
                    || (!candidateNaN && !bestNaN
                        && (kind == AggregateReductionKind.ARG_MIN
                                ? comparison < 0 : comparison > 0))
                    || comparison == 0 && last) best = i;
        }
        return new long[] {best};
    }

    private static int compare(Object input, int left, int right) {
        if (input instanceof double[] values) return preferred(values[left], values[right]);
        if (input instanceof float[] values) return preferred(values[left], values[right]);
        if (input instanceof short[] values) return preferred(
                Float.intBitsToFloat(Short.toUnsignedInt(values[left]) << 16),
                Float.intBitsToFloat(Short.toUnsignedInt(values[right]) << 16));
        if (input instanceof int[] values) return Integer.compare(values[left], values[right]);
        long[] values = (long[]) input;
        return Long.compare(values[left], values[right]);
    }

    private static boolean nan(Object input, int index) {
        if (input instanceof double[] values) return Double.isNaN(values[index]);
        if (input instanceof float[] values) return Float.isNaN(values[index]);
        if (input instanceof short[] values) return Float.isNaN(
                Float.intBitsToFloat(Short.toUnsignedInt(values[index]) << 16));
        return false;
    }

    private static int preferred(double left, double right) {
        if (Double.isNaN(left)) return Double.isNaN(right) ? 0 : 1;
        if (Double.isNaN(right)) return -1;
        return Double.compare(left, right);
    }

    private static int preferred(float left, float right) {
        if (Float.isNaN(left)) return Float.isNaN(right) ? 0 : 1;
        if (Float.isNaN(right)) return -1;
        return Float.compare(left, right);
    }
}
