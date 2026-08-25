package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.AccessFlag;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CpuMaskedReductionGeneratedKernelTest {
    @Test void generalAndZeroStrideLayoutsPreserveCanariesAndBroadcastMapping() throws Throwable {
        Shape dataShape = Shape.of(2, 3), maskShape = Shape.of(1, 3), outputShape = Shape.of(2);
        var base = CpuMaskedReductionLoweringTest.context(AggregateReductionKind.SUM,
                DataType.FLOAT64, dataShape, maskShape, 1,
                LayoutDescriptor.of(dataShape, new long[] {4, 1}, 1, true),
                LayoutDescriptor.of(maskShape, new long[] {0, 2}, 1, true),
                LayoutDescriptor.of(outputShape, new long[] {2}, 1, true));
        var plan = plan(base, List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.BYTE_ARRAY,
                CarrierAccess.DOUBLE_ARRAY));
        double[] data = {-90, 1, 2, 3, -90, 5, 6, 7, -90};
        byte[] mask = {-9, 1, -9, 0, -9, 1, -9};
        double[] output = {-77, -77, -77, -77, -77};
        invoke(plan, data, mask, output);
        assertArrayEquals(new double[] {-77, 4, -77, 12, -77}, output);
        assertArrayEquals(new byte[] {-9, 1, -9, 0, -9, 1, -9}, mask);
    }

    @Test void everyTypedHeapSegmentPatternProducesTheSameRepresentedResult() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (var kind : List.of(AggregateReductionKind.SUM, AggregateReductionKind.MEAN)) {
                for (int pattern = 0; pattern < 8; pattern++) {
                    Object dataArray = type == DataType.FLOAT64 ? new double[] {1, 2}
                            : type == DataType.FLOAT32 ? new float[] {1, 2}
                            : new short[] {(short) 0x3f80, (short) 0x4000};
                    byte[] maskArray = {1, 1}; Object outputArray = array(type, 1);
                    Object data = (pattern & 1) == 0 ? dataArray : segment(dataArray);
                    Object mask = (pattern & 2) == 0 ? maskArray : MemorySegment.ofArray(maskArray);
                    Object output = (pattern & 4) == 0 ? outputArray : segment(outputArray);
                    CarrierAccess numeric = (pattern & 1) == 0 ? arrayCarrier(type)
                            : CarrierAccess.MEMORY_SEGMENT;
                    CarrierAccess maskCarrier = (pattern & 2) == 0 ? CarrierAccess.BYTE_ARRAY
                            : CarrierAccess.MEMORY_SEGMENT;
                    CarrierAccess outputCarrier = (pattern & 4) == 0 ? arrayCarrier(type)
                            : CarrierAccess.MEMORY_SEGMENT;
                    invoke(plan(kind, type, Shape.of(1, 2), Shape.of(2),
                            List.of(numeric, maskCarrier, outputCarrier)), data, mask, output);
                    long expected = type == DataType.FLOAT64
                            ? Double.doubleToRawLongBits(kind == AggregateReductionKind.SUM
                                    ? 3.0 : 1.5)
                            : type == DataType.FLOAT32
                            ? Integer.toUnsignedLong(Float.floatToRawIntBits(
                                    kind == AggregateReductionKind.SUM ? 3.0f : 1.5f))
                            : kind == AggregateReductionKind.SUM ? 0x4040L : 0x3fc0L;
                    assertEquals(expected, bits(type, outputArray), type + "/" + kind
                            + "/pattern=" + pattern);
                }
            }
        }
    }

    @Test void generatedTypedBodySkipsFalseSpecialValuesAndUsesSelectedMeanCount()
            throws Throwable {
        var plan = plan(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(2, 4),
                Shape.of(2, 4), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.BYTE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        double[] data = {1, Double.NaN, 3, Double.POSITIVE_INFINITY,
                -0.0, -0.0, 9, Double.NEGATIVE_INFINITY};
        byte[] mask = {1, 0, 1, 0, 1, 1, 0, 0};
        double[] sumOutput = {99, 99};
        invoke(plan, data, mask, sumOutput);
        assertAll(() -> assertEquals(4.0, sumOutput[0]),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(sumOutput[1])));

        var mean = plan(AggregateReductionKind.MEAN, DataType.FLOAT64, Shape.of(2, 4),
                Shape.of(2, 4), List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.BYTE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
        byte[] varying = {1, 1, 1, 0, 1, 0, 0, 0};
        double[] finite = {1, 2, 6, 100, 8, 20, 30, 40};
        double[] meanOutput = new double[2];
        invoke(mean, finite, varying, meanOutput);
        assertArrayEquals(new double[] {3, 8}, meanOutput);
    }

    @Test void allFalseAndEmptySelectionUseFrozenZeroCountResults() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Object data = type == DataType.FLOAT64 ? new double[] {Double.NaN, 2}
                    : type == DataType.FLOAT32 ? new float[] {Float.NaN, 2}
                    : new short[] {(short) 0x7f81, (short) 0x4000};
            byte[] mask = {0, 0};
            Object sum = array(type, 1), mean = array(type, 1);
            invoke(plan(AggregateReductionKind.SUM, type, Shape.of(1, 2), Shape.of(2),
                    carriers(type)), data, mask, sum);
            invoke(plan(AggregateReductionKind.MEAN, type, Shape.of(1, 2), Shape.of(2),
                    carriers(type)), data, mask, mean);
            assertEquals(0L, bits(type, sum));
            assertEquals(type == DataType.FLOAT64 ? 0x7ff8000000000000L
                    : type == DataType.FLOAT32 ? 0x7fc00000L : 0x7fc0L, bits(type, mean));
        }
    }

    @Test void selectedSpecialValuesAndExactRoundingUseTheOrdinaryExactSumContract()
            throws Throwable {
        var plan = plan(AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(6, 2),
                Shape.of(6, 2), carriers(DataType.FLOAT64));
        double[] data = {Double.MIN_VALUE, Double.MIN_VALUE,
                Double.POSITIVE_INFINITY, 4,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.longBitsToDouble(0xfff0000000000042L), 1,
                -0.0, -0.0,
                0x1.fffffffffffffp1023, 0x1.fffffffffffffp1023};
        byte[] mask = new byte[data.length]; java.util.Arrays.fill(mask, (byte) 1);
        double[] output = new double[6]; invoke(plan, data, mask, output);
        assertAll(
                () -> assertEquals(2L, Double.doubleToRawLongBits(output[0])),
                () -> assertEquals(Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                        Double.doubleToRawLongBits(output[1])),
                () -> assertEquals(0x7ff8000000000000L,
                        Double.doubleToRawLongBits(output[2])),
                () -> assertEquals(0x7ff8000000000000L,
                        Double.doubleToRawLongBits(output[3])),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(output[4])),
                () -> assertEquals(Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                        Double.doubleToRawLongBits(output[5])));
    }

    @Test void classIsDeterministicPublicFieldFreeAndMaskBranchPrecedesDataLoad() {
        var plan = plan(AggregateReductionKind.MEAN, DataType.FLOAT32, Shape.of(2, 4),
                Shape.of(4), carriers(DataType.FLOAT32));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        var instructions = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int maskLoad = first(instructions, Opcode.BALOAD);
        int branch = firstAfter(instructions, Opcode.IFEQ, maskLoad);
        int dataLoad = firstAfter(instructions, Opcode.FALOAD, branch);
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        assertAll(() -> assertArrayEquals(bytes, generator.generateClassBytes(
                        route.specialization(), route.kernelIr())),
                () -> assertEquals("([F[B[FLjava/lang/foreign/MemorySegment;[JJJ)V",
                        model.methods().getFirst().methodTypeSymbol().descriptorString()),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("invoke", model.methods().getFirst().methodName().stringValue()),
                () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.PUBLIC)),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.STATIC)),
                () -> assertTrue(maskLoad >= 0 && branch > maskLoad && dataLoad > branch),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()));
    }

    private static int first(List<Instruction> instructions, Opcode opcode) {
        return firstAfter(instructions, opcode, -1);
    }

    private static int firstAfter(List<Instruction> instructions, Opcode opcode, int after) {
        for (int index = after + 1; index < instructions.size(); index++)
            if (instructions.get(index).opcode() == opcode) return index;
        return -1;
    }

    private static void invoke(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            Object data, Object mask, Object output) throws Throwable {
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long bytes = plan.maskedReductionGeometry().orElseThrow().scratchSliceBytes();
        try (Arena arena = Arena.ofConfined()) {
            var scratch = arena.allocate(bytes, Long.BYTES);
            artifact.entryPoint().invokeWithArguments(data, mask, output, scratch,
                    plan.maskedReductionGeometry().orElseThrow().pack(new long[3], 0),
                    0L, plan.elementCount());
        }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan(
            AggregateReductionKind kind, DataType type, Shape data, Shape mask,
            List<CarrierAccess> carriers) {
        var base = CpuMaskedReductionLoweringTest.context(kind, type, data, mask, 1);
        return plan(base, carriers);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan(
            PrepareContext<CpuPartitionAnalysisInputs> base, List<CarrierAccess> carriers) {
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static List<CarrierAccess> carriers(DataType type) {
        CarrierAccess numeric = arrayCarrier(type);
        return List.of(numeric, CarrierAccess.BYTE_ARRAY, numeric);
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            default -> throw new AssertionError();
        };
    }

    private static MemorySegment segment(Object array) {
        if (array instanceof double[] value) return MemorySegment.ofArray(value);
        if (array instanceof float[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((short[]) array);
    }

    private static Object array(DataType type, int size) {
        return type == DataType.FLOAT64 ? new double[size]
                : type == DataType.FLOAT32 ? new float[size] : new short[size];
    }

    private static long bits(DataType type, Object values) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(((double[]) values)[0])
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                        Float.floatToRawIntBits(((float[]) values)[0]))
                : Short.toUnsignedLong(((short[]) values)[0]);
    }
}
