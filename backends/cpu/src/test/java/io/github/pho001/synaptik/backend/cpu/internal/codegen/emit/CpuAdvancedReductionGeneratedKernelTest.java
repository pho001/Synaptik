package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.AccessFlag;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CpuAdvancedReductionGeneratedKernelTest {
    @Test void focusedEmittersOwnNumericalBodiesWithoutSelectorDelegation() {
        assertAll(
                () -> assertTrue(java.util.Arrays.stream(CpuLogSumExpEmitter.class
                        .getDeclaredMethods()).noneMatch(method -> method.getName()
                                .equals("emitAdvanced"))),
                () -> assertTrue(java.util.Arrays.stream(CpuStatisticalReductionEmitter.class
                        .getDeclaredMethods()).noneMatch(method -> method.getName()
                                .equals("emitAdvanced"))),
                () -> assertTrue(java.util.Arrays.stream(CpuNormEmitter.class.getDeclaredFields())
                        .noneMatch(field -> field.getType() == int.class
                                && java.lang.reflect.Modifier.isStatic(field.getModifiers()))));
    }

    @Test void allFiveDoubleBodiesProduceFrozenFiniteAndSpecialResults() throws Throwable {
        assertEquals(3 + StrictMath.log(StrictMath.exp(-2) + StrictMath.exp(-1) + 1),
                invoke(AggregateReductionKind.LOG_SUM_EXP, new double[] {1, 2, 3}, 0), 2e-15);
        assertEquals(2.0, invoke(AggregateReductionKind.VARIANCE, new double[] {1, 3}, 1));
        assertEquals(StrictMath.sqrt(2.0), invoke(AggregateReductionKind.STANDARD_DEVIATION,
                new double[] {1, 3}, 1));
        assertEquals(3.0, invoke(AggregateReductionKind.L1_NORM,
                new double[] {-1, 2}, 0));
        assertEquals(5.0, invoke(AggregateReductionKind.L2_NORM,
                new double[] {3, 4}, 0));
        assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(invoke(
                AggregateReductionKind.L1_NORM,
                new double[] {Double.POSITIVE_INFINITY, Double.NaN}, 0)));
    }

    @Test void generatedShapeIsDeterministicTypedFieldFreeAndRuntimeHelperFree() {
        var plan = plan(AggregateReductionKind.L2_NORM, 3, 0);
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertAll(() -> assertArrayEquals(bytes, generator.generateClassBytes(
                        route.specialization(), route.kernelIr())),
                () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()), () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("([D[D[JJJ)V",
                        model.methods().getFirst().methodTypeSymbol().descriptorString()),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik"))),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()));
    }

    @Test void everyFloatingTypeAndHeapSegmentPairUsesTheSameRepresentedResult() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (int pattern = 0; pattern < 4; pattern++) {
                Object input = represented(type, new double[] {-3, 4});
                Object output = represented(type, new double[] {0});
                Object inputCarrier = (pattern & 1) == 0 ? input : segment(input);
                Object outputCarrier = (pattern & 2) == 0 ? output : segment(output);
                var plan = plan(AggregateReductionKind.L2_NORM, type, 2, 0,
                        List.of((pattern & 1) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT,
                                (pattern & 2) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT));
                invoke(plan, inputCarrier, outputCarrier);
                assertEquals(type == DataType.FLOAT64 ? Double.doubleToRawLongBits(5.0)
                        : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                            Float.floatToRawIntBits(5.0f)) : 0x40a0L,
                        bits(type, output), type + "/pattern=" + pattern);
            }
        }
    }

    @Test void fifteenKindTypeTargetsNarrowOnceToTheExpectedRepresentation() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.LOG_SUM_EXP,
                    AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION,
                    AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM)) {
                boolean statistics = kind == AggregateReductionKind.VARIANCE
                        || kind == AggregateReductionKind.STANDARD_DEVIATION;
                double[] values = kind == AggregateReductionKind.L2_NORM
                        ? new double[] {3, 4} : kind == AggregateReductionKind.L1_NORM
                        ? new double[] {-1, 2} : new double[] {1, 3};
                double expected = switch (kind) {
                    case LOG_SUM_EXP -> 3 + StrictMath.log1p(StrictMath.exp(-2));
                    case VARIANCE -> 2;
                    case STANDARD_DEVIATION -> StrictMath.sqrt(2);
                    case L1_NORM -> 3;
                    case L2_NORM -> 5;
                    default -> throw new AssertionError();
                };
                Object input = represented(type, values), output = represented(type, new double[] {0});
                var plan = plan(kind, type, 2, statistics ? 1 : 0,
                        List.of(arrayCarrier(type), arrayCarrier(type)));
                invoke(plan, input, output);
                assertEquals(bits(type, represented(type, new double[] {expected})),
                        bits(type, output), type + "/" + kind);
            }
        }
    }

    @Test void exactSpecialPrioritySignedZeroAndFiniteOverflowAreFrozen() throws Throwable {
        assertAll(
                () -> assertEquals(Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
                        Double.doubleToRawLongBits(invoke(AggregateReductionKind.LOG_SUM_EXP,
                                new double[] {Double.NEGATIVE_INFINITY,
                                        Double.NEGATIVE_INFINITY}, 0))),
                () -> assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(invoke(
                        AggregateReductionKind.LOG_SUM_EXP,
                        new double[] {Double.POSITIVE_INFINITY,
                                Double.longBitsToDouble(0x7ff0000000000042L)}, 0))),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(invoke(
                        AggregateReductionKind.LOG_SUM_EXP, new double[] {-0.0}, 0))),
                () -> assertEquals(0L, Double.doubleToRawLongBits(invoke(
                        AggregateReductionKind.VARIANCE, new double[] {-0.0, -0.0}, 0))),
                () -> assertEquals(0x7ff8000000000000L, Double.doubleToRawLongBits(invoke(
                        AggregateReductionKind.STANDARD_DEVIATION,
                        new double[] {1.0, Double.POSITIVE_INFINITY}, 0))),
                () -> assertEquals(Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                        Double.doubleToRawLongBits(invoke(AggregateReductionKind.VARIANCE,
                                new double[] {Double.MAX_VALUE, -Double.MAX_VALUE}, 0))),
                () -> assertEquals(Double.doubleToRawLongBits(0x1.0p53),
                        Double.doubleToRawLongBits(invoke(AggregateReductionKind.L1_NORM,
                                new double[] {0x1.0p53, 1.0}, 0))),
                () -> assertEquals(Double.MIN_VALUE, invoke(AggregateReductionKind.L2_NORM,
                        new double[] {Double.MIN_VALUE, 0.0}, 0)));
    }

    @Test void nonadjacentAxesGeneralOutputAndZeroStrideReadsPreserveCanaries() throws Throwable {
        Shape inputShape = Shape.of(2, 3, 4), outputShape = Shape.of(3);
        var inputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, inputShape,
                LayoutDescriptor.of(inputShape, new long[] {0, 4, 1}, 1, true));
        var outputDescriptor = CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, outputShape,
                LayoutDescriptor.of(outputShape, new long[] {2}, 1, true));
        var base = CpuScatterLoweringTest.context(new Operation(AggregateReductionKind.L1_NORM,
                new MultiAxisReductionAttrs(List.of(2, 0), false)), List.of(0),
                List.of(inputDescriptor), outputDescriptor);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                    List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        double[] input = new double[13];
        for (int row = 0; row < 3; row++) for (int column = 0; column < 4; column++)
            input[1 + row * 4 + column] = row + 1;
        double[] output = new double[7]; java.util.Arrays.fill(output, -77);
        invoke(plan, input, output);
        assertArrayEquals(new double[] {-77, 8, -77, 16, -77, 24, -77}, output);
    }

    @Test void emptyAxesArePointDomainsAndSelectedZeroExtentsUseExactIdentities()
            throws Throwable {
        double[] points = {-0.0, 2.0, -3.0};
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.LOG_SUM_EXP,
                AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION,
                AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM)) {
            boolean statistics = kind == AggregateReductionKind.VARIANCE
                    || kind == AggregateReductionKind.STANDARD_DEVIATION;
            OperationAttrs attrs = statistics
                    ? new StatisticalReductionAttrs(List.of(), false, 0)
                    : new MultiAxisReductionAttrs(List.of(), false);
            var pointPlan = plan(kind, DataType.FLOAT64, Shape.of(3), attrs, Shape.of(3),
                    List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
            double[] output = {91, 92, 93}; invoke(pointPlan, points, output);
            double[] expected = statistics ? new double[] {0, 0, 0}
                    : kind == AggregateReductionKind.LOG_SUM_EXP ? points
                    : new double[] {0, 2, 3};
            for (int index = 0; index < expected.length; index++) assertEquals(
                    Double.doubleToRawLongBits(expected[index]),
                    Double.doubleToRawLongBits(output[index]), kind + "/point/" + index);
        }
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.LOG_SUM_EXP,
                AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM)) {
            var emptyPlan = plan(kind, DataType.FLOAT64, Shape.of(2, 0),
                    new MultiAxisReductionAttrs(List.of(1), false), Shape.of(2),
                    List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
            double[] output = {91, 92}; invoke(emptyPlan, new double[0], output);
            long expected = kind == AggregateReductionKind.LOG_SUM_EXP
                    ? Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY) : 0L;
            assertAll(() -> assertEquals(expected, Double.doubleToRawLongBits(output[0])),
                    () -> assertEquals(expected, Double.doubleToRawLongBits(output[1])));
        }
    }

    @Test void invalidStatisticalCorrectionsFailBeforeGeneration() {
        for (long correction : new long[] {2, Long.MAX_VALUE}) assertThrows(
                IllegalArgumentException.class, () -> plan(AggregateReductionKind.VARIANCE,
                        DataType.FLOAT64, Shape.of(2),
                        new StatisticalReductionAttrs(List.of(0), false, correction),
                        Shape.scalar(), List.of(CarrierAccess.DOUBLE_ARRAY,
                                CarrierAccess.DOUBLE_ARRAY)));
        assertThrows(IllegalArgumentException.class, () -> plan(AggregateReductionKind.VARIANCE,
                DataType.FLOAT64, Shape.of(2, 0),
                new StatisticalReductionAttrs(List.of(1), false, 0), Shape.of(2),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
    }

    private static double invoke(AggregateReductionKind kind, double[] input, long correction)
            throws Throwable {
        var plan = plan(kind, input.length, correction);
        double[] output = {99}; invoke(plan, input, output); return output[0];
    }

    private static void invoke(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            Object input, Object output) throws Throwable {
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long bytes = plan.advancedReductionGeometry().orElseThrow()
                .scratchSliceBytes();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment scratch = bytes == 0 ? null : arena.allocate(bytes, Long.BYTES);
            if (scratch == null) artifact.entryPoint().invokeWithArguments(input, output,
                    plan.advancedReductionGeometry().orElseThrow().pack(new long[2]), 0L,
                    plan.elementCount());
            else artifact.entryPoint().invokeWithArguments(input, output, scratch,
                    plan.advancedReductionGeometry().orElseThrow().pack(new long[2]), 0L,
                    plan.elementCount());
        }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(AggregateReductionKind kind, int count, long correction) {
        return plan(kind, DataType.FLOAT64, count, correction,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(AggregateReductionKind kind, DataType type, int count, long correction,
                    List<CarrierAccess> carriers) {
        OperationAttrs attrs = kind == AggregateReductionKind.VARIANCE
                || kind == AggregateReductionKind.STANDARD_DEVIATION
                ? new StatisticalReductionAttrs(List.of(0), false, correction)
                : new MultiAxisReductionAttrs(List.of(0), false);
        return plan(kind, type, Shape.of(count), attrs, Shape.scalar(), carriers);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(AggregateReductionKind kind, DataType type, Shape input, OperationAttrs attrs,
                    Shape output, List<CarrierAccess> carriers) {
        var base = CpuAggregateLoweringTest.context(kind, type, input, attrs, output);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static Object represented(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) result[i] = (short) (Float.floatToRawIntBits(
                (float) values[i]) >>> 16);
        return result;
    }

    private static MemorySegment segment(Object array) {
        if (array instanceof double[] value) return MemorySegment.ofArray(value);
        if (array instanceof float[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((short[]) array);
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY;
    }

    private static long bits(DataType type, Object array) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(((double[]) array)[0])
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                    Float.floatToRawIntBits(((float[]) array)[0]))
                : Short.toUnsignedLong(((short[]) array)[0]);
    }
}
