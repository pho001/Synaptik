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
    @Test void controlledStatisticsAndLogSumExpSkipInactiveNonleadingRows() throws Throwable {
        for (var kind : List.of(AggregateReductionKind.LOG_SUM_EXP, AggregateReductionKind.VARIANCE,
                AggregateReductionKind.STANDARD_DEVIATION)) {
            OperationAttrs attrs = kind == AggregateReductionKind.LOG_SUM_EXP
                    ? new MultiAxisReductionAttrs(List.of(2), false)
                    : new StatisticalReductionAttrs(List.of(2), false, 0);
            var plan = plan(kind, DataType.FLOAT64, Shape.of(2, 2, 2), attrs, Shape.of(2, 2),
                    List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY));
            var route = plan.units().getFirst().portablePlan();
            var inputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 2);
            var outputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 1);
            var control = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                    List.of(inputRow, outputRow), List.of(inputRow, outputRow));
            var owner = java.lang.constant.ClassDesc.of(
                    "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.ControlledAdvancedRows");
            var inventory = CpuClassFileKernelGenerator.methodInventory(
                    owner, "row$", route.specialization(), route.kernelIr(), true);
            byte[] bytes = ClassFile.of().build(owner, builder -> {
                builder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL);
                CpuClassFileKernelGenerator.emitMethods(builder, owner, "row$",
                        route.specialization(), route.kernelIr(), control);
            });
            CpuClassFileKernelGenerator.verifyMethodInventory(bytes, inventory);
            var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes, true);
            var entry = lookup.findStatic(lookup.lookupClass(), "row$invoke",
                    inventory.getFirst().type().resolveConstantDesc(lookup));
            try (Arena arena = Arena.ofConfined()) {
                var activity = arena.allocate(18, 8);
                activity.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 1);
                activity.set(java.lang.foreign.ValueLayout.JAVA_LONG, 8, 2L);
                activity.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 16, (byte) 1);
                double[] input = {1, 3, Double.NaN, Double.NaN, 3, 5};
                double[] output = {-9, -9, -9, -9};
                var geometry = plan.advancedReductionGeometry().orElseThrow();
                if (route.specialization().scratchParameter()) {
                    var scratch = arena.allocate(geometry.scratchSliceBytes(), 8);
                    entry.invokeWithArguments(input, output, scratch, geometry.pack(new long[2]),
                            0L, 4L, activity);
                } else {
                    entry.invokeWithArguments(input, output, geometry.pack(new long[2]), 0L, 4L, activity);
                }
                double first = kind == AggregateReductionKind.LOG_SUM_EXP
                        ? 3 + Math.log1p(Math.exp(-2)) : 1;
                double second = kind == AggregateReductionKind.LOG_SUM_EXP ? first + 2 : 1;
                assertArrayEquals(new double[]{first, 0, second, 0}, output, 2e-15, kind.name());
                assertEquals(0L, Double.doubleToRawLongBits(output[1]));
                assertEquals(0L, Double.doubleToRawLongBits(output[3]));
            }
        }
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

    @Test void generatedShapeIsDeterministicTypedFieldFreeAndRuntimeHelperFree() throws Throwable {
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
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            for (int pattern = 0; pattern < 4; pattern++) {
                int carrierPattern = pattern;
                { Object input = represented(type, new double[] {-3, 4});
                Object output = represented(type, new double[] {0});
                Object inputCarrier = (carrierPattern & 1) == 0 ? input : segment(input);
                Object outputCarrier = (carrierPattern & 2) == 0 ? output : segment(output);
                var plan = plan(AggregateReductionKind.L2_NORM, type, 2, 0,
                        List.of((carrierPattern & 1) == 0 ? arrayCarrier(type)
                                        : CarrierAccess.MEMORY_SEGMENT,
                                (carrierPattern & 2) == 0 ? arrayCarrier(type)
                                        : CarrierAccess.MEMORY_SEGMENT));
                invoke(plan, inputCarrier, outputCarrier);
                assertEquals(bits(type, represented(type, new double[] {5.0})),
                        bits(type, output), type + "/pattern=" + carrierPattern); }
            }
        }
    }

    @Test void allFloatingKindTargetsNarrowOnceToTheExpectedRepresentation() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.LOG_SUM_EXP,
                    AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION,
                    AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM)) {
                { boolean statistics = kind == AggregateReductionKind.VARIANCE
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
                Object input = represented(type, values);
                Object output = represented(type, new double[] {0});
                var plan = plan(kind, type, 2, statistics ? 1 : 0,
                        List.of(arrayCarrier(type), arrayCarrier(type)));
                invoke(plan, input, output);
                assertEquals(bits(type, represented(type, new double[] {expected})),
                        bits(type, output), type + "/" + kind); }
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
                () -> assertEquals(0.0, invoke(AggregateReductionKind.L2_NORM,
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
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
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

    @Test void invalidStatisticalCorrectionsFailBeforeGeneration() throws Throwable {
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

    @Test void lowWorkingSquareOverflowRemainsInfinityAfterLaterZeroContributors() throws Throwable {
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.VARIANCE,
                AggregateReductionKind.STANDARD_DEVIATION)) {
            var selected = plan(kind, DataType.BFLOAT16, 4, 0,
                    List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.SHORT_ARRAY));
            Object output = represented(DataType.BFLOAT16, new double[] {0});
            invoke(selected, represented(DataType.BFLOAT16,
                    new double[] {0x1p64, -0x1p64, 0, 0}), output);
            assertEquals(0x7f80L, bits(DataType.BFLOAT16, output), kind.name());
        }
    }

    @Test void squareAndTotalOverflowSurviveFiniteTailsInEveryApplicableWorkingType() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            double individual = type == DataType.FLOAT64 ? Double.MAX_VALUE : 0x1p64;
            double total = type == DataType.FLOAT64 ? 0x1p511 : 0x1p63;
            for (double[] values : List.of(new double[] {individual, -individual, 0, 0},
                    new double[] {total, -total, total, -total, 0, 0})) {
                for (AggregateReductionKind kind : List.of(AggregateReductionKind.VARIANCE,
                        AggregateReductionKind.STANDARD_DEVIATION, AggregateReductionKind.L2_NORM)) {
                    var selected = plan(kind, type, values.length, 0,
                            List.of(arrayCarrier(type), arrayCarrier(type)));
                    Object output = represented(type, new double[] {-9});
                    invoke(selected, represented(type, values), output);
                    assertEquals(bits(type, represented(type, new double[] {Double.POSITIVE_INFINITY})),
                            bits(type, output), type + "/" + kind + "/square or total overflow");
                }
            }
            for (double special : new double[] {Double.NaN, Double.POSITIVE_INFINITY}) {
                for (AggregateReductionKind kind : List.of(AggregateReductionKind.VARIANCE,
                        AggregateReductionKind.STANDARD_DEVIATION)) {
                    var selected = plan(kind, type, 4, 0, List.of(arrayCarrier(type), arrayCarrier(type)));
                    Object output = represented(type, new double[] {0});
                    invoke(selected, represented(type, new double[] {individual, -individual, special, 0}), output);
                    assertTrue(Double.isNaN(CpuLayerNormGeneratedKernelTest.decoded(type, output)[0]),
                            type + "/" + kind + "/input special dominates overflow");
                }
            }
        }
    }

    @Test void roundedMeanSubtractionAndSquareUseTheDeclaredPrecisionWithoutResidualCorrection() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            double base = type == DataType.FLOAT32 ? 0x1p24 : 0x1p53;
            var selected = plan(AggregateReductionKind.VARIANCE, type, 2, 0,
                    List.of(arrayCarrier(type), arrayCarrier(type)));
            Object output = represented(type, new double[] {0});
            invoke(selected, represented(type, new double[] {base, base + 2}), output);
            // The rounded mean is base: centered squares are 0 and 4, not exact-real variance 1.
            assertEquals(bits(type, represented(type, new double[] {2})), bits(type, output), type.name());
        }
    }

    @Test void floatLogSumExpRoundsEachNamedWorkingSite() throws Throwable {
        for (float tail : new float[] {.1f, .5f, 1f, 2f, 10f, 50f}) {
            var selected = plan(AggregateReductionKind.LOG_SUM_EXP, DataType.FLOAT32, 2, 0,
                    List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
            float[] output = {-9};
            invoke(selected, new float[] {0, -tail}, output);
            float term = (float) StrictMath.exp(-tail);
            float sum = 1f + term;
            float expected = (float) StrictMath.log(sum);
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(output[0]),
                    "binary32 shifted EXP/sum/LOG for " + tail);
        }
        var selected = plan(AggregateReductionKind.LOG_SUM_EXP, DataType.FLOAT32, 2, 0,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        float[] output = {0};
        invoke(selected, new float[] {Float.POSITIVE_INFINITY, 0}, output);
        assertEquals(Float.POSITIVE_INFINITY, output[0]);
        invoke(selected, new float[] {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, output);
        assertEquals(Float.NEGATIVE_INFINITY, output[0]);
        invoke(selected, new float[] {Float.POSITIVE_INFINITY, Float.NaN}, output);
        assertTrue(Float.isNaN(output[0]));
    }

    @Test void changedArithmeticRejectsRetiredArtifactsButL1RetainsBothIdentityStreams() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            for (AggregateReductionKind kind : List.of(AggregateReductionKind.VARIANCE,
                    AggregateReductionKind.STANDARD_DEVIATION, AggregateReductionKind.L2_NORM)) {
                var selected = plan(kind, type, 4, 0, List.of(arrayCarrier(type), arrayCarrier(type)));
                var route = selected.units().getFirst().portablePlan();
                var current = route.specialization();
                var retired = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                        current.loweringFingerprint(), current.executionStrategy(), current.boundaryDataTypes(),
                        current.carrierPattern(), 0, -1, List.of(), current.scratchParameter(),
                        type == DataType.FLOAT64 || type == DataType.FLOAT32 ? 52 : 77);
                assertFalse(java.util.Arrays.equals(retired.compatibilityBytes(), current.compatibilityBytes()));
                assertFalse(java.util.Arrays.equals(retired.classIdentityBytes(), current.classIdentityBytes()));
                assertThrows(IllegalArgumentException.class, () -> new CpuClassFileKernelGenerator()
                        .generateClassBytes(retired, route.kernelIr()));
            }
            var control = plan(AggregateReductionKind.L1_NORM, type, 4, 0,
                    List.of(arrayCarrier(type), arrayCarrier(type))).units().getFirst().portablePlan().specialization();
            var retained = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                    control.loweringFingerprint(), control.executionStrategy(), control.boundaryDataTypes(),
                    control.carrierPattern(), 0, -1, List.of(), control.scratchParameter(),
                    type == DataType.FLOAT64 || type == DataType.FLOAT32 ? 52 : 77);
            assertArrayEquals(retained.compatibilityBytes(), control.compatibilityBytes());
            assertArrayEquals(retained.classIdentityBytes(), control.classIdentityBytes());
        }
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
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
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
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static Object represented(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        if (type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2) {
            byte[] result = new byte[values.length];
            for (int i = 0; i < values.length; i++) result[i] = type == DataType.FLOAT8_E4M3
                    ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromDouble(values[i])
                    : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromDouble(values[i]);
            return result;
        }
        short[] result = new short[values.length];
        if (type == DataType.FLOAT16) {
            for (int i = 0; i < values.length; i++) result[i] = Float.floatToFloat16((float) values[i]);
            return result;
        }
        for (int i = 0; i < values.length; i++) result[i] = (short) (Float.floatToRawIntBits(
                (float) values[i]) >>> 16);
        return result;
    }

    private static MemorySegment segment(Object array) {
        if (array instanceof double[] value) return MemorySegment.ofArray(value);
        if (array instanceof float[] value) return MemorySegment.ofArray(value);
        if (array instanceof byte[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((short[]) array);
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY
                : type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2
                    ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY;
    }

    private static long bits(DataType type, Object array) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(((double[]) array)[0])
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                    Float.floatToRawIntBits(((float[]) array)[0]))
                : type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2
                    ? Byte.toUnsignedLong(((byte[]) array)[0]) : Short.toUnsignedLong(((short[]) array)[0]);
    }
}
