package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuDataMovementGeneratedKernelTest {
    @Test void windowArtifactIdentityIncludesOnlyLockedStructuralFacts() {
        var window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false);
        var input = CpuNonAffineMovementLoweringTest.descriptor(
                DataType.FLOAT32, Shape.of(1, 1, 3, 3));
        var output = CpuNonAffineMovementLoweringTest.descriptor(
                DataType.FLOAT32, Shape.of(1, 4, 16));
        var directContext = context(new Operation(WindowTransformKind.UNFOLD2D, window),
                List.of(0), List.of(input), output);
        var direct = route(directContext);
        var typedZero = route(context(new Operation(WindowTransformKind.UNFOLD2D,
                new Unfold2dAttrs(window, ScalarValue.float32(+0.0f))),
                List.of(0), List.of(input), output));
        var nanOne = route(context(new Operation(WindowTransformKind.UNFOLD2D,
                new Unfold2dAttrs(window, ScalarValue.float32(
                        Float.intBitsToFloat(0x7fc0_0041)))), List.of(0), List.of(input), output));
        var nanTwo = route(context(new Operation(WindowTransformKind.UNFOLD2D,
                new Unfold2dAttrs(window, ScalarValue.float32(
                        Float.intBitsToFloat(0x7fc0_0042)))), List.of(0), List.of(input), output));

        var coldOne = route(context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(0, 2, 1)), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(2, 5))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(1, 5, 2))));
        var coldTwo = route(context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 3, 2)), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(1, 9))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(1, 4, 3))));
        var generator = new CpuClassFileKernelGenerator();
        var segmentContext = new PrepareContext<>(directContext.partition(), directContext.nodes(),
                directContext.values(), directContext.memoryRequirements(), directContext.constants(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT)));
        var segment = route(segmentContext);
        var doubleType = route(context(new Operation(WindowTransformKind.UNFOLD2D, window),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.FLOAT64, Shape.of(1, 1, 3, 3))),
                CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.FLOAT64, Shape.of(1, 4, 16))));
        byte[] directBytes = generator.generateClassBytes(direct.specialization(), direct.kernelIr());
        byte[] typedZeroBytes = generator.generateClassBytes(
                typedZero.specialization(), typedZero.kernelIr());
        byte[] coldOneBytes = generator.generateClassBytes(
                coldOne.specialization(), coldOne.kernelIr());
        byte[] coldTwoBytes = generator.generateClassBytes(
                coldTwo.specialization(), coldTwo.kernelIr());
        assertAll(
                () -> assertEquals(direct.specialization(), typedZero.specialization()),
                () -> assertArrayEquals(directBytes, typedZeroBytes),
                () -> assertNotEquals(nanOne.specialization(), nanTwo.specialization()),
                () -> assertFalse(Arrays.equals(
                        generator.generateClassBytes(nanOne.specialization(), nanOne.kernelIr()),
                        generator.generateClassBytes(nanTwo.specialization(), nanTwo.kernelIr()))),
                () -> assertEquals(coldOne.specialization(), coldTwo.specialization()),
                () -> assertArrayEquals(coldOneBytes, coldTwoBytes),
                () -> assertNotEquals(direct.specialization(), segment.specialization()),
                () -> assertNotEquals(direct.specialization(), doubleType.specialization()));
    }

    @Test void executesAxisAndTwoDimensionalWindowsAcrossArbitraryRanges() throws Throwable {
        int[] axisOutput = new int[8];
        var axisContext = context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 2, 1)), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(2, 3))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2, 2)));
        invoke(axisContext, List.of(new int[]{1, 2, 3, 4, 5, 6}, axisOutput), 1, 7);
        int[] axisReference = new int[8];
        invokeReference(axisContext, List.of(new int[]{1, 2, 3, 4, 5, 6}, axisReference), 1, 7);

        int[] rankThreeOutput = new int[8];
        var rankThree = context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(2, 2, 2)), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(1, 1, 5))),
                CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(1, 1, 2, 2)));
        invoke(rankThree, List.of(new int[]{1, 2, 3, 4, 5}, rankThreeOutput), 0, 4);

        Window2dAttrs window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false);
        float[] imageOutput = new float[64];
        var imageContext = context(new Operation(WindowTransformKind.UNFOLD2D,
                        new Unfold2dAttrs(window, ScalarValue.float32(-0.0f))), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.FLOAT32, Shape.of(1, 1, 3, 3))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.FLOAT32, Shape.of(1, 4, 16)));
        invoke(imageContext, List.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, imageOutput), 3, 61);
        float[] imageReference = new float[64];
        invokeReference(imageContext,
                List.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, imageReference), 3, 61);
        assertAll(
                () -> assertArrayEquals(axisReference, axisOutput),
                () -> assertArrayEquals(new int[]{1, 2, 3, 4, 0, 0, 0, 0}, rankThreeOutput),
                () -> assertArrayEquals(imageReference, imageOutput),
                () -> assertEquals(Float.floatToRawIntBits(-0.0f),
                        Float.floatToRawIntBits(imageOutput[3])),
                () -> assertEquals(1.0f, imageOutput[5]));
    }

    @Test void honorsOffsetStridedReadZeroInputAndInjectiveStridedOutput() throws Throwable {
        Shape inputShape = Shape.of(2, 3), outputShape = Shape.of(2, 2, 2);
        var input = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, inputShape,
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(inputShape,
                        new long[]{0, 2}, 1, true));
        var output = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, outputShape,
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(outputShape,
                        new long[]{10, 4, 1}, 2, true));
        var context = context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 2, 1)), List.of(0), List.of(input), output);
        int[] source = {99, 1, 99, 2, 99, 3};
        int[] generated = new int[18], reference = new int[18];
        Arrays.fill(generated, -7);
        Arrays.fill(reference, -7);
        invoke(context, List.of(source, generated), 1, 7);
        invokeReference(context, List.of(source, reference), 1, 7);
        assertArrayEquals(reference, generated);
    }

    @Test void generatedAndReferenceWindowsAgreeForEveryAdmittedRepresentedType() throws Throwable {
        var axisRows = List.of(
                new WindowRow(DataType.FLOAT64, new double[]{1, -0.0, 3}, new double[4]),
                new WindowRow(DataType.FLOAT32, new float[]{1, -0.0f, 3}, new float[4]),
                new WindowRow(DataType.BFLOAT16,
                        new short[]{0x3f80, (short) 0x8000, 0x4040}, new short[4]),
                new WindowRow(DataType.INT32, new int[]{1, Integer.MIN_VALUE, 3}, new int[4]),
                new WindowRow(DataType.INT64, new long[]{1, Long.MIN_VALUE, 3}, new long[4]),
                new WindowRow(DataType.BOOL, new byte[]{1, 0, 1}, new byte[4]));
        for (WindowRow row : axisRows) {
            var axis = context(new Operation(WindowTransformKind.UNFOLD_AXIS,
                            new UnfoldAxisAttrs(0, 2, 1)), List.of(0),
                    List.of(CpuNonAffineMovementLoweringTest.descriptor(row.type(), Shape.of(3))),
                    CpuNonAffineMovementLoweringTest.descriptor(row.type(), Shape.of(2, 2)));
            Object generated = copy(row.output()), reference = copy(row.output());
            invoke(axis, List.of(row.input(), generated), 0, 4);
            invokeReference(axis, List.of(row.input(), reference), 0, 4);
            assertTrue(equalBits(generated, reference), "UNFOLD_AXIS " + row.type());
        }

        var window = new Window2dAttrs(1, 1, 2, 2, 1, 1, 1, 1, true);
        var imageRows = List.of(
                new ImageWindowRow(DataType.FLOAT64,
                        ScalarValue.float64(Double.longBitsToDouble(0x7ff8_0000_0000_0042L)),
                        new double[]{-0.0}, new double[4]),
                new ImageWindowRow(DataType.FLOAT32,
                        ScalarValue.float32(Float.intBitsToFloat(0x7fc0_0042)),
                        new float[]{-0.0f}, new float[4]),
                new ImageWindowRow(DataType.BFLOAT16,
                        ScalarValue.bfloat16Bits((short) 0x7fc2),
                        new short[]{(short) 0x8000}, new short[4]));
        for (ImageWindowRow row : imageRows) {
            var image = context(new Operation(WindowTransformKind.UNFOLD2D,
                            new Unfold2dAttrs(window, row.padding())), List.of(0),
                    List.of(CpuNonAffineMovementLoweringTest.descriptor(
                            row.type(), Shape.of(1, 1, 1, 1))),
                    CpuNonAffineMovementLoweringTest.descriptor(row.type(), Shape.of(1, 1, 4)));
            Object generated = copy(row.output()), reference = copy(row.output());
            invoke(image, List.of(row.input(), generated), 0, 4);
            invokeReference(image, List.of(row.input(), reference), 0, 4);
            assertAll(
                    () -> assertTrue(equalBits(generated, reference), "UNFOLD2D " + row.type()),
                    () -> assertAllPaddingBits(generated, row.padding()));
        }
    }

    @Test void executesPadTileConcatAndStackWithArbitraryRanges() throws Throwable {
        int[] padded = new int[5];
        invoke(context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-7))),
                        List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5))),
                List.of(new int[]{10, 20}, padded), 0, 5);
        int[] tiled = new int[6];
        invoke(context(new Operation(TileKind.TILE, new TileAttrs(List.of(3L))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(6))),
                List.of(new int[]{1, 2}, tiled), 1, 5);
        int[] concatenated = new int[5];
        invoke(context(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(1))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5))),
                List.of(new int[]{1, 2}, new int[]{3}, concatenated), 0, 5);
        int[] stacked = new int[4];
        invoke(context(new Operation(TensorCompositionKind.STACK,
                        new CompositionAxisAttrs(1)), List.of(0, 1),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(
                                        DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2))),
                List.of(new int[]{1, 2}, new int[]{3, 4}, stacked), 0, 4);
        assertAll(
                () -> assertArrayEquals(new int[]{-7, 10, 20, -7, -7}, padded),
                () -> assertArrayEquals(new int[]{0, 2, 1, 2, 1, 0}, tiled),
                () -> assertArrayEquals(new int[]{1, 2, 3, 1, 2}, concatenated),
                () -> assertArrayEquals(new int[]{1, 3, 2, 4}, stacked));
    }

    @Test void preservesRawBfloatPaddingBitsAndHasNoSemanticHotDependencies() throws Throwable {
        var context = context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(1L),
                                ScalarValue.bfloat16Bits((short) 0x7fc1))),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.BFLOAT16, Shape.of(1))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.BFLOAT16, Shape.of(3)));
        short[] output = new short[3];
        byte[] bytes = invoke(context, List.of(new short[]{(short) 0x8000}, output), 0, 3);
        String constants = new String(bytes, StandardCharsets.ISO_8859_1);
        assertAll(
                () -> assertArrayEquals(new short[]{(short) 0x7fc1, (short) 0x8000,
                        (short) 0x7fc1}, output),
                () -> assertFalse(constants.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(constants.contains("java/lang/reflect")),
                () -> assertFalse(constants.contains("java/util/Map")));
    }

    @Test void generatedAndReferencePaddingAgreeForEveryRepresentedType() throws Throwable {
        var rows = List.of(
                new PadRow(DataType.FLOAT64, ScalarValue.float64(
                        Double.longBitsToDouble(0x7ff8_0000_0000_0042L)),
                        new double[]{-0.0d}, new double[3]),
                new PadRow(DataType.FLOAT32, ScalarValue.float32(
                        Float.intBitsToFloat(0x7fc0_0042)), new float[]{-0.0f}, new float[3]),
                new PadRow(DataType.BFLOAT16, ScalarValue.bfloat16Bits((short) 0x7fc1),
                        new short[]{(short) 0x8000}, new short[3]),
                new PadRow(DataType.INT32, ScalarValue.int32(Integer.MIN_VALUE),
                        new int[]{17}, new int[3]),
                new PadRow(DataType.INT64, ScalarValue.int64(Long.MIN_VALUE),
                        new long[]{17}, new long[3]),
                new PadRow(DataType.BOOL, ScalarValue.bool(true),
                        new byte[]{0}, new byte[3]));
        for (PadRow row : rows) {
            var context = context(new Operation(PadKind.PAD,
                            new PadAttrs(List.of(1L), List.of(1L), row.padding())),
                    List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                            row.type(), Shape.of(1))),
                    CpuNonAffineMovementLoweringTest.descriptor(row.type(), Shape.of(3)));
            Object generated = copy(row.output());
            Object reference = copy(row.output());
            invoke(context, List.of(row.input(), generated), 0, 3);
            invokeReference(context, List.of(row.input(), reference), 0, 3);
            assertTrue(equalBits(generated, reference), row.type().name());
        }
    }

    @Test void referenceAgreesWithGeneratedForEveryMovementFamily() throws Throwable {
        var cases = List.of(
                new MovementCase(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(1L), ScalarValue.int32(-3))),
                        List.of(0), List.of(Shape.of(2)), Shape.of(4),
                        List.of(new int[]{4, 5})),
                new MovementCase(new Operation(TileKind.TILE, new TileAttrs(List.of(3L))),
                        List.of(0), List.of(Shape.of(2)), Shape.of(6),
                        List.of(new int[]{4, 5})),
                new MovementCase(new Operation(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                        List.of(Shape.of(2), Shape.of(1)), Shape.of(5),
                        List.of(new int[]{4, 5}, new int[]{9})),
                new MovementCase(new Operation(TensorCompositionKind.STACK,
                        new CompositionAxisAttrs(1)), List.of(0, 1),
                        List.of(Shape.of(2), Shape.of(2)), Shape.of(2, 2),
                        List.of(new int[]{4, 5}, new int[]{8, 9})));
        for (MovementCase movement : cases) {
            var descriptors = movement.inputs().stream().map(shape ->
                    CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, shape)).toList();
            var context = context(movement.operation(), movement.occurrences(), descriptors,
                    CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, movement.output()));
            int count = Math.toIntExact(movement.output().knownElementCount().orElseThrow());
            int[] generated = new int[count], reference = new int[count];
            var generatedArguments = new ArrayList<Object>(movement.carriers());
            generatedArguments.add(generated);
            var referenceArguments = new ArrayList<Object>(movement.carriers());
            referenceArguments.add(reference);
            invoke(context, generatedArguments, 0, count);
            invokeReference(context, referenceArguments, 0, count);
            assertArrayEquals(reference, generated, movement.operation().kind().toString());
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            List<Integer> occurrences,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output) {
        var base = CpuNonAffineMovementLoweringTest.context(operation, occurrences, inputs, output);
        CarrierAccess carrier = switch (output.dataType()) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(inputs.size() + 1, carrier)));
    }

    private static byte[] invoke(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers, long start, long end) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        long[] bases = new long[carriers.size()];
        long[] geometry = plan.movementGeometry().orElseThrow().pack(bases, start, end);
        var arguments = new ArrayList<Object>(carriers);
        arguments.add(geometry);
        arguments.add(start);
        arguments.add(end);
        artifact.entryPoint().invokeWithArguments(arguments);
        return bytes;
    }

    private static CpuPortableRoutePlan route(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }

    private static void invokeReference(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers, long start, long end) {
        var lowered = new CpuPartitionLowering().lower(context);
        var ir = (CpuDataMovementIr) lowered.portableKernelIr();
        var arguments = new ArrayList<CpuBufferArgument>(carriers.size());
        for (int index = 0; index < carriers.size(); index++) {
            arguments.add(argument(ir.dataType(), carriers.get(index),
                    index + 1 < carriers.size()));
        }
        CpuScalarReferenceKernel.execute(ir, lowered.movementGeometry().orElseThrow(),
                arguments, start, end);
    }

    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes = Math.multiplyExact((long) java.lang.reflect.Array.getLength(carrier),
                type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0, bytes, readOnly);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) carrier, 0, bytes, readOnly);
        };
    }

    private static Object copy(Object array) {
        if (array instanceof double[] value) return value.clone();
        if (array instanceof float[] value) return value.clone();
        if (array instanceof short[] value) return value.clone();
        if (array instanceof int[] value) return value.clone();
        if (array instanceof long[] value) return value.clone();
        return ((byte[]) array).clone();
    }

    private static boolean equalBits(Object left, Object right) {
        if (left instanceof double[] value) return Arrays.equals(value, (double[]) right);
        if (left instanceof float[] value) return Arrays.equals(value, (float[]) right);
        if (left instanceof short[] value) return Arrays.equals(value, (short[]) right);
        if (left instanceof int[] value) return Arrays.equals(value, (int[]) right);
        if (left instanceof long[] value) return Arrays.equals(value, (long[]) right);
        return Arrays.equals((byte[]) left, (byte[]) right);
    }

    private static void assertAllPaddingBits(Object values, ScalarValue padding) {
        if (values instanceof double[] array) for (double value : array) assertEquals(
                Double.doubleToRawLongBits(padding.float64Value()), Double.doubleToRawLongBits(value));
        else if (values instanceof float[] array) for (float value : array) assertEquals(
                Float.floatToRawIntBits(padding.float32Value()), Float.floatToRawIntBits(value));
        else for (short value : (short[]) values) assertEquals(padding.bfloat16Bits(), value);
    }

    private record PadRow(DataType type, ScalarValue padding, Object input, Object output) { }
    private record WindowRow(DataType type, Object input, Object output) { }
    private record ImageWindowRow(DataType type, ScalarValue padding, Object input, Object output) { }
    private record MovementCase(Operation operation, List<Integer> occurrences,
            List<Shape> inputs, Shape output, List<Object> carriers) { }
}
