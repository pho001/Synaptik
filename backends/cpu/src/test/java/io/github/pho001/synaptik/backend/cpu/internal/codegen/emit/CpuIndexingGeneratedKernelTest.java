package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuIndexingGeneratedKernelTest {
    @Test void generatedClassesEmbedTypedLoopsForEveryFamilyAndGeneralSegments() {
        var gather = code(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, Shape.of(2, 3)),
                        desc(DataType.INT32, Shape.of(2))), desc(DataType.FLOAT32, Shape.of(2, 2))));
        var elements = code(context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                new IndexAxisAttrs(1)), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, Shape.of(2, 3)),
                        desc(DataType.INT32, Shape.of(2, 2))),
                desc(DataType.FLOAT32, Shape.of(2, 2))));
        var nd = code(context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                List.of(0, 1), List.of(desc(DataType.BOOL, Shape.of(2, 3)),
                        desc(DataType.INT32, Shape.of(2, 1))),
                desc(DataType.BOOL, Shape.of(2, 3))));
        var hot = code(context(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(0),
                List.of(desc(DataType.INT64, Shape.of(2))), desc(DataType.BOOL, Shape.of(2, 3))));
        var general = code(context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                        new IndexAxisAttrs(1)), List.of(0, 1),
                List.of(desc(DataType.FLOAT32, Shape.of(2, 3)),
                        desc(DataType.INT32, Shape.of(2, 2))),
                desc(DataType.FLOAT32, Shape.of(2, 2)),
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT)));
        assertAll(
                () -> assertTypedLoop(gather, Opcode.FALOAD, Opcode.IALOAD, Opcode.FASTORE),
                () -> assertTypedLoop(elements, Opcode.FALOAD, Opcode.IALOAD, Opcode.FASTORE),
                () -> assertTypedLoop(nd, Opcode.BALOAD, Opcode.IALOAD, Opcode.BASTORE),
                () -> assertTypedLoop(hot, Opcode.LALOAD, Opcode.BASTORE),
                () -> assertTrue(opcodeCount(general, Opcode.LMUL) > 0),
                () -> assertTrue(opcodeCount(general, Opcode.LADD) > 0),
                () -> assertTrue(invokes(general).stream().allMatch(call -> {
                    String owner = call.owner().asInternalName();
                    return owner.startsWith("java/lang/foreign/")
                            || owner.equals("java/nio/ByteOrder");
                })),
                () -> assertTrue(java.util.stream.Stream.of(gather, elements, nd, hot, general)
                        .flatMap(body -> invokes(body).stream()).noneMatch(call ->
                                call.type().stringValue().contains("Ljava/lang/Object;")
                                || call.owner().asInternalName().contains("CpuIndexingEmitter")
                                || call.owner().asInternalName().startsWith("java/lang/reflect/")
                                || call.owner().asInternalName().equals("java/util/Map")
                                || call.owner().asInternalName().startsWith(
                                        "io/github/pho001/synaptik/runtime/"))),
                () -> assertTrue(java.util.stream.Stream.of(gather, elements, nd, hot, general)
                        .allMatch(body -> body.elementStream().noneMatch(element -> element instanceof
                                java.lang.classfile.instruction.NewObjectInstruction))));
    }

    @Test void generatedWritersExecuteAllFourMappings() throws Throwable {
        float[] gather = new float[4];
        invoke(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(0,1), List.of(desc(DataType.FLOAT32, Shape.of(2,3)),
                        desc(DataType.INT64, Shape.of(2))), desc(DataType.FLOAT32, Shape.of(2,2))),
                List.of(new float[]{10,11,12,20,21,22}, new long[]{2,0}, gather));
        long[] elements = new long[4];
        invoke(context(new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                List.of(0,1), List.of(desc(DataType.INT64, Shape.of(2,3)),
                        desc(DataType.INT32, Shape.of(2,2))), desc(DataType.INT64, Shape.of(2,2))),
                List.of(new long[]{10,11,12,20,21,22}, new int[]{1,0,2,1}, elements));
        byte[] nd = new byte[6];
        invoke(context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                List.of(0,1), List.of(desc(DataType.BOOL, Shape.of(2,3)),
                        desc(DataType.INT32, Shape.of(2,1))), desc(DataType.BOOL, Shape.of(2,3))),
                List.of(new byte[]{1,0,1,0,1,0}, new int[]{1,0}, nd));
        byte[] hot = new byte[6];
        invoke(context(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(0),
                List.of(desc(DataType.INT64, Shape.of(2))), desc(DataType.BOOL, Shape.of(2,3))),
                List.of(new long[]{2,0}, hot));
        assertAll(() -> assertArrayEquals(new float[]{12,10,22,20}, gather),
                () -> assertArrayEquals(new long[]{11,10,22,21}, elements),
                () -> assertArrayEquals(new byte[]{0,1,0,1,0,1}, nd),
                () -> assertArrayEquals(new byte[]{0,0,1,1,0,0}, hot));
    }

    @Test void gatherCopiesEveryRepresentedTypeWithBothIndexWidthsAndScalarIndices()
            throws Throwable {
        for (DataType type : DataType.values()) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                Object source = values(type, 10, 21, 31);
                Object indices = indexType == DataType.INT32 ? new int[]{2, 0}
                        : new long[]{2, 0};
                Object output = values(type, 0, 0);
                invoke(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                                List.of(0, 1),
                                List.of(desc(type, Shape.of(3)), desc(indexType, Shape.of(2))),
                                desc(type, Shape.of(2))),
                        List.of(source, indices, output));
                assertCarrierEquals(values(type, 31, 10), output,
                        type + "/" + indexType);

                Object elementOutput = values(type, 0, 0, 0, 0);
                Object elementIndices = indexType == DataType.INT32
                        ? new int[]{2, 0, 1, 2} : new long[]{2, 0, 1, 2};
                invoke(context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                                        new IndexAxisAttrs(1)), List.of(0, 1),
                                List.of(desc(type, Shape.of(2, 3)),
                                        desc(indexType, Shape.of(2, 2))),
                                desc(type, Shape.of(2, 2))),
                        List.of(values(type, 10, 21, 31, 40, 51, 61),
                                elementIndices, elementOutput));
                assertCarrierEquals(values(type, 31, 10, 51, 61), elementOutput,
                        type + "/" + indexType + " elements");

                Object ndOutput = values(type, 0, 0, 0, 0, 0, 0);
                Object ndIndices = indexType == DataType.INT32 ? new int[]{1, 0}
                        : new long[]{1, 0};
                invoke(context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                                List.of(0, 1),
                                List.of(desc(type, Shape.of(2, 3)),
                                        desc(indexType, Shape.of(2, 1))),
                                desc(type, Shape.of(2, 3))),
                        List.of(values(type, 10, 21, 31, 40, 51, 61), ndIndices, ndOutput));
                assertCarrierEquals(values(type, 40, 51, 61, 10, 21, 31), ndOutput,
                        type + "/" + indexType + " nd");
            }
            Object scalarOutput = values(type, 0);
            invoke(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                            List.of(0, 1),
                            List.of(desc(type, Shape.of(3)), desc(DataType.INT64, Shape.scalar())),
                            desc(type, Shape.scalar())),
                    List.of(values(type, 10, 21, 31), new long[]{1}, scalarOutput));
            assertCarrierEquals(values(type, 21), scalarOutput, type + " scalar");
        }
    }

    @Test void generatedMappingsHonorBatchTupleDepthZeroStrideOffsetsAndMixedSegments()
            throws Throwable {
        var dataShape = Shape.of(2, 3);
        var indexShape = Shape.of(2);
        var outputShape = Shape.of(2, 2);
        var data = CpuIndexingLoweringTest.descriptor(DataType.INT32, dataShape,
                LayoutDescriptor.of(dataShape, new long[]{0, 2}, 1, true));
        var indices = CpuIndexingLoweringTest.descriptor(DataType.INT64, indexShape,
                LayoutDescriptor.of(indexShape, new long[]{0}, 1, true));
        var output = CpuIndexingLoweringTest.descriptor(DataType.INT32, outputShape,
                LayoutDescriptor.of(outputShape, new long[]{5, 2}, 1, true));
        int[] result = new int[9]; Arrays.fill(result, -1);
        invoke(context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                        List.of(0, 1), List.of(data, indices), output),
                List.of(new int[]{99,10,99,20,99,30}, new long[]{7,2}, result));
        assertArrayEquals(new int[]{-1,30,-1,30,-1,-1,30,-1,30}, result);

        var ndContext = context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1)),
                List.of(0, 1),
                List.of(desc(DataType.INT64, Shape.of(2, 3, 2)),
                        desc(DataType.INT32, Shape.of(2, 2, 1))),
                desc(DataType.INT64, Shape.of(2, 2, 2)),
                List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT));
        int[] indexBits = {2, 0, 1, 0};
        long[] ndOutputBits = new long[8];
        invoke(ndContext, List.of(
                new long[]{0,1,10,11,20,21,100,101,110,111,120,121},
                MemorySegment.ofArray(indexBits), MemorySegment.ofArray(ndOutputBits)));
        assertArrayEquals(new long[]{20,21,0,1,110,111,100,101}, ndOutputBits);
    }

    @Test void guardedGatherCursorHonorsArbitrarySubrangesAndRetainsGeneralFallback()
            throws Throwable {
        int width = 256;
        int count = 1024 * width;
        var data = MemorySegment.ofArray(new double[1024 * width]);
        long[] indices = new long[1024];
        indices[0] = 17;
        indices[1] = 23;
        var output = MemorySegment.ofArray(new double[count * 2 + 7]);
        data.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (17L * width + 255) * Double.BYTES, 11.0);
        data.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (23L * width) * Double.BYTES, 22.0);
        data.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, (23L * width + 1) * Double.BYTES, 33.0);
        Shape dataShape = Shape.of(1024, width);
        Shape indexShape = Shape.of(1024);
        Shape outputShape = Shape.of(1024, width);
        var exact = context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                List.of(0, 1), List.of(desc(DataType.FLOAT64, dataShape),
                        desc(DataType.INT64, indexShape)),
                CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, outputShape,
                        LayoutDescriptor.of(outputShape, new long[]{width * 2L, 2}, 3, true)),
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT));
        invoke(exact, List.of(data, indices, output), new long[3], 255, 258);
        assertAll(
                () -> assertEquals(11.0, output.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (3L + 2L * 255) * Double.BYTES)),
                () -> assertEquals(22.0, output.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (3L + 2L * 256) * Double.BYTES)),
                () -> assertEquals(33.0, output.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (3L + 2L * 257) * Double.BYTES)),
                () -> assertEquals(0.0, output.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (3L + 2L * 254) * Double.BYTES)),
                () -> assertEquals(0.0, output.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (3L + 2L * 258) * Double.BYTES)));

        Shape smallDataShape = Shape.of(3, 2);
        Shape smallIndexShape = Shape.of(2);
        Shape smallOutputShape = Shape.of(2, 2);
        var fallbackOutput = MemorySegment.ofArray(new double[4]);
        var fallback = context(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                List.of(0, 1), List.of(desc(DataType.FLOAT64, smallDataShape),
                        desc(DataType.INT64, smallIndexShape)), desc(DataType.FLOAT64,
                        smallOutputShape), List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.LONG_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        invoke(fallback, List.of(MemorySegment.ofArray(new double[]{10, 11, 20, 21, 30, 31}),
                new long[]{2, 0}, fallbackOutput));
        assertArrayEquals(new double[]{30, 31, 10, 11}, fallbackOutput.toArray(
                ValueLayout.JAVA_DOUBLE));
    }

    @Test void guardedGatherNdCursorHonorsTupleBoundarySubrangesAndRetainsGeneralFallback()
            throws Throwable {
        int tuples = 4096;
        int suffix = 16;
        int count = 4 * tuples * suffix;
        float[] data = new float[4 * 64 * 64 * suffix];
        int[] indices = new int[4 * tuples * 2];
        indices[0] = 3;
        indices[1] = 7;
        indices[2] = 5;
        indices[3] = 11;
        data[(3 * 64 + 7) * suffix + 15] = 11.0f;
        data[(5 * 64 + 11) * suffix] = 22.0f;
        data[(5 * 64 + 11) * suffix + 1] = 33.0f;
        var output = MemorySegment.ofArray(new float[count * 2 + 9]);
        Shape dataShape = Shape.of(4, 64, 64, suffix);
        Shape indexShape = Shape.of(4, tuples, 2);
        Shape outputShape = Shape.of(4, tuples, suffix);
        var exact = context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, dataShape),
                        desc(DataType.INT32, indexShape)),
                CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, outputShape,
                        LayoutDescriptor.of(outputShape,
                                new long[]{tuples * suffix * 2L, suffix * 2L, 2}, 4, true)),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT));
        invoke(exact, List.of(data, indices, output), new long[3], 15, 18);
        assertAll(
                () -> assertEquals(11.0f, output.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (4L + 2L * 15) * Float.BYTES)),
                () -> assertEquals(22.0f, output.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (4L + 2L * 16) * Float.BYTES)),
                () -> assertEquals(33.0f, output.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (4L + 2L * 17) * Float.BYTES)),
                () -> assertEquals(0.0f, output.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (4L + 2L * 14) * Float.BYTES)),
                () -> assertEquals(0.0f, output.get(ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (4L + 2L * 18) * Float.BYTES)));

        Shape smallDataShape = Shape.of(1, 2, 2, 2);
        Shape smallIndexShape = Shape.of(1, 2, 2);
        Shape smallOutputShape = Shape.of(1, 2, 2);
        var fallbackOutput = MemorySegment.ofArray(new float[4]);
        var fallback = context(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1)),
                List.of(0, 1), List.of(desc(DataType.FLOAT32, smallDataShape),
                        desc(DataType.INT32, smallIndexShape)), desc(DataType.FLOAT32,
                        smallOutputShape), List.of(CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT));
        invoke(fallback, List.of(new float[]{10, 11, 20, 21, 30, 31, 40, 41},
                new int[]{1, 0, 0, 1}, fallbackOutput));
        assertArrayEquals(new float[]{30, 31, 20, 21}, fallbackOutput.toArray(
                ValueLayout.JAVA_FLOAT));
    }

    private static void invoke(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        invoke(context, carriers, new long[carriers.size()], 0, plan.elementCount());
    }

    private static void invoke(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers, long[] bases, long start, long end) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        long[] geometry = plan.indexingGeometry().orElseThrow().pack(bases, start, end);
        var args = new ArrayList<Object>(carriers); args.add(geometry); args.add(start);
        args.add(end); artifact.entryPoint().invokeWithArguments(args);
    }
    private static java.lang.classfile.CodeModel code(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
        return ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr())).methods().getFirst().code().orElseThrow();
    }
    private static void assertTypedLoop(java.lang.classfile.CodeModel code,
            Opcode... required) {
        assertAll(() -> assertTrue(invokes(code).isEmpty()),
                () -> assertTrue(java.util.Arrays.stream(required)
                        .allMatch(opcode -> opcodeCount(code, opcode) > 0)),
                () -> assertTrue(opcodeCount(code, Opcode.GOTO) > 0));
    }
    private static List<InvokeInstruction> invokes(
            java.lang.classfile.CodeModel code) {
        return code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
    }
    private static long opcodeCount(java.lang.classfile.CodeModel code,
            Opcode opcode) {
        return code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(value -> value.opcode() == opcode).count();
    }
    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            List<Integer> occurrences,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output) {
        var carriers = new ArrayList<CarrierAccess>();
        for (var input : inputs) carriers.add(heap(input.dataType()));
        carriers.add(heap(output.dataType()));
        return context(operation, occurrences, inputs, output, carriers);
    }
    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            List<Integer> occurrences,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output,
            List<CarrierAccess> carriers) {
        var base = CpuIndexingLoweringTest.context(operation, occurrences, inputs, output);
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
    }
    private static CarrierAccess heap(DataType t) { return switch(t) {
        case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
        case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
        case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY; }; }
    private static io.github.pho001.synaptik.model.tensor.TensorDescriptor desc(DataType t, Shape s) {
        return CpuIndexingLoweringTest.descriptor(t,s);
    }

    private static Object values(DataType type, int... values) {
        return switch (type) {
            case FLOAT64 -> Arrays.stream(values).asDoubleStream().toArray();
            case FLOAT32 -> { float[] result = new float[values.length];
                for (int i = 0; i < values.length; i++) result[i] = values[i]; yield result; }
            case BFLOAT16 -> { short[] result = new short[values.length];
                for (int i = 0; i < values.length; i++) result[i] = (short) (0x3f00 + values[i]);
                yield result; }
            case INT32 -> values.clone();
            case INT64 -> Arrays.stream(values).asLongStream().toArray();
            case BOOL -> { byte[] result = new byte[values.length];
                for (int i = 0; i < values.length; i++) result[i] = (byte) (values[i] & 1);
                yield result; }
        };
    }

    private static void assertCarrierEquals(Object expected, Object actual, String message) {
        if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else if (expected instanceof long[] value) assertArrayEquals(value, (long[]) actual, message);
        else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }
}
