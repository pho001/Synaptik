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
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuIndexingGeneratedKernelTest {
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

    private static void invoke(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        long[] bases = new long[carriers.size()];
        long[] geometry = plan.indexingGeometry().orElseThrow().pack(bases, 0, plan.elementCount());
        var args = new ArrayList<Object>(carriers); args.add(geometry); args.add(0L);
        args.add(plan.elementCount()); artifact.entryPoint().invokeWithArguments(args);
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
