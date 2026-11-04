package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuOrderingGeneratedKernelTest {
    @Test void controlledOrderingSkipsNonleadingRowsBeforeScratchAndOperandLoads() throws Throwable {
        for (boolean segment : List.of(false, true)) {
            var base = CpuOrderingLoweringTest.context(
                    new Operation(OrderingKind.SORT, new SortAttrs(2, false)),
                    DataType.FLOAT32, Shape.of(2, 2, 3), Shape.of(2, 2, 3), false);
            var pattern = List.of(segment ? CarrierAccess.MEMORY_SEGMENT : CarrierAccess.FLOAT_ARRAY,
                    CarrierAccess.FLOAT_ARRAY);
            var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                    base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                    base.constants(), new CpuPartitionAnalysisInputs(false, pattern));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var row = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 3);
            var entry = CpuAffineCopyGeneratedKernelTest.controlledEntry(
                    plan.units().getFirst().portablePlan(), List.of(row, row));
            float[] input = {3, 1, 2, Float.NaN, Float.NaN, Float.NaN, 9, 7, 8};
            float[] output = new float[12];
            Arrays.fill(output, -99);
            try (Arena arena = Arena.ofConfined()) {
                var scratch = arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
                scratch.fill((byte) 0x5a);
                entry.invokeWithArguments(segment ? java.lang.foreign.MemorySegment.ofArray(input)
                                : input, output, scratch,
                        plan.orderingGeometry().orElseThrow().pack(new long[2], 0, 4, 0),
                        0L, 4L, CpuAffineCopyGeneratedKernelTest.control(new byte[]{1, 0}));
                assertArrayEquals(new float[]{1, 2, 3, 0, 0, 0, 7, 8, 9, 0, 0, 0}, output);
                for (int cell : new int[]{3, 4, 5, 9, 10, 11})
                    assertEquals(0, Float.floatToRawIntBits(output[cell]));
                scratch.fill((byte) 0x5a);
                entry.invokeWithArguments(segment ? java.lang.foreign.MemorySegment.ofArray(
                                new float[0]) : new float[0], output, scratch,
                        plan.orderingGeometry().orElseThrow().pack(new long[2], 0, 4, 0),
                        0L, 4L, CpuAffineCopyGeneratedKernelTest.control(new byte[]{0, 0}));
                for (byte value : scratch.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
                    assertEquals((byte) 0x5a, value);
            }
        }
    }

    @Test void controlledArgsortAndTopKZeroEveryInactiveOutputWithoutInputStorage() throws Throwable {
        for (Operation operation : List.of(
                new Operation(OrderingKind.ARGSORT, new SortAttrs(1, false)),
                new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, false)))) {
            boolean topK = operation.kind() == TopKKind.TOP_K;
            var base = CpuOrderingLoweringTest.context(operation, DataType.FLOAT32,
                    Shape.of(2, 3), Shape.of(2, topK ? 2 : 3), topK);
            var pattern = topK
                    ? List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                            CarrierAccess.LONG_ARRAY)
                    : List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY);
            var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                    base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                    base.constants(), new CpuPartitionAnalysisInputs(false, pattern));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var inputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 3);
            var outputRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(
                    0, topK ? 2 : 3);
            var entry = CpuAffineCopyGeneratedKernelTest.controlledEntry(
                    plan.units().getFirst().portablePlan(), topK
                            ? List.of(inputRow, outputRow, outputRow) : List.of(inputRow, outputRow));
            long[] indices = new long[topK ? 4 : 6];
            Arrays.fill(indices, -99);
            float[] values = {-99, -99, -99, -99};
            try (Arena arena = Arena.ofConfined()) {
                var scratch = arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
                var arguments = new ArrayList<Object>();
                arguments.add(new float[0]);
                if (topK) arguments.add(values);
                arguments.add(indices);
                arguments.add(scratch);
                arguments.add(plan.orderingGeometry().orElseThrow().pack(
                        new long[pattern.size()], 0, 2, 0));
                arguments.add(0L);
                arguments.add(2L);
                arguments.add(CpuAffineCopyGeneratedKernelTest.control(new byte[]{0, 0}));
                entry.invokeWithArguments(arguments);
                assertArrayEquals(new long[indices.length], indices);
                if (topK) for (float value : values)
                    assertEquals(0, Float.floatToRawIntBits(value));
            }
        }
    }

    @Test void preservesStableNanLastSignedZeroAndRepresentedBitsBothDirections() throws Throwable {
        double nan1 = Double.longBitsToDouble(0x7ff8000000000001L);
        double nan2 = Double.longBitsToDouble(0xfff8000000000002L);
        double[] input = {3, nan1, -0.0, +0.0, 3, nan2,
                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};
        double[] ascending = new double[8], descending = new double[8];
        invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, false)), DataType.FLOAT64,
                Shape.of(8), Shape.of(8), false, input, ascending, null);
        invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, true)), DataType.FLOAT64,
                Shape.of(8), Shape.of(8), false, input, descending, null);
        assertArrayEquals(new long[]{Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
                0x8000000000000000L, 0L, Double.doubleToRawLongBits(3),
                Double.doubleToRawLongBits(3), Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                0x7ff8000000000001L, 0xfff8000000000002L}, raw(ascending));
        assertArrayEquals(new long[]{Double.doubleToRawLongBits(Double.POSITIVE_INFINITY),
                Double.doubleToRawLongBits(3), Double.doubleToRawLongBits(3), 0L,
                0x8000000000000000L, Double.doubleToRawLongBits(Double.NEGATIVE_INFINITY),
                0x7ff8000000000001L, 0xfff8000000000002L}, raw(descending));
    }

    @Test void argsortAndSortedOrUnsortedTopKUseLogicalAxisIndices() throws Throwable {
        float[] input = {3, Float.NaN, -0.0f, +0.0f, 3};
        long[] indices = new long[5];
        invoke(new Operation(OrderingKind.ARGSORT, new SortAttrs(0, false)), DataType.FLOAT32,
                Shape.of(5), Shape.of(5), false, input, indices, null);
        assertArrayEquals(new long[]{2, 3, 0, 4, 1}, indices);
        float[] values = new float[3];
        long[] top = new long[3];
        invoke(new Operation(TopKKind.TOP_K, new TopKAttrs(0, 3, true, false)), DataType.FLOAT32,
                Shape.of(5), Shape.of(3), true, input, values, top);
        assertArrayEquals(new long[]{0, 3, 4}, top);
        assertArrayEquals(new int[]{Float.floatToRawIntBits(3), 0, Float.floatToRawIntBits(3)},
                new int[]{Float.floatToRawIntBits(values[0]), Float.floatToRawIntBits(values[1]),
                        Float.floatToRawIntBits(values[2])});
    }

    @Test void generatedMultiSliceIntegralResultMatchesIndependentReference() throws Throwable {
        var operation = new Operation(TopKKind.TOP_K, new TopKAttrs(1, 3, false, true));
        var context = CpuOrderingLoweringTest.context(operation, DataType.INT32,
                Shape.of(2, 5), Shape.of(2, 3), true);
        var lowered = new CpuPartitionLowering().lower(context);
        int[] input = {4, 1, 1, -2, 9, 7, 3, 3, 8, 0};
        int[] expectedValues = new int[6], actualValues = new int[6];
        long[] expectedIndices = new long[6], actualIndices = new long[6];
        CpuScalarReferenceKernel.execute((CpuOrderingIr) lowered.portableKernelIr(),
                lowered.orderingGeometry().orElseThrow(), List.of(argument(DataType.INT32, input, true),
                        argument(DataType.INT32, expectedValues, false),
                        argument(DataType.INT64, expectedIndices, false)));
        invoke(operation, DataType.INT32, Shape.of(2, 5), Shape.of(2, 3), true,
                input, actualValues, actualIndices);
        assertAll(() -> assertArrayEquals(expectedValues, actualValues),
                () -> assertArrayEquals(expectedIndices, actualIndices));
    }

    @Test void generatedOrderingCoversBfloatInt64AndCanonicalBoolRepresentations() throws Throwable {
        short[] bf = {(short) 0x3f80, (short) 0x8000, (short) 0x0000, (short) 0x7fc1};
        short[] bfOut = new short[4];
        invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, false)), DataType.BFLOAT16,
                Shape.of(4), Shape.of(4), false, bf, bfOut, null);
        assertArrayEquals(new short[]{(short) 0x8000, 0, (short) 0x3f80, (short) 0x7fc1}, bfOut);
        long[] longs = {Long.MAX_VALUE, -1, Long.MIN_VALUE, -1};
        long[] longOut = new long[4];
        invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, false)), DataType.INT64,
                Shape.of(4), Shape.of(4), false, longs, longOut, null);
        assertArrayEquals(new long[]{Long.MIN_VALUE, -1, -1, Long.MAX_VALUE}, longOut);
        byte[] bools = {1, 0, 1, 0};
        long[] boolIndices = new long[4];
        invoke(new Operation(OrderingKind.ARGSORT, new SortAttrs(0, true)), DataType.BOOL,
                Shape.of(4), Shape.of(4), false, bools, boolIndices, null);
        assertArrayEquals(new long[]{0, 2, 1, 3}, boolIndices);
    }

    @Test void fp8SortAndArgsortKeepFiniteExtremesZeroSignsAndStableNanBits() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            byte maximum = (byte) (type == DataType.FLOAT8_E4M3 ? 126 : 123);
            byte nan1 = (byte) (type == DataType.FLOAT8_E4M3 ? 127 : 125);
            byte nan2 = (byte) 255;
            byte[] input = {maximum, nan1, (byte) 128, 0, maximum, nan2};
            for (boolean descending : List.of(false, true)) {
                byte[] values = new byte[6];
                long[] indices = new long[6];
                invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, descending)), type,
                        Shape.of(6), Shape.of(6), false, input, values, null);
                invoke(new Operation(OrderingKind.ARGSORT, new SortAttrs(0, descending)), type,
                        Shape.of(6), Shape.of(6), false, input, indices, null);
                assertArrayEquals(descending ? new byte[] {maximum, maximum, 0, (byte) 128, nan1, nan2}
                        : new byte[] {(byte) 128, 0, maximum, maximum, nan1, nan2}, values);
                assertArrayEquals(descending ? new long[] {0, 4, 3, 2, 1, 5}
                        : new long[] {2, 3, 0, 4, 1, 5}, indices);
            }
        }
    }

    @Test void generalSegmentAndMixedCarrierFormsPreserveRepresentedOutputs() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            var input = arena.allocate(5L * Float.BYTES, 1);
            var output = arena.allocate(5L * Float.BYTES, 1);
            int[] bits = {Float.floatToRawIntBits(3), 0x80000000, 0,
                    Float.floatToRawIntBits(-2), 0x7fc01234};
            for (int i = 0; i < bits.length; i++) input.set(
                    java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, i * 4L, bits[i]);
            invokeCarriers(new Operation(OrderingKind.SORT, new SortAttrs(0, false)),
                    DataType.FLOAT32, Shape.of(5), Shape.of(5), false,
                    List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT),
                    List.of(input, output));
            assertArrayEquals(new int[]{Float.floatToRawIntBits(-2), 0x80000000, 0,
                    Float.floatToRawIntBits(3), 0x7fc01234},
                    java.util.stream.IntStream.range(0, 5).map(i -> output.get(
                            java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, i * 4L)).toArray());

            var bfInput = arena.allocate(4L * Short.BYTES, 1);
            short[] represented = {(short) 0x3f80, (short) 0x8000, 0, (short) 0x7fc1};
            for (int i = 0; i < represented.length; i++) bfInput.set(
                    java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED, i * 2L, represented[i]);
            short[] values = new short[2];
            var indices = arena.allocate(2L * Long.BYTES, 1);
            invokeCarriers(new Operation(TopKKind.TOP_K, new TopKAttrs(0, 2, false, false)),
                    DataType.BFLOAT16, Shape.of(4), Shape.of(2), true,
                    List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.SHORT_ARRAY,
                            CarrierAccess.MEMORY_SEGMENT), List.of(bfInput, values, indices));
            assertAll(() -> assertArrayEquals(new short[]{(short) 0x8000, 0}, values),
                    () -> assertArrayEquals(new long[]{1, 2}, new long[]{indices.get(
                            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, 0), indices.get(
                            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, 8)}));
        }
    }

    @Test void generalZeroStrideInputAndOffsetStridedOutputHonorPartialSliceRange() throws Throwable {
        Shape shape = Shape.of(2, 4);
        var inputLayout = io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(shape,
                new long[]{0, 0}, 0, true);
        var outputLayout = io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(shape,
                new long[]{9, 2}, 3, false);
        var operation = new Operation(OrderingKind.SORT, new SortAttrs(1, false));
        var context = customContext(operation, DataType.FLOAT32, shape, shape,
                inputLayout, outputLayout);
        var carriers = List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY);
        var selected = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(context.partition(), context.nodes(), context.values(), context.memoryRequirements(), context.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(selected).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] input = {Float.intBitsToFloat(0x7fc12345)};
        float[] output = new float[19];
        Arrays.fill(output, Float.intBitsToFloat(0x7f800001));
        try (Arena arena = Arena.ofConfined()) {
            var scratch = arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
            long[] packed = plan.orderingGeometry().orElseThrow().pack(new long[2], 1, 2, 0);
            artifact.entryPoint().invokeExact(input, output, scratch, packed, 1L, 2L);
        }
        for (int i = 0; i < output.length; i++) {
            boolean written = i == 12 || i == 14 || i == 16 || i == 18;
            assertEquals(written ? 0x7fc12345 : 0x7f800001, Float.floatToRawIntBits(output[i]));
        }
    }

    private static void invoke(Operation operation, DataType type, Shape inputShape,
            Shape outputShape, boolean topK, Object input, Object first, Object second) throws Throwable {
        var base = CpuOrderingLoweringTest.context(operation, type, inputShape, outputShape, topK);
        var carriers = new ArrayList<CarrierAccess>();
        carriers.add(heap(type));
        carriers.add(operation.kind() == OrderingKind.ARGSORT ? CarrierAccess.LONG_ARRAY : heap(type));
        if (topK) carriers.add(CarrierAccess.LONG_ARRAY);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (Arena arena = Arena.ofConfined()) {
            var scratch = arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
            long[] packed = plan.orderingGeometry().orElseThrow().pack(new long[carriers.size()],
                    0, plan.elementCount(), 0);
            var args = new ArrayList<Object>();
            args.add(input);
            args.add(first);
            if (topK) {
                args.add(second);
            }
            args.add(scratch);
            args.add(packed);
            args.add(0L);
            args.add(plan.elementCount());
            artifact.entryPoint().invokeWithArguments(args);
        }
    }


    private static void invokeCarriers(Operation operation, DataType type, Shape inputShape,
            Shape outputShape, boolean topK, List<CarrierAccess> carriers, List<Object> arguments)
            throws Throwable {
        var base = CpuOrderingLoweringTest.context(operation, type, inputShape, outputShape, topK);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (Arena scratchArena = Arena.ofConfined()) {
            var scratch = scratchArena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
            long[] packed = plan.orderingGeometry().orElseThrow().pack(new long[carriers.size()],
                    0, plan.elementCount(), 0);
            var invocation = new ArrayList<Object>(arguments);
            invocation.add(scratch);
            invocation.add(packed);
            invocation.add(0L);
            invocation.add(plan.elementCount());
            artifact.entryPoint().invokeWithArguments(invocation);
        }
    }

    private static io.github.pho001.synaptik.prepare.analysis.PrepareContext<CpuPartitionAnalysisInputs>
            customContext(Operation operation, DataType type, Shape inputShape, Shape outputShape,
                    io.github.pho001.synaptik.model.layout.LayoutDescriptor inputLayout,
                    io.github.pho001.synaptik.model.layout.LayoutDescriptor outputLayout) {
        var input = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(type, inputShape,
                Optional.of(inputLayout), false);
        var output = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(type, outputShape,
                Optional.of(outputLayout), false);
        var inputId = new io.github.pho001.synaptik.model.graph.ValueId(0);
        var outputId = new io.github.pho001.synaptik.model.graph.ValueId(1);
        var node = new io.github.pho001.synaptik.model.graph.CompiledNode(
                new io.github.pho001.synaptik.model.graph.NodeId(0), operation,
                List.of(inputId), List.of(outputId));
        var partition = new io.github.pho001.synaptik.planning.partition.PlannedPartition(
                io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        var values = List.of(new io.github.pho001.synaptik.model.graph.GraphValue(inputId, input),
                new io.github.pho001.synaptik.model.graph.GraphValue(outputId, output));
        var memory = List.of(new io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement(
                        inputId, input, Optional.empty(), List.of(partition), false),
                new io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement(outputId,
                        output, Optional.of(partition), List.of(), true));
        return new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(partition, List.of(node), values, memory, Map.of(), CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static CarrierAccess heap(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY;
            case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> CarrierAccess.BYTE_ARRAY;
        };
    }
    private static long[] raw(double[] values) { long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Double.doubleToRawLongBits(values[i]);
        }
        return result; }
    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes = Math.multiplyExact(java.lang.reflect.Array.getLength(carrier), type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16, FLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0, bytes, readOnly);
            case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> new CpuBufferArgument.Bytes((byte[]) carrier, 0, bytes, readOnly);
        };
    }
}
