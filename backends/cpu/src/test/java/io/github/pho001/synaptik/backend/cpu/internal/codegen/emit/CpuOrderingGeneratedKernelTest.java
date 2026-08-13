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
        float[] values = new float[3]; long[] top = new long[3];
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
        long[] longs = {Long.MAX_VALUE, -1, Long.MIN_VALUE, -1}; long[] longOut = new long[4];
        invoke(new Operation(OrderingKind.SORT, new SortAttrs(0, false)), DataType.INT64,
                Shape.of(4), Shape.of(4), false, longs, longOut, null);
        assertArrayEquals(new long[]{Long.MIN_VALUE, -1, -1, Long.MAX_VALUE}, longOut);
        byte[] bools = {1, 0, 1, 0}; long[] boolIndices = new long[4];
        invoke(new Operation(OrderingKind.ARGSORT, new SortAttrs(0, true)), DataType.BOOL,
                Shape.of(4), Shape.of(4), false, bools, boolIndices, null);
        assertArrayEquals(new long[]{0, 2, 1, 3}, boolIndices);
    }

    private static void invoke(Operation operation, DataType type, Shape inputShape,
            Shape outputShape, boolean topK, Object input, Object first, Object second) throws Throwable {
        var base = CpuOrderingLoweringTest.context(operation, type, inputShape, outputShape, topK);
        var carriers = new ArrayList<CarrierAccess>(); carriers.add(heap(type));
        carriers.add(operation.kind() == OrderingKind.ARGSORT ? CarrierAccess.LONG_ARRAY : heap(type));
        if (topK) carriers.add(CarrierAccess.LONG_ARRAY);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (Arena arena = Arena.ofConfined()) {
            var scratch = arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8);
            long[] packed = plan.orderingGeometry().orElseThrow().pack(new long[carriers.size()],
                    0, plan.elementCount(), 0);
            var args = new ArrayList<Object>(); args.add(input); args.add(first);
            if (topK) args.add(second); args.add(scratch); args.add(packed);
            args.add(0L); args.add(plan.elementCount());
            artifact.entryPoint().invokeWithArguments(args);
        }
    }

    private static CarrierAccess heap(DataType type) { return switch (type) {
        case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
        case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
        case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY;
    }; }
    private static long[] raw(double[] values) { long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) result[i] = Double.doubleToRawLongBits(values[i]);
        return result; }
    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes = Math.multiplyExact(java.lang.reflect.Array.getLength(carrier), type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0, bytes, readOnly);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) carrier, 0, bytes, readOnly);
        };
    }
}
