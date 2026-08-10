package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.*;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class CpuAffineCopyGeneratedKernelTest {
    @Test void preservesEveryRepresentedTypeAcrossEveryArraySegmentPair() {
        for (DataType type : DataType.values()) for (boolean sourceHeap : List.of(false, true))
                for (boolean resultHeap : List.of(false, true)) {
            Object source = source(type);
            Object result = array(type, 8);
            List<CarrierAccess> carriers = List.of(sourceHeap ? CarrierAccess.SHORT_ARRAY
                            : CarrierAccess.MEMORY_SEGMENT,
                    resultHeap ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT);
            if (sourceHeap) carriers = List.of(arrayCarrier(type), carriers.get(1));
            var analysis = new CpuPartitionPreparer().analyze(
                    CpuAffineLayoutLoweringTest.select(type, carriers));
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
            CpuBufferRepresentation input = sourceHeap ? borrowed(type, source)
                    : CpuNativeBuffer.allocate(type, Math.multiplyExact(9, type.byteWidth()),
                            type.byteWidth());
            CpuBufferRepresentation output = resultHeap ? borrowed(type, result)
                    : CpuNativeBuffer.allocate(type, Math.multiplyExact(8, type.byteWidth()),
                            type.byteWidth());
            if (!sourceHeap) MemorySegment.copy(segment(source), 0,
                    ((CpuNativeBuffer) input).segment(), 0, Math.multiplyExact(9, type.byteWidth()));
            var state = new RunState(executable.memoryPlan(), List.of(
                    List.of(new BufferRepresentationBinding(input, sourceHeap
                            ? RunResourceOwnership.BORROWED : RunResourceOwnership.RUN_OWNED)),
                    List.of(new BufferRepresentationBinding(output, resultHeap
                            ? RunResourceOwnership.BORROWED : RunResourceOwnership.RUN_OWNED))), List.of());
            try {
                assertDoesNotThrow(() -> executable.bind(state).execute());
                if (!resultHeap) MemorySegment.copy(((CpuNativeBuffer) output).segment(), 0,
                        segment(result), 0, Math.multiplyExact(8, type.byteWidth()));
                Object finalResult = result;
                assertAll(type + " " + sourceHeap + "->" + resultHeap,
                        () -> assertRepresentedEquals(type, source, 1, finalResult, 1),
                        () -> assertRepresentedEquals(type, source, 4, finalResult, 4),
                        () -> assertRepresentedEquals(type, source, 7, finalResult, 7));
            } finally { state.close(); }
        }
    }

    @Test void generatedEntryHonorsArbitraryRangesAndContainsNoSemanticHotDependencies()
            throws Throwable {
        var analysis = new CpuPartitionPreparer().analyze(CpuAffineLayoutLoweringTest.select(
                DataType.INT32, List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY)));
        var route = analysis.plan().units().getFirst().portablePlan();
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(),
                new CpuClassFileKernelGenerator().generateClassBytes(
                        route.specialization(), route.kernelIr()));
        int[] source = {0,11,2,3,44,5,6,77,8};
        int[] result = new int[8];
        artifact.entryPoint().invokeWithArguments(source, result,
                analysis.plan().affineAddressPairs(), 1L, 2L);
        String constants = new String(artifact.classBytes(), StandardCharsets.ISO_8859_1);
        assertAll(
                () -> assertEquals(0, result[1]),
                () -> assertEquals(44, result[4]),
                () -> assertEquals(0, result[7]),
                () -> assertFalse(constants.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(constants.contains("java/lang/reflect")),
                () -> assertFalse(constants.contains("java/util/Map")));
    }

    @Test void preparedExecutableRetainsAffineGeometryForArbitraryRanges() {
        var analysis = new CpuPartitionPreparer().analyze(CpuAffineLayoutLoweringTest.select(
                DataType.INT32, List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty())
                .forRange(1, 2);
        int[] source = {0,11,2,3,44,5,6,77,8};
        int[] result = new int[8];
        var state = new RunState(executable.memoryPlan(), List.of(
                List.of(new BufferRepresentationBinding(borrowed(DataType.INT32, source),
                        RunResourceOwnership.BORROWED)),
                List.of(new BufferRepresentationBinding(borrowed(DataType.INT32, result),
                        RunResourceOwnership.BORROWED))), List.of());
        try {
            executable.bind(state).execute();
            assertArrayEquals(new int[]{0, 0, 0, 0, 44, 0, 0, 0}, result);
        } finally {
            state.close();
        }
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
    }
    private static Object source(DataType type) {
        return switch (type) {
            case FLOAT64 -> new double[]{1, Double.longBitsToDouble(0x7ff8000000000042L), 2,
                    3, Double.NEGATIVE_INFINITY, 4, 5, -0.0, 6};
            case FLOAT32 -> new float[]{1, Float.intBitsToFloat(0x7fc00042), 2,
                    3, Float.NEGATIVE_INFINITY, 4, 5, -0.0f, 6};
            case BFLOAT16 -> new short[]{1, (short)0x7fc1, 2, 3, (short)0xff80, 4,
                    5, (short)0x8000, (short)0xffff};
            case INT32 -> new int[]{0, Integer.MIN_VALUE, -1, 1, Integer.MAX_VALUE, 7, 8, -9, 10};
            case INT64 -> new long[]{0, Long.MIN_VALUE, -1, 1, Long.MAX_VALUE, 7, 8, -9, 10};
            case BOOL -> new byte[]{0, 1, 0, 1, 0, 1, 0, 1, 0};
        };
    }
    private static Object array(DataType type, int size) {
        return switch (type) {
            case FLOAT64 -> new double[size]; case FLOAT32 -> new float[size];
            case BFLOAT16 -> new short[size]; case INT32 -> new int[size];
            case INT64 -> new long[size]; case BOOL -> new byte[size];
        };
    }
    private static MemorySegment segment(Object array) {
        if (array instanceof double[] value) return MemorySegment.ofArray(value);
        if (array instanceof float[] value) return MemorySegment.ofArray(value);
        if (array instanceof short[] value) return MemorySegment.ofArray(value);
        if (array instanceof int[] value) return MemorySegment.ofArray(value);
        if (array instanceof long[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((byte[]) array);
    }
    private static CpuBorrowedBuffer borrowed(DataType type, Object values) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type,
                java.lang.reflect.Array.getLength(values), segment(values)));
    }
    private static void assertRepresentedEquals(DataType type, Object expected, int expectedIndex,
            Object actual, int actualIndex) {
        switch (type) {
            case FLOAT64 -> assertEquals(Double.doubleToRawLongBits(((double[]) expected)[expectedIndex]),
                    Double.doubleToRawLongBits(((double[]) actual)[actualIndex]));
            case FLOAT32 -> assertEquals(Float.floatToRawIntBits(((float[]) expected)[expectedIndex]),
                    Float.floatToRawIntBits(((float[]) actual)[actualIndex]));
            case BFLOAT16 -> assertEquals(((short[]) expected)[expectedIndex], ((short[]) actual)[actualIndex]);
            case INT32 -> assertEquals(((int[]) expected)[expectedIndex], ((int[]) actual)[actualIndex]);
            case INT64 -> assertEquals(((long[]) expected)[expectedIndex], ((long[]) actual)[actualIndex]);
            case BOOL -> assertEquals(((byte[]) expected)[expectedIndex], ((byte[]) actual)[actualIndex]);
        }
    }
}
