package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
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
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.AccessFlag;
import org.junit.jupiter.api.Test;

class CpuAffineCopyGeneratedKernelTest {
    @Test void provedDenseSegmentResultLoadsInitialAddressOnceAndAdvancesForSubranges()
            throws Throwable {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 1,
                List.of(CpuAccessPlan.AxisRole.STRIDED), 0);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var affine = new CpuAffineCopyIr(DataType.FLOAT64, read, write,
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.CONTIGUOUS,
                        1, 1, List.of())), CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(affine.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT64, DataType.FLOAT64),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.MEMORY_SEGMENT),
                0, -1, List.of(), false);
        var generator = new CpuClassFileKernelGenerator();
        assertTrue(CpuAffineCopyEmitter.ownsGeneralLongDenseResultCarrierAccess(
                specialization, affine.encodedKernelIr()));
        byte[] bytes = generator.generateClassBytes(specialization, affine.encodedKernelIr());
        var artifact = generator.defineClassBytes(specialization, bytes);
        double[] source = {10, 20, 30, 40, 50, 60};
        long[] geometry = {5, 4, 4, 5, 3, 6, 2, 700, 1, 800, 0, 900};
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            MemorySegment result = arena.allocate(12 * 8L, 8);
            artifact.entryPoint().invokeExact(source, result, geometry, 2L, 5L);
            assertAll(
                    () -> assertEquals(40, result.get(ValueLayout.JAVA_DOUBLE, 6 * 8L)),
                    () -> assertEquals(30, result.get(ValueLayout.JAVA_DOUBLE, 7 * 8L)),
                    () -> assertEquals(20, result.get(ValueLayout.JAVA_DOUBLE, 8 * 8L)),
                    () -> assertEquals(0, result.get(ValueLayout.JAVA_DOUBLE, 9 * 8L)));
            assertDoesNotThrow(() -> artifact.entryPoint().invokeWithArguments(
                    source, result, geometry, 6L, 6L));
        }
        var code = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
        String constants = new String(bytes, StandardCharsets.ISO_8859_1);
        assertAll(
                () -> assertEquals(2, opcodeCount(code, Opcode.LALOAD)),
                () -> assertEquals(1, opcodeCount(code, Opcode.IFGE)),
                () -> assertEquals(1, opcodeCount(code, Opcode.IFLT)),
                () -> assertEquals(0, opcodeCount(code, Opcode.GOTO)),
                () -> assertEquals(0, opcodeCount(code, Opcode.ASTORE)),
                () -> assertEquals(0, opcodeCount(code, Opcode.NEW)),
                () -> assertEquals(0, opcodeCount(code, Opcode.ANEWARRAY)),
                () -> assertEquals(0, opcodeCount(code, Opcode.NEWARRAY)),
                () -> assertFalse(constants.contains("nativeOrder")),
                () -> assertFalse(constants.contains("withOrder")));
    }

    @Test void guardedFrozenBfloat16PermuteSlicePreservesRangesAndRetainsFallback()
            throws Throwable {
        var analysis = new CpuPartitionPreparer().analyze(
                CpuAffineLayoutLoweringTest.frozenBfloat16PermuteSlice());
        var route = analysis.plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertFalse(CpuAffineCopyEmitter.ownsGeneralLongDenseResultCarrierAccess(
                route.specialization(), route.kernelIr()));
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        long[] geometry = analysis.plan().affineAddressPairs();
        int count = 256 * 32 * 32;
        short canary = (short) 0x55aa;
        short[] input = new short[count * 2 + 16];
        for (int index = 0; index < count; index++)
            input[5 + 2 * index] = (short) (0x7f00 ^ index);
        short[] immutableInput = input.clone();
        short[] output = new short[input.length];
        Arrays.fill(output, canary);
        for (long[] range : List.of(new long[]{0, 1}, new long[]{30, 34},
                new long[]{8_190, 8_194}, new long[]{count - 2L, count}))
            artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(input), output,
                    geometry, range[0], range[1]);
        assertFrozenRanges(output, immutableInput, canary,
                List.of(new long[]{0, 1}, new long[]{30, 34},
                        new long[]{8_190, 8_194}, new long[]{count - 2L, count}));

        long[] fallbackGeometry = geometry.clone();
        fallbackGeometry[62]++;
        short[] fallback = new short[input.length];
        Arrays.fill(fallback, canary);
        artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(input), fallback,
                fallbackGeometry, 8_192L, 8_194L);
        assertFrozenRanges(fallback, immutableInput, canary,
                List.of(new long[]{8_192, 8_194}));
        assertArrayEquals(immutableInput, input);

        var model = ClassFile.of().parse(bytes);
        var method = model.methods().getFirst();
        var code = method.code().orElseThrow();
        var invokes = code.elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        var constantPool = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).toList();
        assertAll(
                () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertTrue(method.flags().has(AccessFlag.STATIC)),
                () -> assertEquals("(Ljava/lang/foreign/MemorySegment;[S[JJJ)V",
                        method.methodTypeSymbol().descriptorString()),
                () -> assertEquals(0, opcodeCount(code, Opcode.IDIV)),
                () -> assertEquals(0, opcodeCount(code, Opcode.IREM)),
                () -> assertEquals(0, opcodeCount(code, Opcode.LDIV)),
                () -> assertEquals(0, opcodeCount(code, Opcode.LREM)),
                () -> assertEquals(0, opcodeCount(code, Opcode.NEW)),
                () -> assertEquals(0, opcodeCount(code, Opcode.ANEWARRAY)),
                () -> assertEquals(0, opcodeCount(code, Opcode.NEWARRAY)),
                () -> assertEquals(0, opcodeCount(code, Opcode.MULTIANEWARRAY)),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(constantPool.stream()
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(constantPool.stream()
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(constantPool.stream().filter(MemberRefEntry.class::isInstance)
                        .map(MemberRefEntry.class::cast).noneMatch(member -> member.owner()
                                .asInternalName().startsWith(
                                        "io/github/pho001/synaptik"))),
                () -> assertTrue(opcodeCount(code, Opcode.LALOAD) >= 14),
                () -> assertEquals(2, invokes.stream().filter(invoke -> invoke.owner()
                        .asInternalName().equals("java/lang/foreign/MemorySegment")
                        && invoke.name().stringValue().equals("get")).count()));
    }

    @Test void denseArrayBodyNarrowsOnceAndGeneralBodyRetainsLongAddressPairs() {
        var denseAnalysis = new CpuPartitionPreparer().analyze(CpuAffineLayoutLoweringTest.contiguous(
                DataType.FLOAT64, List.of(CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY), 17));
        var generalAnalysis = new CpuPartitionPreparer().analyze(CpuAffineLayoutLoweringTest.select(
                DataType.FLOAT64, List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT)));
        var generator = new CpuClassFileKernelGenerator();
        var denseRoute = denseAnalysis.plan().units().getFirst().portablePlan();
        var generalRoute = generalAnalysis.plan().units().getFirst().portablePlan();
        assertAll(
                () -> assertFalse(CpuAffineCopyEmitter
                        .ownsGeneralLongDenseResultCarrierAccess(denseRoute.specialization(),
                                denseRoute.kernelIr())),
                () -> assertFalse(CpuAffineCopyEmitter
                        .ownsGeneralLongDenseResultCarrierAccess(generalRoute.specialization(),
                                generalRoute.kernelIr())));
        var dense = ClassFile.of().parse(generator.generateClassBytes(
                denseRoute.specialization(), denseRoute.kernelIr()))
                .methods().getFirst().code().orElseThrow();
        var general = ClassFile.of().parse(generator.generateClassBytes(
                generalRoute.specialization(), generalRoute.kernelIr()))
                .methods().getFirst().code().orElseThrow();
        assertAll(
                () -> assertEquals(4, opcodeCount(dense, Opcode.L2I)),
                () -> assertTrue(opcodeCount(dense, Opcode.IINC) >= 3),
                () -> assertEquals(2, opcodeCount(general, Opcode.L2I)),
                () -> assertTrue(opcodeCount(general, Opcode.LMUL) >= 1));
    }

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

    private static void assertFrozenRanges(short[] actual, short[] source, short canary,
            List<long[]> ranges) {
        int count = 256 * 32 * 32;
        for (int ordinal = 0; ordinal < count; ordinal++) {
            boolean selected = false;
            for (long[] range : ranges)
                if (ordinal >= range[0] && ordinal < range[1]) selected = true;
            int x = ordinal >>> 13;
            int y = (ordinal >>> 5) & 255;
            int z = ordinal & 31;
            int address = 5 + x * 64 + y * 2_048 + z * 2;
            assertEquals(selected ? source[address] : canary, actual[address],
                    "ordinal " + ordinal + " address " + address);
        }
    }
    private static long opcodeCount(java.lang.classfile.CodeModel code,
            Opcode opcode) {
        return code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(instruction -> instruction.opcode() == opcode)
                .count();
    }
}
