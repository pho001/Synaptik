package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import java.lang.classfile.ClassFile;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.AccessFlag;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class CpuClassFileKernelGeneratorTest {
    private static final CpuLoweringFingerprint LOWERING = CpuLoweringFingerprint.of(
            new byte[] {0x50, 0x52, 0x4f, 0x42, 0x45});

    @Test void generatesDeterministicDistinctHiddenArtifactsAndInvokesHeapInAllFourModes() throws Throwable {
        for (CpuPortableExecutionMode mode : CpuPortableExecutionMode.values()) {
            CpuKernelSpecialization specialization = heapSpecialization(mode, false);
            var generator = new CpuClassFileKernelGenerator();
            CpuGeneratedKernel first = generator.generate(specialization, new CopyEmitter());
            CpuGeneratedKernel second = generator.generate(specialization, new CopyEmitter());
            assertArrayEquals(first.classBytes(), second.classBytes());
            assertNotSame(first, second); assertNotSame(first.hiddenClass(), second.hiddenClass());
            assertTrue(first.hiddenClass().isHidden()); assertSame(first.hiddenClass(), first.hiddenLookup().lookupClass());
            assertEquals(specialization.entryType(), first.entryPoint().type());
            float[] source = new float[20]; float[] target = new float[20];
            for (int i = 0; i < 16; i++) source[i + 1] = i + 0.25F;
            invokeHeap(first.entryPoint(), mode, source, target);
            assertArrayEquals(Arrays.copyOfRange(source, 1, 17), Arrays.copyOfRange(target, 2, 18));
        }
    }

    @Test void invokesExactSegmentAndMixedSignaturesWithoutCopying() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            var segment = arena.allocate(8L * Float.BYTES, Float.BYTES);
            for (int i = 0; i < 8; i++) segment.set(ValueLayout.JAVA_FLOAT, i * 4L, i + 1F);
            float[] target = new float[8];
            var specialization = new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                    LOWERING, CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(
                            CpuKernelSpecializationTest.argument(DataType.FLOAT32,
                                    CpuKernelSpecialization.Carrier.MEMORY_SEGMENT, BufferAccess.READ_ONLY),
                            CpuKernelSpecializationTest.argument(DataType.FLOAT32,
                                    CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.WRITE_ONLY)),
                    List.of(8L), 0, null, java.nio.ByteOrder.nativeOrder(), 1, 4,
                    CpuKernelSpecialization.Tail.NONE,
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
            MethodHandle entry = new CpuClassFileKernelGenerator().generate(
                    specialization, new CopyEmitter()).entryPoint();
            entry.invokeExact(segment, target, 8L);
            assertArrayEquals(new float[] {1,2,3,4,5,6,7,8}, target);
        }
    }

    @Test void invokesVectorSegmentAndHeapSignatureWithoutCopying() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            var segment = arena.allocate(8L * Float.BYTES, Float.BYTES);
            for (int i = 0; i < 8; i++) segment.set(ValueLayout.JAVA_FLOAT, i * 4L, i + 1F);
            float[] target = new float[8];
            var specialization = new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                    LOWERING, CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD, List.of(
                            CpuKernelSpecializationTest.argument(DataType.FLOAT32,
                                    CpuKernelSpecialization.Carrier.MEMORY_SEGMENT, BufferAccess.READ_ONLY),
                            CpuKernelSpecializationTest.argument(DataType.FLOAT32,
                                    CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.WRITE_ONLY)),
                    List.of(8L), 0, new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4),
                    java.nio.ByteOrder.nativeOrder(), 1, 4, CpuKernelSpecialization.Tail.NONE,
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
            MethodHandle entry = new CpuClassFileKernelGenerator().generate(
                    specialization, new CopyEmitter()).entryPoint();
            entry.invokeExact(segment, target, 8L);
            assertArrayEquals(new float[] {1,2,3,4,5,6,7,8}, target);
        }
    }

    @Test void scalarCarrierPlumbingExecutesAllSixPrimitiveHeapForms() throws Throwable {
        copyDoubles(new double[] {1,2}, new double[2]);
        copyFloats(new float[] {1,2}, new float[2]);
        copyShorts(new short[] {1,2}, new short[2]);
        copyInts(new int[] {1,2}, new int[2]);
        copyLongs(new long[] {1,2}, new long[2]);
        copyBytes(new byte[] {1,2}, new byte[2]);
    }

    @Test void tailAndReductionCallbacksAreStructuralAndAccessFailuresPropagateExactly() {
        var scalarTailCalls = new AtomicInteger(); var maskedTailCalls = new AtomicInteger();
        var partialCalls = new AtomicInteger(); var combineCalls = new AtomicInteger();
        var specialization = new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                LOWERING, CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD, List.of(), List.of(), 0,
                new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4),
                java.nio.ByteOrder.nativeOrder(), 1, 4, CpuKernelSpecialization.Tail.MASKED,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
        new CpuClassFileKernelGenerator().generate(specialization, new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { fail("scalar callback"); }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                loops.emitTail(index -> scalarTailCalls.incrementAndGet(),
                        index -> maskedTailCalls.incrementAndGet());
                reductions.emitPartials(DataType.FLOAT32, (index, partial) -> partialCalls.incrementAndGet());
                reductions.emitCombine(DataType.FLOAT32,
                        (left, right, result) -> combineCalls.incrementAndGet());
                vector.code().return_();
            }
        });
        assertEquals(0, scalarTailCalls.get()); assertEquals(1, maskedTailCalls.get());
        assertEquals(1, partialCalls.get()); assertEquals(1, combineCalls.get());

        var selectedScalarTailCalls = new AtomicInteger();
        var rejectedMaskedTailCalls = new AtomicInteger();
        var scalarTail = new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                LOWERING, CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD, List.of(), List.of(), 0,
                new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4),
                java.nio.ByteOrder.nativeOrder(), 1, 4, CpuKernelSpecialization.Tail.SCALAR,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
        new CpuClassFileKernelGenerator().generate(scalarTail, new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { fail("scalar callback"); }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                loops.emitTail(index -> selectedScalarTailCalls.incrementAndGet(),
                        index -> rejectedMaskedTailCalls.incrementAndGet());
                vector.code().return_();
            }
        });
        assertEquals(1, selectedScalarTailCalls.get());
        assertEquals(0, rejectedMaskedTailCalls.get());

        var readOnly = scalarSpecialization(DataType.FLOAT32,
                CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.READ_ONLY);
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () ->
                new CpuClassFileKernelGenerator().generate(readOnly, new CpuFamilyKernelEmitter() {
                    @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
                    @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                            CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                        scalar.code().loadConstant(1F); carriers.emitScalarStore(0, 0);
                    }
                    @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                            CpuLoopEmitter loops, CpuReductionEmitter reductions) {}
                }));
        assertEquals("read-only argument cannot be stored", failure.getMessage());
    }

    @Test void usesWorkerGroupOnlyOutsideGeneratedParallelEntry() {
        var specialization = heapSpecialization(CpuPortableExecutionMode.SCALAR_PARALLEL, true);
        MethodHandle entry = new CpuClassFileKernelGenerator().generate(
                specialization, new CopyEmitter()).entryPoint();
        float[] source = new float[16]; float[] target = new float[16];
        for (int i = 0; i < source.length; i++) source[i] = i;
        try (var workers = new CpuWorkerGroup(2)) {
            workers.execute(0, 16, 8, true, (start, end, range) -> {
                try { entry.invokeExact(source, target, start, end, range); }
                catch (Throwable failure) { throw new AssertionError(failure); }
            });
        }
        assertArrayEquals(source, target);
    }

    @Test void rejectsMismatchedFamilyBeforeEmission() {
        var specialization = heapSpecialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, true);
        CpuFamilyKernelEmitter mismatch = new CopyEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() {
                return CpuLoweringFingerprint.of(new byte[] {9});
            }
        };
        assertEquals("familyEmitter lowering fingerprint does not match specialization",
                assertThrows(IllegalArgumentException.class, () ->
                        new CpuClassFileKernelGenerator().generate(specialization, mismatch)).getMessage());
    }

    @Test void separatesDeterministicEmissionFromStoredByteDefinitionAndRejectsCorruption() {
        var specialization = heapSpecialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, true);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(specialization, new CopyEmitter());
        CpuGeneratedKernel loaded = generator.defineClassBytes(specialization, bytes);
        assertArrayEquals(bytes, loaded.classBytes());
        assertEquals(specialization.entryType(), loaded.entryPoint().type());
        assertNotSame(loaded, generator.defineClassBytes(specialization, bytes));

        byte[] corrupt = bytes.clone();
        corrupt[0] ^= 1;
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> generator.defineClassBytes(specialization, corrupt)).getMessage()
                .startsWith("generated class verification failed"));
        assertAll(
                () -> assertEquals("specialization", assertThrows(NullPointerException.class,
                        () -> generator.defineClassBytes(null, bytes)).getMessage()),
                () -> assertEquals("classBytes", assertThrows(NullPointerException.class,
                        () -> generator.defineClassBytes(specialization, null)).getMessage()));
    }

    @Test void storedDefinitionRejectsEveryExactClassShapeDifference() {
        var specialization = heapSpecialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, true);
        var generator = new CpuClassFileKernelGenerator();
        String expectedName = CpuGeneratorSchema.generatedBinaryName(specialization);
        MethodTypeDesc expectedType = MethodTypeDesc.ofDescriptor(
                specialization.entryType().descriptorString());

        List<byte[]> wrongShapes = List.of(
                classBytes(expectedName, ClassFile.JAVA_25_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 1, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName + "Wrong", ClassFile.JAVA_26_VERSION, 0,
                        AccessFlag.FINAL, ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Number"), List.of(), false,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.PUBLIC,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(ClassDesc.of("java.lang.Runnable")),
                        false, "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), true,
                        "invoke", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "wrong", expectedType, AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", MethodTypeDesc.ofDescriptor("()V"), AccessFlag.STATIC),
                classBytes(expectedName, ClassFile.JAVA_26_VERSION, 0, AccessFlag.FINAL,
                        ClassDesc.of("java.lang.Object"), List.of(), false,
                        "invoke", expectedType),
                classWithTwoMethods(expectedName, expectedType),
                classWithoutCode(expectedName, expectedType));

        for (byte[] wrongShape : wrongShapes) {
            assertThrows(IllegalArgumentException.class,
                    () -> generator.defineClassBytes(specialization, wrongShape));
        }
    }

    private static byte[] classBytes(String name, int major, int minor, AccessFlag classFlag,
            ClassDesc superclass, List<ClassDesc> interfaces, boolean field, String methodName,
            MethodTypeDesc methodType, AccessFlag... methodFlags) {
        return ClassFile.of().build(ClassDesc.of(name), classBuilder -> {
            classBuilder.withVersion(major, minor).withFlags(classFlag)
                    .withSuperclass(superclass).withInterfaceSymbols(interfaces);
            if (field) classBuilder.withField("state", ClassDesc.ofDescriptor("I"),
                    AccessFlag.STATIC.mask());
            classBuilder.withMethodBody(methodName, methodType,
                    java.util.Arrays.stream(methodFlags).mapToInt(AccessFlag::mask)
                            .reduce(0, (left, right) -> left | right),
                    code -> code.return_());
        });
    }

    private static byte[] classWithTwoMethods(String name, MethodTypeDesc type) {
        return ClassFile.of().build(ClassDesc.of(name), classBuilder -> classBuilder
                .withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL)
                .withMethodBody("invoke", type, AccessFlag.STATIC.mask(), code -> code.return_())
                .withMethodBody("second", MethodTypeDesc.ofDescriptor("()V"),
                        AccessFlag.STATIC.mask(), code -> code.return_()));
    }

    private static byte[] classWithoutCode(String name, MethodTypeDesc type) {
        return ClassFile.of().build(ClassDesc.of(name), classBuilder -> classBuilder
                .withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL)
                .withMethod("invoke", type,
                        AccessFlag.STATIC.mask() | AccessFlag.NATIVE.mask(), method -> {}));
    }

    @Test void validatesGeneratorInputsAndPreservesUncheckedEmitterFailure() {
        var generator = new CpuClassFileKernelGenerator();
        var specialization = heapSpecialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, true);
        assertAll(
                () -> assertEquals("specialization", assertThrows(NullPointerException.class,
                        () -> generator.generate(null, new CopyEmitter())).getMessage()),
                () -> assertEquals("familyEmitter", assertThrows(NullPointerException.class,
                        () -> generator.generate(specialization, null)).getMessage()));
        var expected = new IllegalStateException("emission failure");
        CpuFamilyKernelEmitter failing = new CopyEmitter() {
            @Override public void emitScalar(CpuScalarEmitter scalar,
                    CpuCarrierEmitter carriers, CpuLoopEmitter loops,
                    CpuReductionEmitter reductions) {
                throw expected;
            }
        };
        assertSame(expected, assertThrows(IllegalStateException.class,
                () -> generator.generate(specialization, failing)));
    }

    @Test void executionModeDispatchesExactlyOneMatchingFamilyCallback() {
        for (CpuPortableExecutionMode mode : CpuPortableExecutionMode.values()) {
            var scalarCalls = new AtomicInteger();
            var vectorCalls = new AtomicInteger();
            var specialization = heapSpecialization(mode, true);
            new CpuClassFileKernelGenerator().generate(specialization, new CpuFamilyKernelEmitter() {
                @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
                @Override public void emitScalar(CpuScalarEmitter scalar,
                        CpuCarrierEmitter carriers, CpuLoopEmitter loops,
                        CpuReductionEmitter reductions) {
                    scalarCalls.incrementAndGet();
                    assertSame(specialization, scalar.specialization());
                    assertSame(specialization, carriers.specialization());
                    assertSame(specialization, loops.specialization());
                    assertSame(specialization, reductions.specialization());
                    scalar.code().return_();
                }
                @Override public void emitVector(CpuVectorEmitter vector,
                        CpuCarrierEmitter carriers, CpuLoopEmitter loops,
                        CpuReductionEmitter reductions) {
                    vectorCalls.incrementAndGet();
                    assertSame(specialization, vector.specialization());
                    assertSame(specialization, carriers.specialization());
                    assertSame(specialization, loops.specialization());
                    assertSame(specialization, reductions.specialization());
                    vector.code().return_();
                }
            });
            assertEquals(mode.vectorized() ? 0 : 1, scalarCalls.get());
            assertEquals(mode.vectorized() ? 1 : 0, vectorCalls.get());
        }
    }

    @Test void dynamicPrimitiveArrayOffsetsRemainUncheckedInvocationArguments() {
        var specialization = heapSpecialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, false);
        CpuGeneratedKernel kernel = assertDoesNotThrow(() ->
                new CpuClassFileKernelGenerator().generate(specialization, new CopyEmitter()));
        assertEquals(long.class, specialization.entryType().parameterType(1));
        assertEquals(long.class, specialization.entryType().parameterType(3));
        assertTrue(ClassFile.of().verify(kernel.classBytes()).isEmpty());
    }

    private static void invokeHeap(MethodHandle entry, CpuPortableExecutionMode mode,
            float[] source, float[] target) throws Throwable {
        if (mode.parallel()) {
            try (var workers = new CpuWorkerGroup(2)) {
                workers.execute(0, 16, 8, true, (start, end, range) -> {
                    try { entry.invokeExact(source, 4L, target, 8L, start, end, range); }
                    catch (Throwable failure) { throw new AssertionError(failure); }
                });
            }
        } else entry.invokeExact(source, 4L, target, 8L, 16L);
    }

    private static CpuKernelSpecialization heapSpecialization(CpuPortableExecutionMode mode,
            boolean bakedOffsets) {
        var vector = mode.vectorized()
                ? new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4) : null;
        var args = List.of(argument(BufferAccess.READ_ONLY, bakedOffsets),
                argument(BufferAccess.WRITE_ONLY, bakedOffsets));
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION, LOWERING, mode, args,
                List.of(16L), 0, vector, java.nio.ByteOrder.nativeOrder(), 1, 4,
                CpuKernelSpecialization.Tail.NONE,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    private static CpuKernelSpecialization.Argument argument(BufferAccess access, boolean baked) {
        return new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                CpuKernelSpecialization.Carrier.FLOAT_ARRAY, access, baked, 0, List.of(1L));
    }

    private static CpuKernelSpecialization scalarSpecialization(DataType type,
            CpuKernelSpecialization.Carrier carrier, BufferAccess access) {
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION, LOWERING,
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                access == BufferAccess.READ_ONLY
                        ? List.of(CpuKernelSpecializationTest.argument(type, carrier, access))
                        : List.of(CpuKernelSpecializationTest.argument(type, carrier, BufferAccess.READ_ONLY),
                                CpuKernelSpecializationTest.argument(type, carrier, BufferAccess.WRITE_ONLY)),
                List.of(2L), 0, null, java.nio.ByteOrder.nativeOrder(), 1, 1,
                CpuKernelSpecialization.Tail.NONE,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    private static MethodHandle scalarCopy(DataType type, CpuKernelSpecialization.Carrier carrier) {
        return new CpuClassFileKernelGenerator().generate(
                scalarSpecialization(type, carrier, BufferAccess.WRITE_ONLY), new CopyEmitter()).entryPoint();
    }
    private static void copyDoubles(double[] source, double[] target) throws Throwable {
        scalarCopy(DataType.FLOAT64, CpuKernelSpecialization.Carrier.DOUBLE_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }
    private static void copyFloats(float[] source, float[] target) throws Throwable {
        scalarCopy(DataType.FLOAT32, CpuKernelSpecialization.Carrier.FLOAT_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }
    private static void copyShorts(short[] source, short[] target) throws Throwable {
        scalarCopy(DataType.BFLOAT16, CpuKernelSpecialization.Carrier.SHORT_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }
    private static void copyInts(int[] source, int[] target) throws Throwable {
        scalarCopy(DataType.INT32, CpuKernelSpecialization.Carrier.INT_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }
    private static void copyLongs(long[] source, long[] target) throws Throwable {
        scalarCopy(DataType.INT64, CpuKernelSpecialization.Carrier.LONG_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }
    private static void copyBytes(byte[] source, byte[] target) throws Throwable {
        scalarCopy(DataType.BOOL, CpuKernelSpecialization.Carrier.BYTE_ARRAY)
                .invokeExact(source, target, 2L); assertArrayEquals(source, target); }

    private static class CopyEmitter implements CpuFamilyKernelEmitter {
        @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
        @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                CpuLoopEmitter loops, CpuReductionEmitter reductions) {
            loops.emitRange(index -> { carriers.emitScalarLoad(0, index); carriers.emitScalarStore(1, index); });
            scalar.code().return_();
        }
        @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                CpuLoopEmitter loops, CpuReductionEmitter reductions) {
            loops.emitTiles((start, end, tile) -> {
                carriers.emitVectorLoad(0, start); carriers.emitVectorStore(1, start);
            });
            vector.code().return_();
        }
    }
}
