package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import java.lang.classfile.ClassFile;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodType;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuKernelSpecializationTest {
    @Test void loweringFingerprintIsImmutableContentIdentityWithFixedVector() {
        byte[] source = "probe-v1".getBytes(StandardCharsets.UTF_8);
        CpuLoweringFingerprint fingerprint = CpuLoweringFingerprint.of(source);
        source[0] = 0;
        assertEquals("817e02e9fe9fd870b220fa35931d3d3b08d5c71e4efb670cc158fd91bad2b307",
                fingerprint.toString());
        assertEquals(fingerprint, CpuLoweringFingerprint.of(
                "probe-v1".getBytes(StandardCharsets.UTF_8)));
        assertNotEquals(fingerprint, CpuLoweringFingerprint.of(new byte[] {1}));
        byte[] returned = fingerprint.bytes(); returned[0] = 0;
        assertNotEquals(0, fingerprint.bytes()[0]);
        assertEquals("canonicalFamilyBytes", assertThrows(NullPointerException.class,
                () -> CpuLoweringFingerprint.of(null)).getMessage());
        assertEquals("canonicalFamilyBytes must not be empty", assertThrows(
                IllegalArgumentException.class, () -> CpuLoweringFingerprint.of(new byte[0])).getMessage());

        byte[] digest = new byte[CpuGeneratorSchema.FINGERPRINT_BYTE_COUNT];
        for (int index = 0; index < digest.length; index++) digest[index] = (byte) index;
        CpuLoweringFingerprint trusted = CpuLoweringFingerprint.fromDigest(digest);
        digest[0] = 99;
        assertAll(
                () -> assertEquals(
                        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                        trusted.toString()),
                () -> assertNotEquals(CpuLoweringFingerprint.of(new byte[32]),
                        CpuLoweringFingerprint.fromDigest(new byte[32])),
                () -> assertEquals("digest", assertThrows(NullPointerException.class,
                        () -> CpuLoweringFingerprint.fromDigest(null)).getMessage()),
                () -> assertEquals("digest must contain exactly 32 bytes", assertThrows(
                        IllegalArgumentException.class,
                        () -> CpuLoweringFingerprint.fromDigest(new byte[31])).getMessage()));
    }

    @Test void snapshotsListsDerivesExactOrderedEntryTypeAndUsesStructuralEquality() {
        var strides = new ArrayList<>(List.of(2L));
        var arguments = new ArrayList<>(List.of(
                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.READ_ONLY,
                        false, 0, strides),
                new CpuKernelSpecialization.Argument(DataType.INT64,
                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT, BufferAccess.READ_WRITE,
                        true, 0, List.of())));
        var extents = new ArrayList<>(List.of(8L));
        var first = specialization(CpuPortableExecutionMode.SCALAR_PARALLEL, arguments, extents,
                2, null, CpuKernelSpecialization.Tail.NONE, 4);
        var second = specialization(CpuPortableExecutionMode.SCALAR_PARALLEL, List.copyOf(arguments),
                List.of(8L), 2, null, CpuKernelSpecialization.Tail.NONE, 4);
        strides.set(0, 99L); arguments.clear(); extents.set(0, 99L);
        assertEquals(List.of(2L), first.arguments().getFirst().bakedElementStrides());
        assertEquals(List.of(8L), first.bakedExtents());
        assertEquals(MethodType.methodType(void.class, float[].class, long.class,
                MemorySegment.class, long.class, long.class, long.class, long.class, int.class),
                first.entryType());
        assertEquals(first, second); assertEquals(first.hashCode(), second.hashCode());
        assertEquals(first.specializationFingerprint(), second.specializationFingerprint());
        assertEquals("3552566101c3eae39b74886121546a52f626a298390e99b0b7127f4959b61a8f",
                first.specializationFingerprint().toString());
        assertNotEquals(first, specialization(CpuPortableExecutionMode.SCALAR_PARALLEL,
                second.arguments(), List.of(8L), 2, null, CpuKernelSpecialization.Tail.NONE, 8));
    }

    @Test void validatesDescriptorInStableOrder() {
        assertEquals("dataType", assertThrows(NullPointerException.class, () ->
                new CpuKernelSpecialization.Argument(null, null, null, false, -1, null)).getMessage());
        assertEquals("bakedByteOffset must be zero when byteOffsetBaked is false", assertThrows(
                IllegalArgumentException.class, () -> new CpuKernelSpecialization.Argument(
                        DataType.FLOAT32, CpuKernelSpecialization.Carrier.FLOAT_ARRAY,
                        BufferAccess.READ_ONLY, false, 4, List.of())).getMessage());
        assertEquals("MEMORY_SEGMENT requires a baked zero byte offset", assertThrows(
                IllegalArgumentException.class, () -> new CpuKernelSpecialization.Argument(
                        DataType.FLOAT32, CpuKernelSpecialization.Carrier.MEMORY_SEGMENT,
                        BufferAccess.READ_ONLY, false, 0, List.of())).getMessage());
        assertEquals("carrier does not match dataType", assertThrows(IllegalArgumentException.class,
                () -> argument(DataType.FLOAT32, CpuKernelSpecialization.Carrier.INT_ARRAY,
                        BufferAccess.READ_ONLY)).getMessage());
        assertEquals("bakedByteOffset must be aligned to dataType byte width", assertThrows(
                IllegalArgumentException.class, () -> new CpuKernelSpecialization.Argument(
                        DataType.FLOAT32, CpuKernelSpecialization.Carrier.FLOAT_ARRAY,
                        BufferAccess.READ_ONLY, true, 2, List.of())).getMessage());
        assertDoesNotThrow(() -> new CpuKernelSpecialization.Argument(
                DataType.FLOAT32, CpuKernelSpecialization.Carrier.FLOAT_ARRAY,
                BufferAccess.READ_ONLY, false, 0, List.of()));
        assertEquals("laneType must be FLOAT64, FLOAT32, INT32, or INT64", assertThrows(
                IllegalArgumentException.class, () -> new CpuKernelSpecialization.VectorShape(
                        DataType.BOOL, 128, 16)).getMessage());
        assertEquals("dynamicExtentCount must be non-negative", assertThrows(
                IllegalArgumentException.class, () -> specialization(
                        CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(), List.of(),
                        -1, null, CpuKernelSpecialization.Tail.NONE, 4)).getMessage());
        assertEquals("vectorShape is required for vector execution mode", assertThrows(
                IllegalArgumentException.class, () -> specialization(
                        CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD, List.of(), List.of(),
                        0, null, CpuKernelSpecialization.Tail.NONE, 4)).getMessage());
    }

    @Test void validatesArgumentFailuresInTheSpecifiedOrder() {
        assertAll(
                () -> assertEquals("carrier", assertThrows(NullPointerException.class, () ->
                        new CpuKernelSpecialization.Argument(DataType.FLOAT32, null, null,
                                false, -1, null)).getMessage()),
                () -> assertEquals("access", assertThrows(NullPointerException.class, () ->
                        new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                CpuKernelSpecialization.Carrier.FLOAT_ARRAY, null,
                                false, -1, null)).getMessage()),
                () -> assertEquals("bakedByteOffset must be zero when byteOffsetBaked is false",
                        assertThrows(IllegalArgumentException.class, () ->
                                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                        CpuKernelSpecialization.Carrier.INT_ARRAY,
                                        BufferAccess.READ_ONLY, false, -1, null)).getMessage()),
                () -> assertEquals("bakedByteOffset must be non-negative", assertThrows(
                        IllegalArgumentException.class, () ->
                                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                        CpuKernelSpecialization.Carrier.INT_ARRAY,
                                        BufferAccess.READ_ONLY, true, -1, null)).getMessage()),
                () -> assertEquals("bakedElementStrides", assertThrows(NullPointerException.class,
                        () -> new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                CpuKernelSpecialization.Carrier.INT_ARRAY,
                                BufferAccess.READ_ONLY, true, 2, null)).getMessage()),
                () -> assertEquals("bakedElementStrides[0]", assertThrows(
                        NullPointerException.class, () -> new CpuKernelSpecialization.Argument(
                                DataType.FLOAT32, CpuKernelSpecialization.Carrier.INT_ARRAY,
                                BufferAccess.READ_ONLY, true, 2,
                                java.util.Arrays.asList((Long) null))).getMessage()),
                () -> assertEquals("bakedElementStrides[0] must be non-negative", assertThrows(
                        IllegalArgumentException.class, () ->
                                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                        CpuKernelSpecialization.Carrier.INT_ARRAY,
                                        BufferAccess.READ_ONLY, true, 2, List.of(-1L))).getMessage()),
                () -> assertEquals("MEMORY_SEGMENT requires a baked zero byte offset", assertThrows(
                        IllegalArgumentException.class, () ->
                                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT,
                                        BufferAccess.READ_ONLY, true, 2, List.of())).getMessage()),
                () -> assertEquals("carrier does not match dataType", assertThrows(
                        IllegalArgumentException.class, () ->
                                new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                                        CpuKernelSpecialization.Carrier.INT_ARRAY,
                                        BufferAccess.READ_ONLY, true, 2, List.of())).getMessage()));
    }

    @Test void scalarModesAcceptOnlyNoneAndVectorModesAcceptEveryTail() {
        for (CpuPortableExecutionMode mode : CpuPortableExecutionMode.values()) {
            CpuKernelSpecialization.VectorShape vector = mode.vectorized()
                    ? new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4) : null;
            for (CpuKernelSpecialization.Tail tail : CpuKernelSpecialization.Tail.values()) {
                if (mode.vectorized() || tail == CpuKernelSpecialization.Tail.NONE) {
                    assertDoesNotThrow(() -> specialization(
                            mode, List.of(), List.of(), 0, vector, tail, 4));
                } else {
                    assertEquals("scalar execution mode requires Tail.NONE", assertThrows(
                            IllegalArgumentException.class, () -> specialization(
                                    mode, List.of(), List.of(), 0, vector, tail, 4)).getMessage());
                }
            }
        }
    }

    @Test void modesAreExactlyClosedFourValueVocabulary() {
        assertArrayEquals(new CpuPortableExecutionMode[] {
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                CpuPortableExecutionMode.SCALAR_PARALLEL,
                CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD,
                CpuPortableExecutionMode.VECTOR_API_PARALLEL}, CpuPortableExecutionMode.values());
        assertFalse(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD.vectorized());
        assertTrue(CpuPortableExecutionMode.VECTOR_API_PARALLEL.vectorized());
        assertTrue(CpuPortableExecutionMode.SCALAR_PARALLEL.parallel());
    }

    static CpuKernelSpecialization specialization(CpuPortableExecutionMode mode,
            List<CpuKernelSpecialization.Argument> arguments, List<Long> extents, int dynamics,
            CpuKernelSpecialization.VectorShape vector, CpuKernelSpecialization.Tail tail, long tile) {
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                CpuLoweringFingerprint.of("probe-v1".getBytes(StandardCharsets.UTF_8)), mode,
                arguments, extents, dynamics, vector, ByteOrder.nativeOrder(), 1, tile, tail,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    static CpuKernelSpecialization.Argument argument(DataType type,
            CpuKernelSpecialization.Carrier carrier, BufferAccess access) {
        return new CpuKernelSpecialization.Argument(type, carrier, access, true, 0, List.of(1L));
    }
}
