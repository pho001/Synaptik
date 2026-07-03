package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryAllocationTest {
    @Test
    void allocatesExactPrimitiveCarrierWithDefaultZeroContentsForEveryDataType() {
        assertCarrier(DataType.FLOAT64, double[].class);
        assertCarrier(DataType.FLOAT32, float[].class);
        assertCarrier(DataType.BFLOAT16, short[].class);
        assertCarrier(DataType.INT32, int[].class);
        assertCarrier(DataType.INT64, long[].class);
        assertCarrier(DataType.BOOL, byte[].class);
    }

    @Test
    void allocatesReferencedSpanForScalarEmptyDenseOffsetStridedAndBroadcastGeometry() {
        assertAllocatedSpan(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), 1);

        Shape emptyShape = Shape.of(2, 0, 4);
        assertAllocatedSpan(
                emptyShape,
                LayoutDescriptor.of(emptyShape, new long[] {0, 4, 1}, 9, true),
                0);

        Shape denseShape = Shape.of(2, 3);
        assertAllocatedSpan(denseShape, LayoutDescriptor.contiguous(denseShape), 6);
        assertAllocatedSpan(
                denseShape,
                LayoutDescriptor.of(denseShape, new long[] {3, 1}, 5, true),
                11);

        Shape stridedShape = Shape.of(2, 2);
        assertAllocatedSpan(
                stridedShape,
                LayoutDescriptor.of(stridedShape, new long[] {5, 1}, 0, true),
                7);

        assertAllocatedSpan(
                denseShape,
                LayoutDescriptor.of(denseShape, new long[] {0, 1}, 0, true),
                3);
    }

    @Test
    void retainsDescriptorNormalizesLabelAndAttachesExactAllocatedStorage() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor descriptor = resolved(
                DataType.FLOAT32, shape, LayoutDescriptor.contiguous(shape));

        Tensor tensor = TensorFactory.allocate(descriptor, Optional.of("  weights\n"));
        MemorySegmentStorage storage =
                assertInstanceOf(MemorySegmentStorage.class, tensor.hostStorage().orElseThrow());

        assertAll(
                () -> assertSame(descriptor, tensor.descriptor()),
                () -> assertEquals(Optional.of("weights"), tensor.label()),
                () -> assertSame(DataType.FLOAT32, storage.dataType()),
                () -> assertEquals(6, storage.elementCapacity()),
                () -> assertEquals(24, storage.byteSize()),
                () -> assertFalse(storage.isReadOnly()),
                () -> assertTrue(storage.isAlive()),
                () -> assertInstanceOf(float[].class, storage.segment().heapBase().orElseThrow()));
    }

    @Test
    void rejectsNullUnresolvedAndOverLimitInputsWithoutConsumingIdentifiers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor staticUnresolved = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        TensorDescriptor dynamicUnresolved = unresolved(DataType.FLOAT32, dynamicShape);
        Shape oversizedShape = Shape.of((long) Integer.MAX_VALUE + 1);
        TensorDescriptor oversized = resolved(
                DataType.BOOL,
                oversizedShape,
                LayoutDescriptor.contiguous(oversizedShape));

        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class, () -> TensorFactory.allocate(null));
        NullPointerException nullLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.allocate(staticUnresolved, null));
        IllegalArgumentException staticFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.allocate(staticUnresolved));
        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.allocate(dynamicUnresolved));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.allocate(oversized));

        assertAll(
                () -> assertEquals("descriptor", nullDescriptor.getMessage()),
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals(
                        "tensor allocation requires a resolved layout",
                        staticFailure.getMessage()),
                () -> assertEquals(
                        "tensor allocation requires a resolved layout",
                        dynamicFailure.getMessage()),
                () -> assertEquals(
                        "tensor allocation span exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void delegatesBlankLabelAfterAllocationAndConsumesIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = resolved(
                DataType.FLOAT32, shape, LayoutDescriptor.contiguous(shape));

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.allocate(descriptor, Optional.of(" \t\n ")));
        Tensor subsequent = TensorFactory.allocate(descriptor);

        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(before + 1, subsequent.id().value()),
                () -> assertEquals(before + 2, next.get()));
    }

    @Test
    void automaticHeapScopeRetainsCarrierAndAllowsAccessFromAnotherThread() throws Exception {
        Tensor tensor = allocateScalarFromHelper();
        MemorySegment segment = tensor.hostStorage().orElseThrow().segment();
        Object heapBase = segment.heapBase().orElseThrow();

        try (ExecutorService executor = Executors.newSingleThreadExecutor()) {
            Byte secondThreadValue = executor.submit(() -> segment.get(JAVA_BYTE, 0)).get();
            assertEquals(Byte.valueOf((byte) 0), secondThreadValue);
        }

        assertAll(
                () -> assertFalse(segment.isNative()),
                () -> assertTrue(segment.scope().isAlive()),
                () -> assertInstanceOf(long[].class, heapBase),
                () -> assertEquals(1, ((long[]) heapBase).length),
                () -> assertEquals(0L, ((long[]) heapBase)[0]));
    }

    private static Tensor allocateScalarFromHelper() {
        Shape shape = Shape.scalar();
        return TensorFactory.allocate(resolved(
                DataType.INT64, shape, LayoutDescriptor.contiguous(shape)));
    }

    private static void assertCarrier(DataType dataType, Class<?> expectedCarrier) {
        Shape shape = Shape.of(3);
        Tensor tensor = TensorFactory.allocate(resolved(
                dataType, shape, LayoutDescriptor.contiguous(shape)));
        MemorySegmentStorage storage =
                assertInstanceOf(MemorySegmentStorage.class, tensor.hostStorage().orElseThrow());
        Object heapBase = storage.segment().heapBase().orElseThrow();

        assertAll(
                dataType.name(),
                () -> assertSame(dataType, storage.dataType()),
                () -> assertEquals(3, storage.elementCapacity()),
                () -> assertEquals(3L * dataType.byteWidth(), storage.byteSize()),
                () -> assertEquals(expectedCarrier, heapBase.getClass()),
                () -> assertEquals(3, java.lang.reflect.Array.getLength(heapBase)),
                () -> assertTrue(allRawElementsAreZero(heapBase)),
                () -> assertFalse(storage.isReadOnly()),
                () -> assertTrue(storage.isAlive()),
                () -> assertFalse(storage.segment().isNative()));
    }

    private static boolean allRawElementsAreZero(Object array) {
        if (array instanceof double[] values) {
            return Arrays.equals(values, new double[values.length]);
        }
        if (array instanceof float[] values) {
            return Arrays.equals(values, new float[values.length]);
        }
        if (array instanceof short[] values) {
            return Arrays.equals(values, new short[values.length]);
        }
        if (array instanceof int[] values) {
            return Arrays.equals(values, new int[values.length]);
        }
        if (array instanceof long[] values) {
            return Arrays.equals(values, new long[values.length]);
        }
        return Arrays.equals((byte[]) array, new byte[((byte[]) array).length]);
    }

    private static void assertAllocatedSpan(
            Shape shape, LayoutDescriptor layout, long expectedSpan) {
        TensorDescriptor descriptor = resolved(DataType.FLOAT32, shape, layout);
        Tensor tensor = TensorFactory.allocate(descriptor);
        MemorySegmentStorage storage =
                assertInstanceOf(MemorySegmentStorage.class, tensor.hostStorage().orElseThrow());
        Object heapBase = storage.segment().heapBase().orElseThrow();

        assertAll(
                () -> assertEquals(layout.referencedElementSpan(), expectedSpan),
                () -> assertEquals(expectedSpan, storage.elementCapacity()),
                () -> assertEquals(Math.multiplyExact(expectedSpan, 4), storage.byteSize()),
                () -> assertInstanceOf(float[].class, heapBase),
                () -> assertEquals(Math.toIntExact(expectedSpan), ((float[]) heapBase).length));
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(
            DataType dataType, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
