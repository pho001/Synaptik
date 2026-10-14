package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU execution of exact heap-backed input bits through the default segment route. */
final class EngineCpuHeapContiguousIntegrationTest {
    @Test void directlyComputesEveryHeapPrimitiveTypeWithoutChangingStoredBits() {
        Shape shape = Shape.of(2);
        try (Engine engine = Engine.standard()) {
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.FLOAT64, shape),
                    Optional.empty(), new double[] {-0.0,
                            Double.longBitsToDouble(0x7ff8000000000042L)}),
                    ByteBuffer.allocate(16).putLong(0x8000000000000000L)
                            .putLong(0x7ff8000000000042L).array());
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.FLOAT32, shape),
                    Optional.empty(), new float[] {-0.0f,
                            Float.intBitsToFloat(0x7fc00042)}),
                    ByteBuffer.allocate(8).putInt(0x80000000).putInt(0x7fc00042).array());
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.BFLOAT16, shape),
                    Optional.empty(), new short[] {(short) 0x8000, (short) 0x7fc2}),
                    ByteBuffer.allocate(4).putShort((short) 0x8000)
                            .putShort((short) 0x7fc2).array());
            assertBits(engine, TensorFactory.fromFlatFloat16Array(descriptor(DataType.FLOAT16,
                    shape), Optional.empty(), new short[] {(short) 0x8000, (short) 0x7e42}),
                    ByteBuffer.allocate(4).putShort((short) 0x8000)
                            .putShort((short) 0x7e42).array());
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.INT32, shape),
                    Optional.empty(), new int[] {Integer.MIN_VALUE, 0x12345678}),
                    ByteBuffer.allocate(8).putInt(Integer.MIN_VALUE)
                            .putInt(0x12345678).array());
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.INT64, shape),
                    Optional.empty(), new long[] {Long.MIN_VALUE, 0x123456789abcdef0L}),
                    ByteBuffer.allocate(16).putLong(Long.MIN_VALUE)
                            .putLong(0x123456789abcdef0L).array());
            assertBits(engine, TensorFactory.fromFlatArray(descriptor(DataType.BOOL, shape),
                    Optional.empty(), new byte[] {0, 1}), new byte[] {0, 1});
        }
    }

    @Test void rankZeroEmptyAndOffsetHeapSliceUseExactSelectedRegion() {
        try (Engine engine = Engine.standard()) {
            assertBits(engine, TensorFactory.fromFlatArray(
                    descriptor(DataType.FLOAT64, Shape.scalar()), Optional.empty(),
                    new double[] {-0.0}), ByteBuffer.allocate(8)
                            .putLong(0x8000000000000000L).array());
            assertBits(engine, TensorFactory.fromFlatArray(
                    descriptor(DataType.INT32, Shape.of(0)), Optional.empty(), new int[0]),
                    new byte[0]);

            double[] carrier = {99, -0.0,
                    Double.longBitsToDouble(0x7ff8000000000042L), 99};
            MemorySegment selected = MemorySegment.ofArray(carrier).asSlice(8, 16);
            Tensor slice = TensorFactory.create(descriptor(DataType.FLOAT64, Shape.of(2)),
                    Optional.empty(), Optional.of(new MemorySegmentStorage(DataType.FLOAT64, 2,
                            selected)));
            assertBits(engine, slice, ByteBuffer.allocate(16).putLong(0x8000000000000000L)
                    .putLong(0x7ff8000000000042L).array());
        }
    }

    @Test void nativeAndHeapInputsTakeTheSamePreparedSegmentRoute() {
        double[] values = {-0.0, Double.longBitsToDouble(0x7ff8000000000042L)};
        var descriptor = descriptor(DataType.FLOAT64, Shape.of(2));
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            MemorySegment nativeSegment = arena.allocate(16, Double.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(values), 0, nativeSegment, 0, 16);
            Tensor nativeTensor = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT64, 2, nativeSegment)));
            Tensor heapTensor = TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
            assertArrayEquals(bytes(engine.compute(nativeTensor.contiguous())),
                    bytes(engine.compute(heapTensor.contiguous())));
        }
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static void assertBits(Engine engine, Tensor input, byte[] expected) {
        HostTensorValue result = engine.compute(input.contiguous());
        assertEquals(input.descriptor().dataType(), result.dataType());
        assertEquals(input.descriptor().shape(), result.shape());
        assertArrayEquals(expected, bytes(result));
    }

    private static byte[] bytes(HostTensorValue value) {
        ByteBuffer buffer = value.bytes();
        byte[] result = new byte[buffer.remaining()];
        buffer.get(result);
        return result;
    }
}
