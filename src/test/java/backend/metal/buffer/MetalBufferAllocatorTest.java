package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalBufferAllocatorTest {
    @Test
    void createOutputBindingAllocatesFromLayoutByteLength() {
        AtomicLong allocatedBytes = new AtomicLong();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                allocatedBytes.set(byteLength);
                return new MetalBufferHandle(MemorySegment.ofAddress(9), byteLength, "shared", "test", false);
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );

        MetalBufferBinding binding = allocator.createOutputBinding(9, layout);

        assertEquals(24, allocatedBytes.get());
        assertEquals(layout, binding.layout());
        assertEquals(MetalBufferAccess.READ_WRITE, binding.access());
    }

    @Test
    void createOutputBindingRejectsBroadcastZeroStrideLayoutBeforeNativeAllocation() {
        AtomicInteger allocations = new AtomicInteger();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                allocations.incrementAndGet();
                return new MetalBufferHandle(MemorySegment.ofAddress(9), byteLength, "shared", "test", false);
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
        AcceleratorBufferLayout zeroOffsetView = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 2},
                new int[]{0, 1},
                0,
                4
        );

        assertThrows(UnsupportedOperationException.class, () -> allocator.createOutputBinding(9, zeroOffsetView));
        assertEquals(0, allocations.get());
    }

    @Test
    void materializerSupportsPermutedLogicalViewForGraphOutput() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        Tensor target = permutedTarget();
        MetalBufferBinding binding = bindingFor(7, target);

        assertTrue(materializer.supports(binding, target, CpuMaterializationReason.GRAPH_OUTPUT));
    }

    @Test
    void materializerSupportsNonZeroOffsetLogicalViewForCpuConsumer() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        Tensor target = nonZeroOffsetTarget();
        MetalBufferBinding binding = bindingFor(8, target);

        assertTrue(materializer.supports(binding, target, CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void materializerSupportsZeroOffsetLogicalViewForGradientPublication() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        Tensor target = zeroOffsetTarget();
        MetalBufferBinding binding = bindingFor(9, target);

        assertTrue(materializer.supports(binding, target, CpuMaterializationReason.GRADIENT_PUBLICATION));
    }

    @Test
    void materializerRejectsBroadcastZeroStrideLogicalView() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        Tensor base = new Tensor(new float[]{0f, 0f}, new int[]{2, 1}, null, "base", DataType.FLOAT32);
        Tensor target = base.expand(2, 3);
        TensorInternalAccess.aliasRuntimeFrom(target, base);
        MetalBufferBinding binding = bindingFor(10, target);

        assertFalse(materializer.supports(binding, target, CpuMaterializationReason.GRAPH_OUTPUT));
    }

    @Test
    void materializerRejectsUnsupportedLogicalView() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        Tensor target = new Tensor(
                new int[]{2},
                new int[]{-1},
                0,
                null,
                null,
                "target",
                DataType.FLOAT32
        );
        MetalBufferBinding binding = bindingFor(11, target);

        assertFalse(materializer.supports(binding, target, CpuMaterializationReason.GRAPH_OUTPUT));
    }

    @Test
    void readToCpuRejectsBindingStrideMismatchBeforeNativeRead() {
        AtomicInteger reads = new AtomicInteger();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                reads.incrementAndGet();
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 2}, new int[]{1, 2}, 0, 4),
                new MetalBufferHandle(MemorySegment.ofAddress(7), 16, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );
        Tensor destination = new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, null, "destination", DataType.FLOAT32);

        assertThrows(
                IllegalArgumentException.class,
                () -> allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS)
        );
        assertEquals(0, reads.get());
    }

    @Test
    void readToCpuScattersPermutedDenseBufferIntoDestinationStorage() {
        Tensor destination = permutedTarget();
        MetalBufferAllocator allocator = allocatorReading(1f, 2f, 3f, 4f, 5f, 6f);
        MetalBufferBinding binding = bindingFor(7, destination);

        allocator.readToCpu(binding, destination, CpuMaterializationReason.GRAPH_OUTPUT);

        assertFloatArrayEquals(new float[]{1f, 3f, 5f, 2f, 4f, 6f}, destination.getFloat32Data());
    }

    @Test
    void readToCpuScattersNonZeroOffsetDenseBufferIntoDestinationStorage() {
        Tensor destination = nonZeroOffsetTarget();
        MetalBufferAllocator allocator = allocatorReading(10f, 20f, 30f);
        MetalBufferBinding binding = bindingFor(8, destination);

        allocator.readToCpu(binding, destination, CpuMaterializationReason.CPU_CONSUMER);

        assertFloatArrayEquals(new float[]{0f, 0f, 0f, 10f, 20f, 30f}, destination.getFloat32Data());
    }

    private static MetalBufferAllocator unusedAllocator() {
        return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
    }

    private static MetalBufferAllocator allocatorReading(float... values) {
        return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                throw new UnsupportedOperationException("not used");
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                for (int i = 0; i < values.length; i++) {
                    destination.setAtIndex(JAVA_FLOAT, i, values[i]);
                }
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
    }

    private static Tensor permutedTarget() {
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f, 0f, 0f}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor target = base.permute(1, 0);
        TensorInternalAccess.aliasRuntimeFrom(target, base);
        return target;
    }

    private static Tensor nonZeroOffsetTarget() {
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f, 0f, 0f}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor target = base.select(0, 1);
        TensorInternalAccess.aliasRuntimeFrom(target, base);
        return target;
    }

    private static Tensor zeroOffsetTarget() {
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f, 0f, 0f}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor target = base.select(1, 0);
        TensorInternalAccess.aliasRuntimeFrom(target, base);
        return target;
    }

    private static MetalBufferBinding bindingFor(int nodeId, Tensor target) {
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.fromTensor(target);
        return new MetalBufferBinding(
                nodeId,
                layout,
                new MetalBufferHandle(MemorySegment.ofAddress(nodeId), layout.logicalByteLength(), "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );
    }

    private static void assertFloatArrayEquals(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 0.00001f, "mismatch at " + i);
        }
    }
}
