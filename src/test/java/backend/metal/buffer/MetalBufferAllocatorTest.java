package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.memory.CpuMaterializationReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    void createOutputBindingRejectsNonDenseLayoutBeforeNativeAllocation() {
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
                new int[]{4, 2},
                0,
                4
        );

        assertThrows(UnsupportedOperationException.class, () -> allocator.createOutputBinding(9, zeroOffsetView));
        assertEquals(0, allocations.get());
    }

    @Test
    void materializerSupportRequiresStrideAndStorageOffsetMatch() {
        MetalDeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(unusedAllocator());
        MetalBufferBinding denseBinding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0, 4),
                new MetalBufferHandle(MemorySegment.ofAddress(7), 16, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );
        Tensor squareBase = new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, null, "square", DataType.FLOAT32);
        Tensor permutedTarget = squareBase.permute(1, 0);

        assertFalse(materializer.supports(denseBinding, permutedTarget, CpuMaterializationReason.PUBLIC_DATA_ACCESS));

        MetalBufferBinding zeroOffsetBinding = new MetalBufferBinding(
                8,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{3}, new int[]{1}, 0, 3),
                new MetalBufferHandle(MemorySegment.ofAddress(8), 12, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );
        Tensor rowBase = new Tensor(new float[]{0f, 0f, 0f, 0f, 0f, 0f}, new int[]{2, 3}, null, "rowBase", DataType.FLOAT32);
        Tensor offsetTarget = rowBase.select(0, 1);

        assertFalse(materializer.supports(zeroOffsetBinding, offsetTarget, CpuMaterializationReason.PUBLIC_DATA_ACCESS));
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
    void readToCpuRejectsNonContiguousDestinationBeforeNativeRead() {
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
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor destination = base.permute(1, 0);
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.fromTensor(destination),
                new MetalBufferHandle(MemorySegment.ofAddress(7), 16, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );

        assertThrows(
                UnsupportedOperationException.class,
                () -> allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS)
        );
        assertEquals(0, reads.get());
    }

    @Test
    void readToCpuRejectsNonZeroOffsetDestinationBeforeNativeRead() {
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
        Tensor base = new Tensor(new float[]{0f, 0f, 0f, 0f, 0f, 0f}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor destination = base.select(0, 1);
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.fromTensor(destination),
                new MetalBufferHandle(MemorySegment.ofAddress(7), 12, "shared", "test", false),
                MetalBufferAccess.READ_WRITE
        );

        assertThrows(
                UnsupportedOperationException.class,
                () -> allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS)
        );
        assertEquals(0, reads.get());
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
}
