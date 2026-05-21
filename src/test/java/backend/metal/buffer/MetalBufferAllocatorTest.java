package backend.metal.buffer;

import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceToCpuMaterializer;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeFloat32Storage;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalBufferAllocatorTest {
    @Test
    void nativeFloat32InputBindingUploadsSegmentWithoutJavaArrayCopy() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor source = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "source", DataType.FLOAT32);
        NativeFloat32Storage storage = (NativeFloat32Storage) new NativeCpuStorageFactory()
                .allocate(DataType.FLOAT32, 3, "native-source");
        storage.setFloat32At(0, 1.25f);
        storage.setFloat32At(1, -2.5f);
        storage.setFloat32At(2, 3.75f);

        MetalBufferBinding binding = allocator.createNativeFloat32InputBinding(5, source, storage);

        assertEquals(DataType.FLOAT32, binding.layout().dataType());
        assertEquals(3L * Float.BYTES, binding.logicalByteLength());
        assertArrayEquals(new float[]{1.25f, -2.5f, 3.75f}, nativeAccess.floatValues(binding.handle()), 0.0f);
    }

    @Test
    void nativeFloat32ReadbackWritesSegmentWithoutJavaArrayCopy() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor destination = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "destination", DataType.FLOAT32);
        NativeFloat32Storage storage = (NativeFloat32Storage) new NativeCpuStorageFactory()
                .allocate(DataType.FLOAT32, 3, "native-destination");
        MetalBufferBinding binding = allocator.createOutputBinding(
                7,
                backend.accelerator.buffer.AcceleratorBufferLayout.fromTensor(destination)
        );
        nativeAccess.putFloatValues(binding.handle(), new float[]{4f, 5f, 6f});

        allocator.readToNativeFloat32(binding, destination, storage, CpuMaterializationReason.CPU_CONSUMER);

        assertEquals(4f, storage.getFloat32At(0), 0.0f);
        assertEquals(5f, storage.getFloat32At(1), 0.0f);
        assertEquals(6f, storage.getFloat32At(2), 0.0f);
    }

    @Test
    void bfloat16InputBindingAndReadbackPreserveRawStorageBits() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        short[] raw = new short[]{(short) 0x3f80, (short) 0xc020, (short) 0x0000};
        Tensor source = new Tensor(raw.clone(), new int[]{3}, null, "bf16", DataType.BFLOAT16);

        MetalBufferBinding binding = allocator.createInputBinding(7, source);
        Tensor destination = new Tensor(new short[]{0, 0, 0}, new int[]{3}, null, "out", DataType.BFLOAT16);
        allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

        assertEquals(DataType.BFLOAT16, binding.layout().dataType());
        assertEquals(raw.length * Short.BYTES, binding.logicalByteLength());
        assertArrayEquals(raw, destination.toBFloat16BitsArrayCopy());
    }

    @Test
    void bfloat16OutputBindingAllocatesLogicalByteLength() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor output = new Tensor(new short[]{0, 0, 0, 0}, new int[]{2, 2}, null, "out", DataType.BFLOAT16);

        MetalBufferBinding binding = allocator.createOutputBinding(9, backend.accelerator.buffer.AcceleratorBufferLayout.fromTensor(output));

        assertEquals(DataType.BFLOAT16, binding.layout().dataType());
        assertEquals(4L * Short.BYTES, binding.logicalByteLength());
        assertEquals(4L * Short.BYTES, binding.handle().byteLength());
    }

    @Test
    void int32IndexInputBindingAllocatesLogicalByteLength() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor indices = new Tensor(new int[]{2, 0, 1}, new int[]{3}, null, "indices", DataType.INT32);

        MetalBufferBinding binding = allocator.createInputBinding(11, indices);

        assertEquals(DataType.INT32, binding.layout().dataType());
        assertEquals(3L * Integer.BYTES, binding.logicalByteLength());
        assertEquals(3L * Integer.BYTES, binding.handle().byteLength());
    }

    @Test
    void broadcastZeroStrideViewReadbackCopiesPhysicalSpanAndPreservesLogicalValues() {
        FakeNativeAccess nativeAccess = new FakeNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        Tensor source = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "source", DataType.FLOAT32);
        Tensor expanded = source.expand(2, 3);
        Tensor destination = new Tensor(
                expanded.getShapeUnsafe().clone(),
                expanded.getStridesUnsafe().clone(),
                expanded.getStorageOffsetUnsafe(),
                null,
                null,
                "expandedDestination",
                DataType.FLOAT32
        );

        MetalBufferBinding sourceBinding = allocator.createInputBinding(3, source);
        MetalBufferBinding viewBinding = MetalBufferBinding.viewOf(
                4,
                backend.accelerator.buffer.AcceleratorBufferLayout.fromTensor(expanded),
                sourceBinding,
                MetalBufferAccess.READ
        );
        DeviceToCpuMaterializer materializer = new MetalDeviceToCpuMaterializer(allocator);

        assertEquals(3L * Float.BYTES, viewBinding.handle().byteLength());
        assertEquals(6L * Float.BYTES, viewBinding.logicalByteLength());
        assertEquals(backend.accelerator.buffer.AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW,
                viewBinding.layout().layoutClass());
        assertTrue(materializer.supports(viewBinding, destination, CpuMaterializationReason.GRADIENT_PUBLICATION));

        allocator.readToCpu(viewBinding, destination, CpuMaterializationReason.GRADIENT_PUBLICATION);

        assertArrayEquals(new float[]{1f, 2f, 3f, 0f, 0f, 0f}, TensorInternalAccess.float32Data(destination), 0.0f);
        assertArrayEquals(new double[]{1d, 2d, 3d, 1d, 2d, 3d}, destination.toDoubleArrayCopy(), 0.0d);
    }

    private static final class FakeNativeAccess implements MetalBufferAllocator.NativeAccess {
        private long nextAddress = 1_000L;
        private final Map<Long, byte[]> buffers = new HashMap<>();

        @Override
        public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
            long address = nextAddress++;
            byte[] storage = new byte[Math.toIntExact(byteLength)];
            if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                byte[] initial = new byte[Math.toIntExact(initialDataBytes)];
                MemorySegment.ofArray(initial).copyFrom(initialData.reinterpret(initialDataBytes));
                System.arraycopy(initial, 0, storage, 0, Math.min(initial.length, storage.length));
            }
            buffers.put(address, storage);
            return new MetalBufferHandle(MemorySegment.ofAddress(address), byteLength, storageMode == 1 ? "shared" : "test", "test", true);
        }

        @Override
        public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            byte[] storage = buffers.get(handle.nativeHandle().address());
            byte[] source = Arrays.copyOf(storage, Math.toIntExact(byteLength));
            destination.copyFrom(MemorySegment.ofArray(source));
        }

        @Override
        public void destroyBuffer(MetalBufferHandle handle) {
            buffers.remove(handle.nativeHandle().address());
        }

        void putFloatValues(MetalBufferHandle handle, float[] values) {
            byte[] bytes = new byte[values.length * Float.BYTES];
            MemorySegment.ofArray(bytes).copyFrom(MemorySegment.ofArray(values));
            buffers.put(handle.nativeHandle().address(), bytes);
        }

        float[] floatValues(MetalBufferHandle handle) {
            byte[] storage = buffers.get(handle.nativeHandle().address());
            float[] values = new float[storage.length / Float.BYTES];
            MemorySegment.ofArray(values).copyFrom(MemorySegment.ofArray(storage));
            return values;
        }
    }
}
