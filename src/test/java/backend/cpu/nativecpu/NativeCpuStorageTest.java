package backend.cpu.nativecpu;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuStorageTest {
    @Test
    void float32StorageAllocatesExpectedBytesAndRoundTripsValues() {
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 3, "f32-test");

        try {
            assertTrue(storage instanceof NativeFloat32Storage);
            assertEquals(12L, storage.byteSize());
            assertEquals(4L, storage.elementSizeBytes());
            NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
            long before = f32.version();

            f32.setFloat32At(0, 1.25f);
            f32.setFloat32At(1, -2.5f);

            assertEquals(1.25f, f32.getFloat32At(0), 0f);
            assertEquals(-2.5f, f32.getFloat32At(1), 0f);
            assertTrue(f32.version() > before);
        } finally {
            storage.close();
        }
    }

    @Test
    void float64StorageAllocatesExpectedBytesAndRoundTripsValues() {
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT64, 2, "f64-test");

        try {
            assertTrue(storage instanceof NativeFloat64Storage);
            assertEquals(16L, storage.byteSize());
            assertEquals(8L, storage.elementSizeBytes());
            NativeFloat64Storage f64 = (NativeFloat64Storage) storage;

            f64.setFloat64At(0, 1.25d);
            f64.setFloat64At(1, -2.5d);

            assertEquals(1.25d, f64.getFloat64At(0), 0d);
            assertEquals(-2.5d, f64.getFloat64At(1), 0d);
        } finally {
            storage.close();
        }
    }

    @Test
    void bfloat16StorageAllocatesTwoBytesPerElementAndPreservesRawBits() {
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.BFLOAT16, 3, "bf16-test");

        try {
            assertTrue(storage instanceof NativeBFloat16Storage);
            assertEquals(6L, storage.byteSize());
            assertEquals(2L, storage.elementSizeBytes());
            NativeBFloat16Storage bf16 = (NativeBFloat16Storage) storage;

            bf16.setBFloat16BitsAt(0, (short) 0x3f80);
            bf16.setBFloat16BitsAt(1, (short) 0xc020);

            assertEquals((short) 0x3f80, bf16.getBFloat16BitsAt(0));
            assertEquals((short) 0xc020, bf16.getBFloat16BitsAt(1));
        } finally {
            storage.close();
        }
    }

    @Test
    void closeIsIdempotentAndUseAfterCloseFailsClearly() {
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 1, "close-test");

        storage.close();
        storage.close();

        IllegalStateException error = assertThrows(IllegalStateException.class, storage::segment);
        assertTrue(error.getMessage().contains("closed"));
    }

    @Test
    void float32ArrayNativeArrayMaterializationPreservesValues() {
        Tensor source = new Tensor(new float[]{1.5f, -2f, 3.25f}, new int[]{3}, null, "source", DataType.FLOAT32);
        Tensor target = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "target", DataType.FLOAT32);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 3, "f32-copy");

        try {
            NativeCpuMaterializer.arrayToNative(source, storage);
            NativeCpuMaterializer.nativeToArray(storage, target);

            assertArrayEquals(source.getFloat32Data(), target.getFloat32Data(), 0f);
        } finally {
            storage.close();
        }
    }

    @Test
    void float64ArrayNativeArrayMaterializationPreservesValues() {
        Tensor source = new Tensor(new double[]{1.5d, -2d, 3.25d}, new int[]{3}, null, "source", DataType.FLOAT64);
        Tensor target = new Tensor(new double[]{0d, 0d, 0d}, new int[]{3}, null, "target", DataType.FLOAT64);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT64, 3, "f64-copy");

        try {
            NativeCpuMaterializer.arrayToNative(source, storage);
            NativeCpuMaterializer.nativeToArray(storage, target);

            assertArrayEquals(source.getFloat64Data(), target.getFloat64Data(), 0d);
        } finally {
            storage.close();
        }
    }

    @Test
    void bfloat16ArrayNativeArrayMaterializationPreservesRawBits() {
        short[] bits = new short[]{(short) 0x3f80, (short) 0x7fc1, (short) 0x8000};
        Tensor source = new Tensor(bits.clone(), new int[]{3}, null, "source", DataType.BFLOAT16);
        Tensor target = new Tensor(new short[]{0, 0, 0}, new int[]{3}, null, "target", DataType.BFLOAT16);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.BFLOAT16, 3, "bf16-copy");

        try {
            NativeCpuMaterializer.arrayToNative(source, storage);
            NativeCpuMaterializer.nativeToArray(storage, target);

            assertArrayEquals(bits, target.getBFloat16Data());
        } finally {
            storage.close();
        }
    }

    @Test
    void materializerRejectsStridedViewsInMvp() {
        Tensor source = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "source", DataType.FLOAT32);
        Tensor selected = source.select(0, 1);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, selected.getFlatDataSize(), "strided-reject");

        try {
            UnsupportedOperationException error = assertThrows(
                    UnsupportedOperationException.class,
                    () -> NativeCpuMaterializer.arrayToNative(selected, storage)
            );
            assertTrue(error.getMessage().contains("dense contiguous"));
        } finally {
            storage.close();
        }
    }
}
