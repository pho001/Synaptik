package backend.cpu.nativecpu;

import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    void allocatorTracksLeaseStatsAndRetainState() {
        NativeCpuAllocator allocator = new NativeCpuAllocator();
        NativeCpuAllocation allocation = allocator.allocate(0L, "stats-test");

        assertFalse(allocation.released());
        assertEquals(0L, allocation.byteSize());
        assertEquals(1L, allocation.allocatedBytes());

        var allocated = allocator.statsSnapshot();
        assertEquals(1L, allocated.allocationCount());
        assertEquals(0L, allocated.requestedBytes());
        assertEquals(1L, allocated.allocatedBytes());
        assertEquals(1L, allocated.currentLiveBytes());
        assertEquals(1L, allocated.peakLiveBytes());

        allocation.retain("published output");
        allocation.retain("ignored duplicate");

        assertTrue(allocation.retainedAfterExecute());
        assertEquals("published output", allocation.retainedReason());
        var retained = allocator.statsSnapshot();
        assertEquals(1L, retained.retainCount());
        assertEquals(1L, retained.retainedBytes());

        allocation.release();
        allocation.close();

        assertTrue(allocation.released());
        var released = allocator.statsSnapshot();
        assertEquals(1L, released.releaseCount());
        assertEquals(0L, released.currentLiveBytes());
        assertEquals(0L, released.retainedBytes());
        assertThrows(IllegalStateException.class, () -> allocation.retain("too late"));
    }

    @Test
    void disabledAllocatorNeverReportsPoolHits() {
        NativeCpuAllocator allocator = new NativeCpuAllocator(NativeCpuMemoryConfig.disabled());
        NativeCpuAllocation first = allocator.allocate(32L, "first");
        first.release();
        NativeCpuAllocation second = allocator.allocate(32L, "second");
        second.release();

        var stats = allocator.statsSnapshot();
        assertEquals(NativeMemoryPoolPolicy.DISABLED, allocator.effectivePoolPolicy());
        assertEquals(2L, stats.allocationCount());
        assertEquals(0L, stats.poolHitCount());
        assertEquals(0L, stats.poolMissCount());
    }

    @Test
    void perPreparedPolicyIsAcceptedButNotPooledInThisWave() {
        NativeCpuAllocator allocator = new NativeCpuAllocator(new NativeCpuMemoryConfig(
                NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION,
                1024L,
                64,
                false,
                false
        ));
        NativeCpuAllocation first = allocator.allocate(32L, "first");
        first.release();
        NativeCpuAllocation second = allocator.allocate(32L, "second");
        second.release();

        var stats = allocator.statsSnapshot();
        assertEquals(NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION, allocator.requestedPoolPolicy());
        assertEquals(NativeMemoryPoolPolicy.DISABLED, allocator.effectivePoolPolicy());
        assertEquals(0L, stats.poolHitCount());
        assertEquals(0L, stats.poolMissCount());
    }

    @Test
    void perExecutionAllocatorReusesReleasedSameSizeBlock() {
        NativeCpuAllocator allocator = new NativeCpuAllocator(NativeCpuMemoryConfig.perExecution(1024L));
        NativeCpuAllocation first = allocator.allocate(32L, "first");
        first.release();
        NativeCpuAllocation second = allocator.allocate(32L, "second");

        assertEquals(64L, first.allocatedBytes());
        assertEquals(64L, second.allocatedBytes());
        assertThrows(IllegalStateException.class, first::segment);
        second.release();
        allocator.drainPool();

        var stats = allocator.statsSnapshot();
        assertEquals(NativeMemoryPoolPolicy.PER_EXECUTION, allocator.effectivePoolPolicy());
        assertEquals(2L, stats.allocationCount());
        assertEquals(1L, stats.poolHitCount());
        assertEquals(1L, stats.poolMissCount());
        assertEquals(64L, stats.reusedBytes());
        assertEquals(0L, stats.pooledBytes());
    }

    @Test
    void perExecutionAllocatorDoesNotPoolRetainedOrOverBudgetBlocks() {
        NativeCpuAllocator retainedAllocator = new NativeCpuAllocator(NativeCpuMemoryConfig.perExecution(1024L));
        NativeCpuAllocation retained = retainedAllocator.allocate(32L, "retained");
        retained.retain("publication");
        retained.release();
        NativeCpuAllocation afterRetain = retainedAllocator.allocate(32L, "after-retain");
        afterRetain.release();
        retainedAllocator.drainPool();

        var retainedStats = retainedAllocator.statsSnapshot();
        assertEquals(0L, retainedStats.poolHitCount());
        assertEquals(2L, retainedStats.poolMissCount());
        assertTrue(retainedStats.discardedBytes() >= 64L);

        NativeCpuAllocator budgetAllocator = new NativeCpuAllocator(NativeCpuMemoryConfig.perExecution(32L));
        NativeCpuAllocation overBudget = budgetAllocator.allocate(64L, "over-budget");
        overBudget.release();
        NativeCpuAllocation afterBudget = budgetAllocator.allocate(64L, "after-budget");
        afterBudget.release();

        var budgetStats = budgetAllocator.statsSnapshot();
        assertEquals(0L, budgetStats.poolHitCount());
        assertEquals(2L, budgetStats.poolMissCount());
        assertEquals(0L, budgetStats.pooledBytes());
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
        short[] bits = new short[]{
                (short) 0x3f80,
                (short) 0x7fc1,
                (short) 0x7f80,
                (short) 0xff80,
                (short) 0x8000,
                (short) 0x0001,
                (short) 0x8001
        };
        Tensor source = new Tensor(bits.clone(), new int[]{bits.length}, null, "source", DataType.BFLOAT16);
        Tensor target = new Tensor(new short[bits.length], new int[]{bits.length}, null, "target", DataType.BFLOAT16);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.BFLOAT16, bits.length, "bf16-copy");

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
