package backend.metal.buffer;

import runtime.device.buffer.AcceleratorBufferAccessMode;
import runtime.device.buffer.AcceleratorBufferLayout;
import runtime.device.buffer.AcceleratorBufferLayoutClass;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalBufferBindingTest {
    @Test
    void bindingReportsLogicalPayloadCoverage() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 16, "shared", "test", false);
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 2}, new int[]{2, 1}, 0, 4),
                handle,
                MetalBufferAccess.READ
        );

        assertTrue(binding.bufferCoversLogicalPayload());
        assertTrue(binding.available());
        assertEquals("GPU_METAL", binding.backendId());
        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, binding.layout().layoutClass());
        assertArrayEquals(new int[]{2, 1}, binding.layout().strides());
        assertEquals(0, binding.layout().storageOffset());
        assertTrue(binding.nativeHandleIdentity().contains("GPU_METAL"));
        assertTrue(binding.nativeHandleIdentity().contains("owner=test"));
        assertTrue(binding.nativeHandleIdentity().contains("storageMode=shared"));
        assertTrue(binding.nativeHandleIdentity().contains("bytes=16"));
        assertEquals(AcceleratorBufferAccessMode.READ, binding.accessMode());
        assertTrue(binding.describe().contains("nodeId=7"));
        assertTrue(binding.describe().contains("layoutClass=DENSE_CONTIGUOUS"));
        assertTrue(binding.describe().contains("strides=[2, 1]"));
        assertTrue(binding.describe().contains("storageOffset=0"));
        assertTrue(binding.describe().contains("bytes=16"));
        assertTrue(binding.describe().contains("handleBytes=16"));
    }

    @Test
    void broadcastViewCoverageUsesPhysicalSpanRatherThanDenseLogicalBytes() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 12, "shared", "test:logical-view", false);
        MetalBufferBinding binding = new MetalBufferBinding(
                7,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 3}, new int[]{0, 1}, 0, 6),
                handle,
                MetalBufferAccess.READ
        );

        assertEquals(24, binding.logicalByteLength());
        assertTrue(binding.bufferCoversLogicalPayload());
        assertTrue(binding.available());
    }

    @Test
    void handleIdentifiesOnlySharedStorageAsHostShared() {
        MetalBufferHandle shared = new MetalBufferHandle(MemorySegment.ofAddress(1), 16, "shared", "test", false);
        MetalBufferHandle mixedCaseShared = new MetalBufferHandle(MemorySegment.ofAddress(2), 16, "SHARED", "test", false);
        MetalBufferHandle privateStorage = new MetalBufferHandle(MemorySegment.ofAddress(3), 16, "private", "test", false);
        MetalBufferHandle unknownStorage = new MetalBufferHandle(MemorySegment.ofAddress(4), 16, "", "test", false);

        assertTrue(shared.hostShared());
        assertTrue(mixedCaseShared.hostShared());
        assertFalse(privateStorage.hostShared());
        assertFalse(unknownStorage.hostShared());
    }

    @Test
    void layoutAccessorsReturnDefensiveCopiesThroughBinding() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(1), 32, "shared", "test", false);
        int[] shape = {2, 4};
        int[] strides = {4, 1};
        AcceleratorBufferLayout layout = AcceleratorBufferLayout.of(DataType.FLOAT32, shape, strides, 0, 8);
        MetalBufferBinding binding = new MetalBufferBinding(3, layout, handle, MetalBufferAccess.WRITE);

        shape[0] = 99;
        strides[0] = 99;
        int[] read = binding.layout().shape();
        int[] readStrides = binding.layout().strides();
        read[1] = 99;
        readStrides[1] = 99;

        assertArrayEquals(new int[]{2, 4}, binding.layout().shape());
        assertArrayEquals(new int[]{4, 1}, binding.layout().strides());
        assertEquals(AcceleratorBufferAccessMode.WRITE, binding.accessMode());
    }

    @Test
    void unavailableOrSmallHandleDoesNotCoverPayload() {
        MetalBufferBinding unavailable = new MetalBufferBinding(
                1,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2}, new int[]{1}, 0, 2),
                new MetalBufferHandle(MemorySegment.NULL, 8, "shared", "test", false),
                MetalBufferAccess.READ
        );
        MetalBufferBinding tooSmall = new MetalBufferBinding(
                2,
                AcceleratorBufferLayout.of(DataType.FLOAT64, new int[]{2}, new int[]{1}, 0, 2),
                new MetalBufferHandle(MemorySegment.ofAddress(1), 8, "shared", "test", false),
                MetalBufferAccess.READ
        );

        assertFalse(unavailable.bufferCoversLogicalPayload());
        assertFalse(tooSmall.bufferCoversLogicalPayload());
    }

    @Test
    void viewOfReusesHandleWithTargetLayoutAndNodeId() {
        MetalBufferHandle handle = new MetalBufferHandle(MemorySegment.ofAddress(77), 32, "shared", "test", true);
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 4},
                new int[]{4, 1},
                0,
                8
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{4, 2},
                new int[]{1, 4},
                0,
                8
        );
        MetalBufferBinding source = new MetalBufferBinding(3, sourceLayout, handle, MetalBufferAccess.READ);

        MetalBufferBinding view = MetalBufferBinding.viewOf(9, targetLayout, source, MetalBufferAccess.READ_WRITE);

        assertEquals(9, view.nodeId());
        assertEquals(targetLayout, view.layout());
        assertEquals(source.handle().nativeHandle(), view.handle().nativeHandle());
        assertTrue(view.handle().owner().contains(":logical-view"));
        assertFalse(view.handle().ownsHandle());
        assertTrue(view.available());
        assertEquals(AcceleratorBufferAccessMode.READ_WRITE, view.accessMode());
    }
}
