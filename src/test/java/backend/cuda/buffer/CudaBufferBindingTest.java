package backend.cuda.buffer;

import runtime.device.buffer.AcceleratorBufferAccessMode;
import runtime.device.buffer.AcceleratorBufferLayout;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaBufferBindingTest {
    @Test
    void viewOfReusesHandleWithTargetLayoutAndNodeId() {
        CudaBufferHandle handle = new CudaBufferHandle(MemorySegment.ofAddress(88), 32, true);
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
        CudaBufferBinding source = new CudaBufferBinding(3, sourceLayout, handle, CudaBufferAccess.READ);

        CudaBufferBinding view = CudaBufferBinding.viewOf(9, targetLayout, source, CudaBufferAccess.READ_WRITE);

        assertEquals(9, view.nodeId());
        assertEquals(targetLayout, view.layout());
        assertTrue(view.handle() == source.handle());
        assertEquals(source.nativeHandleIdentity(), view.nativeHandleIdentity());
        assertTrue(view.available());
        assertEquals(AcceleratorBufferAccessMode.READ_WRITE, view.accessMode());
    }
}
