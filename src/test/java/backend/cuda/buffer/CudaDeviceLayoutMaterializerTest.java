package backend.cuda.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformRequest;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaDeviceLayoutMaterializerTest {
    @Test
    void materializesDenseFloat32TargetThroughBridge() {
        RecordingBridge bridge = new RecordingBridge();
        CudaBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        CudaBufferBinding source = new CudaBufferBinding(
                1,
                sourceLayout,
                new CudaBufferHandle(MemorySegment.ofAddress(10), 24, false),
                CudaBufferAccess.READ
        );

        var materializer = new CudaDeviceLayoutMaterializer(bridge, context(), allocator);
        var materialized = materializer.materialize(decision(sourceLayout, targetLayout), source, null);

        assertTrue(bridge.materializeCalled.get());
        assertEquals(2, materialized.nodeId());
        assertEquals(targetLayout, materialized.layout());
    }

    @Test
    void rejectsUnsupportedTargetDTypeBeforeBridgeCall() {
        RecordingBridge bridge = new RecordingBridge();
        CudaBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2},
                new int[]{1},
                0,
                2
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.BOOL,
                new int[]{2},
                new int[]{1},
                0,
                2
        );
        CudaBufferBinding source = new CudaBufferBinding(
                1,
                sourceLayout,
                new CudaBufferHandle(MemorySegment.ofAddress(10), 8, false),
                CudaBufferAccess.READ
        );
        var materializer = new CudaDeviceLayoutMaterializer(bridge, context(), allocator);

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> materializer.materialize(decision(sourceLayout, targetLayout), source, null)
        );

        assertTrue(failure.getMessage().contains("NATIVE_LAYOUT_DTYPE_UNSUPPORTED"));
        assertTrue(failure.getMessage().contains("role=COMPUTE_OUTPUT"));
        assertTrue(failure.getMessage().contains("dtype=BOOL"));
        assertEquals(false, bridge.materializeCalled.get());
    }

    @Test
    void rejectsBroadcastMaterializationWithCudaSpecificReason() {
        RecordingBridge bridge = new RecordingBridge();
        CudaBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{1, 2},
                new int[]{0, 1},
                0,
                2
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        CudaBufferBinding source = new CudaBufferBinding(
                1,
                sourceLayout,
                new CudaBufferHandle(MemorySegment.ofAddress(10), 8, false),
                CudaBufferAccess.READ
        );
        var materializer = new CudaDeviceLayoutMaterializer(bridge, context(), allocator);

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> materializer.materialize(broadcastDecision(sourceLayout, targetLayout), source, null)
        );

        assertTrue(failure.getMessage().contains("CUDA_LAYOUT_BROADCAST_UNSUPPORTED"));
        assertEquals(false, bridge.materializeCalled.get());
    }

    @Test
    void rejectsNonDenseTargetLayoutWithStableReason() {
        RecordingBridge bridge = new RecordingBridge();
        CudaBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 2},
                new int[]{2, 1},
                0,
                4
        );
        AcceleratorBufferLayout targetLayout = new AcceleratorBufferLayout(
                DataType.FLOAT32,
                new int[]{2, 2},
                new int[]{1, 2},
                0,
                4,
                16,
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW
        );
        CudaBufferBinding source = new CudaBufferBinding(
                1,
                sourceLayout,
                new CudaBufferHandle(MemorySegment.ofAddress(10), 16, false),
                CudaBufferAccess.READ
        );
        var materializer = new CudaDeviceLayoutMaterializer(bridge, context(), allocator);

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> materializer.materialize(decision(sourceLayout, targetLayout), source, null)
        );

        assertTrue(failure.getMessage().contains("CUDA_LAYOUT_TARGET_UNSUPPORTED"));
        assertTrue(failure.getMessage().contains("layoutClass=PERMUTED_OR_STRIDED_VIEW"));
        assertEquals(false, bridge.materializeCalled.get());
    }

    private static AcceleratorLayoutTransformDecision decision(
            AcceleratorBufferLayout sourceLayout,
            AcceleratorBufferLayout targetLayout
    ) {
        return AcceleratorLayoutTransformDecision.denseGpuMaterialization(
                new AcceleratorLayoutTransformRequest(
                        "GPU_CUDA",
                        1,
                        2,
                        Operation.OpType.CONTIGUOUS,
                        sourceLayout,
                        targetLayout,
                        null,
                        false
                ),
                "test materialization"
        );
    }

    private static AcceleratorLayoutTransformDecision broadcastDecision(
            AcceleratorBufferLayout sourceLayout,
            AcceleratorBufferLayout targetLayout
    ) {
        return AcceleratorLayoutTransformDecision.broadcastGpuMaterialization(
                new AcceleratorLayoutTransformRequest(
                        "GPU_CUDA",
                        1,
                        2,
                        Operation.OpType.EXPAND,
                        sourceLayout,
                        targetLayout,
                        null,
                        false
                ),
                "test broadcast materialization"
        );
    }

    private static CudaBridgeContext context() {
        return new CudaBridgeContext(true, MemorySegment.ofAddress(1), "");
    }

    private static CudaBufferAllocator allocator() {
        return CudaBufferAllocator.available(new CudaBufferAllocator.NativeAccess() {
            @Override
            public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
                return new CudaBufferHandle(MemorySegment.ofAddress(20), byteLength, false);
            }

            @Override
            public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
            }

            @Override
            public void destroyBuffer(CudaBufferHandle handle) {
            }
        });
    }

    private static final class RecordingBridge implements CudaGraphBridge {
        private final AtomicBoolean materializeCalled = new AtomicBoolean(false);

        @Override
        public boolean isAvailable() {
            return true;
        }

        @Override
        public String unavailableReason() {
            return "";
        }

        @Override
        public CudaBridgeContext createContext() {
            return context();
        }

        @Override
        public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
            return CudaBridgeExecutable.unavailable("not used");
        }

        @Override
        public boolean supportsLayoutMaterialization() {
            return true;
        }

        @Override
        public void materializeLayout(CudaBridgeContext context, CudaBufferBinding source, CudaBufferBinding destination) {
            materializeCalled.set(true);
        }

        @Override
        public void execute(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<tensor.Tensor> externalInputs,
                List<tensor.Tensor> outputs
        ) {
            throw new UnsupportedOperationException("not used");
        }
    }
}
