package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformRequest;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.lowering.MetalPartitionPlan;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalDeviceLayoutMaterializerTest {
    @Test
    void materializesDenseFloat32TargetThroughBridge() {
        RecordingBridge bridge = new RecordingBridge();
        MetalBufferAllocator allocator = allocator();
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
        MetalBufferBinding source = new MetalBufferBinding(
                1,
                sourceLayout,
                new MetalBufferHandle(MemorySegment.ofAddress(10), 24, "shared", "source:logical-view", false),
                MetalBufferAccess.READ
        );

        var materializer = new MetalDeviceLayoutMaterializer(bridge, context(), allocator);
        var materialized = materializer.materialize(decision(sourceLayout, targetLayout), source, null);

        assertTrue(bridge.materializeCalled.get());
        assertEquals(2, materialized.nodeId());
        assertEquals(targetLayout, materialized.layout());
    }

    @Test
    void materializesBroadcastFloat32TargetThroughBridge() {
        RecordingBridge bridge = new RecordingBridge();
        MetalBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{0, 1},
                0,
                6
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );
        MetalBufferBinding source = new MetalBufferBinding(
                1,
                sourceLayout,
                new MetalBufferHandle(MemorySegment.ofAddress(10), 12, "shared", "source:broadcast-view", false),
                MetalBufferAccess.READ
        );

        var materializer = new MetalDeviceLayoutMaterializer(bridge, context(), allocator);
        var materialized = materializer.materialize(
                AcceleratorLayoutTransformDecision.broadcastGpuMaterialization(
                        request(sourceLayout, targetLayout),
                        "test broadcast materialization"
                ),
                source,
                null
        );

        assertTrue(bridge.materializeCalled.get());
        assertEquals(2, materialized.nodeId());
        assertEquals(targetLayout, materialized.layout());
    }

    @Test
    void rejectsUnsupportedTargetDTypeBeforeBridgeCall() {
        RecordingBridge bridge = new RecordingBridge();
        MetalBufferAllocator allocator = allocator();
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2},
                new int[]{1},
                0,
                2
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.INT32,
                new int[]{2},
                new int[]{1},
                0,
                2
        );
        MetalBufferBinding source = new MetalBufferBinding(
                1,
                sourceLayout,
                new MetalBufferHandle(MemorySegment.ofAddress(10), 8, "shared", "source", false),
                MetalBufferAccess.READ
        );
        var materializer = new MetalDeviceLayoutMaterializer(bridge, context(), allocator);

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> materializer.materialize(decision(sourceLayout, targetLayout), source, null)
        );

        assertTrue(failure.getMessage().contains("FLOAT32 only"));
        assertTrue(failure.getMessage().contains("NATIVE_LAYOUT_DTYPE_UNSUPPORTED"));
        assertEquals(false, bridge.materializeCalled.get());
    }

    private static AcceleratorLayoutTransformDecision decision(
            AcceleratorBufferLayout sourceLayout,
            AcceleratorBufferLayout targetLayout
    ) {
        return AcceleratorLayoutTransformDecision.denseGpuMaterialization(
                request(sourceLayout, targetLayout),
                "test materialization"
        );
    }

    private static AcceleratorLayoutTransformRequest request(
            AcceleratorBufferLayout sourceLayout,
            AcceleratorBufferLayout targetLayout
    ) {
        return new AcceleratorLayoutTransformRequest(
                "GPU_METAL",
                1,
                2,
                Operation.OpType.CONTIGUOUS,
                sourceLayout,
                targetLayout,
                null,
                false
        );
    }

    private static MetalMpsBridgeContext context() {
        return new MetalMpsBridgeContext(true, MemorySegment.ofAddress(1), "");
    }

    private static MetalBufferAllocator allocator() {
        return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                return new MetalBufferHandle(MemorySegment.ofAddress(20), byteLength, "shared", "test", false);
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
            }
        });
    }

    private static final class RecordingBridge implements MetalMpsGraphBridge {
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
        public MetalMpsBridgeContext createContext() {
            return context();
        }

        @Override
        public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
            return MetalMpsBridgeExecutable.unavailable("not used");
        }

        @Override
        public boolean supportsLayoutMaterialization() {
            return true;
        }

        @Override
        public void materializeLayout(MetalMpsBridgeContext context, MetalBufferBinding source, MetalBufferBinding destination) {
            materializeCalled.set(true);
        }

        @Override
        public backend.metal.bridge.MetalMpsBridgeExecutionStats execute(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<tensor.Tensor> externalInputs,
                List<tensor.Tensor> outputs
        ) {
            throw new UnsupportedOperationException("not used");
        }
    }
}
