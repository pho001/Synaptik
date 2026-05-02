package backend.metal.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformKind;
import backend.memory.DeviceBufferBinding;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import graph.execution.DeviceLayoutMaterializer;
import tensor.DataType;

import java.util.Objects;

/**
 * Metal implementation of dense accelerator-side layout materialization.
 */
public final class MetalDeviceLayoutMaterializer implements DeviceLayoutMaterializer {
    private final MetalMpsGraphBridge bridge;
    private final MetalMpsBridgeContext bridgeContext;
    private final MetalBufferAllocator allocator;

    public MetalDeviceLayoutMaterializer(
            MetalMpsGraphBridge bridge,
            MetalMpsBridgeContext bridgeContext,
            MetalBufferAllocator allocator
    ) {
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = Objects.requireNonNull(bridgeContext, "bridgeContext cannot be null");
        this.allocator = Objects.requireNonNull(allocator, "allocator cannot be null");
    }

    @Override
    public DeviceBufferBinding materialize(
            AcceleratorLayoutTransformDecision decision,
            DeviceBufferBinding source,
            ExecutionContext context
    ) {
        Objects.requireNonNull(decision, "decision cannot be null");
        if (!isSupportedMaterialization(decision.kind())) {
            throw new UnsupportedOperationException("Metal layout materializer requires a GPU materialization route, got "
                    + decision.kind());
        }
        if (!(source instanceof MetalBufferBinding metalSource)) {
            throw new UnsupportedOperationException("Metal layout materializer requires a Metal source binding, got "
                    + (source == null ? "null" : source.getClass().getName()));
        }
        if (!bridge.supportsLayoutMaterialization()) {
            throw new UnsupportedOperationException("Metal bridge does not support dense layout materialization.");
        }
        if (!allocator.available()) {
            throw new UnsupportedOperationException(allocator.unavailableReason().isBlank()
                    ? "Metal buffer allocator is unavailable."
                    : allocator.unavailableReason());
        }

        AcceleratorBufferLayout targetLayout = decision.targetLayout();
        if (targetLayout.dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("Metal dense layout materialization supports FLOAT32 only; got "
                    + targetLayout.dataType());
        }
        if (targetLayout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            throw new UnsupportedOperationException("Metal dense layout materialization requires dense target layout; got "
                    + targetLayout.layoutClass());
        }

        MetalBufferBinding destination = allocator.createOutputBinding(decision.targetNodeId(), targetLayout);
        if (context != null) {
            context.registerResource(new MetalBufferResource(allocator, destination.handle()));
        }
        bridge.materializeLayout(bridgeContext, metalSource, destination);
        return destination;
    }

    private static boolean isSupportedMaterialization(AcceleratorLayoutTransformKind kind) {
        return kind == AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                || kind == AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION;
    }
}
