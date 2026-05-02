package backend.cuda.buffer;

import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.accelerator.buffer.AcceleratorLayoutTransformKind;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaGraphBridge;
import backend.memory.DeviceBufferBinding;
import backend.runtime.ExecutionContext;
import graph.execution.DeviceLayoutMaterializer;
import tensor.DataType;

import java.util.Objects;

/**
 * CUDA implementation of dense accelerator-side layout materialization.
 */
public final class CudaDeviceLayoutMaterializer implements DeviceLayoutMaterializer {
    private final CudaGraphBridge bridge;
    private final CudaBridgeContext bridgeContext;
    private final CudaBufferAllocator allocator;

    public CudaDeviceLayoutMaterializer(
            CudaGraphBridge bridge,
            CudaBridgeContext bridgeContext,
            CudaBufferAllocator allocator
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
        if (decision.kind() != AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION) {
            throw new UnsupportedOperationException("CUDA layout materializer requires DENSE_GPU_MATERIALIZATION, got "
                    + decision.kind());
        }
        if (!(source instanceof CudaBufferBinding cudaSource)) {
            throw new UnsupportedOperationException("CUDA layout materializer requires a CUDA source binding, got "
                    + (source == null ? "null" : source.getClass().getName()));
        }
        if (!bridge.supportsLayoutMaterialization()) {
            throw new UnsupportedOperationException("CUDA bridge does not support dense layout materialization.");
        }
        if (!allocator.available()) {
            throw new UnsupportedOperationException(allocator.unavailableReason().isBlank()
                    ? "CUDA buffer allocator is unavailable."
                    : allocator.unavailableReason());
        }

        AcceleratorBufferLayout targetLayout = decision.targetLayout();
        if (targetLayout.dataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("CUDA dense layout materialization supports FLOAT32 only; got "
                    + targetLayout.dataType());
        }
        if (targetLayout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            throw new UnsupportedOperationException("CUDA dense layout materialization requires dense target layout; got "
                    + targetLayout.layoutClass());
        }

        CudaBufferBinding destination = allocator.createOutputBinding(decision.targetNodeId(), targetLayout);
        if (context != null) {
            context.registerResource(new CudaBufferResource(allocator, destination.handle()));
        }
        bridge.materializeLayout(bridgeContext, cudaSource, destination);
        return destination;
    }
}
