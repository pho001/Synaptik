package backend.cuda.buffer;

import runtime.device.buffer.AcceleratorBufferLayout;
import runtime.device.buffer.AcceleratorBufferLayoutClass;
import runtime.device.buffer.AcceleratorLayoutTransformDecision;
import runtime.device.buffer.AcceleratorLayoutTransformKind;
import backend.cuda.CudaDTypeRolePolicy;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaGraphBridge;
import runtime.device.buffer.DeviceBufferBinding;
import runtime.execution.ExecutionContext;
import graph.execution.device.DeviceLayoutMaterializer;
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
        if (decision.kind() == AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION) {
            throw new UnsupportedOperationException("CUDA_LAYOUT_BROADCAST_UNSUPPORTED: CUDA layout materializer "
                    + "does not implement broadcast/zero-stride GPU materialization yet.");
        }
        if (decision.kind() == AcceleratorLayoutTransformKind.STRIDED_NATIVE_COMPUTE) {
            throw new UnsupportedOperationException("CUDA_STRIDED_COMPUTE_UNSUPPORTED: CUDA layout materializer "
                    + "cannot execute arbitrary strided native compute.");
        }
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
            throw new UnsupportedOperationException("NATIVE_LAYOUT_DTYPE_UNSUPPORTED: CUDA dense layout materialization "
                    + "requires native compute/output role FLOAT32; "
                    + CudaDTypeRolePolicy.computeOutput(targetLayout.dataType()).detail());
        }
        if (targetLayout.layoutClass() != AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
            throw new UnsupportedOperationException("CUDA_LAYOUT_TARGET_UNSUPPORTED: CUDA dense layout materialization "
                    + "requires dense target layout; layoutClass=" + targetLayout.layoutClass());
        }

        CudaBufferBinding destination = allocator.createOutputBinding(decision.targetNodeId(), targetLayout);
        if (context != null) {
            context.registerResource(new CudaBufferResource(allocator, destination.handle()));
        }
        bridge.materializeLayout(bridgeContext, cudaSource, destination);
        return destination;
    }
}
