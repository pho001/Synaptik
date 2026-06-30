package graph.execution.device;

import runtime.device.buffer.AcceleratorLayoutTransformDecision;
import runtime.device.buffer.DeviceBufferBinding;
import backend.runtime.ExecutionContext;

/**
 * Run-scoped backend service for dense GPU layout materialization.
 */
public interface DeviceLayoutMaterializer {
    DeviceBufferBinding materialize(
            AcceleratorLayoutTransformDecision decision,
            DeviceBufferBinding source,
            ExecutionContext context
    );
}
