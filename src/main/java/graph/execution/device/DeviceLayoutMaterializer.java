package graph.execution.device;

import runtime.device.buffer.AcceleratorLayoutTransformDecision;
import runtime.device.buffer.DeviceBufferBinding;
import runtime.execution.ExecutionContext;

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
