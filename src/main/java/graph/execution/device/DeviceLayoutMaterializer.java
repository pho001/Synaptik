package graph.execution.device;

import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.memory.DeviceBufferBinding;
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
