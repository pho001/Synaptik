package backend.accelerator.buffer;

import backend.memory.DeviceBufferBinding;

import java.util.List;

/**
 * Typed backend buffer bindings selected for one accelerator execution.
 */
public record AcceleratorBufferBindings<B extends DeviceBufferBinding>(
        List<B> inputs,
        List<B> outputs
) {
    public AcceleratorBufferBindings {
        inputs = List.copyOf(inputs == null ? List.of() : inputs);
        outputs = List.copyOf(outputs == null ? List.of() : outputs);
    }
}
