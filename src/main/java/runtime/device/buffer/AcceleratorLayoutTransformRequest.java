package runtime.device.buffer;

import operations.Operation;

import java.util.Objects;

/**
 * Backend-neutral request describing a layout/view transform between compiled graph nodes.
 */
public record AcceleratorLayoutTransformRequest(
        String backendId,
        int sourceNodeId,
        int targetNodeId,
        Operation.OpType opType,
        AcceleratorBufferLayout sourceLayout,
        AcceleratorBufferLayout targetLayout,
        DeviceBufferBinding sourceBinding,
        boolean requiresBackward
) {
    public AcceleratorLayoutTransformRequest {
        backendId = requireNonBlank(backendId, "backendId");
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(sourceLayout, "sourceLayout cannot be null");
        Objects.requireNonNull(targetLayout, "targetLayout cannot be null");
    }

    private static String requireNonBlank(String value, String label) {
        Objects.requireNonNull(value, label + " cannot be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(label + " cannot be blank");
        }
        return value;
    }
}
