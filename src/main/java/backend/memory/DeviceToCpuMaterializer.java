package backend.memory;

import tensor.Tensor;

/**
 * Backend hook that synchronizes a device-current runtime value into CPU-readable tensor storage.
 *
 * <p>This is deliberately a per-run execution hook, not a global tensor feature. A materializer is
 * registered on {@code ExecutionState} or {@code ExecutionContext} for one backend id such as
 * {@code GPU_METAL}. When execution later needs CPU storage for a device-current node, the state passes
 * the runtime tensor and the active {@link DeviceBufferBinding} to the registered materializer.</p>
 *
 * <p>Implementations must update {@code target}'s CPU-visible storage before returning. Returning only a
 * successful result without copying bytes is a correctness bug, because the execution state will mark CPU
 * storage current after this method completes.</p>
 */
public interface DeviceToCpuMaterializer {
    /**
     * Returns whether this materializer can synchronize the supplied binding into the target tensor.
     *
     * @param binding active device buffer binding for the compiled node
     * @param target runtime tensor whose CPU storage must be updated
     * @param reason reason CPU-readable storage is required
     * @return true when {@link #materialize(DeviceBufferBinding, Tensor, CpuMaterializationReason)} can be called
     */
    default boolean supports(DeviceBufferBinding binding, Tensor target, CpuMaterializationReason reason) {
        return binding != null && binding.available() && target != null;
    }

    /**
     * Synchronizes the device value into the runtime tensor's CPU-visible storage.
     *
     * @param binding active device buffer binding for the compiled node
     * @param target runtime tensor whose CPU storage must be updated
     * @param reason reason CPU-readable storage is required
     * @return timing and diagnostic result; {@code null} is treated as an unmeasured successful sync
     */
    CpuMaterializationResult materialize(
            DeviceBufferBinding binding,
            Tensor target,
            CpuMaterializationReason reason
    );
}
