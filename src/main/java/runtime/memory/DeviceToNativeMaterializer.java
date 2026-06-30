package runtime.memory;

import runtime.contract.CpuMaterializationReason;
import runtime.device.buffer.DeviceBufferBinding;

import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

/**
 * Backend hook that synchronizes a device-current runtime value directly into native CPU storage.
 */
public interface DeviceToNativeMaterializer {
    /**
     * Returns whether this materializer can synchronize the supplied binding into native CPU storage.
     *
     * @param binding active device buffer binding for the compiled node
     * @param target runtime tensor whose logical layout must match the binding
     * @param nativeStorage destination native CPU storage
     * @param reason reason native CPU-readable storage is required
     * @return true when direct device-to-native materialization is supported
     */
    default boolean supports(
            DeviceBufferBinding binding,
            Tensor target,
            NativeTensorStorage nativeStorage,
            CpuMaterializationReason reason
    ) {
        return binding != null && binding.available() && target != null && nativeStorage != null;
    }

    /**
     * Synchronizes the device value into native CPU storage.
     *
     * @param binding active device buffer binding for the compiled node
     * @param target runtime tensor whose logical layout must match the binding
     * @param nativeStorage destination native CPU storage
     * @param reason reason native CPU-readable storage is required
     * @return timing and diagnostic result; {@code null} is treated as an unmeasured successful sync
     */
    CpuMaterializationResult materialize(
            DeviceBufferBinding binding,
            Tensor target,
            NativeTensorStorage nativeStorage,
            CpuMaterializationReason reason
    );
}
