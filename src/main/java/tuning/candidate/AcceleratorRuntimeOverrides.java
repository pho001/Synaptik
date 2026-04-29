package tuning.candidate;

import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorConfig;
import config.runtime.RuntimeConfig;

/**
 * Runtime overrides shared by candidate spaces that tune accelerator execution policy.
 */
public final class AcceleratorRuntimeOverrides {
    private AcceleratorRuntimeOverrides() {
    }

    /**
     * Creates an override that applies one buffer-binding mode to all accelerator backend configs.
     *
     * @param modeName enum name from {@link AcceleratorBufferBindingMode}
     * @return runtime override
     */
    public static RuntimeConfigOverride bufferBindingMode(String modeName) {
        AcceleratorBufferBindingMode mode = AcceleratorBufferBindingMode.valueOf(modeName);
        return runtime -> withAcceleratorBufferMode(runtime, mode);
    }

    private static RuntimeConfig withAcceleratorBufferMode(RuntimeConfig runtime, AcceleratorBufferBindingMode mode) {
        AcceleratorConfig accelerator = runtime.accelerator();
        return runtime.withAccelerator(new AcceleratorConfig(
                withBufferMode(accelerator.cuda(), mode),
                withBufferMode(accelerator.opencl(), mode),
                withBufferMode(accelerator.metal(), mode)
        ));
    }

    private static AcceleratorBackendConfig withBufferMode(
            AcceleratorBackendConfig backend,
            AcceleratorBufferBindingMode mode
    ) {
        return backend.withBuffer(backend.buffer().withBindingMode(mode));
    }
}
