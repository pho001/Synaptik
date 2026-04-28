package config.profile;

import config.runtime.AcceleratorConfig;

import java.util.Objects;

/**
 * Platform-calibrated accelerator policy for all supported accelerator backend families.
 *
 * <p>This profile section is part of {@link PlatformRuntimeProfile}. It is stored per platform and
 * converted back to {@link AcceleratorConfig} when execution profiles are assembled.</p>
 *
 * @param cuda CUDA backend policy; {@code null} uses backend defaults
 * @param opencl OpenCL backend policy; {@code null} uses backend defaults
 * @param metal Metal backend policy; {@code null} uses backend defaults
 */
public record AcceleratorPlatformProfile(
        AcceleratorBackendPlatformProfile cuda,
        AcceleratorBackendPlatformProfile opencl,
        AcceleratorBackendPlatformProfile metal
) {
    public AcceleratorPlatformProfile {
        cuda = Objects.requireNonNullElse(cuda, AcceleratorBackendPlatformProfile.defaults());
        opencl = Objects.requireNonNullElse(opencl, AcceleratorBackendPlatformProfile.defaults());
        metal = Objects.requireNonNullElse(metal, AcceleratorBackendPlatformProfile.defaults());
    }

    /**
     * Creates the default accelerator platform profile from runtime defaults.
     *
     * @return default accelerator platform profile
     */
    public static AcceleratorPlatformProfile defaults() {
        return fromRuntimeConfig(AcceleratorConfig.defaults());
    }

    /**
     * Converts runtime accelerator config into a platform profile section.
     *
     * @param config runtime accelerator config; {@code null} uses runtime defaults
     * @return platform profile section
     */
    public static AcceleratorPlatformProfile fromRuntimeConfig(AcceleratorConfig config) {
        AcceleratorConfig resolved = config == null ? AcceleratorConfig.defaults() : config;
        return new AcceleratorPlatformProfile(
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.cuda()),
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.opencl()),
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.metal())
        );
    }

    /**
     * Converts this platform profile section back into runtime config.
     *
     * @return runtime accelerator config
     */
    public AcceleratorConfig toRuntimeConfig() {
        return new AcceleratorConfig(
                cuda.toRuntimeConfig(),
                opencl.toRuntimeConfig(),
                metal.toRuntimeConfig()
        );
    }
}
