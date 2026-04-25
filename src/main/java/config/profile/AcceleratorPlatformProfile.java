package config.profile;

import config.runtime.AcceleratorConfig;

import java.util.Objects;

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

    public static AcceleratorPlatformProfile defaults() {
        return fromRuntimeConfig(AcceleratorConfig.defaults());
    }

    public static AcceleratorPlatformProfile fromRuntimeConfig(AcceleratorConfig config) {
        AcceleratorConfig resolved = config == null ? AcceleratorConfig.defaults() : config;
        return new AcceleratorPlatformProfile(
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.cuda()),
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.opencl()),
                AcceleratorBackendPlatformProfile.fromRuntimeConfig(resolved.metal())
        );
    }

    public AcceleratorConfig toRuntimeConfig() {
        return new AcceleratorConfig(
                cuda.toRuntimeConfig(),
                opencl.toRuntimeConfig(),
                metal.toRuntimeConfig()
        );
    }
}
