package tuning.candidate;

import config.runtime.RuntimeConfig;

/**
 * Runtime-profile override applied by candidate spaces that intentionally mutate runtime policy.
 */
@FunctionalInterface
public interface RuntimeConfigOverride {
    /**
     * Returns an unchanged runtime config.
     *
     * @return identity override
     */
    static RuntimeConfigOverride identity() {
        return runtime -> runtime;
    }

    /**
     * Applies this override to a calibrated runtime config.
     *
     * @param runtime calibrated runtime config
     * @return runtime config for the generated candidate
     */
    RuntimeConfig apply(RuntimeConfig runtime);
}
