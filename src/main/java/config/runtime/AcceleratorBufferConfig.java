package config.runtime;

import java.util.Objects;

/**
 * Per-backend native buffer-binding policy.
 *
 * <p>This config is stored under {@link AcceleratorBackendConfig}. It is deliberately shared by
 * accelerator backends, while concrete handle ownership and native ABI remain backend-specific.</p>
 *
 * @param bindingMode whether buffer bindings are disabled, automatic, or required
 * @param allowPreparedInputMaterialization whether non-contiguous external inputs may be remapped into
 *                                         prepared contiguous tensors before upload/binding
 * @param minimumEstimatedWork minimum estimated work before buffer-binding preflight is attempted
 */
public record AcceleratorBufferConfig(
        AcceleratorBufferBindingMode bindingMode,
        boolean allowPreparedInputMaterialization,
        long minimumEstimatedWork
) {
    public AcceleratorBufferConfig {
        bindingMode = Objects.requireNonNullElse(bindingMode, AcceleratorBufferBindingMode.AUTO);
        minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
    }

    /**
     * Returns the production default buffer policy.
     *
     * @return default buffer policy
     */
    public static AcceleratorBufferConfig defaults() {
        return new AcceleratorBufferConfig(AcceleratorBufferBindingMode.AUTO, true, 0L);
    }

    /**
     * Returns a policy that bypasses buffer bindings.
     *
     * @return disabled buffer policy
     */
    public static AcceleratorBufferConfig disabled() {
        return new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, true, 0L);
    }

    /**
     * Returns a copy with a different binding mode.
     *
     * @param newBindingMode replacement binding mode; {@code null} uses {@link AcceleratorBufferBindingMode#AUTO}
     * @return updated config
     */
    public AcceleratorBufferConfig withBindingMode(AcceleratorBufferBindingMode newBindingMode) {
        return new AcceleratorBufferConfig(newBindingMode, allowPreparedInputMaterialization, minimumEstimatedWork);
    }

    /**
     * Returns a copy with a different prepared-input materialization flag.
     *
     * @param newAllowPreparedInputMaterialization replacement flag
     * @return updated config
     */
    public AcceleratorBufferConfig withAllowPreparedInputMaterialization(boolean newAllowPreparedInputMaterialization) {
        return new AcceleratorBufferConfig(bindingMode, newAllowPreparedInputMaterialization, minimumEstimatedWork);
    }

    /**
     * Returns a copy with a different minimum estimated work threshold.
     *
     * @param newMinimumEstimatedWork replacement threshold; negative values are normalized to {@code 0}
     * @return updated config
     */
    public AcceleratorBufferConfig withMinimumEstimatedWork(long newMinimumEstimatedWork) {
        return new AcceleratorBufferConfig(bindingMode, allowPreparedInputMaterialization, newMinimumEstimatedWork);
    }
}
