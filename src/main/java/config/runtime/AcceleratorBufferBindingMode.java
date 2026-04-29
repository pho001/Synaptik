package config.runtime;

/**
 * Runtime policy for accelerator native buffer-binding execution.
 *
 * <p>The mode controls whether an accelerator prepared executable may use backend-owned/native buffers
 * instead of the tensor-array bridge. The enum is backend-neutral: Metal uses it today and CUDA can
 * adopt the same contract when a CUDA buffer ABI exists.</p>
 */
public enum AcceleratorBufferBindingMode {
    /**
     * Never attempt native buffer bindings.
     */
    OFF,

    /**
     * Attempt native buffer bindings only when preflight says the path is legal.
     */
    AUTO,

    /**
     * Require native buffer bindings and fail instead of silently falling back.
     */
    REQUIRE
}
