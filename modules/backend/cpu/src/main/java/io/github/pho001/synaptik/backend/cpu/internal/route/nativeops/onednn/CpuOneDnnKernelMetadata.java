package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import java.util.Objects;

/**
 * Cold descriptor facts safe to retain before shared assignment; contains no native resource.
 * For a fixed pipeline the single allocation includes aligned literal intermediates/constants
 * and each atomic operation's user scratch exactly once; invocation controls are not data scratch.
 *
 * @param scratchpadBytes non-negative exact complete invocation-data allocation in bytes
 * @param implementation non-null non-blank copied atomic or fixed-sequence implementation identity
 */
public record CpuOneDnnKernelMetadata(long scratchpadBytes, String implementation) {
    /**
     * Validates copied descriptor facts without acquiring a library lease.
     *
     * @param scratchpadBytes non-negative complete invocation-data byte count
     * @param implementation non-null non-blank implementation identity
     * @throws NullPointerException if implementation is null
     * @throws IllegalArgumentException if the byte count is negative or identity blank
     */
    public CpuOneDnnKernelMetadata {
        Objects.requireNonNull(implementation, "implementation");
        if (scratchpadBytes < 0 || implementation.isBlank()) {
            throw new IllegalArgumentException("invalid oneDNN descriptor metadata");
        }
    }
}
