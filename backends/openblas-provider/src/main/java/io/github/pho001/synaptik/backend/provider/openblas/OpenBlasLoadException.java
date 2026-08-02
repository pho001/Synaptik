package io.github.pho001.synaptik.backend.provider.openblas;

import java.util.Objects;

/**
 * Reports failure to load or completely bind an explicitly selected OpenBLAS library.
 *
 * <p>The exception preserves the original loading, native-access, symbol-resolution, or binding
 * failure as its cause. A failure while closing a partial lookup is suppressed on that cause. The
 * exception carries no fallback or availability decision; the caller owns any policy applied
 * after this failure.
 */
public final class OpenBlasLoadException extends IllegalStateException {
    /**
     * Creates a provider loading failure for use by the package-private loading implementation.
     *
     * @param message the stable description identifying the caller-supplied library; must not be
     *                {@code null}
     * @param cause the original loading or binding failure; must not be {@code null}
     * @throws NullPointerException if either argument is {@code null}
     */
    OpenBlasLoadException(String message, Throwable cause) {
        super(Objects.requireNonNull(message, "message"), Objects.requireNonNull(cause, "cause"));
    }
}
