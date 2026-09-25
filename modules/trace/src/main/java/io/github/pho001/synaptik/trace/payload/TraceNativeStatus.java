package io.github.pho001.synaptik.trace.payload;

import java.util.Objects;

/**
 * Retains one exact backend-stream-local native status code and its trace-owned meaning.
 *
 * <p>The signed code is preserved without imposing a global numeric mapping across backends.
 * Success is represented only by kind {@link TraceNativeStatusKind#SUCCESS} with code {@code 0};
 * every nonzero code uses a non-success kind, including {@link TraceNativeStatusKind#UNKNOWN} when
 * the producer does not recognize it. Equality, hashing, and diagnostic text use ordinary record
 * component semantics.</p>
 *
 * @param kind non-null trace-owned interpretation of the exact native status
 * @param code exact signed native status code scoped by the producing backend stream
 */
public record TraceNativeStatus(TraceNativeStatusKind kind, int code) {
    /**
     * Validates the success/code invariant and retains both components unchanged.
     *
     * @param kind non-null status interpretation
     * @param code exact signed native status code
     * @throws NullPointerException if {@code kind} is {@code null}
     * @throws IllegalArgumentException if {@code SUCCESS} has a nonzero code or code {@code 0} has
     *     a non-success kind
     */
    public TraceNativeStatus {
        Objects.requireNonNull(kind, "kind");
        if (kind == TraceNativeStatusKind.SUCCESS && code != 0) {
            throw new IllegalArgumentException("SUCCESS requires code 0");
        }
        if (kind != TraceNativeStatusKind.SUCCESS && code == 0) {
            throw new IllegalArgumentException("code 0 requires SUCCESS");
        }
    }
}
