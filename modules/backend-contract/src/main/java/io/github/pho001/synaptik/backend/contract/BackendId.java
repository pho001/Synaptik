package io.github.pho001.synaptik.backend.contract;

/**
 * Identifies a backend ownership domain through an open, backend-defined string value.
 *
 * <p>Names such as {@code "cpu"}, {@code "metal"}, and {@code "cuda"} are examples rather
 * than a closed vocabulary. The exact nonblank {@link String} reference supplied by the caller is
 * retained without trimming, case folding, Unicode normalization, syntax validation, interning,
 * or alias resolution. Consequently, equality, hashing, and diagnostic text follow ordinary
 * record semantics over that exact string content.</p>
 *
 * <p>This immutable identity can describe compile-time backend ownership. It does not register,
 * discover, locate, or prove the availability or capability of a backend and provides no live
 * backend service or execution behavior.</p>
 *
 * @param value open nonblank backend identity value to retain by reference without normalization;
 *     must not be {@code null}
 */
public record BackendId(String value) {
    /**
     * Creates a backend identity from an exact caller-supplied string.
     *
     * @param value open nonblank backend identity value to retain by reference without
     *     normalization; must not be {@code null}
     * @throws NullPointerException if {@code value} is {@code null}; the exception message is
     *     {@code value}
     * @throws IllegalArgumentException if {@code value} is blank according to
     *     {@link String#isBlank()}; the exception message is {@code value must not be blank}
     */
    public BackendId(String value) {
        if (value == null) {
            throw new NullPointerException("value");
        }
        if (value.isBlank()) {
            throw new IllegalArgumentException("value must not be blank");
        }
        this.value = value;
    }

    /**
     * Returns the exact backend identity string supplied at construction.
     *
     * @return the stored nonblank string by the same reference supplied by the caller; never
     *     {@code null}
     */
    public String value() {
        return value;
    }
}
