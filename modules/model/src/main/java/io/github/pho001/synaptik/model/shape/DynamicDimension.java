package io.github.pho001.synaptik.model.shape;

import java.util.Objects;

/**
 * A dimension whose numeric size is represented by a stable symbolic name.
 *
 * <p>The symbol expresses compile-time identity only. It is not mutable runtime state and does not
 * bind a concrete size. Leading and trailing Unicode whitespace is removed before the value is
 * stored, making equality and hashing use the canonical stripped symbol.</p>
 *
 * @param symbol non-null, non-blank symbolic dimension name
 */
public record DynamicDimension(String symbol) implements Dimension {
    /**
     * Creates a symbolic dynamic dimension and canonicalizes its name with {@link String#strip()}.
     *
     * @param symbol non-null name containing at least one non-whitespace character
     * @throws NullPointerException if {@code symbol} is {@code null}
     * @throws IllegalArgumentException if the stripped symbol is blank
     */
    public DynamicDimension {
        symbol = Objects.requireNonNull(symbol, "symbol").strip();
        if (symbol.isBlank()) {
            throw new IllegalArgumentException("Dynamic dimension symbol must not be blank");
        }
    }

    /**
     * Returns the canonical symbolic name of this dimension.
     *
     * @return non-null, non-blank symbol with leading and trailing whitespace removed
     */
    public String symbol() {
        return symbol;
    }
}
