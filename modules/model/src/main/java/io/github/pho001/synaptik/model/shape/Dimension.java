package io.github.pho001.synaptik.model.shape;

import java.util.Optional;
import java.util.OptionalLong;

/**
 * Represents one immutable axis extent in a {@link Shape}.
 *
 * <p>A dimension is either a known non-negative {@link StaticDimension}, a named symbolic
 * {@link DynamicDimension}, or an {@link ExpressionDimension} that retains a derived formula or
 * constrained unknown extent. Dynamic dimensions are explicit values rather than negative numeric
 * sentinels, so static validation never needs to reinterpret a numeric size.</p>
 */
public sealed interface Dimension permits StaticDimension, DynamicDimension, ExpressionDimension {
    /**
     * Reports whether this dimension has a statically known numeric size.
     *
     * @return {@code true} for {@link StaticDimension}; {@code false} for
     *     {@link DynamicDimension} and {@link ExpressionDimension}
     */
    default boolean isStatic() {
        return this instanceof StaticDimension;
    }

    /**
     * Reports whether this dimension lacks a statically known numeric size.
     *
     * @return {@code true} for {@link DynamicDimension} and {@link ExpressionDimension};
     *     {@code false} for {@link StaticDimension}
     */
    default boolean isDynamic() {
        return !isStatic();
    }

    /**
     * Returns the known numeric size when this is a static dimension.
     *
     * @return a present non-negative size for {@link StaticDimension}, or an empty optional for
     *     {@link DynamicDimension} or {@link ExpressionDimension}; never {@code null}
     */
    default OptionalLong staticSize() {
        if (this instanceof StaticDimension staticDimension) {
            return OptionalLong.of(staticDimension.size());
        }
        return OptionalLong.empty();
    }

    /**
     * Returns the canonical symbol when this is a named dynamic dimension.
     *
     * @return a present non-blank symbol for {@link DynamicDimension}, or an empty optional for
     *     {@link StaticDimension} or {@link ExpressionDimension}; never {@code null}
     */
    default Optional<String> dynamicSymbol() {
        if (this instanceof DynamicDimension dynamicDimension) {
            return Optional.of(dynamicDimension.symbol());
        }
        return Optional.empty();
    }
}
