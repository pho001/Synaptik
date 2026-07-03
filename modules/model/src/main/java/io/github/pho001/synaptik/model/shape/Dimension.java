package io.github.pho001.synaptik.model.shape;

import java.util.Optional;
import java.util.OptionalLong;

/**
 * Represents one immutable axis extent in a {@link Shape}.
 *
 * <p>A dimension is either a known non-negative {@link StaticDimension} or a named symbolic
 * {@link DynamicDimension}. Dynamic dimensions are explicit values rather than negative numeric
 * sentinels, so static validation never needs to reinterpret a numeric size.</p>
 */
public sealed interface Dimension permits StaticDimension, DynamicDimension {
    /**
     * Reports whether this dimension has a statically known numeric size.
     *
     * @return {@code true} for {@link StaticDimension}; {@code false} for
     *     {@link DynamicDimension}
     */
    default boolean isStatic() {
        return this instanceof StaticDimension;
    }

    /**
     * Reports whether this dimension is identified by a symbolic name.
     *
     * @return {@code true} for {@link DynamicDimension}; {@code false} for
     *     {@link StaticDimension}
     */
    default boolean isDynamic() {
        return this instanceof DynamicDimension;
    }

    /**
     * Returns the known numeric size when this is a static dimension.
     *
     * @return a present non-negative size for {@link StaticDimension}, or an empty optional for
     *     {@link DynamicDimension}; never {@code null}
     */
    default OptionalLong staticSize() {
        if (this instanceof StaticDimension staticDimension) {
            return OptionalLong.of(staticDimension.size());
        }
        return OptionalLong.empty();
    }

    /**
     * Returns the canonical symbol when this is a dynamic dimension.
     *
     * @return a present non-blank symbol for {@link DynamicDimension}, or an empty optional for
     *     {@link StaticDimension}; never {@code null}
     */
    default Optional<String> dynamicSymbol() {
        if (this instanceof DynamicDimension dynamicDimension) {
            return Optional.of(dynamicDimension.symbol());
        }
        return Optional.empty();
    }
}
