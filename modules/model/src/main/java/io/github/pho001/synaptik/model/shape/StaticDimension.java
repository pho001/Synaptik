package io.github.pho001.synaptik.model.shape;

/**
 * A dimension whose non-negative numeric size is known in the model.
 *
 * <p>Size zero is valid and represents an empty axis. The value uses {@code long} so the logical
 * shape model is not constrained by Java array-size limits.</p>
 *
 * @param size non-negative logical extent of the axis
 */
public record StaticDimension(long size) implements Dimension {
    /**
     * Creates a statically sized dimension.
     *
     * @param size non-negative logical extent; zero represents an empty axis
     * @throws IllegalArgumentException if {@code size} is negative
     */
    public StaticDimension {
        if (size < 0) {
            throw new IllegalArgumentException("Static dimension size must be non-negative: " + size);
        }
    }

    /**
     * Returns the known logical extent of this dimension.
     *
     * @return non-negative size, including zero for an empty axis
     */
    public long size() {
        return size;
    }
}
