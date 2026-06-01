package backend.cpu1.plan;

import java.util.Objects;

/**
 * Prepare-time logical iteration metadata for one cpu1 unit.
 */
public record Cpu1IterationPlan(
        int elementCount,
        int[] shape,
        boolean contiguousFastPath
) {
    public Cpu1IterationPlan {
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount cannot be negative");
        }
        shape = Objects.requireNonNull(shape, "shape cannot be null").clone();
    }

    public int[] shape() {
        return shape.clone();
    }

    public static Cpu1IterationPlan contiguous(int elementCount, int[] shape) {
        return new Cpu1IterationPlan(elementCount, shape, true);
    }
}
