package backend.cpu1.plan;

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
        if (shape == null) {
            throw new IllegalArgumentException("shape cannot be null");
        }
        shape = shape.clone();
    }

    public int[] shape() {
        return shape.clone();
    }

    public static Cpu1IterationPlan contiguous(int elementCount, int[] shape) {
        return new Cpu1IterationPlan(elementCount, shape, true);
    }
}
