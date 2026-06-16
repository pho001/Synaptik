package backend.cpu1.offset;

final class Cpu1OffsetPlanValidation {
    private Cpu1OffsetPlanValidation() {
    }

    static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " cannot be negative");
        }
    }

    static void requireMatchingRank(int[] shape, int[] strides) {
        if (shape == null) {
            throw new IllegalArgumentException("shape cannot be null");
        }
        if (strides == null) {
            throw new IllegalArgumentException("strides cannot be null");
        }
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have matching rank");
        }
    }

    static int requirePositiveExtent(int extent, int dim) {
        if (extent <= 0) {
            throw new IllegalArgumentException("Invalid non-positive extent at dim=" + dim + ": " + extent);
        }
        return extent;
    }

    static void requireElementCount(int expectedCount, int elementCount) {
        if (expectedCount != elementCount) {
            throw new IllegalArgumentException("shape element count " + expectedCount
                    + " does not match elementCount " + elementCount);
        }
    }

    static void requireInRange(int linearIndex, int elementCount) {
        if (linearIndex < 0 || linearIndex >= elementCount) {
            throw new IndexOutOfBoundsException("linearIndex=" + linearIndex
                    + " is outside [0, " + elementCount + ")");
        }
    }
}
