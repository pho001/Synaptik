package backend.cpu1.offset;

import backend.cpu1.exec.Cpu1TensorView;

import java.util.Objects;

/**
 * Generic rank-N strided offset plan.
 */
public final class Cpu1GenericOffsetPlan {
    private final int baseOffset;
    private final int[] shape;
    private final int[] strides;
    private final int elementCount;

    private Cpu1GenericOffsetPlan(int baseOffset, int[] shape, int[] strides, int elementCount) {
        Cpu1OffsetPlanValidation.requireNonNegative(baseOffset, "baseOffset");
        Cpu1OffsetPlanValidation.requireMatchingRank(shape, strides);
        Cpu1OffsetPlanValidation.requireNonNegative(elementCount, "elementCount");
        this.baseOffset = baseOffset;
        this.shape = Objects.requireNonNull(shape, "shape cannot be null").clone();
        this.strides = Objects.requireNonNull(strides, "strides cannot be null").clone();
        this.elementCount = elementCount;
        int expectedCount = 1;
        for (int dim = 0; dim < this.shape.length; dim++) {
            expectedCount *= Cpu1OffsetPlanValidation.requirePositiveExtent(this.shape[dim], dim);
        }
        Cpu1OffsetPlanValidation.requireElementCount(expectedCount, elementCount);
    }

    public static Cpu1GenericOffsetPlan forView(Cpu1TensorView view) {
        Objects.requireNonNull(view, "view cannot be null");
        return new Cpu1GenericOffsetPlan(
                view.storageOffset(),
                view.shape(),
                view.strides(),
                view.elementCount()
        );
    }

    /**
     * Fast offset mapping. Callers must pass a valid linear index from the launch range.
     */
    public int offset(int linearIndex) {
        int offset = baseOffset;
        int remainder = linearIndex;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int coordinate = remainder % shape[dim];
            remainder /= shape[dim];
            offset += coordinate * strides[dim];
        }
        return offset;
    }

    public int elementCount() {
        return elementCount;
    }

    public int checkedOffset(int linearIndex) {
        Cpu1OffsetPlanValidation.requireInRange(linearIndex, elementCount);
        return offset(linearIndex);
    }
}
