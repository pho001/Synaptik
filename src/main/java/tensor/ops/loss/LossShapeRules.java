package tensor.ops.loss;

import tensor.Tensor;

final class LossShapeRules {
    private LossShapeRules() {
    }

    static int sampleCount(int[] shape, int classDimension) {
        int count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (i != classDimension) {
                count *= shape[i];
            }
        }
        return Math.max(1, count);
    }

    static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    static void validateClassWeightsShape(Tensor classWeights, int expectedClasses) {
        int[] shape = classWeights.getShape();
        if (shape.length != 1 || shape[0] != expectedClasses) {
            throw new IllegalArgumentException("classWeights must have shape [" + expectedClasses + "].");
        }
    }
}
