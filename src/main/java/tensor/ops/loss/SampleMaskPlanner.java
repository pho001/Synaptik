package tensor.ops.loss;

import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

final class SampleMaskPlanner {
    private SampleMaskPlanner() {
    }

    static Tensor align(Tensor mask, int[] sampleShape, String opName) {
        if (mask == null) {
            throw new IllegalArgumentException(opName + " mask cannot be null");
        }
        if (mask.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException(opName + " mask must have BOOL dtype.");
        }
        int[] maskShape = mask.getShapeUnsafe();
        if (maskShape.length > sampleShape.length) {
            throw new IllegalArgumentException(opName + " mask rank cannot exceed sample rank.");
        }
        for (int[] candidate : sampleMaskCandidates(maskShape, sampleShape.length)) {
            try {
                Tensor reshaped = Arrays.equals(candidate, maskShape) ? mask : mask.reshape(candidate);
                return reshaped.expand(sampleShape);
            } catch (IllegalArgumentException ignored) {
                // Try the next placement candidate.
            }
        }
        throw new IllegalArgumentException(opName + " mask shape " + Arrays.toString(maskShape)
                + " is not broadcastable to sample shape " + Arrays.toString(sampleShape) + ".");
    }

    private static int[][] sampleMaskCandidates(int[] maskShape, int targetRank) {
        if (maskShape.length == targetRank) {
            return new int[][]{maskShape.clone()};
        }
        int[] append = new int[targetRank];
        Arrays.fill(append, 1);
        System.arraycopy(maskShape, 0, append, 0, maskShape.length);

        int[] prepend = new int[targetRank];
        Arrays.fill(prepend, 1);
        System.arraycopy(maskShape, 0, prepend, targetRank - maskShape.length, maskShape.length);
        return new int[][]{append, prepend};
    }
}
