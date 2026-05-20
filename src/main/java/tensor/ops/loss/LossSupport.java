package tensor.ops.loss;

import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class LossSupport {
    private LossSupport() {
    }

    static List<Tensor> asInputs(Tensor first, Tensor... rest) {
        List<Tensor> inputs = new ArrayList<>(1 + rest.length);
        inputs.add(first);
        for (Tensor tensor : rest) {
            inputs.add(tensor);
        }
        return List.copyOf(inputs);
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

    static Tensor applyLossReduction(Tensor perSampleLoss, Tensor reductionWeights, LossReduction reduction) {
        return switch (reduction) {
            case NONE -> perSampleLoss;
            case SUM -> perSampleLoss.sum();
            case MEAN -> {
                if (reductionWeights == null) {
                    yield perSampleLoss.mean();
                }
                Tensor validCount = reductionWeights.sum();
                Tensor totalLoss = perSampleLoss.sum();
                yield totalLoss.div(validCount.clampMin(1.0));
            }
        };
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

    static Tensor buildIgnoreMask(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        byte[] mask = new byte[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            mask[i] = value == ignoreIndex ? (byte) 0 : (byte) 1;
        }
        return new Tensor(mask, targetIndices.getShape().clone(), null, "index_valid_mask", DataType.BOOL);
    }

    static Tensor buildSafeIndices(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        if (targetIndices.getDataType() == DataType.INT32) {
            int[] safe = new int[size];
            for (int i = 0; i < size; i++) {
                long value = readIntegralIndex(targetIndices, i);
                safe[i] = value == ignoreIndex ? 0 : (int) value;
            }
            return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", DataType.INT32);
        }
        if (targetIndices.getDataType() == DataType.INT64) {
            long[] safe = new long[size];
            for (int i = 0; i < size; i++) {
                long value = readIntegralIndex(targetIndices, i);
                safe[i] = value == ignoreIndex ? 0L : value;
            }
            return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", DataType.INT64);
        }
        double[] safe = new double[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            safe[i] = value == ignoreIndex ? 0.0 : (double) value;
        }
        return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", targetIndices.getDataType());
    }

    static Tensor alignSampleMask(Tensor mask, int[] sampleShape, String opName) {
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

    private static long readIntegralIndex(Tensor indices, int flatIndex) {
        if (indices.getDataType() == DataType.INT32 || indices.getDataType() == DataType.INT64) {
            return indices.getIntegralByFlatIndex(flatIndex);
        }
        double raw = indices.getByFlatIndex(flatIndex);
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value.");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return integral;
    }
}
