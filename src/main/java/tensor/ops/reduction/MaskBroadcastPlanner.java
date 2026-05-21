package tensor.ops.reduction;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

final class MaskBroadcastPlanner {
    private MaskBroadcastPlanner() {
    }

    static Tensor alignToShape(Tensor mask, int[] targetShape, int preferredAxis, String opName) {
        if (mask == null) {
            throw new IllegalArgumentException(opName + " mask cannot be null");
        }
        if (mask.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException(opName + " mask must have BOOL dtype.");
        }
        int[] maskShape = mask.getShapeUnsafe();
        if (maskShape.length > targetShape.length) {
            throw new IllegalArgumentException(opName + " mask rank cannot exceed input rank.");
        }
        for (int[] candidate : maskBroadcastCandidates(maskShape, targetShape, preferredAxis)) {
            try {
                Tensor reshaped = Arrays.equals(maskShape, candidate) ? mask : mask.reshape(candidate);
                return reshaped.expand(targetShape);
            } catch (IllegalArgumentException ignored) {
                // Try the next legal placement candidate.
            }
        }
        throw new IllegalArgumentException(opName + " mask shape " + Arrays.toString(maskShape)
                + " is not broadcastable to input shape " + Arrays.toString(targetShape) + ".");
    }

    private static List<int[]> maskBroadcastCandidates(int[] maskShape, int[] targetShape, int preferredAxis) {
        List<MaskCandidate> candidates = new ArrayList<>();
        addMaskCandidate(candidates, normalizeCandidate(maskShape, targetShape.length), maskShape, targetShape, preferredAxis);
        if (maskShape.length < targetShape.length) {
            int[] append = new int[targetShape.length];
            Arrays.fill(append, 1);
            System.arraycopy(maskShape, 0, append, 0, maskShape.length);
            addMaskCandidate(candidates, append, maskShape, targetShape, preferredAxis);

            int[] prepend = new int[targetShape.length];
            Arrays.fill(prepend, 1);
            System.arraycopy(maskShape, 0, prepend, targetShape.length - maskShape.length, maskShape.length);
            addMaskCandidate(candidates, prepend, maskShape, targetShape, preferredAxis);
        }
        placeMaskDims(candidates, new int[targetShape.length], maskShape, targetShape, preferredAxis, 0, 0, new int[maskShape.length]);
        candidates.sort(Comparator.comparingInt(MaskCandidate::score));
        List<int[]> out = new ArrayList<>(candidates.size());
        for (MaskCandidate candidate : candidates) {
            out.add(candidate.shape());
        }
        return out;
    }

    private static int[] normalizeCandidate(int[] shape, int rank) {
        if (shape.length == rank) {
            return shape.clone();
        }
        int[] out = new int[rank];
        Arrays.fill(out, 1);
        System.arraycopy(shape, 0, out, rank - shape.length, shape.length);
        return out;
    }

    private static void placeMaskDims(
            List<MaskCandidate> candidates,
            int[] workingShape,
            int[] maskShape,
            int[] targetShape,
            int preferredAxis,
            int maskIndex,
            int targetStart,
            int[] positions
    ) {
        if (maskIndex == 0) {
            Arrays.fill(workingShape, 1);
        }
        if (maskIndex == maskShape.length) {
            addMaskCandidate(candidates, workingShape, maskShape, targetShape, preferredAxis, positions.clone());
            return;
        }
        int remaining = maskShape.length - maskIndex - 1;
        for (int targetIndex = targetStart; targetIndex < targetShape.length - remaining; targetIndex++) {
            int dim = maskShape[maskIndex];
            if (dim != 1 && dim != targetShape[targetIndex]) {
                continue;
            }
            int old = workingShape[targetIndex];
            workingShape[targetIndex] = dim;
            positions[maskIndex] = targetIndex;
            placeMaskDims(candidates, workingShape, maskShape, targetShape, preferredAxis, maskIndex + 1, targetIndex + 1, positions);
            workingShape[targetIndex] = old;
        }
    }

    private static void addMaskCandidate(
            List<MaskCandidate> candidates,
            int[] shape,
            int[] maskShape,
            int[] targetShape,
            int preferredAxis
    ) {
        int[] positions = new int[maskShape.length];
        Arrays.fill(positions, -1);
        int source = 0;
        for (int i = 0; i < shape.length && source < maskShape.length; i++) {
            if (shape[i] == maskShape[source]) {
                positions[source++] = i;
            }
        }
        addMaskCandidate(candidates, shape, maskShape, targetShape, preferredAxis, positions);
    }

    private static void addMaskCandidate(
            List<MaskCandidate> candidates,
            int[] shape,
            int[] maskShape,
            int[] targetShape,
            int preferredAxis,
            int[] positions
    ) {
        for (int i = 0; i < shape.length; i++) {
            int dim = shape[i];
            if (dim != 1 && dim != targetShape[i]) {
                return;
            }
        }
        for (MaskCandidate existing : candidates) {
            if (Arrays.equals(existing.shape(), shape)) {
                return;
            }
        }
        candidates.add(new MaskCandidate(shape.clone(), maskCandidateScore(positions, preferredAxis)));
    }

    private static int maskCandidateScore(int[] positions, int preferredAxis) {
        int score = 0;
        boolean coversPreferredAxis = false;
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] < 0) {
                score += 100;
                continue;
            }
            if (positions[i] == preferredAxis) {
                coversPreferredAxis = true;
            }
            score += Math.abs(positions[i] - i);
        }
        return coversPreferredAxis ? score : score + 50;
    }

    private record MaskCandidate(int[] shape, int score) {
    }
}
