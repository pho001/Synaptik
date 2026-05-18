package tensor.ops.layout;

import operations.layout.tile;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code tile}.
 */
public final class TileOp {
    private TileOp() {
    }

    public static Tensor build(Tensor input, int[] repeats) {
        if (input == null) {
            throw new IllegalArgumentException("tile input cannot be null");
        }
        int rank = input.getShapeUnsafe().length;
        if (repeats == null || repeats.length != rank) {
            throw new IllegalArgumentException("tile repeats length must match input rank.");
        }
        int[] normalizedRepeats = repeats.clone();
        int[] outShape = input.getShape();
        for (int d = 0; d < rank; d++) {
            if (normalizedRepeats[d] <= 0) {
                throw new IllegalArgumentException("tile repeats must be positive.");
            }
            outShape[d] = Math.multiplyExact(outShape[d], normalizedRepeats[d]);
        }
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                outShape,
                new tile(normalizedRepeats),
                "tile",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input);
            int[] inputShape = input.getShapeUnsafe();
            int repeatCount = 1;
            for (int repeat : normalizedRepeats) {
                repeatCount = Math.multiplyExact(repeatCount, repeat);
            }
            for (int logical = 0; logical < repeatCount; logical++) {
                int tmp = logical;
                int[] starts = new int[rank];
                int[] ends = new int[rank];
                for (int d = rank - 1; d >= 0; d--) {
                    int repeatCoord = tmp % normalizedRepeats[d];
                    tmp /= normalizedRepeats[d];
                    starts[d] = repeatCoord * inputShape[d];
                    ends[d] = starts[d] + inputShape[d];
                }
                grad = grad.add(outGrad.slice(starts, ends, LayoutSupport.allAxes(rank), LayoutSupport.ones(rank)));
            }
            LayoutSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
