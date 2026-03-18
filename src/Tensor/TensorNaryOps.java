package Tensor;

import java.util.ArrayList;
import java.util.List;

final class TensorNaryOps {
    private TensorNaryOps() {}

    static List<Tensor> asInputs(Tensor first, Tensor... rest) {
        List<Tensor> inputs = new ArrayList<>(1 + rest.length);
        inputs.add(first);
        for (Tensor t : rest) {
            inputs.add(t);
        }
        return List.copyOf(inputs);
    }

    static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor runningMean,
            Tensor runningVar,
            double epsilon,
            boolean training
    ) {
        throw new UnsupportedOperationException(
                "BatchNorm op is not implemented yet. " +
                "Use TensorNaryOps as extension point for multi-input operations."
        );
    }
}
