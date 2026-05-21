package tensor.ops.loss;

import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

final class LossInputList {
    private LossInputList() {
    }

    static List<Tensor> of(Tensor first, Tensor... rest) {
        List<Tensor> inputs = new ArrayList<>(1 + rest.length);
        inputs.add(first);
        for (Tensor tensor : rest) {
            inputs.add(tensor);
        }
        return List.copyOf(inputs);
    }
}
