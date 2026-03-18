package Tensor;

import Operations.Operation;
import Operations.contiguous;

import java.util.List;

final class TensorLayoutOps {
    private TensorLayoutOps() {}

    static Tensor contiguous(Tensor input) {
        Operation op = new contiguous();
        return new Tensor(input.getShape(), List.of(input), op, "contiguous");
    }
}
