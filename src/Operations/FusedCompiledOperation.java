package Operations;

import Tensor.Tensor;

import java.util.List;

public interface FusedCompiledOperation {
    void applyRangeScalar(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive);

    default void applyRangeVector(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive) {
        applyRangeScalar(inputs, out, startInclusive, endExclusive);
    }
}
