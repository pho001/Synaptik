package Operations;
import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;

public interface Operation {
    boolean isElementWise();
    void apply(List<Tensor> inputs, Tensor out);

    void gradient(List<Tensor> inputs, Tensor out);
    ComputeBackend getPreferredBackend();
    boolean supportsBackend(ComputeBackend backend);

    String getExpression();

    boolean requiresOutputForGradient();


}