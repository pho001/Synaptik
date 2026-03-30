package operations;
import tensor.Tensor;

import java.util.List;

public interface iFusedOperation
{
    void apply(List<Tensor> inputs,Tensor node);
}
