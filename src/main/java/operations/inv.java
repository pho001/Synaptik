package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class inv implements Operation {

    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Inv operation requires exactly 1 input tensor.");
        }

        double[] inputData = inputs.getFirst().getData();
        double[] result = node.getData();

        for (int i = 0; i < result.length; i++) {
            result[i] = 1.0 / inputData[i];
        }

        node.setData(result);
    }

    @Override
    public OpType opType() {
        return OpType.INV;
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public boolean isCheap() {
        return false;
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return true;
    }

    @Override
    public String getExpression() {
        return "inv";
    }
}
