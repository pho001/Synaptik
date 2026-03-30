package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import java.util.List;

public class sqrt implements Operation {

    @Override
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Sqrt operation requires exactly 1 input tensor.");
        }

        double[] inputData = inputs.getFirst().getData();
        double[] result = node.getData();

        // Standardní CPU výpočet pomocí Math.sqrt
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.sqrt(inputData[i]);
        }

        node.setData(result);
    }

    @Override
    public OpType opType() {
        return OpType.SQRT;
    }

    @Override
    public String getExpression() {
        // Použijeme název funkce pro generátor výrazů
        return "sqrt";
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU || backend == ComputeBackend.GPU_CUDA;
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public boolean isCheap() {
        // Jak jsme si řekli, sqrt je dražší, takže raději materiálizovat než počítat 2x
        return false;
    }
}
