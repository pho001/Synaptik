package Backend.kernels.cpu;

import Operations.Operation;
import Operations.sum;
import Tensor.Tensor;

import java.util.List;

public class CpuSumKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        Tensor input = inputs.getFirst();
        int[] shape = input.getShape();
        int dimension = ((sum) op).getDimension();

        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds");
        }

        double[] data = input.getData();
        int[] strides = input.getStrides();
        double[] result = node.getData();

        if (dimension == -1) {
            result[0] = 0.0;
            for (double v : data) result[0] += v;
            return;
        }

        int reducedDim = shape[dimension];
        int stride = strides[dimension];

        if (dimension == shape.length - 1) {
            for (int i = 0; i < result.length; i++) {
                double acc = 0.0;
                int base = i * reducedDim;
                for (int j = 0; j < reducedDim; j++) acc += data[base + j];
                result[i] = acc;
            }
            return;
        }

        for (int i = 0; i < result.length; i++) {
            int outerBlock = i / stride;
            int innerOffset = i % stride;
            int baseIndex = outerBlock * (reducedDim * stride) + innerOffset;
            double acc = 0.0;
            for (int j = 0; j < reducedDim; j++) acc += data[baseIndex + j * stride];
            result[i] = acc;
        }
    }
}
