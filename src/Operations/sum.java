package Operations;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import Backend.ComputeBackend;
import Tensor.Tensor;


public class sum implements Operation{
    int dimension;
    public sum(int dimension){
        this.dimension = dimension;
    }

    // Sum in direction of dim
    // Dim is dimension of tensor which values you want to sum. Example for matrix (2x4) - if you sum around dimension 1 (4 columns),
    // you are summing up values from this dimension, meaning result will have 2 values: sums of all column values in each row.
    public void apply(List<Tensor> inputs, Tensor node) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("The input array must contain exactly 1 element");
        }

        Tensor input=inputs.getFirst();
        int[] shape=input.getShape();
        if (!input.isContiguous()){
            input.contiguous();
        }



        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds");
        }

        // Kontrola výstupního tvaru
        int expectedSize = 1;
        if (dimension != -1) {
            for (int i = 0; i < shape.length; i++) {
                if (i != dimension) expectedSize *= shape[i];
            }
        }
        if (node.getData().length != expectedSize) {
            throw new IllegalArgumentException("Output tensor has wrong size");
        }

        double[] data = input.getData();
        int[] strides = input.getStrides();
        double[] result = node.getData();
        if (dimension == -1){
            result[0] = 0;
            for (int i=0; i<data.length; i++){
                result[0]+=data[i];
            }
            return;
        }


        int reducedDim = shape[dimension];
        int stride = strides[dimension];  //stride of reduced dimension

        // Last dim optimization
        if (dimension == shape.length - 1) {
            for (int i = 0; i < result.length; i++) {
                double sum = 0;
                for (int j = 0; j < reducedDim; j++) {
                    sum += data[i * reducedDim + j];
                }
                result[i] = sum;
            }
            return;
        }

        // general situation - any dimension
        for (int i = 0; i < result.length; i++) {
            int outerBlock = i / stride;
            int innerOffset = i % stride;
            int baseIndex = outerBlock * (reducedDim * stride) + innerOffset;
            double sum = 0;
            for (int j = 0; j < reducedDim; j++) {
                double y = data[baseIndex + j * stride];
                sum = sum + y;
            }
            result[i] = sum;
        }
    }

    @Override
    public OpType opType() {
        return OpType.SUM;
    }







    public void gradient(List<Tensor> inputs, Tensor output){


    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.GPU_CUDA ||
                backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return "tanh";
    }

    @Override
    public boolean isElementWise(){
        return false;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return false;
    }

    public int getDimension() {
        return dimension;
    }


}
