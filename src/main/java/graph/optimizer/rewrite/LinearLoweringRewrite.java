package graph.optimizer.rewrite;

import operations.Operation;
import operations.linalg.linear;
import operations.linalg.matmul;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

/**
 * Lowers {@code matmul + bias} patterns to the specialized linear operation.
 */
public class LinearLoweringRewrite extends AbstractRewriteRule {
    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (op == null || op.opType() != Operation.OpType.ADD) {
            return tensor;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return tensor;
        }

        Tensor candidate = tryLower(inputs.get(0), inputs.get(1), tensor);
        if (candidate != tensor) {
            return candidate;
        }
        return tryLower(inputs.get(1), inputs.get(0), tensor);
    }

    private Tensor tryLower(Tensor first, Tensor second, Tensor originalAdd) {
        if (first == null || second == null) {
            return originalAdd;
        }
        if (!(first.getOperation() instanceof matmul)) {
            return originalAdd;
        }
        if (!isBias(second)) {
            return originalAdd;
        }

        Tensor input = first.getPrevTensors().get(0);
        Tensor weight = first.getPrevTensors().get(1);
        if (!matchesLinearShape(input, weight, second, first)) {
            return originalAdd;
        }

        return input.linear(weight, second);
    }

    private boolean isBias(Tensor tensor) {
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            return false;
        }
        int[] shape = tensor.getShapeUnsafe();
        return shape.length == 1 && shape[0] > 0;
    }

    private boolean matchesLinearShape(Tensor input, Tensor weight, Tensor bias, Tensor matmulOut) {
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        int[] biasShape = bias.getShapeUnsafe();
        int[] outShape = matmulOut.getShapeUnsafe();

        if (inputShape.length < 2 || weightShape.length != 2 || biasShape.length != 1 || outShape.length != inputShape.length) {
            return false;
        }
        int inFeatures = inputShape[inputShape.length - 1];
        int outFeatures = weightShape[1];
        if (weightShape[0] != inFeatures || biasShape[0] != outFeatures) {
            return false;
        }
        if (outShape[outShape.length - 1] != outFeatures) {
            return false;
        }
        for (int i = 0; i < outShape.length - 1; i++) {
            if (outShape[i] != inputShape[i]) {
                return false;
            }
        }
        return true;
    }
}
