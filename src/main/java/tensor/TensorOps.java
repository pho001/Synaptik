package tensor;

public final class TensorOps {
    private TensorOps() {}

    public static Tensor contiguous(Tensor input) {
        return TensorLayoutOps.contiguous(input);
    }

    public static Tensor reshape(Tensor input, int[] newShape) {
        return TensorLayoutOps.reshape(input, newShape);
    }

    public static Tensor expand(Tensor input, int[] newShape) {
        return TensorLayoutOps.expand(input, newShape);
    }

    public static Tensor permute(Tensor input, int[] axes) {
        return TensorLayoutOps.permute(input, axes);
    }

    public static Tensor expandDims(Tensor input, int axis) {
        return TensorLayoutOps.expandDims(input, axis);
    }

    public static Tensor squeeze(Tensor input, int axis) {
        return TensorLayoutOps.squeeze(input, axis);
    }

    public static Tensor add(Tensor first, Tensor second) {
        return TensorBinaryOps.add(first, second);
    }

    public static Tensor sub(Tensor first, Tensor second) {
        return TensorBinaryOps.sub(first, second);
    }

    public static Tensor mul(Tensor first, Tensor second) {
        return TensorBinaryOps.mul(first, second);
    }

    public static Tensor div(Tensor first, Tensor second) {
        return TensorBinaryOps.div(first, second);
    }

    public static Tensor min(Tensor first, Tensor second) {
        return TensorBinaryOps.min(first, second);
    }

    public static Tensor max(Tensor first, Tensor second) {
        return TensorBinaryOps.max(first, second);
    }

    public static Tensor matmul(Tensor first, Tensor second) {
        return TensorMatMulOps.matmul(first, second);
    }

    public static Tensor neg(Tensor input) {
        return TensorUnaryOps.neg(input);
    }

    public static Tensor log(Tensor input) {
        return TensorUnaryOps.log(input);
    }

    public static Tensor exp(Tensor input) {
        return TensorUnaryOps.exp(input);
    }

    public static Tensor fastExp(Tensor input) {
        return TensorUnaryOps.fastExp(input);
    }

    public static Tensor fastTanh(Tensor input) {
        return TensorUnaryOps.fastTanh(input);
    }

    public static Tensor pow(Tensor input, double exponent) {
        return TensorUnaryOps.pow(input, exponent);
    }

    public static Tensor mulScalar(Tensor input, double scalar) {
        return TensorUnaryOps.mulScalar(input, scalar);
    }

    public static Tensor inv(Tensor input) {
        return TensorUnaryOps.inv(input);
    }

    public static Tensor sqrt(Tensor input) {
        return TensorUnaryOps.sqrt(input);
    }

    public static Tensor sigmoid(Tensor input) {
        return TensorUnaryOps.sigmoid(input);
    }

    public static Tensor tanh(Tensor input) {
        return TensorUnaryOps.tanh(input);
    }

    public static Tensor sum(Tensor input, int dimension) {
        return TensorReduceOps.sum(input, dimension);
    }

    public static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.sum(input, dimension, keepDims);
    }

    public static Tensor sumAll(Tensor input) {
        return TensorReduceOps.sumAll(input);
    }

    public static Tensor mean(Tensor input, int dimension) {
        return TensorReduceOps.mean(input, dimension);
    }

    public static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.mean(input, dimension, keepDims);
    }

    public static Tensor meanAll(Tensor input) {
        return TensorReduceOps.meanAll(input);
    }

    public static Tensor min(Tensor input, int dimension) {
        return TensorReduceOps.min(input, dimension);
    }

    public static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.min(input, dimension, keepDims);
    }

    public static Tensor minAll(Tensor input) {
        return TensorReduceOps.minAll(input);
    }

    public static Tensor max(Tensor input, int dimension) {
        return TensorReduceOps.max(input, dimension);
    }

    public static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.max(input, dimension, keepDims);
    }

    public static Tensor maxAll(Tensor input) {
        return TensorReduceOps.maxAll(input);
    }

    public static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor runningMean,
            Tensor runningVar,
            double epsilon,
            boolean training
    ) {
        return TensorNaryOps.batchNorm(input, gamma, beta, runningMean, runningVar, epsilon, training);
    }
}
