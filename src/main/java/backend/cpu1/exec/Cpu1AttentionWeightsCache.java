package backend.cpu1.exec;

import tensor.DataType;

import java.util.Arrays;

/**
 * Run-scoped cpu1 attention weights cache keyed by the runtime attention output tensor.
 */
public final class Cpu1AttentionWeightsCache {
    private final DataType dataType;
    private final int[] shape;
    private final float[] f32Weights;
    private final double[] f64Weights;

    private Cpu1AttentionWeightsCache(DataType dataType, int[] shape, float[] f32Weights, double[] f64Weights) {
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64) {
            throw new IllegalArgumentException("cpu1 attention cache supports only FLOAT32/FLOAT64, got " + dataType);
        }
        if (shape == null || shape.length < 2) {
            throw new IllegalArgumentException("attention cache shape must have rank >= 2");
        }
        int elementCount = elementCount(shape);
        if (dataType == DataType.FLOAT32) {
            if (f32Weights == null || f32Weights.length != elementCount || f64Weights != null) {
                throw new IllegalArgumentException("FLOAT32 attention cache requires exactly " + elementCount
                        + " F32 weights");
            }
        } else if (f64Weights == null || f64Weights.length != elementCount || f32Weights != null) {
            throw new IllegalArgumentException("FLOAT64 attention cache requires exactly " + elementCount
                    + " F64 weights");
        }
        this.dataType = dataType;
        this.shape = shape.clone();
        this.f32Weights = f32Weights;
        this.f64Weights = f64Weights;
    }

    public static Cpu1AttentionWeightsCache f32(int[] shape) {
        return new Cpu1AttentionWeightsCache(DataType.FLOAT32, shape, new float[elementCount(shape)], null);
    }

    public static Cpu1AttentionWeightsCache f64(int[] shape) {
        return new Cpu1AttentionWeightsCache(DataType.FLOAT64, shape, null, new double[elementCount(shape)]);
    }

    public DataType dataType() {
        return dataType;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int elementCount() {
        return elementCount(shape);
    }

    public float[] requireF32Weights() {
        if (f32Weights == null) {
            throw new IllegalStateException("attention weights cache is not FLOAT32");
        }
        return f32Weights;
    }

    public double[] requireF64Weights() {
        if (f64Weights == null) {
            throw new IllegalStateException("attention weights cache is not FLOAT64");
        }
        return f64Weights;
    }

    public boolean matches(DataType expectedDataType, int[] expectedShape) {
        return dataType == expectedDataType && Arrays.equals(shape, expectedShape);
    }

    private static int elementCount(int[] shape) {
        int product = 1;
        for (int dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("attention cache shape dimensions must be positive: "
                        + Arrays.toString(shape));
            }
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }
}
