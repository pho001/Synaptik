package tensor;

import java.util.Arrays;

public final class TensorMetadata {
    public static final DataType DEFAULT_DATA_TYPE = DataType.FLOAT32;

    private final int[] shape;
    private final int[] strides;
    private final int storageOffset;
    private final boolean contiguous;
    private String label;
    private boolean requiresGrad;
    private boolean trainableParameter;
    private DataType dataType;

    public TensorMetadata(int[] shape, String label, boolean requiresGrad) {
        this(shape, label, requiresGrad, DEFAULT_DATA_TYPE);
    }

    public TensorMetadata(int[] shape, String label, boolean requiresGrad, DataType dataType) {
        int[] normalizedShape = normalizeShape(shape);
        this.shape = normalizedShape;
        this.strides = computeStrides(normalizedShape);
        this.storageOffset = 0;
        this.contiguous = computeContiguous(this.shape, this.strides);
        this.label = label;
        this.requiresGrad = requiresGrad;
        this.trainableParameter = false;
        this.dataType = dataType == null ? DEFAULT_DATA_TYPE : dataType;
    }

    public TensorMetadata(int[] shape, int[] strides, String label, boolean requiresGrad) {
        this(shape, strides, label, requiresGrad, DEFAULT_DATA_TYPE);
    }

    public TensorMetadata(int[] shape, int[] strides, String label, boolean requiresGrad, DataType dataType) {
        this(shape, strides, 0, label, requiresGrad, dataType);
    }

    public TensorMetadata(int[] shape, int[] strides, int storageOffset, String label, boolean requiresGrad, DataType dataType) {
        int[] normalizedShape = normalizeShape(shape);
        int[] normalizedStrides;
        if (strides == null) {
            throw new IllegalArgumentException("Strides cannot be null.");
        }
        if (strides.length == 0 && normalizedShape.length == 1 && normalizedShape[0] == 1) {
            normalizedStrides = new int[]{1};
        } else {
            if (strides.length != normalizedShape.length) {
                throw new IllegalArgumentException("Strides length must match shape length.");
            }
            normalizedStrides = strides.clone();
        }

        this.shape = normalizedShape;
        this.strides = normalizedStrides;
        this.storageOffset = normalizeStorageOffset(storageOffset);
        this.contiguous = computeContiguous(this.shape, this.strides);
        this.label = label;
        this.requiresGrad = requiresGrad;
        this.trainableParameter = false;
        this.dataType = dataType == null ? DEFAULT_DATA_TYPE : dataType;
    }

    public int[] getShape() {
        return shape.clone();
    }

    int[] shapeRef() {
        return shape;
    }

    public int[] getStrides() {
        return strides.clone();
    }

    int[] stridesRef() {
        return strides;
    }

    public int getStorageOffset() {
        return storageOffset;
    }

    public int getStride(int index) {
        return strides[index];
    }

    public int getDimensionAt(int index) {
        return shape[index];
    }

    public int rank() {
        return shape.length;
    }

    public int getFlatSize() {
        int size = 1;
        for (int dimension : shape) {
            size *= dimension;
        }
        return size;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public void setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        if (!requiresGrad) {
            this.trainableParameter = false;
        }
    }

    public boolean trainableParameter() {
        return trainableParameter;
    }

    public void setTrainableParameter(boolean trainableParameter) {
        this.trainableParameter = trainableParameter;
        if (trainableParameter) {
            this.requiresGrad = true;
        }
    }

    public DataType getDataType() {
        return dataType;
    }

    public void setDataType(DataType dataType) {
        this.dataType = dataType == null ? DEFAULT_DATA_TYPE : dataType;
    }

    public boolean isContiguous() {
        return contiguous;
    }

    public boolean hasZeroStride() {
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    public boolean isBroadcastView() {
        return hasZeroStride();
    }

    public boolean hasStorageOffset() {
        return storageOffset != 0;
    }

    private static boolean computeContiguous(int[] shape, int[] strides) {
        int expectedStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
    }

    public int getFlatIndex(int[] indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Incorrect number of indices provided.");
        }

        int[] denseStrides = computeStrides(shape);
        int flatIndex = 0;
        for (int i = 0; i < shape.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index out of bounds for dimension " + i + ".");
            }
            flatIndex += indices[i] * denseStrides[i];
        }
        return flatIndex;
    }

    public int[] getSpatialIndex(int index) {
        int flatSize = getFlatSize();
        if (index < 0 || index >= flatSize) {
            throw new IndexOutOfBoundsException("Flat index out of bounds.");
        }
        int[] denseStrides = computeStrides(shape);
        int[] indices = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            indices[i] = (index / denseStrides[i]) % shape[i];
        }
        return indices;
    }

    public TensorMetadata copy() {
        TensorMetadata copy = new TensorMetadata(shape, strides, storageOffset, label, requiresGrad, dataType);
        copy.setTrainableParameter(trainableParameter);
        return copy;
    }

    public static int[] computeStrides(int[] shape) {
        int[] normalizedShape = normalizeShape(shape);
        int[] strides = new int[normalizedShape.length];
        int stride = 1;
        for (int i = normalizedShape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= normalizedShape[i];
        }
        return strides;
    }

    private static int[] normalizeShape(int[] shape) {
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (shape.length == 0) {
            return new int[]{1};
        }
        return shape.clone();
    }

    private static int normalizeStorageOffset(int storageOffset) {
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative.");
        }
        return storageOffset;
    }

    @Override
    public String toString() {
        return "TensorMetadata{" +
                "shape=" + Arrays.toString(shape) +
                ", strides=" + Arrays.toString(strides) +
                ", storageOffset=" + storageOffset +
                ", label='" + label + '\'' +
                ", requiresGrad=" + requiresGrad +
                ", dataType=" + dataType +
                '}';
    }
}
