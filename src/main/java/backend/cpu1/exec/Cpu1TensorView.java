package backend.cpu1.exec;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.TensorStorage;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;
import java.util.Objects;

/**
 * Runtime tensor view used by cpu1 kernels.
 */
public final class Cpu1TensorView {
    private final Tensor tensor;
    private final Cpu1BufferView buffer;
    private final int[] shape;
    private final int[] strides;
    private final int storageOffset;
    private final int elementCount;
    private final boolean contiguous;

    private Cpu1TensorView(
            Tensor tensor,
            Cpu1BufferView buffer,
            int[] shape,
            int[] strides,
            int storageOffset,
            int elementCount,
            boolean contiguous
    ) {
        this.tensor = Objects.requireNonNull(tensor, "tensor cannot be null");
        this.buffer = Objects.requireNonNull(buffer, "buffer cannot be null");
        this.shape = shape == null ? new int[0] : shape.clone();
        this.strides = strides == null ? new int[0] : strides.clone();
        this.storageOffset = storageOffset;
        this.elementCount = elementCount;
        this.contiguous = contiguous;
    }

    public static Cpu1TensorView fromTensor(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        DataType dataType = tensor.getDataType();
        Cpu1BufferView buffer = bufferView(tensor, dataType);
        return new Cpu1TensorView(
                tensor,
                buffer,
                tensor.getShape(),
                tensor.getStrides(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize(),
                tensor.isContiguous()
        );
    }

    public static Cpu1TensorView fromNativeStorage(Tensor tensor, NativeTensorStorage nativeStorage) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        Objects.requireNonNull(nativeStorage, "nativeStorage cannot be null");
        DataType dataType = tensor.getDataType();
        if (nativeStorage.getType() != dataType) {
            throw new IllegalArgumentException("Native storage dtype does not match tensor dtype. native="
                    + nativeStorage.getType() + ", tensor=" + dataType);
        }
        return new Cpu1TensorView(
                tensor,
                Cpu1BufferView.segment(dataType, nativeStorage.segment()),
                tensor.getShape(),
                tensor.getStrides(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize(),
                tensor.isContiguous()
        );
    }

    public DataType dataType() {
        return buffer.dataType();
    }

    public Cpu1BufferView buffer() {
        return buffer;
    }

    public backend.cpu1.storage.Cpu1StorageKind storageKind() {
        return buffer.storageKind();
    }

    public Cpu1TensorView broadcastToShape(int[] targetShape) {
        Objects.requireNonNull(targetShape, "targetShape cannot be null");
        if (Arrays.equals(shape, targetShape)) {
            return this;
        }
        if (shape.length > targetShape.length) {
            throw new IllegalArgumentException("Cannot broadcast rank " + shape.length + " tensor to rank "
                    + targetShape.length);
        }
        int[] broadcastShape = targetShape.clone();
        int[] broadcastStrides = new int[targetShape.length];
        int sourceOffset = targetShape.length - shape.length;
        boolean hasBroadcastStride = false;
        for (int dim = 0; dim < targetShape.length; dim++) {
            int sourceDim = dim < sourceOffset ? 1 : shape[dim - sourceOffset];
            int sourceStride = dim < sourceOffset ? 0 : strides[dim - sourceOffset];
            int targetDim = targetShape[dim];
            if (sourceDim == targetDim) {
                broadcastStrides[dim] = sourceStride;
            } else if (sourceDim == 1) {
                broadcastStrides[dim] = 0;
                hasBroadcastStride = true;
            } else {
                throw new IllegalArgumentException("Cannot broadcast shape " + Arrays.toString(shape)
                        + " to " + Arrays.toString(targetShape));
            }
        }
        return new Cpu1TensorView(
                tensor,
                buffer,
                broadcastShape,
                broadcastStrides,
                storageOffset,
                product(targetShape),
                contiguous && !hasBroadcastStride && Arrays.equals(strides, broadcastStrides)
        );
    }

    public float[] float32Array() {
        return buffer.float32Array();
    }

    public double[] float64Array() {
        return buffer.float64Array();
    }

    public short[] bfloat16Array() {
        return buffer.bfloat16Array();
    }

    public byte[] boolArray() {
        return buffer.boolArray();
    }

    public int[] int32Array() {
        return buffer.int32Array();
    }

    public long[] int64Array() {
        return buffer.int64Array();
    }

    public java.lang.foreign.MemorySegment segment() {
        return buffer.segment();
    }

    public int[] shape() {
        return shape.clone();
    }

    public int rank() {
        return shape.length;
    }

    public int shape(int dim) {
        return shape[dim];
    }

    public int[] strides() {
        return strides.clone();
    }

    public int stride(int dim) {
        return strides[dim];
    }

    public int storageOffset() {
        return storageOffset;
    }

    public int elementCount() {
        return elementCount;
    }

    public boolean contiguous() {
        return contiguous;
    }

    public void markStorageModified() {
        TensorInternalAccess.markStorageModified(tensor);
    }

    private static Cpu1BufferView bufferView(Tensor tensor, DataType dataType) {
        TensorStorage storage = TensorInternalAccess.storage(tensor);
        if (storage instanceof NativeTensorStorage nativeStorage) {
            if (nativeStorage.getType() != dataType) {
                throw new IllegalArgumentException("Native storage dtype does not match tensor dtype. native="
                        + nativeStorage.getType() + ", tensor=" + dataType);
            }
            return Cpu1BufferView.segment(dataType, nativeStorage.segment());
        }
        Object array = switch (dataType) {
            case FLOAT32 -> TensorInternalAccess.float32Data(tensor);
            case FLOAT64 -> TensorInternalAccess.float64Data(tensor);
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(tensor);
            case BOOL -> TensorInternalAccess.boolData(tensor);
            case INT32 -> TensorInternalAccess.int32Data(tensor);
            case INT64 -> TensorInternalAccess.int64Data(tensor);
        };
        return Cpu1BufferView.array(dataType, array);
    }

    private static int product(int[] shape) {
        int product = 1;
        for (int dimension : shape) {
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }
}
