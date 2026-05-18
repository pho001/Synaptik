package tensor;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import operations.index.ScatterReduction;
import operations.layout.noop;
import tensor.factory.TensorArrayData;
import tensor.loss.LossReduction;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.*;


/**
 * Mutable tensor object that records computation graph metadata and typed storage.
 *
 * <p>A tensor combines shape/stride metadata, a {@link TensorStorage} buffer,
 * optional autograd links to previous tensors, and an optional operation node.
 * Tensor instances are mutable and are not thread-safe: labels, gradient flags,
 * storage, and typed backing arrays can be changed after construction. Layout
 * operations may return views that share storage with their source tensor, so
 * writes through one alias can be observed through another.</p>
 *
 * <p>Public shape accessors generally return defensive copies unless the method
 * name contains {@code Unsafe}. Typed storage getters return mutable backing
 * arrays without copying.</p>
 */
public class Tensor {
    /**
     * Internal label used to mark synthetic forward-output graph nodes.
     */
    public static final String SYSTEM_FORWARD_OUTPUT_LABEL = "System_Forward_Output";
    private TensorStorage storage;
    private TensorMetadata metadata;
    private Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private ComputeBackend forcedBackend = null;
    private Runnable backwardFunction;
    private boolean isBackward = false;





    /**
     * Creates a tensor from a rectangular multidimensional Java array using the default dtype.
     *
     * @param multiDimArray rectangular nested array of doubles; must be non-null
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @throws IllegalArgumentException if the array cannot be flattened as numeric data
     */
    public Tensor(Object multiDimArray, List<Tensor> previous, String label) {
        this(multiDimArray, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a tensor from a rectangular multidimensional Java array.
     *
     * @param multiDimArray rectangular nested array of doubles; must be non-null
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage conversion; must be non-null
     * @throws IllegalArgumentException if the array cannot be flattened as numeric data
     */
    public Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType) {
        int[] computedShape = TensorArrayData.inferShape(multiDimArray);
        this.metadata = new TensorMetadata(computedShape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        storage = TensorStorageSupport.fromDoubleArray(metadata, TensorArrayData.flattenToDouble(multiDimArray, metadata.getFlatSize()));
    }

    /**
     * Creates an empty tensor with the default dtype and contiguous row-major layout.
     *
     * @param dimensions tensor shape; must be non-null and contain valid dimensions
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(int[] dimensions, List<Tensor> previous, String label) {
        this(dimensions, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates an empty tensor with contiguous row-major layout.
     *
     * @param dimensions tensor shape; must be non-null and contain valid dimensions
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage allocation; must be non-null
     */
    public Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

        this.metadata = new TensorMetadata(dimensions, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        storage = TensorStorageSupport.emptyStorage(metadata);
    }

    /**
     * Creates an operation-backed empty tensor with the default dtype.
     *
     * @param shape output shape; must be non-null
     * @param previous parent tensors used by autograd; null is treated as an empty list
     * @param operation operation that produces this tensor, may be null for leaf-like tensors
     * @param label tensor label, may be null
     */
    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        this(shape, previous, operation, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates an operation-backed empty tensor with contiguous row-major layout.
     *
     * @param shape output shape; must be non-null
     * @param previous parent tensors used by autograd; null is treated as an empty list
     * @param operation operation that produces this tensor, may be null for leaf-like tensors
     * @param label tensor label, may be null
     * @param dataType dtype used for storage allocation; must be non-null
     */
    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        storage = TensorStorageSupport.emptyStorage(metadata, calculateSize(shape));
    }

    /**
     * Creates an empty strided tensor view descriptor with zero storage offset.
     *
     * @param shape logical shape; must be non-null
     * @param strides logical-to-storage strides; must match {@code shape.length}
     * @param previous parent tensors used by autograd; null is treated as an empty list
     * @param operation operation that produces this tensor, may be null
     * @param label tensor label, may be null
     * @param dataType dtype used for storage allocation; must be non-null
     */
    public Tensor(int[] shape, int[] strides, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this(shape, strides, 0, previous, operation, label, dataType);
    }

    /**
     * Creates an empty strided tensor descriptor.
     *
     * @param shape logical shape; must be non-null
     * @param strides logical-to-storage strides; must match {@code shape.length}
     * @param storageOffset physical storage offset for logical element zero
     * @param previous parent tensors used by autograd; null is treated as an empty list
     * @param operation operation that produces this tensor, may be null
     * @param label tensor label, may be null
     * @param dataType dtype used for storage allocation; must be non-null
     */
    public Tensor(int[] shape, int[] strides, int storageOffset, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, strides, storageOffset, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        storage = TensorStorageSupport.emptyStorage(metadata);
    }

    /**
     * Creates a tensor from double values using the default dtype.
     *
     * @param data storage-order source values; length must equal the product of {@code shape}
     * @param shape logical shape; must be non-null
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a contiguous tensor from double values.
     *
     * @param data storage-order source values; length must equal the product of {@code shape}
     * @param shape logical shape; must be non-null
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for conversion; must be non-null
     * @throws IllegalArgumentException if data length does not match the shape size
     */
    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided tensor from double values using the default dtype.
     *
     * @param data storage-order source values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a strided tensor from double values.
     *
     * @param data storage-order source values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for conversion; must be non-null
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        storage = TensorStorageSupport.fromDoubleArray(metadata, data);
    }

    /**
     * Creates a contiguous tensor from float values using the default dtype.
     *
     * @param data storage-order source values; length must equal the product of {@code shape}
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(float[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a contiguous tensor from float values.
     *
     * @param data storage-order source values; length must equal the product of {@code shape}
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for conversion; typically {@link DataType#FLOAT32}
     * @throws IllegalArgumentException if data length does not match the shape size
     */
    public Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided tensor from float values using the default dtype.
     *
     * @param data storage-order source values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a strided tensor from float values.
     *
     * @param data storage-order source values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for conversion; typically {@link DataType#FLOAT32}
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        storage = TensorStorageSupport.fromFloatArray(metadata, data);
    }

    /**
     * Creates a contiguous tensor from raw bfloat16 bit values using the default dtype.
     *
     * @param data storage-order bfloat16 bit patterns; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(short[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a contiguous tensor from raw bfloat16 bit values.
     *
     * @param data storage-order bfloat16 bit patterns; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; typically {@link DataType#BFLOAT16}
     * @throws IllegalArgumentException if data length does not match the shape size
     */
    public Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided tensor from raw bfloat16 bit values using the default dtype.
     *
     * @param data storage-order bfloat16 bit patterns; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a strided tensor from raw bfloat16 bit values.
     *
     * @param data storage-order bfloat16 bit patterns; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; typically {@link DataType#BFLOAT16}
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        storage = TensorStorageSupport.fromBFloat16Array(metadata, data);
    }

    /**
     * Creates a contiguous BOOL tensor from byte values.
     *
     * @param data storage-order boolean bytes; non-zero values represent true
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(byte[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, DataType.BOOL);
    }

    /**
     * Creates a contiguous boolean tensor from byte values.
     *
     * @param data storage-order boolean bytes; non-zero values represent true
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#BOOL}
     * @throws IllegalArgumentException if data length does not match the shape size
     */
    public Tensor(byte[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided BOOL tensor from byte values.
     *
     * @param data storage-order boolean bytes; non-zero values represent true
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, DataType.BOOL);
    }

    /**
     * Creates a strided boolean tensor from byte values.
     *
     * @param data storage-order boolean bytes; non-zero values represent true
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#BOOL}
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        storage = TensorStorageSupport.fromBoolArray(metadata, data);
    }

    /**
     * Creates a contiguous INT32 tensor from integer values.
     *
     * @param data storage-order integer values; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(int[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, DataType.INT32);
    }

    /**
     * Creates a contiguous int32 tensor from integer values.
     *
     * @param data storage-order integer values; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#INT32}
     * @throws IllegalArgumentException if data length does not match shape size
     */
    public Tensor(int[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided int32 tensor from integer values.
     *
     * @param data storage-order integer values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#INT32}
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(int[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        storage = TensorStorageSupport.fromIntArray(metadata, data);
    }

    /**
     * Creates a contiguous INT64 tensor from long values.
     *
     * @param data storage-order integer values; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     */
    public Tensor(long[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, DataType.INT64);
    }

    /**
     * Creates a contiguous int64 tensor from long values.
     *
     * @param data storage-order integer values; length must equal shape size
     * @param shape logical shape
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#INT64}
     * @throws IllegalArgumentException if data length does not match shape size
     */
    public Tensor(long[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    /**
     * Creates a strided int64 tensor from long values.
     *
     * @param data storage-order integer values; length must match logical flat size
     * @param shape logical shape
     * @param strides logical-to-storage strides
     * @param previous parent tensors for autograd metadata; null is treated as an empty list
     * @param label tensor label, may be null
     * @param dataType dtype used for storage; must be {@link DataType#INT64}
     * @throws IllegalArgumentException if data length does not match logical flat size
     */
    public Tensor(long[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "long[]");
        storage = TensorStorageSupport.fromLongArray(metadata, data);
    }

    /**
     * Creates a shape {@code [1]} scalar tensor using the default dtype.
     *
     * @param value scalar value
     * @return new scalar tensor
     */
    public static Tensor scalar(double value) {
        return scalar(value, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a shape {@code [1]} scalar tensor using the requested dtype.
     *
     * @param value scalar value; must be integral for {@link DataType#INT32} or {@link DataType#INT64}
     * @param dataType requested dtype; must be non-null
     * @return new scalar tensor
     * @throws IllegalArgumentException if an INT32/INT64 scalar is requested with a non-integral value
     */
    public static Tensor scalar(double value, DataType dataType) {
        return TensorDataFactory.scalar(value, dataType);
    }

    /**
     * Creates a tensor filled with zeros using the default dtype.
     *
     * @param shape output shape
     * @return newly allocated tensor
     */
    public static Tensor zeros(int[] shape) {
        return zeros(shape, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * <p>All current public dtypes are supported, including {@link DataType#BOOL},
     * {@link DataType#INT32}, and {@link DataType#INT64}. The returned tensor is a
     * leaf tensor with no graph predecessors.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType output dtype; must be non-null
     * @return newly allocated tensor
     */
    public static Tensor zeros(int[] shape, DataType dataType) {
        return zeros(shape, dataType, "zeros");
    }

    /**
     * Creates a labeled tensor filled with zeros.
     *
     * <p>All current public dtypes are supported, including {@link DataType#BOOL},
     * {@link DataType#INT32}, and {@link DataType#INT64}. The returned tensor is a
     * leaf tensor with no graph predecessors.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType output dtype; must be non-null
     * @param label tensor label
     * @return newly allocated tensor
     */
    public static Tensor zeros(int[] shape, DataType dataType, String label) {
        return TensorDataFactory.zeros(shape, dataType, label);
    }

    /**
     * Creates a tensor filled with ones using the default dtype.
     *
     * @param shape output shape
     * @return newly allocated tensor
     */
    public static Tensor ones(int[] shape) {
        return ones(shape, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    /**
     * Creates a tensor filled with ones.
     *
     * <p>For {@link DataType#BOOL}, one means {@code true}. For integer dtypes,
     * one is stored as {@code 1} or {@code 1L}. The returned tensor is a leaf tensor
     * with no graph predecessors.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType output dtype; must be non-null
     * @return newly allocated tensor
     */
    public static Tensor ones(int[] shape, DataType dataType) {
        return ones(shape, dataType, "ones");
    }

    /**
     * Creates a labeled tensor filled with ones.
     *
     * <p>For {@link DataType#BOOL}, one means {@code true}. For integer dtypes,
     * one is stored as {@code 1} or {@code 1L}. The returned tensor is a leaf tensor
     * with no graph predecessors.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param dataType output dtype; must be non-null
     * @param label tensor label
     * @return newly allocated tensor
     */
    public static Tensor ones(int[] shape, DataType dataType, String label) {
        return TensorDataFactory.ones(shape, dataType, label);
    }

    /**
     * Creates a tensor filled from a standard normal distribution using the default dtype.
     *
     * @param shape output shape
     * @return newly allocated tensor
     */
    public static Tensor randn(int[] shape) {
        return randn(shape, 0.0d, 1.0d, TensorMetadata.DEFAULT_DATA_TYPE, "randn");
    }

    /**
     * Creates a labeled floating tensor filled from a normal distribution.
     *
     * <p>The sampled values follow {@code mean + gaussian * stdDev}. This factory
     * supports only floating dtypes because random normal values do not have a
     * well-defined integer or boolean storage contract.</p>
     *
     * @param shape output shape; must be non-null and contain positive dimensions
     * @param mean distribution mean
     * @param stdDev finite non-negative distribution standard deviation
     * @param dataType floating output dtype; must be FLOAT64, FLOAT32, or BFLOAT16
     * @param label tensor label
     * @return newly allocated tensor
     * @throws IllegalArgumentException if {@code stdDev} is negative/non-finite or dtype is non-floating
     */
    public static Tensor randn(int[] shape, double mean, double stdDev, DataType dataType, String label) {
        return TensorDataFactory.randn(shape, mean, stdDev, dataType, label);
    }

    /**
     * Creates a rank-1 tensor containing an arithmetic integer range.
     *
     * <p>The range includes {@code start}, excludes {@code end}, and advances by
     * {@code step}. Empty ranges are rejected so the public tensor shape remains
     * non-empty. {@link DataType#BOOL} is not supported.</p>
     *
     * @param start inclusive start
     * @param end exclusive end
     * @param step non-zero step
     * @param dataType output dtype; must be numeric and non-BOOL
     * @return rank-1 range tensor
     * @throws IllegalArgumentException if step is zero, cannot reach end, produces an empty range, or dtype is BOOL
     */
    public static Tensor arange(int start, int end, int step, DataType dataType) {
        return TensorDataFactory.arange(start, end, step, dataType);
    }

    /**
     * Returns the product of dimensions in a shape array.
     *
     * @param dimensions dimensions to multiply; must be non-null
     * @return flat element count
     */
    public int calculateSize(int[] dimensions) {
        return TensorShape.checkedFlatSize(dimensions);
    }

    /**
     * Returns this tensor's rank.
     *
     * <p>Rank is the number of logical axes in the tensor shape. For example,
     * shape {@code [batch, time, features]} has rank {@code 3}.</p>
     *
     * @return number of dimensions in the logical shape
     */
    public int rank() {
        return metadata.shapeRef().length;
    }

    /**
     * Returns this tensor's logical element count.
     *
     * <p>This is equivalent to {@link #getFlatDataSize()} and is named for
     * ergonomic public code.</p>
     *
     * @return product of shape dimensions
     */
    public int size() {
        return getFlatDataSize();
    }

    /**
     * Returns the final logical dimension size.
     *
     * <p>This is useful for operations whose contract is based on the last axis,
     * such as N-D {@link #linear(Tensor, Tensor)} where input shape is
     * {@code [..., inFeatures]}.</p>
     *
     * @return last shape dimension
     * @throws IllegalStateException if this tensor has no dimensions
     */
    public int lastDim() {
        int[] shape = metadata.shapeRef();
        if (shape.length == 0) {
            throw new IllegalStateException("lastDim() requires rank >= 1.");
        }
        return shape[shape.length - 1];
    }

    /**
     * Compares this tensor's logical shape with an expected shape.
     *
     * @param shape expected shape; varargs are not retained
     * @return true when shapes match exactly
     */
    public boolean shapeEquals(int... shape) {
        return Arrays.equals(metadata.shapeRef(), shape);
    }

    /**
     * Returns a defensive copy of this tensor's shape.
     *
     * <p>This is the explicit safe alternative to {@link #getShapeUnsafe()} for
     * application or framework code that may store or mutate the returned array.</p>
     *
     * @return copied shape array
     */
    public int[] shapeCopy() {
        return getShape();
    }

    /**
     * Returns one stride from this tensor's logical layout.
     *
     * @param index stride axis
     * @return storage stride for the axis
     * @throws IndexOutOfBoundsException if the axis is invalid
     */
    public int getStride(int index){
        return metadata.getStride(index);
    }

    /**
     * Reports whether this tensor should accumulate gradients.
     *
     * @return true when autograd should compute gradients for this tensor
     */
    public boolean getRequiresGrad(){
        return metadata.requiresGrad();
    }

    /**
     * Enables or disables gradient accumulation for this tensor.
     *
     * @param requiresGrad true to participate in autograd
     */
    public void setRequiresGrad(boolean requiresGrad){
        metadata.setRequiresGrad(requiresGrad);
    }

    /**
     * Reports whether this tensor is a trainable model parameter.
     *
     * <p>Trainable parameters are the only tensors updated by training optimizers.
     * This is intentionally narrower than {@link #getRequiresGrad()}: intermediate
     * tensors can require gradients without being optimizer-owned parameters.</p>
     *
     * @return true when optimizers may update this tensor
     */
    public boolean isTrainableParameter() {
        return metadata.trainableParameter();
    }

    /**
     * Marks this tensor as a trainable model parameter or clears that role.
     *
     * <p>Setting the flag to true also enables gradient tracking. Clearing the
     * flag leaves {@code requiresGrad} unchanged so callers can keep gradients
     * observable for non-parameter tensors.</p>
     *
     * @param trainableParameter true when optimizers may update this tensor
     */
    public void setTrainableParameter(boolean trainableParameter) {
        metadata.setTrainableParameter(trainableParameter);
    }

    /**
     * Fluent variant of {@link #setTrainableParameter(boolean)}.
     *
     * @return this tensor
     */
    public Tensor trainableParameter() {
        setTrainableParameter(true);
        return this;
    }

    /**
     * Returns this tensor's label.
     *
     * @return label string, possibly null
     */
    public String getLabel() {
        return metadata.getLabel();
    }

    /**
     * Updates this tensor's label.
     *
     * @param label new label, may be null
     */
    public void setLabel(String label) {
        metadata.setLabel(label);
    }


    /**
     * Reads one logical element by row-major flat index.
     *
     * @param index logical flat index in {@code [0, getFlatDataSize())}
     * @return value converted to double
     * @throws IndexOutOfBoundsException if {@code index} is outside the logical tensor size
     */
    public double getByFlatIndex(int index){
        if (index < 0 || index >= getFlatDataSize()) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return getByStorageOffset(TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, index));
    }

    /**
     * Reads one INT64 logical element exactly by row-major flat index.
     *
     * @param index logical flat index in {@code [0, getFlatDataSize())}
     * @return stored int64 value
     * @throws UnsupportedOperationException if this tensor is not INT64-backed
     * @throws IndexOutOfBoundsException if {@code index} is outside the logical tensor size
     */
    public long getInt64ByFlatIndex(int index) {
        if (metadata.getDataType() != DataType.INT64) {
            throw new UnsupportedOperationException("getInt64ByFlatIndex() is only supported for INT64 tensors.");
        }
        if (index < 0 || index >= getFlatDataSize()) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return getInt64Data()[TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, index)];
    }

    /**
     * Reads one integral logical element as a long.
     *
     * @param index logical flat index in {@code [0, getFlatDataSize())}
     * @return stored INT32 or INT64 value
     * @throws UnsupportedOperationException if this tensor is not an integral tensor
     */
    public long getIntegralByFlatIndex(int index) {
        if (index < 0 || index >= getFlatDataSize()) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        int offset = TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, index);
        return switch (metadata.getDataType()) {
            case INT32 -> getInt32Data()[offset];
            case INT64 -> getInt64Data()[offset];
            default -> throw new UnsupportedOperationException("getIntegralByFlatIndex() is only supported for INT32/INT64 tensors.");
        };
    }

    /**
     * Writes one logical element by row-major flat index.
     *
     * @param flatindex logical flat index in {@code [0, getFlatDataSize())}
     * @param value value converted to this tensor's storage dtype
     * @throws UnsupportedOperationException if this tensor is a broadcast view
     * @throws IndexOutOfBoundsException if {@code flatindex} maps outside storage
     */
    public void setDataAt(int flatindex,double value) {
        if (isBroadcastView()) {
            throw new UnsupportedOperationException("Cannot write through broadcast view tensor.");
        }
        setByStorageOffset(TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, flatindex), value);
        markStorageModified();
    }

    /**
     * Returns a defensive copy of this tensor's strides.
     *
     * @return stride array copy
     */
    public int[] getStrides() {
        return metadata.getStrides();
    }

    /**
     * Returns the mutable internal stride array.
     *
     * @return internal stride array; modifying it mutates tensor metadata
     */
    public int[] getStridesUnsafe() {
        return metadata.stridesRef();
    }

    /**
     * Returns this tensor's physical storage offset.
     *
     * @return offset of logical element zero in the backing storage
     */
    public int getStorageOffsetUnsafe() {
        return metadata.getStorageOffset();
    }

    /**
     * Replaces this tensor's storage from double values.
     *
     * @param data storage-order values; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(double[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(double[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        storage = TensorStorageSupport.fromDoubleArray(metadata, data);
    }

    /**
     * Replaces this tensor's storage from float values.
     *
     * @param data storage-order values; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(float[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(float[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        storage = TensorStorageSupport.fromFloatArray(metadata, data);
    }

    /**
     * Replaces this tensor's storage from raw bfloat16 bit values.
     *
     * @param data storage-order bfloat16 bits; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(short[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(short[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        storage = TensorStorageSupport.fromBFloat16Array(metadata, data);
    }

    /**
     * Replaces this tensor's storage from boolean byte values.
     *
     * @param data storage-order boolean bytes; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(byte[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(byte[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        storage = TensorStorageSupport.fromBoolArray(metadata, data);
    }

    /**
     * Replaces this tensor's storage from int32 values.
     *
     * @param data storage-order integers; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(int[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(int[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        storage = TensorStorageSupport.fromIntArray(metadata, data);
    }

    /**
     * Replaces this tensor's storage from int64 values.
     *
     * @param data storage-order integers; must be non-null and match logical flat size
     * @throws IllegalArgumentException if {@code data} is null or has the wrong length
     */
    public void setData(long[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        requireDenseStorageReplacement("setData(long[])");
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "long[]");
        storage = TensorStorageSupport.fromLongArray(metadata, data);
    }

    /**
     * Copies logical data from another tensor of the same shape and dtype.
     *
     * <p>The copy respects non-contiguous layouts and storage offsets. The source
     * tensor is not modified.</p>
     *
     * @param source tensor to copy from; must be non-null
     * @throws IllegalArgumentException if source is null, shape differs, or dtype differs
     */
    public void copyDataFrom(Tensor source) {
        if (source == null) {
            throw new IllegalArgumentException("source cannot be null");
        }
        if (!Arrays.equals(this.getShapeUnsafe(), source.getShapeUnsafe())) {
            throw new IllegalArgumentException("copyDataFrom requires matching shapes.");
        }
        if (this.getDataType() != source.getDataType()) {
            throw new IllegalArgumentException("copyDataFrom requires matching dtypes.");
        }
        if (this == source) {
            return;
        }
        TensorRemap.RemapPlan plan = TensorRemap.buildPlan(source, this);
        TensorRemap.applyTrusted(source, this, plan, Integer.MAX_VALUE);
    }

    /**
     * Replaces storage for a FLOAT32 tensor.
     *
     * @param data storage-order float values; must be non-null and match logical flat size
     * @throws UnsupportedOperationException if this tensor is not {@link DataType#FLOAT32}
     * @throws IllegalArgumentException if data is null or has the wrong length
     */
    public void setFloat32Data(float[] data) {
        if (metadata.getDataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("setFloat32Data() is only supported for FLOAT32 tensors.");
        }
        setData(data);
    }

    /**
     * Converts a multidimensional index into a logical row-major flat index.
     *
     * @param indices one index per dimension
     * @return logical flat index
     * @throws IllegalArgumentException if rank or indices are invalid
     */
    public int getFlatIndex(int[] indices) {
        return metadata.getFlatIndex(indices);
    }

    /**
     * Converts a logical row-major flat index into multidimensional coordinates.
     *
     * @param index logical flat index
     * @return coordinate array
     * @throws IndexOutOfBoundsException if {@code index} is invalid
     */
    public int[] getSpatialIndex(int index){
        return metadata.getSpatialIndex(index);
    }


    /**
     * Returns a defensive copy of this tensor's shape.
     *
     * @return shape array copy
     */
    public int[] getShape() {
        return metadata.getShape();
    }

    /**
     * Returns the mutable internal shape array.
     *
     * @return internal shape array; modifying it mutates tensor metadata
     */
    public int[] getShapeUnsafe() {
        return metadata.shapeRef();
    }

    /**
     * Returns previous tensors in the computation graph.
     *
     * @return unmodifiable view of previous tensors, or null if internal state has been cleared
     */
    public List<Tensor> getPrevTensors() {
        return prevTensors == null ? null : Collections.unmodifiableList(prevTensors);
    }

    /**
     * Returns one shape dimension.
     *
     * @param index axis index
     * @return dimension size
     * @throws IndexOutOfBoundsException if {@code index} is invalid
     */
    public int getDimensionAt(int index) {
        return metadata.getDimensionAt(index);
    }

    /**
     * Returns the gradient associated with this tensor.
     *
     * <p>When called inside an autograd compilation scope, this reads the scoped
     * gradient map. Otherwise it returns this tensor's stored gradient reference.</p>
     *
     * @return gradient tensor, or null when no gradient has been accumulated
     */
    public Tensor getGradient(){
        AutogradCompilationScope scope = AutogradCompilationScope.current();
        if (scope != null) {
            return scope.gradientOf(this);
        }
        return gradient;
    }

    /**
     * Computes row-major contiguous strides for a shape.
     *
     * @param shape shape to analyze; must be non-null
     * @return newly allocated strides array
     */
    public int[] computeStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    /**
     * Reports whether this tensor represents a backward graph tensor.
     *
     * @return true for tensors marked by internal autograd construction
     */
    public boolean isBackward() {
        return isBackward;
    }
    void setBackwardInternal(boolean backward) {
        isBackward = backward;
    }

    /**
     * Returns a defensive copy of this tensor's current strides.
     *
     * @return stride array copy
     */
    public int[] computeStrides() {
        return metadata.getStrides();
    }

    /**
     * Returns this tensor's dtype.
     *
     * @return non-null dtype
     */
    public DataType getDataType() {
        return metadata.getDataType();
    }

    /**
     * Converts this tensor's storage to another floating dtype.
     *
     * <p>INT32, INT64 and BOOL conversions are deliberately restricted because those
     * dtypes have different semantic roles from floating tensors. This method
     * replaces storage and preserves logical values where conversion is allowed.</p>
     *
     * @param dataType target dtype; must be non-null and compatible
     * @throws UnsupportedOperationException if converting to or from INT32/INT64, or
     *                                       between BOOL and numeric dtypes
     */
    public void setDataType(DataType dataType) {
        DataType current = metadata.getDataType();
        if (current == dataType) {
            return;
        }
        if (current == DataType.INT32 || dataType == DataType.INT32
                || current == DataType.INT64 || dataType == DataType.INT64) {
            throw new UnsupportedOperationException("Implicit INT32/INT64 <-> other dtype conversion is not supported.");
        }
        if ((current == DataType.BOOL) != (dataType == DataType.BOOL)) {
            throw new UnsupportedOperationException("Implicit BOOL <-> numeric dtype conversion is not supported.");
        }
        double[] snapshot = TensorDebugSupport.toDoubleStorageOrderCopy(this);
        metadata.setDataType(dataType);
        storage = TensorStorageSupport.fromDoubleArray(metadata, snapshot);
    }

    /**
     * Returns the backing storage object.
     *
     * @return mutable storage backing this tensor
     */
    public TensorStorage getStorage() {
        return storage;
    }

    /**
     * Returns the current backing storage version.
     *
     * @return storage mutation counter
     */
    public long storageVersion() {
        return TensorStorageSupport.version(storage);
    }

    /**
     * Marks this tensor's storage as modified.
     *
     * <p>Call this after direct mutation through a typed array returned by a
     * storage getter if downstream caches rely on storage versions.</p>
     */
    public void markStorageModified() {
        TensorStorageSupport.markModified(storage);
    }

    /**
     * Returns the mutable FLOAT32 backing array.
     *
     * @return storage-order float array
     * @throws UnsupportedOperationException if the tensor storage is not FLOAT32
     */
    public float[] getFloat32Data() {
        return TensorStorageSupport.float32Data(storage);
    }

    /**
     * Returns the mutable FLOAT64 backing array.
     *
     * @return storage-order double array
     * @throws UnsupportedOperationException if the tensor storage is not FLOAT64
     */
    public double[] getFloat64Data() {
        return TensorStorageSupport.float64Data(storage);
    }

    /**
     * Returns the mutable BFLOAT16 backing array.
     *
     * @return storage-order raw bfloat16 bit array
     * @throws UnsupportedOperationException if the tensor storage is not BFLOAT16
     */
    public short[] getBFloat16Data() {
        return TensorStorageSupport.bfloat16Data(storage);
    }

    /**
     * Returns the mutable INT32 backing array.
     *
     * @return storage-order int array
     * @throws UnsupportedOperationException if the tensor storage is not INT32
     */
    public int[] getInt32Data() {
        return TensorStorageSupport.int32Data(storage);
    }

    /**
     * Returns the mutable INT64 backing array.
     *
     * @return storage-order long array
     * @throws UnsupportedOperationException if the tensor storage is not INT64
     */
    public long[] getInt64Data() {
        return TensorStorageSupport.int64Data(storage);
    }

    /**
     * Returns the mutable BOOL backing array.
     *
     * @return storage-order byte array where non-zero values are true
     * @throws UnsupportedOperationException if the tensor storage is not BOOL
     */
    public byte[] getBoolData() {
        return TensorStorageSupport.boolData(storage);
    }

    /**
     * Returns the number of logical tensor elements.
     *
     * @return product of shape dimensions
     */
    public int getFlatDataSize(){
        return metadata.getFlatSize();
    }

    int getStorageSize() {
        return storage != null ? storage.getSize() : metadata.getFlatSize();
    }

    /**
     * Returns mutable double storage for FLOAT64 tensors.
     *
     * @return storage-order double array
     * @throws UnsupportedOperationException if this tensor is not FLOAT64
     */
    public double[] getData() {
        if (metadata.getDataType() != DataType.FLOAT64) {
            throw new UnsupportedOperationException("getData() is only supported for FLOAT64 tensors. Use typed storage getters or toDoubleArrayCopy().");
        }
        return getFloat64Data();
    }

    /**
     * Compatibility hook for older mirrored-data callers.
     *
     * <p>This implementation intentionally does nothing because non-FLOAT64
     * tensors do not maintain a mirrored {@code double[]} view.</p>
     */
    public void markDataViewStale() {
        // no-op: non-F64 tensors no longer maintain a mirrored double[] view
    }

    /**
     * Reports whether the tensor layout is contiguous in row-major order.
     *
     * @return true when shape and strides describe contiguous storage
     */
    public boolean isContiguous() {
        return metadata.isContiguous();
    }

    boolean isBroadcastView() {
        return metadata.isBroadcastView();
    }

    /**
     * Reports whether logical element zero begins after physical storage offset zero.
     *
     * @return true when this tensor is a storage-offset view
     */
    public boolean hasStorageOffset() {
        return metadata.hasStorageOffset();
    }

    /**
     * Returns a debugging representation of shape, dtype, strides, and storage.
     *
     * @return human-readable structure string
     */
    public String toStructString(){
        return TensorDebugSupport.toStructString(this);
    }

    /**
     * Returns the operation that produces this tensor.
     *
     * @return operation node, or null for leaf/constant tensors
     */
    public Operation getOperation(){
        return operation;
    }

    /**
     * Copies all logical values into a new double array.
     *
     * @return logical row-major copy converted to double values
     */
    public double[] toDoubleArrayCopy() {
        return TensorDebugSupport.toDoubleArrayCopy(this);
    }

    /**
     * Copies all logical values into a new boolean array.
     *
     * @return logical row-major boolean copy
     */
    public boolean[] toBooleanArrayCopy() {
        return TensorDebugSupport.toBooleanArrayCopy(this);
    }

    /**
     * Reads a single-element tensor as a double.
     *
     * @return the only logical value converted to double
     * @throws IllegalStateException if the tensor does not contain exactly one element
     */
    public double scalarAsDouble() {
        return TensorDebugSupport.scalarAsDouble(this);
    }

    void setBackendInternal(ComputeBackend backend) {
        this.forcedBackend = backend;
    }

    /**
     * Resolves the backend that should execute this tensor graph.
     *
     * @return forced backend when set, otherwise the configured default backend
     */
    public ComputeBackend resolveBackend() {
        return TensorExecutionSupport.resolveBackend(forcedBackend);
    }

    void setOperationInternal(Operation operation){
        this.operation=operation;
    }


    void setPrevTensorsInternal(List<Tensor> prevTensors) {
        this.prevTensors = prevTensors == null ? null : new ArrayList<>(prevTensors);
    }

    void setGradientInternal(Tensor t) {
        AutogradCompilationScope scope = AutogradCompilationScope.current();
        if (scope != null) {
            scope.setGradient(this, t);
            return;
        }
        this.gradient=t;
    }

    List<Tensor> prevTensorsRef() {
        return prevTensors;
    }



    /**
     * Returns tensors in topological dependency order ending at this tensor.
     *
     * @return list of graph tensors sorted from dependencies toward this tensor
     */
    public List<Tensor> topologicalSort() {
        return TensorGraphTraversal.topologicalSort(this);
    }

    /**
     * Prepares an execution plan for this tensor using an explicit profile.
     *
     * @param profile execution profile; must be non-null
     * @return prepared execution artifact
     */
    public PreparedExecution prepare(ExecutionProfile profile) {
        return TensorExecutionSupport.prepare(this, profile);
    }

    /**
     * Compiles this tensor graph for inference-only execution.
     *
     * @return compiled graph artifact
     */
    public CompiledGraph compile() {
        return TensorExecutionSupport.compile(this, CompileMode.INFERENCE_ONLY);
    }

    /**
     * Compiles this tensor graph with an explicit compile mode.
     *
     * @param compileMode compile mode; null handling is delegated to execution support
     * @return compiled graph artifact
     */
    public CompiledGraph compile(CompileMode compileMode) {
        return TensorExecutionSupport.compile(this, compileMode);
    }

    /**
     * Computes this tensor using default compute options.
     *
     * @return computed output tensor
     */
    public Tensor compute() {
        return TensorExecutionSupport.compute(this);
    }

    /**
     * Computes this tensor using an explicit compile mode.
     *
     * @param compileMode compile mode; null handling is delegated to execution support
     * @return computed output tensor
     */
    public Tensor compute(CompileMode compileMode) {
        return TensorExecutionSupport.compute(this, compileMode);
    }

    /**
     * Computes this tensor using mutable compute options.
     *
     * @param options options controlling compile mode, autotuning, optimizer, and runtime;
     *                null handling is delegated to execution support
     * @return computed output tensor
     */
    public Tensor compute(ComputeOptions options) {
        return TensorExecutionSupport.compute(this, options);
    }

    /**
     * Computes this tensor using an explicit execution profile.
     *
     * @param profile execution profile; must be non-null
     */
    public void compute(ExecutionProfile profile) {
        TensorExecutionSupport.compute(this, profile);
    }

    /**
     * Executes an already prepared execution artifact.
     *
     * @param execution prepared execution; must be non-null
     * @param mode runtime execution mode; must be non-null
     */
    public void compute(PreparedExecution execution, ExecutionMode mode) {
        TensorExecutionSupport.compute(execution, mode);
    }















    //
    // Operations below
    //

    /**
     * Creates a tensor of ones with the same shape and dtype as another tensor.
     *
     * @param other tensor whose metadata is copied; must be non-null
     * @return new tensor filled with ones
     */
    public static Tensor onesLike(Tensor other) {
        return TensorDataFactory.onesLike(other);
    }

    /**
     * Creates a tensor of zeros with the same shape and dtype as another tensor.
     *
     * @param other tensor whose metadata is copied; must be non-null
     * @return new tensor filled with zeros
     */
    public static Tensor zerosLike(Tensor other) {
        return TensorDataFactory.zerosLike(other);
    }

    /**
     * Returns a contiguous-layout tensor with the same logical values.
     *
     * @return contiguous tensor; input storage may be copied by the operation
     */
    public Tensor contiguous(){
        return TensorOps.contiguous(this);
    }

    /**
     * Reshapes this tensor without changing logical element order.
     *
     * <pre>{@code
     * Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
     * Tensor y = x.reshape(3, 2); // y shape is [3, 2], values are [[1, 2], [3, 4], [5, 6]]
     * }</pre>
     *
     * @param newShape target shape; one dimension may be {@code -1} for inference
     * @return tensor with the requested shape
     * @throws IllegalArgumentException if the requested shape is invalid or changes element count
     */
    public Tensor reshape(int... newShape) {
        return TensorOps.reshape(this, newShape);
    }

    /**
     * Broadcasts singleton dimensions to a larger shape.
     *
     * <pre>{@code
     * Tensor row = new Tensor(new double[]{10, 20, 30}, new int[]{1, 3}, null, "row");
     * Tensor grid = row.expand(2, 3); // logical values [[10, 20, 30], [10, 20, 30]]
     * }</pre>
     *
     * @param newShape broadcast target shape
     * @return broadcast view sharing storage with this tensor
     * @throws IllegalArgumentException if shapes are not broadcast-compatible
     */
    public Tensor expand(int... newShape) {
        return TensorOps.expand(this, newShape);
    }

    /**
     * Reorders dimensions using a permutation of axes.
     *
     * @param axes axis permutation; negative axes are normalized
     * @return strided view with permuted shape and strides
     * @throws IllegalArgumentException if axes are duplicated, missing, or out of range
     */
    public Tensor permute(int... axes) {
        return TensorOps.permute(this, axes);
    }

    /**
     * Transposes a rank-2 tensor.
     *
     * @return view with axes {@code [1, 0]}
     * @throws IllegalStateException if this tensor is not rank 2
     */
    public Tensor transpose() {
        int rank = this.getShape().length;
        if (rank != 2) {
            throw new IllegalStateException("transpose() requires rank-2 tensor, got rank=" + rank);
        }
        return this.permute(1, 0);
    }

    /**
     * Inserts a size-1 dimension.
     *
     * @param axis insertion position in {@code [0, rank]}; negative axes are normalized
     * @return view with one additional dimension
     */
    public Tensor expandDims(int axis) {
        return TensorOps.expandDims(this, axis);
    }

    /**
     * Removes a size-1 dimension.
     *
     * @param axis axis to remove; negative axes are normalized
     * @return view with one fewer dimension
     * @throws IllegalArgumentException if the selected dimension is not size 1
     */
    public Tensor squeeze(int axis) {
        return TensorOps.squeeze(this, axis);
    }

    public Tensor slice(int[] starts, int[] ends, int[] axes, int[] steps) {
        return TensorOps.slice(this, starts, ends, axes, steps);
    }

    /**
     * Slices one axis with positive step 1.
     *
     * <p>This is the one-axis ergonomic form of {@link #slice(int[], int[], int[], int[])}.
     * The rank is preserved and only the selected axis length changes.</p>
     *
     * @param axis axis to slice; negative axes are normalized
     * @param fromInclusive inclusive start index
     * @param toExclusive exclusive end index
     * @return sliced tensor view
     */
    public Tensor sliceAxis(int axis, int fromInclusive, int toExclusive) {
        return TensorOps.sliceAxis(this, axis, fromInclusive, toExclusive);
    }

    public Tensor pad(int[] before, int[] after, double constantValue) {
        return TensorOps.pad(this, before, after, constantValue);
    }

    public Tensor tile(int... repeats) {
        return TensorOps.tile(this, repeats);
    }

    public Tensor cast(DataType targetType) {
        return TensorOps.cast(this, targetType);
    }

    public static Tensor concat(int axis, Tensor... inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("concat inputs cannot be null");
        }
        return TensorOps.concat(axis, List.of(inputs));
    }

    /**
     * Stacks tensors by inserting a new axis.
     *
     * <p>All inputs must have the same shape and dtype. The inserted axis has
     * length {@code inputs.length}. For example, stacking three tensors shaped
     * {@code [batch, features]} at axis {@code 1} produces
     * {@code [batch, 3, features]}.</p>
     *
     * @param axis insertion position in {@code [0, rank]}; negative axes are normalized
     * @param inputs same-shaped tensors to stack
     * @return tensor with one additional dimension
     * @throws IllegalArgumentException if inputs are null, empty, null-containing, shape-mismatched, or dtype-mismatched
     */
    public static Tensor stack(int axis, Tensor... inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("stack inputs cannot be null");
        }
        return TensorOps.stack(axis, Arrays.asList(inputs));
    }

    /**
     * Adds another tensor elementwise with broadcasting.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted sum tensor
     */
    public Tensor add (Tensor second){
        return TensorOps.add(this, second);

    }

    /**
     * Subtracts another tensor elementwise with broadcasting.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted difference tensor
     */
    public Tensor sub (Tensor second){
        return TensorOps.sub(this, second);
    }

    /**
     * Multiplies another tensor elementwise with broadcasting.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted product tensor
     */
    public Tensor mul (Tensor second){
        return TensorOps.mul(this, second);

    }

    /**
     * Divides by another tensor elementwise with broadcasting.
     *
     * @param second denominator tensor; must be floating and broadcast-compatible
     * @return broadcasted quotient tensor
     */
    public Tensor div (Tensor second){
        return TensorOps.div(this, second);
    }

    /**
     * Computes elementwise minimum with another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted minimum tensor
     */
    public Tensor min(Tensor second) {
        return TensorOps.min(this, second);
    }

    /**
     * Computes elementwise maximum with another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted maximum tensor
     */
    public Tensor max(Tensor second) {
        return TensorOps.max(this, second);
    }

    /**
     * Raises this tensor to an elementwise tensor exponent with broadcasting.
     *
     * @param exponent exponent tensor; must be floating and broadcast-compatible
     * @return broadcasted power tensor
     */
    public Tensor pow(Tensor exponent) {
        return TensorOps.pow(this, exponent);
    }

    /**
     * Compares whether this tensor is elementwise greater than another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor greaterThan(Tensor second) {
        return TensorOps.greaterThan(this, second);
    }

    /**
     * Compares whether this tensor is elementwise less than another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor lessThan(Tensor second) {
        return TensorOps.lessThan(this, second);
    }

    /**
     * Compares whether this tensor is elementwise greater than or equal to another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor greaterOrEqual(Tensor second) {
        return TensorOps.greaterOrEqual(this, second);
    }

    /**
     * Compares whether this tensor is elementwise less than or equal to another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor lessOrEqual(Tensor second) {
        return TensorOps.lessOrEqual(this, second);
    }

    /**
     * Compares elementwise equality with another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor equalTo(Tensor second) {
        return TensorOps.equalTo(this, second);
    }

    /**
     * Compares elementwise inequality with another tensor.
     *
     * @param second right operand; must be floating and broadcast-compatible
     * @return broadcasted BOOL tensor
     */
    public Tensor notEqualTo(Tensor second) {
        return TensorOps.notEqualTo(this, second);
    }

    /**
     * Selects values from two branch tensors using a BOOL condition.
     *
     * <pre>{@code
     * Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask");
     * Tensor out = Tensor.where(mask, Tensor.scalar(9).expand(3), Tensor.scalar(2).expand(3));
     * // logical values are [9, 2, 9]
     * }</pre>
     *
     * @param condition BOOL tensor broadcast-compatible with both branches
     * @param ifTrue floating branch used where condition is true
     * @param ifFalse floating branch used where condition is false
     * @return broadcasted selected tensor
     */
    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return TensorOps.where(condition, ifTrue, ifFalse);
    }

    /**
     * Selects one index from a dimension and removes that dimension.
     *
     * @param dimension axis to select from; negative axes are normalized
     * @param index element index; negative indices are normalized
     * @return strided view of the selected slice
     */
    public Tensor select(int dimension, int index) {
        return TensorOps.select(this, dimension, index);
    }

    /**
     * Alias for elementwise minimum.
     *
     * @param second right operand
     * @return broadcasted minimum tensor
     */
    public Tensor minimum(Tensor second) {
        return TensorOps.minimum(this, second);
    }

    /**
     * Alias for elementwise maximum.
     *
     * @param second right operand
     * @return broadcasted maximum tensor
     */
    public Tensor maximum(Tensor second) {
        return TensorOps.maximum(this, second);
    }

    /**
     * Computes logical AND with another BOOL tensor.
     *
     * @param second BOOL tensor broadcast-compatible with this tensor
     * @return broadcasted BOOL tensor
     */
    public Tensor logicalAnd(Tensor second) {
        return TensorOps.logicalAnd(this, second);
    }

    /**
     * Computes logical OR with another BOOL tensor.
     *
     * @param second BOOL tensor broadcast-compatible with this tensor
     * @return broadcasted BOOL tensor
     */
    public Tensor logicalOr(Tensor second) {
        return TensorOps.logicalOr(this, second);
    }

    /**
     * Computes logical NOT of a BOOL tensor.
     *
     * @return BOOL tensor with the same shape
     */
    public Tensor logicalNot() {
        return TensorOps.logicalNot(this);
    }

    /**
     * Gathers values from one dimension using integer-like indices.
     *
     * <pre>{@code
     * Tensor x = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "x");
     * Tensor idx = new Tensor(new int[]{2, 0}, new int[]{2}, null, "idx");
     * Tensor y = x.gather(idx, 1); // logical values [30, 40]
     * }</pre>
     *
     * @param indices numeric integral index tensor shaped like this tensor without {@code dimension}
     * @param dimension source axis; negative axes are normalized
     * @return gathered tensor
     */
    public Tensor gather(Tensor indices, int dimension) {
        return TensorOps.gather(this, indices, dimension);
    }

    /**
     * ONNX-style gather that inserts the index tensor shape at the gathered axis.
     *
     * @param indices numeric integral index tensor
     * @param axis source axis; negative axes are normalized
     * @return gathered tensor with shape {@code dataShape[:axis] + indicesShape + dataShape[axis + 1:]}
     */
    public Tensor gatherAxis(Tensor indices, int axis) {
        return TensorOps.gatherAxis(this, indices, axis);
    }

    /**
     * Gathers values from one axis using ONNX Gather-style shape semantics.
     *
     * <p>The index tensor shape is inserted at {@code axis}. Result shape is
     * {@code dataShape[:axis] + indicesShape + dataShape[axis + 1:]}.</p>
     *
     * @param axis source axis; negative axes are normalized
     * @param indices numeric integral index tensor
     * @return gathered tensor
     */
    public Tensor take(int axis, Tensor indices) {
        return TensorOps.take(this, axis, indices);
    }

    /**
     * Gathers values from one axis using a Java index list.
     *
     * <p>The Java array is copied into an INT32 index tensor and then interpreted
     * with the same shape semantics as {@link #take(int, Tensor)}.</p>
     *
     * @param axis source axis; negative axes are normalized
     * @param indices integer indices; must be non-null and non-empty
     * @return gathered tensor
     */
    public Tensor take(int axis, int[] indices) {
        return TensorOps.take(this, axis, indices);
    }

    /**
     * Splits this tensor along one axis and removes that axis from each output.
     *
     * <p>For input shape {@code [2, 3, 4]}, {@code unstack(1)} returns three
     * tensors shaped {@code [2, 4]}. The outputs are graph tensors built from
     * {@code select} operations, so gradients scatter back into the source tensor.</p>
     *
     * @param axis axis to unstack; negative axes are normalized
     * @return one tensor per position along {@code axis}
     */
    public Tensor[] unstack(int axis) {
        return TensorOps.unstack(this, axis);
    }

    /**
     * ONNX GatherND with {@code batch_dims=0}.
     *
     * <p>The final dimension of {@code indices} contains a tuple of coordinates
     * into this tensor. The output shape is
     * {@code indices.shape[:-1] + this.shape[indices.shape[-1]:]}.</p>
     *
     * @param indices numeric integral index tensor whose final dimension is the tuple rank
     * @return gathered tensor with tuple-indexed values or slices
     */
    public Tensor gatherNd(Tensor indices) {
        return TensorOps.gatherNd(this, indices);
    }

    /**
     * ONNX GatherND with explicit leading batch dimensions.
     *
     * <p>The output shape is
     * {@code indices.shape[:batchDims] + indices.shape[batchDims:-1]
     * + this.shape[batchDims + indices.shape[-1]:]}.</p>
     *
     * @param indices numeric integral index tensor whose final dimension is the tuple rank
     * @param batchDims number of leading dimensions shared by this tensor and {@code indices}
     * @return gathered tensor with tuple-indexed values or slices
     */
    public Tensor gatherNd(Tensor indices, int batchDims) {
        return TensorOps.gatherNd(this, indices, batchDims);
    }

    /**
     * Adds source values into indexed positions of this tensor's shape.
     *
     * @param indices numeric integral index tensor
     * @param src source values matching {@code indices} shape
     * @param dimension target axis; negative axes are normalized
     * @return tensor containing this tensor plus scattered additions
     */
    public Tensor scatterAdd(Tensor indices, Tensor src, int dimension) {
        return TensorOps.scatterAdd(this, indices, src, dimension);
    }

    /**
     * Writes updates into a copy of this tensor using rank-preserving axis indices.
     *
     * @param indices numeric integral index tensor with the same rank as this tensor
     * @param updates update values with shape equal to {@code indices}
     * @param axis target axis; negative axes are normalized
     * @return tensor with this tensor's shape and scattered updates
     */
    public Tensor scatterElements(Tensor indices, Tensor updates, int axis) {
        return TensorOps.scatterElements(this, indices, updates, axis);
    }

    /**
     * Writes or reduces updates into a copy of this tensor using rank-preserving axis indices.
     *
     * @param indices numeric integral index tensor with the same rank as this tensor
     * @param updates update values with shape equal to {@code indices}
     * @param axis target axis; negative axes are normalized
     * @param reduction scatter reduction policy
     * @return tensor with this tensor's shape and scattered updates
     */
    public Tensor scatterElements(Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        return TensorOps.scatterElements(this, indices, updates, axis, reduction);
    }

    /**
     * Writes updates into a copy of this tensor using tuple indices.
     *
     * <p>The final dimension of {@code indices} contains a tuple of coordinates
     * into this tensor. {@code updates} must have shape
     * {@code indices.shape[:-1] + this.shape[indices.shape[-1]:]}.</p>
     *
     * @param indices numeric integral index tensor whose final dimension is the tuple rank
     * @param updates update values matching the indexed slice shape
     * @return tensor with this tensor's shape and scattered updates
     */
    public Tensor scatterNd(Tensor indices, Tensor updates) {
        return TensorOps.scatterNd(this, indices, updates);
    }

    /**
     * Writes or reduces updates into a copy of this tensor using tuple indices.
     *
     * @param indices numeric integral index tensor whose final dimension is the tuple rank
     * @param updates update values matching the indexed slice shape
     * @param reduction scatter reduction policy
     * @return tensor with this tensor's shape and scattered updates
     */
    public Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction) {
        return TensorOps.scatterNd(this, indices, updates, reduction);
    }

    /**
     * Writes or reduces updates into a copy of this tensor using tuple indices
     * with optional leading batch dimensions.
     *
     * @param indices numeric integral index tensor whose final dimension is the tuple rank
     * @param updates update values matching the indexed slice shape
     * @param reduction scatter reduction policy
     * @param batchDims leading batch dimensions shared by this tensor and indices
     * @return tensor with this tensor's shape and scattered updates
     */
    public Tensor scatterNd(Tensor indices, Tensor updates, ScatterReduction reduction, int batchDims) {
        return TensorOps.scatterNd(this, indices, updates, reduction, batchDims);
    }

    /**
     * Adds rank-changing axis updates into a copy of this tensor.
     *
     * @param indices numeric integral index tensor used like gatherAxis indices
     * @param updates updates shaped as {@code gatherAxis(this, indices, axis)}
     * @param axis target axis; negative axes are normalized
     * @return tensor with this tensor's shape and scattered additions
     */
    public Tensor scatterAxisAdd(Tensor indices, Tensor updates, int axis) {
        return TensorOps.scatterAxisAdd(this, indices, updates, axis);
    }

    /**
     * Gathers values using an output-shaped index tensor.
     *
     * @param indices numeric integral index tensor whose shape is the output shape
     * @param dimension source axis; negative axes are normalized
     * @return gathered tensor with shape {@code indices.getShape()}
     */
    public Tensor takeAlongAxis(Tensor indices, int dimension) {
        return TensorOps.takeAlongAxis(this, indices, dimension);
    }

    /**
     * Computes elementwise absolute value.
     *
     * @return shape-preserving absolute value tensor
     */
    public Tensor abs() {
        return TensorOps.abs(this);
    }

    /**
     * Multiplies this tensor as a matrix or batch of matrices.
     *
     * @param second right matrix operand
     * @return matrix product tensor
     */
    public Tensor matmul(Tensor second) {
        return TensorOps.matmul(this, second);
    }

    /**
     * Applies a linear projection without bias.
     *
     * <p>The receiver may have any rank greater than or equal to 2. The last
     * dimension is treated as {@code inFeatures}. The weight shape must be
     * {@code [inFeatures, outFeatures]}. The output shape equals the input shape
     * with the last dimension replaced by {@code outFeatures}.</p>
     *
     * @param weight projection weight tensor shaped {@code [inFeatures, outFeatures]}
     * @return projected tensor shaped {@code [..., outFeatures]}
     */
    public Tensor linear(Tensor weight) {
        return TensorOps.linear(this, weight);
    }

    /**
     * Applies a linear projection with bias.
     *
     * <p>The receiver may have any rank greater than or equal to 2. The last
     * dimension is treated as {@code inFeatures}. Bias is broadcast over every
     * leading dimension and may be shaped {@code [outFeatures]} or
     * {@code [1, outFeatures]}.</p>
     *
     * @param weight projection weight tensor shaped {@code [inFeatures, outFeatures]}
     * @param bias bias tensor shaped {@code [outFeatures]} or {@code [1, outFeatures]}
     * @return projected tensor plus bias shaped {@code [..., outFeatures]}
     */
    public Tensor linear(Tensor weight, Tensor bias) {
        return TensorOps.linear(this, weight, bias);
    }

    /**
     * Applies rank-4 NCHW convolution without bias.
     *
     * @param weight convolution weights shaped {@code [C_out, C_in / groups, kH, kW]}
     * @param options convolution options; must be non-null
     * @return convolution output tensor
     */
    public Tensor conv2d(Tensor weight, Conv2dOptions options) {
        return TensorOps.conv2d(this, weight, options);
    }

    /**
     * Applies rank-4 NCHW convolution with bias.
     *
     * @param weight convolution weights shaped {@code [C_out, C_in / groups, kH, kW]}
     * @param bias optional bias shaped {@code [C_out]}; may be null
     * @param options convolution options; must be non-null
     * @return convolution output tensor
     */
    public Tensor conv2d(Tensor weight, Tensor bias, Conv2dOptions options) {
        return TensorOps.conv2d(this, weight, bias, options);
    }

    /**
     * Applies 2-D max pooling to a rank-4 NCHW tensor.
     *
     * @param options pooling options; must be non-null
     * @return pooled output tensor
     */
    public Tensor maxPool2d(Pool2dOptions options) {
        return TensorOps.maxPool2d(this, options);
    }

    /**
     * Applies 2-D average pooling to a rank-4 NCHW tensor.
     *
     * @param options pooling options; must be non-null
     * @return pooled output tensor
     */
    public Tensor avgPool2d(Pool2dOptions options) {
        return TensorOps.avgPool2d(this, options);
    }

    /**
     * Uses this tensor as query in scaled dot-product attention.
     *
     * @param key key tensor
     * @param value value tensor
     * @param options attention options; must be non-null
     * @return attention output tensor
     */
    public Tensor scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options) {
        return TensorOps.scaledDotProductAttention(this, key, value, options);
    }

    /**
     * Uses this tensor as query in scaled dot-product attention with a mask.
     *
     * @param key key tensor
     * @param value value tensor
     * @param mask optional BOOL mask broadcast-compatible with score shape
     * @param options attention options; must be non-null
     * @return attention output tensor
     */
    public Tensor scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options) {
        return TensorOps.scaledDotProductAttention(this, key, value, mask, options);
    }

    /**
     * Negates every element.
     *
     * @return shape-preserving negated tensor
     */
    public Tensor neg (){
        return TensorOps.neg(this);

    }

    /**
     * Computes natural logarithm elementwise.
     *
     * @return shape-preserving log tensor
     */
    public Tensor log (){
        return TensorOps.log(this);
    }

    /**
     * Computes exponential elementwise.
     *
     * @return shape-preserving exponential tensor
     */
    public Tensor exp (){
        return TensorOps.exp(this);
    }

    /**
     * Computes the Gaussian error function elementwise.
     *
     * @return shape-preserving error-function tensor
     */
    public Tensor erf() {
        return TensorOps.erf(this);
    }

    /**
     * Computes implementation-specific approximate exponential elementwise.
     *
     * @return shape-preserving approximate exponential tensor
     */
    public Tensor fastExp() {
        return TensorOps.fastExp(this);
    }

    /**
     * Computes implementation-specific approximate tanh elementwise.
     *
     * @return shape-preserving approximate tanh tensor
     */
    public Tensor fastTanh() {
        return TensorOps.fastTanh(this);
    }

    /**
     * Clamps each element into an inclusive range.
     *
     * @param minValue lower bound
     * @param maxValue upper bound
     * @return shape-preserving clamped tensor
     * @throws IllegalArgumentException if {@code minValue > maxValue}
     */
    public Tensor clamp(double minValue, double maxValue) {
        return TensorOps.clamp(this, minValue, maxValue);
    }

    /**
     * Clamps each element to be at least {@code minValue}.
     *
     * @param minValue lower bound
     * @return shape-preserving clamped tensor
     */
    public Tensor clampMin(double minValue) {
        return TensorOps.clampMin(this, minValue);
    }

    /**
     * Clamps each element to be at most {@code maxValue}.
     *
     * @param maxValue upper bound
     * @return shape-preserving clamped tensor
     */
    public Tensor clampMax(double maxValue) {
        return TensorOps.clampMax(this, maxValue);
    }

    /**
     * Raises each element to a scalar exponent.
     *
     * @param exp exponent
     * @return shape-preserving power tensor
     */
    public Tensor pow(double exp) {
        return TensorOps.pow(this, exp);
    }

    /**
     * Multiplies each element by a scalar.
     *
     * @param scalar scalar multiplier
     * @return shape-preserving scaled tensor
     */
    public Tensor mul(double scalar) {
        return TensorOps.mulScalar(this, scalar);
    }

    /**
     * Computes reciprocal elementwise.
     *
     * @return shape-preserving reciprocal tensor
     */
    public Tensor inv() {
        return TensorOps.inv(this);
    }

    /**
     * Wraps this tensor in a synthetic forward-output marker operation.
     *
     * @return graph tensor with identical shape and dtype
     */
    public Tensor forwardOutput() {
        return TensorPrimitiveBuilder.unary(this, this.getShape(), new noop(), SYSTEM_FORWARD_OUTPUT_LABEL, this.getDataType());
    }

    /**
     * Computes square root elementwise.
     *
     * @return shape-preserving square-root tensor
     */
    public Tensor sqrt() {
        return TensorOps.sqrt(this);
    }

    public Tensor floor() {
        return TensorOps.floor(this);
    }

    public Tensor ceil() {
        return TensorOps.ceil(this);
    }

    public Tensor sign() {
        return TensorOps.sign(this);
    }

    /**
     * Applies logistic sigmoid elementwise.
     *
     * @return shape-preserving sigmoid tensor
     */
    public Tensor sigmoid() {
        return TensorOps.sigmoid(this);
    }

    /**
     * Applies hyperbolic tangent elementwise.
     *
     * @return shape-preserving tanh tensor
     */
    public Tensor tanh() {
        return TensorOps.tanh(this);
    }

    /**
     * Applies rectified linear activation.
     *
     * @return shape-preserving ReLU tensor
     */
    public Tensor relu() {
        return TensorOps.relu(this);
    }


    /**
     * Sums along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced sum tensor
     */
    public Tensor sum(int dimension){
        return TensorOps.sum(this, dimension);

    }

    /**
     * Sums along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced sum tensor
     */
    public Tensor sum(int dimension, boolean keepDims) {
        return TensorOps.sum(this, dimension, keepDims);
    }

    /**
     * Sums along one dimension while ignoring positions where {@code mask} is false.
     *
     * <p>The mask must have dtype {@link DataType#BOOL} and be broadcastable to
     * this tensor. Masked-out values contribute zero to the sum. This method
     * removes {@code dimension}; use explicit composition when a keep-dims masked
     * reduction is needed.</p>
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param mask BOOL mask broadcastable to this tensor
     * @return masked sum tensor
     */
    public Tensor sum(int dimension, Tensor mask) {
        return TensorOps.sum(this, dimension, mask);
    }

    /**
     * Sums all elements into a shape {@code [1]} tensor.
     *
     * @return scalar-like sum tensor
     */
    public Tensor sum(){
        return TensorOps.sumAll(this);
    }

    /**
     * Averages along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced mean tensor
     */
    public Tensor mean(int dimension) {
        return TensorOps.mean(this, dimension);
    }

    /**
     * Averages along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced mean tensor
     */
    public Tensor mean(int dimension, boolean keepDims) {
        return TensorOps.mean(this, dimension, keepDims);
    }

    /**
     * Averages along one dimension while ignoring positions where {@code mask} is false.
     *
     * <p>The mask must have dtype {@link DataType#BOOL} and be broadcastable to
     * this tensor. The denominator is the number of true mask entries for each
     * output element, not the full reduced-axis length. All-masked positions
     * return zero instead of NaN or infinity.</p>
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param mask BOOL mask broadcastable to this tensor
     * @return masked mean tensor
     */
    public Tensor mean(int dimension, Tensor mask) {
        return TensorOps.mean(this, dimension, mask);
    }

    /**
     * Averages all elements into a shape {@code [1]} tensor.
     *
     * @return scalar-like mean tensor
     */
    public Tensor mean() {
        return TensorOps.meanAll(this);
    }

    public Tensor prod(int dimension) {
        return TensorOps.prod(this, dimension);
    }

    public Tensor prod(int dimension, boolean keepDims) {
        return TensorOps.prod(this, dimension, keepDims);
    }

    public Tensor prod() {
        return TensorOps.prodAll(this);
    }

    public Tensor argMax(int dimension) {
        return TensorOps.argMax(this, dimension);
    }

    public Tensor argMax(int dimension, boolean keepDims) {
        return TensorOps.argMax(this, dimension, keepDims);
    }

    public Tensor argMax(int dimension, boolean keepDims, operations.reduction.ArgMaxTiePolicy tiePolicy) {
        return TensorOps.argMax(this, dimension, keepDims, tiePolicy);
    }

    /**
     * Computes cumulative sums along one axis.
     *
     * @param axis axis to scan; negative axes are normalized
     * @return shape-preserving cumulative sum tensor
     */
    public Tensor cumSum(int axis) {
        return TensorOps.cumSum(this, axis);
    }

    /**
     * Computes cumulative sums along one axis.
     *
     * @param axis axis to scan; negative axes are normalized
     * @param exclusive true to exclude the current element from its prefix
     * @param reverse true to scan from the end of the selected axis
     * @return shape-preserving cumulative sum tensor
     */
    public Tensor cumSum(int axis, boolean exclusive, boolean reverse) {
        return TensorOps.cumSum(this, axis, exclusive, reverse);
    }

    /**
     * Applies softmax along one dimension.
     *
     * @param dimension axis over which output values sum to 1
     * @return shape-preserving softmax tensor
     */
    public Tensor softmax(int dimension) {
        return TensorOps.softmax(this, dimension);
    }

    /**
     * Applies log-softmax along one dimension.
     *
     * @param dimension axis over which log probabilities are normalized
     * @return shape-preserving log-softmax tensor
     */
    public Tensor logSoftmax(int dimension) {
        return TensorOps.logSoftmax(this, dimension);
    }

    /**
     * Applies batch normalization using statistics computed from this tensor.
     *
     * @param gamma rank-1 scale parameter
     * @param beta rank-1 bias parameter
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as this tensor
     */
    public Tensor batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon) {
        return TensorOps.batchNorm(this, gamma, beta, channelDimension, epsilon);
    }

    /**
     * Applies batch normalization using supplied mean and variance.
     *
     * @param gamma rank-1 scale parameter
     * @param beta rank-1 bias parameter
     * @param mean rank-1 mean tensor
     * @param variance rank-1 variance tensor
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as this tensor
     */
    public Tensor batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon) {
        return TensorOps.batchNorm(this, gamma, beta, mean, variance, channelDimension, epsilon);
    }

    /**
     * Applies layer normalization over trailing dimensions matching {@code gamma}.
     *
     * @param gamma scale parameter whose shape matches the normalized trailing axes
     * @param beta bias parameter with the same shape as {@code gamma}
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as this tensor
     */
    public Tensor layerNorm(Tensor gamma, Tensor beta, double epsilon) {
        return TensorOps.layerNorm(this, gamma, beta, epsilon);
    }

    /**
     * Applies RMS normalization over trailing dimensions matching {@code gamma}.
     *
     * @param gamma scale parameter whose shape matches the normalized trailing axes
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as this tensor
     */
    public Tensor rmsNorm(Tensor gamma, double epsilon) {
        return TensorOps.rmsNorm(this, gamma, epsilon);
    }

    /**
     * Computes dense-target negative log-likelihood loss.
     *
     * @param targets dense target tensor with the same shape as this tensor
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} loss tensor
     */
    public Tensor nllLoss(Tensor targets, int classDimension) {
        return TensorOps.nllLoss(this, targets, classDimension);
    }

    /**
     * Computes dense-target cross-entropy loss.
     *
     * @param targets dense target tensor with the same shape as this tensor
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} loss tensor
     */
    public Tensor crossEntropyLoss(Tensor targets, int classDimension) {
        return TensorOps.crossEntropyLoss(this, targets, classDimension);
    }

    /**
     * Computes dense-target cross-entropy loss while ignoring masked-out samples.
     *
     * <p>The mask is applied after reducing {@code classDimension}. For logits
     * shaped {@code [batch, time, classes]} and {@code classDimension = 2}, the
     * natural mask shape is {@code [batch, time]}. The mean denominator is the
     * number of true mask entries.</p>
     *
     * @param targets dense target tensor with the same shape as this tensor
     * @param classDimension class axis; negative axes are normalized
     * @param mask BOOL mask broadcastable to this tensor with the class axis removed
     * @return shape {@code [1]} mean loss normalized by valid sample count
     */
    public Tensor crossEntropyLoss(Tensor targets, int classDimension, Tensor mask) {
        return TensorOps.crossEntropyLoss(this, targets, classDimension, mask);
    }

    /**
     * Computes mean negative log-likelihood loss from integer-like target indices.
     *
     * <pre>{@code
     * Tensor logProbs = logits.logSoftmax(1); // logits shape [batch, classes]
     * Tensor target = new Tensor(new int[]{2, 0}, new int[]{2}, null, "target");
     * Tensor loss = logProbs.nllLossFromIndices(target, 1); // shape [1]
     * }</pre>
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @return mean loss tensor with shape {@code [1]}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension);
    }

    /**
     * Computes negative log-likelihood loss from target indices.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, reduction);
    }

    /**
     * Computes weighted negative log-likelihood loss from target indices.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param classWeights rank-1 tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, classWeights, reduction);
    }

    /**
     * Computes mean NLL loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and mean denominator
     * @return mean loss tensor with shape {@code [1]}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex);
    }

    /**
     * Computes NLL loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and reduction weights
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex, reduction);
    }

    /**
     * Computes weighted NLL loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and reduction weights
     * @param classWeights rank-1 tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    /**
     * Computes mean cross-entropy loss from integer-like target indices.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @return mean loss tensor with shape {@code [1]}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension);
    }

    /**
     * Computes cross-entropy loss from integer-like target indices.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, reduction);
    }

    /**
     * Computes index-target cross-entropy loss while ignoring masked-out samples.
     *
     * <p>The mask is broadcast against the per-sample loss shape, which is the
     * logits shape with {@code classDimension} removed. The mean denominator is
     * the number of true mask entries.</p>
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param mask BOOL mask broadcastable to {@code targetIndices}
     * @return shape {@code [1]} mean loss normalized by valid sample count
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor mask) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, mask);
    }

    /**
     * Computes weighted cross-entropy loss from integer-like target indices.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param classWeights rank-1 tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, classWeights, reduction);
    }

    /**
     * Computes mean cross-entropy loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and mean denominator
     * @return mean loss tensor with shape {@code [1]}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex);
    }

    /**
     * Computes cross-entropy loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and reduction weights
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex, reduction);
    }

    /**
     * Computes weighted cross-entropy loss while ignoring one target index value.
     *
     * @param targetIndices class indices shaped like this tensor without {@code classDimension}
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from loss and reduction weights
     * @param classWeights rank-1 tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor shaped according to {@code reduction}
     */
    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    /**
     * Reduces minimum along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced minimum tensor
     */
    public Tensor min(int dimension) {
        return TensorOps.min(this, dimension);
    }

    /**
     * Reduces minimum along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced minimum tensor
     */
    public Tensor min(int dimension, boolean keepDims) {
        return TensorOps.min(this, dimension, keepDims);
    }

    /**
     * Reduces minimum across all elements.
     *
     * @return shape {@code [1]} minimum tensor
     */
    public Tensor min() {
        return TensorOps.minAll(this);
    }

    /**
     * Reduces maximum along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced maximum tensor
     */
    public Tensor max(int dimension) {
        return TensorOps.max(this, dimension);
    }

    /**
     * Reduces maximum along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced maximum tensor
     */
    public Tensor max(int dimension, boolean keepDims) {
        return TensorOps.max(this, dimension, keepDims);
    }

    /**
     * Reduces maximum across all elements.
     *
     * @return shape {@code [1]} maximum tensor
     */
    public Tensor max() {
        return TensorOps.maxAll(this);
    }

    /**
     * Computes logical AND along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced BOOL tensor
     */
    public Tensor all(int dimension) {
        return TensorOps.all(this, dimension);
    }

    /**
     * Computes logical AND along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced BOOL tensor
     */
    public Tensor all(int dimension, boolean keepDims) {
        return TensorOps.all(this, dimension, keepDims);
    }

    /**
     * Computes logical AND across all elements.
     *
     * @return shape {@code [1]} BOOL tensor
     */
    public Tensor all() {
        return TensorOps.allAll(this);
    }

    /**
     * Computes logical OR along one dimension and removes that dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced BOOL tensor
     */
    public Tensor any(int dimension) {
        return TensorOps.any(this, dimension);
    }

    /**
     * Computes logical OR along one dimension.
     *
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to keep the reduced axis with size 1
     * @return reduced BOOL tensor
     */
    public Tensor any(int dimension, boolean keepDims) {
        return TensorOps.any(this, dimension, keepDims);
    }

    /**
     * Computes logical OR across all elements.
     *
     * @return shape {@code [1]} BOOL tensor
     */
    public Tensor any() {
        return TensorOps.anyAll(this);
    }


    //lambda section
    void buildBackwardGraphInternal() {
        if (this.backwardFunction != null) {
            this.backwardFunction.run(); // Spustí se připravená lambda
        }
    }

    void setBackwardFunctionInternal(Runnable backwardFunction) {
        this.backwardFunction = backwardFunction;
    }

    Runnable backwardFunctionInternal() {
        return backwardFunction;
    }

    double getByStorageOffset(int offset) {
        return TensorStorageSupport.getByStorageOffset(storage, getStorageSize(), offset);
    }

    void setByStorageOffset(int offset, double value) {
        TensorStorageSupport.setByStorageOffset(storage, getStorageSize(), offset, value);
    }

    void aliasRuntimeFromInternal(Tensor source) {
        if (source == null) {
            throw new IllegalArgumentException("source tensor cannot be null");
        }
        this.storage = source.storage;
    }

    void replaceStorageInternal(TensorStorage replacement) {
        Objects.requireNonNull(replacement, "replacement storage cannot be null");
        requireDenseStorageReplacement("replaceStorageInternal");
        if (replacement.getType() != metadata.getDataType()) {
            throw new IllegalArgumentException("replacement storage dtype " + replacement.getType()
                    + " does not match tensor dtype " + metadata.getDataType());
        }
        TensorStorageSupport.validateInputLength(replacement.getSize(), metadata.getFlatSize(), "TensorStorage");
        this.storage = replacement;
    }

    private void requireDenseStorageReplacement(String operationName) {
        if (!metadata.isContiguous() || metadata.hasStorageOffset()) {
            throw new UnsupportedOperationException(operationName
                    + " cannot replace storage for non-dense tensor layout. Use contiguous() first.");
        }
    }




}
