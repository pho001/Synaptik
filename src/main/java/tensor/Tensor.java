package tensor;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import operations.layout.noop;
import tensor.factory.TensorArrayData;
import tensor.loss.LossReduction;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.*;


public class Tensor {
    public static final String SYSTEM_FORWARD_OUTPUT_LABEL = "System_Forward_Output";
    private TensorStorage storage;
    private TensorMetadata metadata;
    private Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private ComputeBackend forcedBackend = null;
    private Runnable backwardFunction;
    private boolean isBackward = false;





    public Tensor(Object multiDimArray, List<Tensor> previous, String label) {
        this(multiDimArray, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType) {
        int[] computedShape = TensorArrayData.inferShape(multiDimArray);
        this.metadata = new TensorMetadata(computedShape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        storage = TensorStorageSupport.fromDoubleArray(metadata, TensorArrayData.flattenToDouble(multiDimArray, metadata.getFlatSize()));
    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label) {
        this(dimensions, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

        this.metadata = new TensorMetadata(dimensions, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        storage = TensorStorageSupport.emptyStorage(metadata);
    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        this(shape, previous, operation, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        storage = TensorStorageSupport.emptyStorage(metadata, calculateSize(shape));
    }

    public Tensor(int[] shape, int[] strides, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this(shape, strides, 0, previous, operation, label, dataType);
    }

    public Tensor(int[] shape, int[] strides, int storageOffset, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, strides, storageOffset, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        storage = TensorStorageSupport.emptyStorage(metadata);
    }

    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        storage = TensorStorageSupport.fromDoubleArray(metadata, data);
    }

    public Tensor(float[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    public Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        storage = TensorStorageSupport.fromFloatArray(metadata, data);
    }

    public Tensor(short[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    public Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        storage = TensorStorageSupport.fromBFloat16Array(metadata, data);
    }

    public Tensor(byte[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, DataType.BOOL);
    }

    public Tensor(byte[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    public Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {
        this(data, shape, strides, previous, label, DataType.BOOL);
    }

    public Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        storage = TensorStorageSupport.fromBoolArray(metadata, data);
    }

    public Tensor(int[] data, int[] shape, List<Tensor> previous, String label) {
        this(data, shape, previous, label, DataType.INT32);
    }

    public Tensor(int[] data, int[] shape, List<Tensor> previous, String label, DataType dataType) {
        this(data, shape, TensorMetadata.computeStrides(shape), previous, label, dataType);
    }

    public Tensor(int[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType) {
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        storage = TensorStorageSupport.fromIntArray(metadata, data);
    }



    public static Tensor scalar(double value) {
        return scalar(value, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public static Tensor scalar(double value, DataType dataType) {
        return TensorDataFactory.scalar(value, dataType);
    }

    public int calculateSize(int[] dimensions) {
        return Arrays.stream(dimensions).reduce(1, (a, b) -> a * b);
    }

    public int getStride(int index){
        return metadata.getStride(index);
    }

    public boolean getRequiresGrad(){
        return metadata.requiresGrad();
    }

    public void setRequiresGrad(boolean requiresGrad){
        metadata.setRequiresGrad(requiresGrad);
    }

    public String getLabel() {
        return metadata.getLabel();
    }
    public void setLabel(String label) {
        metadata.setLabel(label);
    }


    public double getByFlatIndex(int index){
        if (index < 0 || index >= getFlatDataSize()) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return getByStorageOffset(TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, index));
    }

    public void setDataAt(int flatindex,double value) {
        if (isBroadcastView()) {
            throw new UnsupportedOperationException("Cannot write through broadcast view tensor.");
        }
        setByStorageOffset(TensorStorageSupport.logicalFlatIndexToStorageOffset(metadata, flatindex), value);
        markStorageModified();
    }

    public int[] getStrides() {
        return metadata.getStrides();
    }

    public int[] getStridesUnsafe() {
        return metadata.stridesRef();
    }

    public int getStorageOffsetUnsafe() {
        return metadata.getStorageOffset();
    }

    public void setData(double[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        storage = TensorStorageSupport.fromDoubleArray(metadata, data);
    }

    public void setData(float[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        storage = TensorStorageSupport.fromFloatArray(metadata, data);
    }

    public void setData(short[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        storage = TensorStorageSupport.fromBFloat16Array(metadata, data);
    }

    public void setData(byte[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        storage = TensorStorageSupport.fromBoolArray(metadata, data);
    }

    public void setData(int[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        TensorStorageSupport.validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        storage = TensorStorageSupport.fromIntArray(metadata, data);
    }

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

    public void setFloat32Data(float[] data) {
        if (metadata.getDataType() != DataType.FLOAT32) {
            throw new UnsupportedOperationException("setFloat32Data() is only supported for FLOAT32 tensors.");
        }
        setData(data);
    }

    public int getFlatIndex(int[] indices) {
        return metadata.getFlatIndex(indices);
    }

    public int[] getSpatialIndex(int index){
        return metadata.getSpatialIndex(index);
    }


    public int[] getShape() {
        return metadata.getShape();
    }

    public int[] getShapeUnsafe() {
        return metadata.shapeRef();
    }

    public List<Tensor> getPrevTensors() {
        return prevTensors == null ? null : Collections.unmodifiableList(prevTensors);
    }

    public int getDimensionAt(int index) {
        return metadata.getDimensionAt(index);
    }

    public Tensor getGradient(){
        AutogradCompilationScope scope = AutogradCompilationScope.current();
        if (scope != null) {
            return scope.gradientOf(this);
        }
        return gradient;
    }

    public int[] computeStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    public boolean isBackward() {
        return isBackward;
    }
    void setBackwardInternal(boolean backward) {
        isBackward = backward;
    }

    public int[] computeStrides() {
        return metadata.getStrides();
    }

    public DataType getDataType() {
        return metadata.getDataType();
    }

    public void setDataType(DataType dataType) {
        DataType current = metadata.getDataType();
        if (current == dataType) {
            return;
        }
        if (current == DataType.INT32 || dataType == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit INT32 <-> other dtype conversion is not supported.");
        }
        if ((current == DataType.BOOL) != (dataType == DataType.BOOL)) {
            throw new UnsupportedOperationException("Implicit BOOL <-> numeric dtype conversion is not supported.");
        }
        double[] snapshot = TensorDebugSupport.toDoubleStorageOrderCopy(this);
        metadata.setDataType(dataType);
        storage = TensorStorageSupport.fromDoubleArray(metadata, snapshot);
    }

    public TensorStorage getStorage() {
        return storage;
    }

    public long storageVersion() {
        return TensorStorageSupport.version(storage);
    }

    public void markStorageModified() {
        TensorStorageSupport.markModified(storage);
    }

    public float[] getFloat32Data() {
        return TensorStorageSupport.float32Data(storage);
    }

    public double[] getFloat64Data() {
        return TensorStorageSupport.float64Data(storage);
    }

    public short[] getBFloat16Data() {
        return TensorStorageSupport.bfloat16Data(storage);
    }

    public int[] getInt32Data() {
        return TensorStorageSupport.int32Data(storage);
    }

    public byte[] getBoolData() {
        return TensorStorageSupport.boolData(storage);
    }

    public int getFlatDataSize(){
        return metadata.getFlatSize();
    }

    int getStorageSize() {
        return storage != null ? storage.getSize() : metadata.getFlatSize();
    }

    public double[] getData() {
        if (metadata.getDataType() != DataType.FLOAT64) {
            throw new UnsupportedOperationException("getData() is only supported for FLOAT64 tensors. Use typed storage getters or toDoubleArrayCopy().");
        }
        return getFloat64Data();
    }

    public void markDataViewStale() {
        // no-op: non-F64 tensors no longer maintain a mirrored double[] view
    }

    public boolean isContiguous() {
        return metadata.isContiguous();
    }

    boolean isBroadcastView() {
        return metadata.isBroadcastView();
    }

    public boolean hasStorageOffset() {
        return metadata.hasStorageOffset();
    }

    public String toStructString(){
        return TensorDebugSupport.toStructString(this);
    }

    public Operation getOperation(){
        return operation;
    }

    public double[] toDoubleArrayCopy() {
        return TensorDebugSupport.toDoubleArrayCopy(this);
    }

    public boolean[] toBooleanArrayCopy() {
        return TensorDebugSupport.toBooleanArrayCopy(this);
    }

    public double scalarAsDouble() {
        return TensorDebugSupport.scalarAsDouble(this);
    }

    void setBackendInternal(ComputeBackend backend) {
        this.forcedBackend = backend;
    }

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



    public List<Tensor> topologicalSort() {
        return TensorGraphTraversal.topologicalSort(this);
    }

    public PreparedExecution prepare(ExecutionProfile profile) {
        return TensorExecutionSupport.prepare(this, profile);
    }

    public CompiledGraph compile() {
        return TensorExecutionSupport.compile(this, CompileMode.INFERENCE_ONLY);
    }

    public CompiledGraph compile(CompileMode compileMode) {
        return TensorExecutionSupport.compile(this, compileMode);
    }

    public Tensor compute() {
        return TensorExecutionSupport.compute(this);
    }

    public Tensor compute(CompileMode compileMode) {
        return TensorExecutionSupport.compute(this, compileMode);
    }

    public Tensor compute(ComputeOptions options) {
        return TensorExecutionSupport.compute(this, options);
    }

    public void compute(ExecutionProfile profile) {
        TensorExecutionSupport.compute(this, profile);
    }

    public void compute(PreparedExecution execution, ExecutionMode mode) {
        TensorExecutionSupport.compute(execution, mode);
    }















    //
    // Operations below
    //

    public static Tensor onesLike(Tensor other) {
        return TensorDataFactory.onesLike(other);
    }

    public static Tensor zerosLike(Tensor other) {
        return TensorDataFactory.zerosLike(other);
    }

    public Tensor contiguous(){
        return TensorOps.contiguous(this);
    }

    public Tensor reshape(int... newShape) {
        return TensorOps.reshape(this, newShape);
    }

    public Tensor expand(int... newShape) {
        return TensorOps.expand(this, newShape);
    }

    public Tensor permute(int... axes) {
        return TensorOps.permute(this, axes);
    }

    public Tensor transpose() {
        int rank = this.getShape().length;
        if (rank != 2) {
            throw new IllegalStateException("transpose() requires rank-2 tensor, got rank=" + rank);
        }
        return this.permute(1, 0);
    }

    public Tensor expandDims(int axis) {
        return TensorOps.expandDims(this, axis);
    }

    public Tensor squeeze(int axis) {
        return TensorOps.squeeze(this, axis);
    }

    public Tensor add (Tensor second){
        return TensorOps.add(this, second);

    }

    public Tensor sub (Tensor second){
        return TensorOps.sub(this, second);
    }

    public Tensor mul (Tensor second){
        return TensorOps.mul(this, second);

    }

    public Tensor div (Tensor second){
        return TensorOps.div(this, second);
    }

    public Tensor min(Tensor second) {
        return TensorOps.min(this, second);
    }

    public Tensor max(Tensor second) {
        return TensorOps.max(this, second);
    }

    public Tensor greaterThan(Tensor second) {
        return TensorOps.greaterThan(this, second);
    }

    public Tensor lessThan(Tensor second) {
        return TensorOps.lessThan(this, second);
    }

    public Tensor greaterOrEqual(Tensor second) {
        return TensorOps.greaterOrEqual(this, second);
    }

    public Tensor lessOrEqual(Tensor second) {
        return TensorOps.lessOrEqual(this, second);
    }

    public Tensor equalTo(Tensor second) {
        return TensorOps.equalTo(this, second);
    }

    public Tensor notEqualTo(Tensor second) {
        return TensorOps.notEqualTo(this, second);
    }

    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return TensorOps.where(condition, ifTrue, ifFalse);
    }

    public Tensor select(int dimension, int index) {
        return TensorOps.select(this, dimension, index);
    }

    public Tensor minimum(Tensor second) {
        return TensorOps.minimum(this, second);
    }

    public Tensor maximum(Tensor second) {
        return TensorOps.maximum(this, second);
    }

    public Tensor logicalAnd(Tensor second) {
        return TensorOps.logicalAnd(this, second);
    }

    public Tensor logicalOr(Tensor second) {
        return TensorOps.logicalOr(this, second);
    }

    public Tensor logicalNot() {
        return TensorOps.logicalNot(this);
    }

    public Tensor gather(Tensor indices, int dimension) {
        return TensorOps.gather(this, indices, dimension);
    }

    public Tensor scatterAdd(Tensor indices, Tensor src, int dimension) {
        return TensorOps.scatterAdd(this, indices, src, dimension);
    }

    public Tensor takeAlongAxis(Tensor indices, int dimension) {
        return TensorOps.takeAlongAxis(this, indices, dimension);
    }

    public Tensor abs() {
        return TensorOps.abs(this);
    }

    public Tensor matmul(Tensor second) {
        return TensorOps.matmul(this, second);
    }

    public Tensor linear(Tensor weight) {
        return TensorOps.linear(this, weight);
    }

    public Tensor linear(Tensor weight, Tensor bias) {
        return TensorOps.linear(this, weight, bias);
    }

    public Tensor conv2d(Tensor weight, Conv2dOptions options) {
        return TensorOps.conv2d(this, weight, options);
    }

    public Tensor conv2d(Tensor weight, Tensor bias, Conv2dOptions options) {
        return TensorOps.conv2d(this, weight, bias, options);
    }

    public Tensor maxPool2d(Pool2dOptions options) {
        return TensorOps.maxPool2d(this, options);
    }

    public Tensor avgPool2d(Pool2dOptions options) {
        return TensorOps.avgPool2d(this, options);
    }

    public Tensor scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options) {
        return TensorOps.scaledDotProductAttention(this, key, value, options);
    }

    public Tensor scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options) {
        return TensorOps.scaledDotProductAttention(this, key, value, mask, options);
    }

    public Tensor neg (){
        return TensorOps.neg(this);

    }

    public Tensor log (){
        return TensorOps.log(this);
    }

    public Tensor exp (){
        return TensorOps.exp(this);
    }

    public Tensor fastExp() {
        return TensorOps.fastExp(this);
    }

    public Tensor fastTanh() {
        return TensorOps.fastTanh(this);
    }

    public Tensor clamp(double minValue, double maxValue) {
        return TensorOps.clamp(this, minValue, maxValue);
    }

    public Tensor clampMin(double minValue) {
        return TensorOps.clampMin(this, minValue);
    }

    public Tensor clampMax(double maxValue) {
        return TensorOps.clampMax(this, maxValue);
    }

    public Tensor pow(double exp) {
        return TensorOps.pow(this, exp);
    }

    public Tensor mul(double scalar) {
        return TensorOps.mulScalar(this, scalar);
    }

    public Tensor inv() {
        return TensorOps.inv(this);
    }

    public Tensor forwardOutput() {
        return TensorPrimitiveBuilder.unary(this, this.getShape(), new noop(), SYSTEM_FORWARD_OUTPUT_LABEL, this.getDataType());
    }

    public Tensor sqrt() {
        return TensorOps.sqrt(this);
    }

    public Tensor sigmoid() {
        return TensorOps.sigmoid(this);
    }

    public Tensor tanh() {
        return TensorOps.tanh(this);
    }

    public Tensor relu() {
        return TensorOps.relu(this);
    }


    public Tensor sum(int dimension){
        return TensorOps.sum(this, dimension);

    }

    public Tensor sum(int dimension, boolean keepDims) {
        return TensorOps.sum(this, dimension, keepDims);
    }

    public Tensor sum(){
        return TensorOps.sumAll(this);
    }

    public Tensor mean(int dimension) {
        return TensorOps.mean(this, dimension);
    }

    public Tensor mean(int dimension, boolean keepDims) {
        return TensorOps.mean(this, dimension, keepDims);
    }

    public Tensor mean() {
        return TensorOps.meanAll(this);
    }

    public Tensor softmax(int dimension) {
        return TensorOps.softmax(this, dimension);
    }

    public Tensor logSoftmax(int dimension) {
        return TensorOps.logSoftmax(this, dimension);
    }

    public Tensor batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon) {
        return TensorOps.batchNorm(this, gamma, beta, channelDimension, epsilon);
    }

    public Tensor batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon) {
        return TensorOps.batchNorm(this, gamma, beta, mean, variance, channelDimension, epsilon);
    }

    public Tensor layerNorm(Tensor gamma, Tensor beta, double epsilon) {
        return TensorOps.layerNorm(this, gamma, beta, epsilon);
    }

    public Tensor rmsNorm(Tensor gamma, double epsilon) {
        return TensorOps.rmsNorm(this, gamma, epsilon);
    }

    public Tensor nllLoss(Tensor targets, int classDimension) {
        return TensorOps.nllLoss(this, targets, classDimension);
    }

    public Tensor crossEntropyLoss(Tensor targets, int classDimension) {
        return TensorOps.crossEntropyLoss(this, targets, classDimension);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, reduction);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, classWeights, reduction);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorOps.nllLossFromIndices(this, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, reduction);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, classWeights, reduction);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorOps.crossEntropyLossFromIndices(this, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    public Tensor min(int dimension) {
        return TensorOps.min(this, dimension);
    }

    public Tensor min(int dimension, boolean keepDims) {
        return TensorOps.min(this, dimension, keepDims);
    }

    public Tensor min() {
        return TensorOps.minAll(this);
    }

    public Tensor max(int dimension) {
        return TensorOps.max(this, dimension);
    }

    public Tensor max(int dimension, boolean keepDims) {
        return TensorOps.max(this, dimension, keepDims);
    }

    public Tensor max() {
        return TensorOps.maxAll(this);
    }

    public Tensor all(int dimension) {
        return TensorOps.all(this, dimension);
    }

    public Tensor all(int dimension, boolean keepDims) {
        return TensorOps.all(this, dimension, keepDims);
    }

    public Tensor all() {
        return TensorOps.allAll(this);
    }

    public Tensor any(int dimension) {
        return TensorOps.any(this, dimension);
    }

    public Tensor any(int dimension, boolean keepDims) {
        return TensorOps.any(this, dimension, keepDims);
    }

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




}
