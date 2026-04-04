package tensor;

import backend.ComputeBackend;
import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import operations.*;

import java.lang.reflect.Array;
import java.util.*;


public class Tensor {
    public static final String SYSTEM_FORWARD_OUTPUT_LABEL = "System_Forward_Output";
    private TensorStorage storage;
    private TensorMetadata metadata;
    private Runnable localgradients;
    public Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private ComputeBackend forcedBackend = null;
    private double [] intermediates;
    private Runnable backwardFunction;
    private boolean isBackward = false;





    public Tensor(Object multiDimArray, List<Tensor> previous, String label) {
        this(multiDimArray, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType) {
        int[] computedShape = calculateShape(multiDimArray);
        this.metadata = new TensorMetadata(computedShape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        initStorageFromDoubleArray(flatten(multiDimArray));
    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label) {
        this(dimensions, previous, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType) {
        int totalSize = 1;
        for (int dim : dimensions) {
            totalSize *= dim;
        }
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

        this.metadata = new TensorMetadata(dimensions, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        initEmptyStorage();
    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        this(shape, previous, operation, label, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        int totalSize=calculateSize(shape);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        initEmptyStorage(totalSize);
    }

    public Tensor(int[] shape, int[] strides, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this(shape, strides, 0, previous, operation, label, dataType);
    }

    public Tensor(int[] shape, int[] strides, int storageOffset, List<Tensor> previous, Operation operation, String label, DataType dataType) {
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, strides, storageOffset, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad), dataType);
        this.operation = operation;
        initEmptyStorage();
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
        validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        initStorageFromDoubleArray(data);
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
        validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        initStorageFromFloatArray(data);
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
        validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        initStorageFromFloat16Array(data);
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
        validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        initStorageFromBoolArray(data);
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
        validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        initStorageFromIntArray(data);
    }



    public static Tensor scalar(double value) {
        return scalar(value, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public static Tensor scalar(double value, DataType dataType) {
        if (dataType == DataType.INT32) {
            long integral = Math.round(value);
            if (Math.abs(value - integral) > 1e-9) {
                throw new IllegalArgumentException("INT32 scalar requires an integral value. got=" + value);
            }
            return new Tensor(new int[]{(int) integral}, new int[]{1}, new ArrayList<>(), "scalar_const", DataType.INT32);
        }
        // Skalár má data o délce 1 a prázdný tvar (nebo [1])
        double[] data = new double[]{value};
        int[] shape = new int[]{1};
        int[] strides = new int[]{1};

        // Vytvoříme tenzor bez operace (považován za konstantu)
        Tensor scalar = new Tensor(data, shape, strides, new ArrayList<>(), "scalar_const", dataType);

        // Pro AlgebraicRewite je klíčové, aby tenzor neměl operaci,
        // čímž signalizuje, že jde o list (vstupy/konstantu).
        return scalar;
    }

    public int calculateSize(int[] dimensions) {
        return Arrays.stream(dimensions).reduce(1, (a, b) -> a * b);
    }

    private int[] calculateShape(Object multiDimArray) {
        int[] dims = new int[getDepth(multiDimArray)];
        Object currentArray = multiDimArray;
        for (int i = 0; i < dims.length; i++) {
            dims[i] = Array.getLength(currentArray);
            if (Array.get(currentArray, 0).getClass().isArray()) {
                currentArray = Array.get(currentArray, 0);
            } else {
                break;
            }
        }
        return dims;
    }

    public int getStride(int index){
        return metadata.getStride(index);
    }

    private int getDepth(Object array) {
        int depth = 0;
        while (array.getClass().isArray()) {
            depth++;
            array = Array.get(array, 0);
        }
        return depth;
    }

    public boolean getRequiresGrad(){
        return metadata.requiresGrad();
    }

    public void setRequiresGrad(boolean requiresGrad){
        metadata.setRequiresGrad(requiresGrad);
    }

    private double[] flatten(Object multiDimArray) {
        int size = metadata.getFlatSize();
        double[] flatArray = new double[size];
        fillFlatArray(multiDimArray, flatArray, 0);
        return flatArray;
    }

    private static int fillFlatArray(Object multiDimArray, double[] flatArray, int startIndex) {
        int length = Array.getLength(multiDimArray);
        if (multiDimArray instanceof double[] row){
            System.arraycopy(row, 0, flatArray, startIndex, row.length);
            return startIndex+row.length;
        }
        else if (multiDimArray instanceof Object[]){
            int currentIndex = startIndex;
            for (Object element : (Object[]) multiDimArray) {
                currentIndex = fillFlatArray(element, flatArray, currentIndex);
            }
            return currentIndex;
        }
        else {
            throw new IllegalArgumentException("Multidimensional data must be either double, or n-dimensional object");
        }
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
        return getByStorageOffset(logicalFlatIndexToStorageOffset(index));
    }

    public void setDataAt(int flatindex,double value) {
        if (isBroadcastView()) {
            throw new UnsupportedOperationException("Cannot write through broadcast view tensor.");
        }
        setByStorageOffset(logicalFlatIndexToStorageOffset(flatindex), value);
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
        validateInputLength(data.length, metadata.getFlatSize(), "double[]");
        initStorageFromDoubleArray(data);
    }

    public void setData(float[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        validateInputLength(data.length, metadata.getFlatSize(), "float[]");
        initStorageFromFloatArray(data);
    }

    public void setData(short[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        validateInputLength(data.length, metadata.getFlatSize(), "short[]");
        initStorageFromFloat16Array(data);
    }

    public void setData(byte[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        validateInputLength(data.length, metadata.getFlatSize(), "byte[]");
        initStorageFromBoolArray(data);
    }

    public void setData(int[] data) {
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        validateInputLength(data.length, metadata.getFlatSize(), "int[]");
        initStorageFromIntArray(data);
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
        return prevTensors;
    }

    public int getDimensionAt(int index) {
        return metadata.getDimensionAt(index);
    }

    public Tensor getGradient(){
        return gradient;
    }

    public int[] computeStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    public boolean isBackward() {
        return isBackward;
    }
    public void setBackward(boolean backward) {
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
        double[] snapshot = toDoubleStorageOrderCopy();
        metadata.setDataType(dataType);
        initStorageFromDoubleArray(snapshot);
    }

    public TensorStorage getStorage() {
        return storage;
    }

    public float[] getFloat32Data() {
        if (storage instanceof Float32Storage s) {
            return s.getFloatArray();
        }
        return null;
    }

    public double[] getFloat64Data() {
        if (storage instanceof Float64Storage s) {
            return s.getDoubleArray();
        }
        return null;
    }

    public short[] getFloat16Data() {
        if (storage instanceof Float16Storage s) {
            return s.getShortArray();
        }
        return null;
    }

    public int[] getInt32Data() {
        if (storage instanceof Int32Storage s) {
            return s.getIntArray();
        }
        return null;
    }

    public byte[] getBoolData() {
        if (storage instanceof BoolStorage s) {
            return s.getByteArray();
        }
        return null;
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

    public void aliasRuntimeFrom(Tensor source) {
        if (source == null) {
            throw new IllegalArgumentException("source tensor cannot be null");
        }
        this.storage = source.storage;
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
        int[] shape = this.getShape();       // Tvar tensoru
        int[] strides = this.getStrides();   // Strides tensoru
        double[] data = this.toDoubleArrayCopy();
        StringBuilder sbData = new StringBuilder();
        StringBuilder sbGrads = new StringBuilder();
        buildTensorString(shape, strides, data, new int[shape.length], 0, sbData);

        String output="tensor.Tensor{" +
                "shape=" + Arrays.toString(shape) +
                ", strides=" + Arrays.toString(strides) +
                ", data=" + sbData +
                '}';

        return output;
    }

    private void buildTensorString(int[] shape, int[] strides, double[] data, int[] indices, int dim, StringBuilder sb) {
        // Pokud jsme v poslední dimenzi, vypíšeme hodnoty
        if (dim == shape.length) {
            int index = 0;
            for (int i = 0; i < indices.length; i++) {
                index += indices[i] * strides[i];
            }
            sb.append(data[index]);
            return;
        }


        sb.append("[");

        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            buildTensorString(shape, strides, data, indices, dim + 1, sb);
            if (i < shape[dim] - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
    }

    public Operation getOperation(){
        return operation;
    }

    public double[] toDoubleArrayCopy() {
        int n = getFlatDataSize();
        double[] out = new double[n];
        if (storage != null) {
            for (int i = 0; i < n; i++) {
                out[i] = getByStorageOffset(logicalFlatIndexToStorageOffset(i));
            }
        }
        return out;
    }

    public boolean[] toBooleanArrayCopy() {
        if (metadata.getDataType() != DataType.BOOL) {
            throw new UnsupportedOperationException("toBooleanArrayCopy() is only supported for BOOL tensors.");
        }
        int n = getFlatDataSize();
        boolean[] out = new boolean[n];
        byte[] data = getBoolData();
        for (int i = 0; i < n; i++) {
            out[i] = data[logicalFlatIndexToStorageOffset(i)] != 0;
        }
        return out;
    }

    private double[] toDoubleStorageOrderCopy() {
        int n = getStorageSize();
        double[] out = new double[n];
        if (storage != null) {
            for (int i = 0; i < n; i++) {
                out[i] = getByStorageOffset(i);
            }
        }
        return out;
    }

    public double scalarAsDouble() {
        if (getFlatDataSize() != 1) {
            throw new IllegalStateException("Tensor is not scalar.");
        }
        if (getDataType() == DataType.BOOL) {
            throw new UnsupportedOperationException("scalarAsDouble() is not supported for BOOL tensors.");
        }
        return getByFlatIndex(0);
    }

    public void setBackend(ComputeBackend backend) {
        this.forcedBackend = backend;
    }

    public ComputeBackend resolveBackend() {
        if (forcedBackend != null) {
            return forcedBackend;
        }
        return ComputeBackend.CPU;
    }

    public void setOperation(Operation operation){
        this.operation=operation;
    }


    public void setPrevTensors(List<Tensor> prevTensors) {
        this.prevTensors = prevTensors; // Zajištění měnitelné kolekce
    }

    public void setGradient(Tensor t) {
        this.gradient=t;
    }



    public List<Tensor> topologicalSort() {
        Deque<Tensor> sorted = new ArrayDeque<>();
        Set<Tensor> visited = new LinkedHashSet<>();
        topologicalSortHelper(this, visited, sorted);
        return new ArrayList<>(sorted); // Převedeme zpět na List
    }

    private void topologicalSortHelper(Tensor tensor, Set<Tensor> visited, Deque<Tensor> sorted) {
        if (!visited.contains(tensor)) {
            visited.add(tensor);

            if (tensor.prevTensors != null) {
                for (Tensor prev : tensor.prevTensors) {
                    topologicalSortHelper(prev, visited, sorted);
                }
            }

            if (tensor.prevTensors == null) {
                sorted.addFirst(tensor);
            } else {
                sorted.addLast(tensor);
            }
        }
    }

    public PreparedExecution prepare(ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return CompiledGraph.compile(this, profile.optimizer()).prepare(profile.runtime());
    }

    public void compute(ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        compute(prepare(profile), profile.mode());
    }

    public void compute(PreparedExecution execution, ExecutionMode mode) {
        if (execution == null) {
            throw new IllegalArgumentException("execution cannot be null");
        }
        execution.execute(mode);
    }















    //
    // Operations below
    //

    public static Tensor onesLike(Tensor other) {
        int size = other.getFlatDataSize();
        if (other.getDataType() == DataType.INT32) {
            int[] data = new int[size];
            java.util.Arrays.fill(data, 1);
            return new Tensor(
                    data,
                    other.getShape().clone(),
                    new java.util.ArrayList<>(),
                    "ones_like",
                    DataType.INT32
            );
        }
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0);
        Tensor out = new Tensor(
                data,
                other.getShape().clone(),
                new java.util.ArrayList<>(), // Žádní předci (je to konstanta)
                "ones_like",
                other.getDataType()
        );
        return out;
    }

    public static Tensor zerosLike(Tensor other) {
        int size = other.getFlatDataSize();
        if (other.getDataType() == DataType.INT32) {
            int[] data = new int[size];
            return new Tensor(
                    data,
                    other.getShape().clone(),
                    new java.util.ArrayList<>(),
                    "zeros_like",
                    DataType.INT32
            );
        }
        double[] data = new double[size]; // Java defaultně inicializuje na 0.0

        Tensor out = new Tensor(
                data,
                other.getShape().clone(),
                new java.util.ArrayList<>(),
                "zeros_like",
                other.getDataType()
        );
        return out;
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
        Operation op = new noop();
        Tensor out = new Tensor(this.getShape(), List.of(this), op, SYSTEM_FORWARD_OUTPUT_LABEL, this.getDataType());
        return out;
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
    public void buildBackwardGraph() {
        if (this.backwardFunction != null) {
            this.backwardFunction.run(); // Spustí se připravená lambda
        }
    }

    public void setBackwardFunction(Runnable backwardFunction) {
        this.backwardFunction = backwardFunction;
    }

    private void initEmptyStorage() {
        initEmptyStorage(metadata.getFlatSize());
    }

    private void initEmptyStorage(int size) {
        DataType type = normalizeDataType(metadata.getDataType());
        if (type == DataType.FLOAT64) {
            storage = new Float64Storage(size);
            return;
        }
        storage = switch (type) {
            case BOOL -> new BoolStorage(size);
            case FLOAT16 -> new Float16Storage(size);
            case FLOAT32 -> new Float32Storage(size);
            case INT32 -> new Int32Storage(size);
            case FLOAT64 -> throw new IllegalStateException("Unexpected dtype branch");
        };
    }

    private void initStorageFromDoubleArray(double[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        }
        int size = source.length;
        if (type == DataType.FLOAT64) {
            storage = new Float64Storage(source);
            return;
        }
        if (type == DataType.FLOAT32) {
            float[] converted = new float[size];
            for (int i = 0; i < size; i++) {
                converted[i] = (float) source[i];
            }
            storage = new Float32Storage(converted);
            return;
        }
        short[] converted = new short[size];
        for (int i = 0; i < size; i++) {
            converted[i] = CpuDTypeOps.toHalfBits((float) source[i]);
        }
        storage = new Float16Storage(converted);
    }

    private void initStorageFromFloatArray(float[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        }
        int size = source.length;
        switch (type) {
            case FLOAT32 -> {
                storage = new Float32Storage(source);
            }
            case FLOAT64 -> {
                double[] converted = new double[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = source[i];
                }
                storage = new Float64Storage(converted);
            }
            case FLOAT16 -> {
                short[] converted = new short[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = CpuDTypeOps.toHalfBits(source[i]);
                }
                storage = new Float16Storage(converted);
            }
        }
    }

    private void initStorageFromFloat16Array(short[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        if (type == DataType.BOOL || type == DataType.INT32) {
            throw new UnsupportedOperationException("Implicit numeric -> BOOL/INT32 storage conversion is not supported.");
        }
        int size = source.length;
        switch (type) {
            case FLOAT16 -> {
                storage = new Float16Storage(source);
            }
            case FLOAT32 -> {
                float[] converted = new float[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = CpuDTypeOps.fromHalfBits(source[i]);
                }
                storage = new Float32Storage(converted);
            }
            case FLOAT64 -> {
                double[] converted = new double[size];
                for (int i = 0; i < size; i++) {
                    converted[i] = CpuDTypeOps.fromHalfBits(source[i]);
                }
                storage = new Float64Storage(converted);
            }
        }
    }

    private void initStorageFromBoolArray(byte[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        if (type != DataType.BOOL) {
            throw new UnsupportedOperationException("Implicit BOOL -> numeric storage conversion is not supported.");
        }
        byte[] normalized = new byte[source.length];
        for (int i = 0; i < source.length; i++) {
            normalized[i] = source[i] == 0 ? (byte) 0 : (byte) 1;
        }
        storage = new BoolStorage(normalized);
    }

    private void initStorageFromIntArray(int[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        if (type != DataType.INT32) {
            throw new UnsupportedOperationException("Implicit INT32 -> other dtype conversion is not supported.");
        }
        storage = new Int32Storage(source);
    }

    private static void validateInputLength(int actual, int expected, String sourceName) {
        if (actual != expected) {
            throw new IllegalArgumentException(sourceName + " length mismatch. expected=" + expected + ", actual=" + actual);
        }
    }

    double getByStorageOffset(int offset) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= getStorageSize()) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        return switch (storage.getType()) {
            case FLOAT64 -> getFloat64Data()[offset];
            case FLOAT32 -> getFloat32Data()[offset];
            case FLOAT16 -> CpuDTypeOps.fromHalfBits(getFloat16Data()[offset]);
            case INT32 -> getInt32Data()[offset];
            case BOOL -> getBoolData()[offset] == 0 ? 0.0d : 1.0d;
        };
    }

    void setByStorageOffset(int offset, double value) {
        if (storage == null) {
            throw new IllegalStateException("Tensor storage is not initialized.");
        }
        if (offset < 0 || offset >= getStorageSize()) {
            throw new IndexOutOfBoundsException("Storage offset out of bounds.");
        }
        switch (storage.getType()) {
            case FLOAT64 -> getFloat64Data()[offset] = value;
            case FLOAT32 -> getFloat32Data()[offset] = (float) value;
            case FLOAT16 -> getFloat16Data()[offset] = CpuDTypeOps.toHalfBits((float) value);
            case INT32 -> {
                long integral = Math.round(value);
                if (Math.abs(value - integral) > 1e-9) {
                    throw new UnsupportedOperationException("Non-integral write into INT32 storage is not supported.");
                }
                getInt32Data()[offset] = (int) integral;
            }
            case BOOL -> throw new UnsupportedOperationException("Numeric write into BOOL storage is not supported.");
        }
    }

    private int logicalFlatIndexToStorageOffset(int logicalIndex) {
        int[] shape = metadata.shapeRef();
        int[] strides = metadata.stridesRef();
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        int rem = logicalIndex;
        int offset = metadata.getStorageOffset();
        for (int dim = 0; dim < shape.length; dim++) {
            int coord = rem / denseStrides[dim];
            rem %= denseStrides[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    private DataType normalizeDataType(DataType dataType) {
        if (dataType != null) {
            return dataType;
        }
        metadata.setDataType(TensorMetadata.DEFAULT_DATA_TYPE);
        return metadata.getDataType();
    }




}
