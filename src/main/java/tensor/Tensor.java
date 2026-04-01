package tensor;

import backend.ComputeBackend;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.OptimizerFactory;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import config.profile.ExecutionProfile;
import config.optimizer.OptimizerConfig;
import operations.*;

import java.lang.reflect.Array;
import java.util.*;


public class Tensor {
    private TensorStorage storage;
    private TensorMetadata metadata;
    private Runnable localgradients;
    public Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private CompiledGraph compiledGraph;
    private GraphOptimizer compiledWithOptimizer;
    private OptimizerConfig compiledWithOptimizerConfig;
    private PreparedExecution lastPreparedExecution;
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



    public static Tensor scalar(double value) {
        return scalar(value, TensorMetadata.DEFAULT_DATA_TYPE);
    }

    public static Tensor scalar(double value, DataType dataType) {
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
        return storage.getAsDoubleAt(index);
    }

    public void setDataAt(int flatindex,double value) {
        storage.setAsDoubleAt(flatindex, value);
    }

    public int[] getStrides() {
        return metadata.getStrides();
    }

    public int[] getStridesUnsafe() {
        return metadata.stridesRef();
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
        double[] snapshot = toDoubleArrayCopy();
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

    public int getFlatDataSize(){
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
                out[i] = storage.getAsDoubleAt(i);
            }
        }
        return out;
    }

    public double scalarAsDouble() {
        if (getFlatDataSize() != 1) {
            throw new IllegalStateException("Tensor is not scalar.");
        }
        return storage.getAsDoubleAt(0);
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

    public CompiledGraph getCompiledGraph() {
        return compiledGraph;
    }

    public void resetCompiledGraph() {
        this.compiledGraph = null;
        this.compiledWithOptimizer = null;
        this.compiledWithOptimizerConfig = null;
        this.lastPreparedExecution = null;
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

    @Deprecated(forRemoval = true)
    public void compute(GraphOptimizer optimizer) {
        compute(optimizer, (RuntimeConfig) null);
    }

    @Deprecated(forRemoval = true)
    public void prepareCompiledGraph(GraphOptimizer optimizer) {
        compile(optimizer);
    }


    @Deprecated(forRemoval = true)
    public void compute(){

        compute((GraphOptimizer) null, (RuntimeConfig) null);
    }


    public void backward(){

        if (lastPreparedExecution == null){
            throw new RuntimeException("Can not compute gradients, forward pass must be done first");
        }
        lastPreparedExecution.backward();
    }

    @Deprecated(forRemoval = true)
    public CompiledGraph compile(GraphOptimizer optimizer) {
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        if (compiledGraph == null || compiledWithOptimizer != optimizer) {
            compiledGraph = CompiledGraph.compile(this, optimizer);
            compiledWithOptimizer = optimizer;
            compiledWithOptimizerConfig = null;
        }
        return compiledGraph;
    }

    @Deprecated(forRemoval = true)
    public CompiledGraph compile(config.optimizer.OptimizerConfig optimizerConfig) {
        if (optimizerConfig == null) {
            throw new IllegalArgumentException("optimizerConfig cannot be null");
        }
        if (compiledGraph == null || !optimizerConfig.equals(compiledWithOptimizerConfig)) {
            compiledGraph = CompiledGraph.compile(this, optimizerConfig);
            compiledWithOptimizer = null;
            compiledWithOptimizerConfig = optimizerConfig;
        }
        return compiledGraph;
    }

    @Deprecated(forRemoval = true)
    public PreparedExecution prepare(GraphOptimizer optimizer, RuntimeConfig runtimeConfig) {
        CompiledGraph graph = compile(optimizer);
        PreparedExecution execution = graph.prepare(runtimeConfig);
        this.lastPreparedExecution = execution;
        return execution;
    }

    public PreparedExecution prepare(RuntimeConfig runtimeConfig) {
        return prepare(defaultOptimizerConfigForPrepare(), runtimeConfig);
    }

    public PreparedExecution prepare(ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        CompiledGraph graph = compile(profile.optimizer());
        PreparedExecution execution = graph.prepare(profile.runtime());
        rememberExecution(graph, execution, profile.optimizer());
        return execution;
    }

    @Deprecated(forRemoval = true)
    public void compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig, ExecutionMode mode) {
        PreparedExecution execution = prepare(optimizer, runtimeConfig);
        compute(execution, mode);
    }

    public void compute(RuntimeConfig runtimeConfig, ExecutionMode mode) {
        compute(prepare(runtimeConfig), mode);
    }

    @Deprecated(forRemoval = true)
    public void compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig) {
        GraphOptimizer effectiveOptimizer = optimizer == null ? OptimizerFactory.createTrainingOptimizer() : optimizer;
        ExecutionMode mode = compile(effectiveOptimizer).supportsBackward()
                ? ExecutionMode.FORWARD_BACKWARD
                : ExecutionMode.FORWARD;
        compute(effectiveOptimizer, runtimeConfig, mode);
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
        this.lastPreparedExecution = execution;
        execution.execute(mode);
    }

    private PreparedExecution prepare(OptimizerConfig optimizerConfig, RuntimeConfig runtimeConfig) {
        CompiledGraph graph = compile(optimizerConfig);
        PreparedExecution execution = graph.prepare(runtimeConfig);
        rememberExecution(graph, execution, optimizerConfig);
        return execution;
    }

    private void rememberExecution(CompiledGraph graph, PreparedExecution execution, OptimizerConfig optimizerConfig) {
        this.compiledGraph = graph;
        this.compiledWithOptimizer = null;
        this.compiledWithOptimizerConfig = optimizerConfig;
        this.lastPreparedExecution = execution;
    }

    private OptimizerConfig defaultOptimizerConfigForPrepare() {
        return hasTrainableLeafInputs() ? OptimizerConfig.trainingDefaults() : OptimizerConfig.inferenceDefaults();
    }

    private boolean hasTrainableLeafInputs() {
        for (Tensor tensor : topologicalSort()) {
            if ((tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) && tensor.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }















    //
    // Operations below
    //

    public static Tensor onesLike(Tensor other) {
        int size = other.getFlatDataSize();
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0);
        Tensor out = new Tensor(
                data,
                other.getShape().clone(),
                other.getStrides().clone(),
                new java.util.ArrayList<>(), // Žádní předci (je to konstanta)
                "ones_like"
        );
        out.setDataType(other.getDataType());
        return out;
    }

    public static Tensor zerosLike(Tensor other) {
        int size = other.getFlatDataSize();
        double[] data = new double[size]; // Java defaultně inicializuje na 0.0

        Tensor out = new Tensor(
                data,
                other.getShape().clone(),
                other.getStrides().clone(),
                new java.util.ArrayList<>(),
                "zeros_like"
        );
        out.setDataType(other.getDataType());
        return out;
    }

    public Tensor contiguous(){
        return TensorOps.contiguous(this);
    }

    public Tensor reshape(int... newShape) {
        return TensorOps.reshape(this, newShape);
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

    public Tensor matmul(Tensor second) {
        return TensorOps.matmul(this, second);
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
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "noop", this.getDataType());
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


    public Tensor sum(int dimension){
        return TensorOps.sum(this, dimension);

    }

    public Tensor sum(){
        return TensorOps.sumAll(this);
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
            case FLOAT16 -> new Float16Storage(size);
            case FLOAT32 -> new Float32Storage(size);
            case FLOAT64 -> throw new IllegalStateException("Unexpected dtype branch");
        };
    }

    private void initStorageFromDoubleArray(double[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        int size = source.length;
        if (type == DataType.FLOAT64) {
            storage = new Float64Storage(source);
            return;
        }
        TensorStorage target = (type == DataType.FLOAT32) ? new Float32Storage(size) : new Float16Storage(size);
        for (int i = 0; i < size; i++) {
            target.setAsDoubleAt(i, source[i]);
        }
        storage = target;
    }

    private void initStorageFromFloatArray(float[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
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
                Float16Storage converted = new Float16Storage(size);
                for (int i = 0; i < size; i++) {
                    converted.setAsDoubleAt(i, source[i]);
                }
                storage = converted;
            }
        }
    }

    private void initStorageFromFloat16Array(short[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source data cannot be null");
        }
        DataType type = normalizeDataType(metadata.getDataType());
        int size = source.length;
        switch (type) {
            case FLOAT16 -> {
                storage = new Float16Storage(source);
            }
            case FLOAT32 -> {
                Float32Storage converted = new Float32Storage(size);
                Float16Storage in = new Float16Storage(source);
                for (int i = 0; i < size; i++) {
                    converted.setAsDoubleAt(i, in.getAsDoubleAt(i));
                }
                storage = converted;
            }
            case FLOAT64 -> {
                double[] converted = new double[size];
                Float16Storage in = new Float16Storage(source);
                for (int i = 0; i < size; i++) {
                    converted[i] = in.getAsDoubleAt(i);
                }
                storage = new Float64Storage(converted);
            }
        }
    }

    private static void validateInputLength(int actual, int expected, String sourceName) {
        if (actual != expected) {
            throw new IllegalArgumentException(sourceName + " length mismatch. expected=" + expected + ", actual=" + actual);
        }
    }

    private DataType normalizeDataType(DataType dataType) {
        if (dataType != null) {
            return dataType;
        }
        metadata.setDataType(TensorMetadata.DEFAULT_DATA_TYPE);
        return metadata.getDataType();
    }




}
