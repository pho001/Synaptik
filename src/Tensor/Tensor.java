package Tensor;

import Backend.ComputeBackend;
import Backend.kernels.cpu.CpuKernel;
import Graph.CompiledGraph;
import Graph.optimizer.GraphOptimizer;
import Operations.*;

import java.lang.reflect.Array;
import java.util.*;


public class Tensor {
    private double[] data;
    private TensorMetadata metadata;
    private Runnable localgradients;
    public Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private boolean isCompiled = false;
    private CompiledGraph compiledGraph;
    private ComputeBackend forcedBackend = null;
    private ComputeBackend resolvedBackend;
    private double [] intermediates;
    private CpuKernel resolvedCpuKernel;
    private Runnable backwardFunction;
    private boolean isBackward = false;





    public Tensor(Object multiDimArray, List<Tensor> previous, String label) {
        int[] computedShape = calculateShape(multiDimArray);
        this.metadata = new TensorMetadata(computedShape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad));
        this.data = flatten(multiDimArray);
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label) {

        int totalSize = 1;
        for (int dim : dimensions) {
            totalSize *= dim;
        }
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

        this.data = new double[totalSize];
        this.metadata = new TensorMetadata(dimensions, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad));

    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        int totalSize=calculateSize(shape);
        //this.data = new double[totalSize];
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad));
        this.operation = operation;
        this.data=new double[totalSize];
    }


    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad));

    }

    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.metadata = new TensorMetadata(shape, strides, label, previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad));

    }

    public static Tensor scalar(double value) {
        // Skalár má data o délce 1 a prázdný tvar (nebo [1])
        double[] data = new double[]{value};
        int[] shape = new int[]{1};
        int[] strides = new int[]{1};

        // Vytvoříme tenzor bez operace (považován za konstantu)
        Tensor scalar = new Tensor(data, shape, strides, new ArrayList<>(), "scalar_const");

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
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return data[index];
    }

    public void setDataAt(int flatindex,double value) {
        if (this.data==null)
            this.data = new double[metadata.getFlatSize()];
        this.data[flatindex]=value;
    }

    public int[] getStrides() {
        return metadata.getStrides();
    }

    public void setData(double[] data) {
        this.data=data;
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

    public int getFlatDataSize(){
        return this.data.length;
    }

    public double[] getData() {
        return data;
    }

    public boolean isContiguous() {
        return metadata.isContiguous();
    }

    public String toStructString(){
        int[] shape = this.getShape();       // Tvar tensoru
        int[] strides = this.getStrides();   // Strides tensoru
        double[] data = this.getData();
        StringBuilder sbData = new StringBuilder();
        StringBuilder sbGrads = new StringBuilder();
        buildTensorString(shape, strides, data, new int[shape.length], 0, sbData);

        String output="Tensor.Tensor{" +
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

    public void setBackend(ComputeBackend backend) {
        if (operation != null && !operation.supportsBackend(backend)) {
            throw new IllegalArgumentException("operations.Operation doesn't support backend: " + backend);
        }
        this.forcedBackend = backend;
        this.resolvedBackend = null;
        this.resolvedCpuKernel = null;
    }

    public ComputeBackend resolveBackend() {
        if (forcedBackend != null) {
            return forcedBackend;
        }
        if (operation == null || operation.getPreferredBackend()==null){
            return ComputeBackend.CPU;
        }
        return operation.getPreferredBackend();
    }

    public void setOperation(Operation operation){
        this.operation=operation;
        this.resolvedBackend = null;
        this.resolvedCpuKernel = null;
    }


    public void setPrevTensors(List<Tensor> prevTensors) {
        this.prevTensors = prevTensors; // Zajištění měnitelné kolekce
    }

    public void setIntermediates(double[] intermediates) {
        this.intermediates = intermediates;
    }

    public double[] getIntermediates() {
        /*
        if (intermediates == null) {
            throw new IllegalStateException("Intermediate values were not initialized");
            //return new double[this.getFlatDataSize()*7];
        }

         */
        return this.intermediates;
    }

    public void setGradient(Tensor t) {
        this.gradient=t;
    }

    public CpuKernel getResolvedCpuKernel() {
        return resolvedCpuKernel;
    }

    public void setResolvedCpuKernel(CpuKernel resolvedCpuKernel) {
        this.resolvedCpuKernel = resolvedCpuKernel;
    }

    public ComputeBackend getResolvedBackend() {
        return resolvedBackend;
    }

    public void setResolvedBackend(ComputeBackend resolvedBackend) {
        this.resolvedBackend = resolvedBackend;
    }

    public CompiledGraph getCompiledGraph() {
        return compiledGraph;
    }

    public void resetCompiledGraph() {
        this.compiledGraph = null;
    }



    public List<Tensor> topologicalSort() {
        Deque<Tensor> sorted = new LinkedList<>();
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



    public void compute(GraphOptimizer optimizer) {
        if (this.compiledGraph==null){

            compiledGraph=new CompiledGraph(this,optimizer);
        }
        //compiledGraph.forward();
        compiledGraph.execute();


    }


    public void compute(){
        GraphOptimizer optimizer = new GraphOptimizer();
        compute(optimizer);
    }


    public void backward(){

        if (compiledGraph==null){
            throw new RuntimeException("Can not compute gradients, forward pass must be done first");
        }
        compiledGraph.backward();
    }















    //
    // Operations below
    //

    public static Tensor onesLike(Tensor other) {
        int size = other.getFlatDataSize();
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0);
        return new Tensor(
                data,
                other.getShape().clone(),
                other.getStrides().clone(),
                new java.util.ArrayList<>(), // Žádní předci (je to konstanta)
                "ones_like"
        );
    }

    public static Tensor zerosLike(Tensor other) {
        int size = other.getFlatDataSize();
        double[] data = new double[size]; // Java defaultně inicializuje na 0.0

        return new Tensor(
                data,
                other.getShape().clone(),
                other.getStrides().clone(),
                new java.util.ArrayList<>(),
                "zeros_like"
        );
    }

    public Tensor contiguous(){
        return TensorOps.contiguous(this);
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

    public Tensor neg (){
        return TensorOps.neg(this);

    }

    public Tensor log (){
        return TensorOps.log(this);
    }

    public Tensor exp (){
        return TensorOps.exp(this);
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
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "noop");
        return out;
    }

    public Tensor sqrt() {
        return TensorOps.sqrt(this);
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




}
