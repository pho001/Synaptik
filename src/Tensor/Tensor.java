package Tensor;

import Backend.ComputeBackend;
import Graph.CompiledGraph;
import Graph.FuseElementWise;
import Graph.GraphOptimizer;
import Operations.*;

import java.lang.reflect.Array;
import java.util.*;


public class Tensor {
    private double[] data;
    private int[] shape;
    private String label;
    private boolean requiresGrad=true;
    private Runnable localgradients;
    private int[] strides;
    public Tensor gradient;
    private Operation operation;
    private List<Tensor> prevTensors=new ArrayList<>();
    private boolean isCompiled = false;
    private CompiledGraph compiledGraph;
    private ComputeBackend forcedBackend = null;
    private double [] intermediates;





    public Tensor(Object multiDimArray, List<Tensor> previous, String label) {
        this.shape = calculateShape(multiDimArray);
        this.data = flatten(multiDimArray);
        this.prevTensors = previous;
        this.label = label;
        this.strides=computeStrides(this.shape);

    }

    public Tensor(int[] dimensions, List<Tensor> previous, String label) {

        int totalSize = 1;
        for (int dim : dimensions) {
            totalSize *= dim;
        }

        this.data = new double[totalSize];
        this.prevTensors = previous;
        this.label = label;
        this.shape = dimensions;
        this.strides=computeStrides(dimensions);
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        int totalSize=calculateSize(shape);
        //this.data = new double[totalSize];
        this.prevTensors = previous;
        this.label = label;
        this.shape = shape;
        this.strides=computeStrides(shape);
        this.operation = operation;
        this.data=new double[calculateSize(shape)];
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);
    }


    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous;
        this.label = label;
        this.shape = shape;
        this.strides=computeStrides(shape);
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

    }

    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous;
        this.label = label;
        this.shape = shape;
        this.strides=strides;
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

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

        if (index < 0 || index >= shape.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return strides[index];
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
        return requiresGrad;
    }

    public void setRequiresGrad(boolean requiresGrad){
        this.requiresGrad=requiresGrad;
    }

    private double[] flatten(Object multiDimArray) {
        int size = calculateSize(shape);
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
        return label;
    }

    public double getByFlatIndex(int index){
        if (index < 0 || index >= data.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }
        return data[index];
    }

    public void setDataAt(int flatindex,double value) {
        if (this.data==null)
            this.data = new double[calculateSize(shape)];
        this.data[flatindex]=value;
    }

    public int[] getStrides() {
        return this.strides;
    }

    public void setData(double[] data) {
        this.data=data;
    }

    public int getFlatIndex(int[] indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Wrong index count");
        }
        int index = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (getStride(i) != 0) {
                index += indices[i] * stride;
            }
            stride *= shape[i];
        }
        return index;
    }

    public int[] getSpatialIndex(int index){
        int[] spatialIndex = new int[shape.length];
        for (int i=0; i<shape.length; i++){
            spatialIndex[i]=index/getStride(i);
            index %= getStride(i);
        }
        return spatialIndex;
    }


    public int[] getShape() {
        return shape;
    }

    public List<Tensor> getPrevTensors() {
        return prevTensors;
    }

    public int getDimensionAt(int index) {
        if (index < 0 || index >= shape.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }

        return shape[index];
    }

    public Tensor getGradient(){
        return gradient;
    }

    public int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    public int[] computeStrides() {
        return computeStrides(shape);
    }

    public int getFlatDataSize(){
        return this.data.length;
    }

    public double[] getData() {
        return data;
    }

    public boolean isContiguous() {
        int expectedStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
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
    }

    public ComputeBackend getEffectiveBackend() {
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
            compiledGraph.forward();
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

    public Tensor contiguous(){
        Operation op=new contiguous();
        this.getShape();
        return new Tensor(this.getShape(),List.of(this),op,"contiguous");
    }

    public Tensor add (Tensor second){
        Operation op= new add();
        return new Tensor(this.getShape(),List.of(this,second),op,"+");
    }

    public Tensor sub (Tensor second){
        Operation op= new sub();
        return new Tensor(this.getShape(),List.of(this,second),op,"-");
    }

    public Tensor mul (Tensor second){
        Operation op= new mul();
        return new Tensor(this.getShape(),List.of(this,second),op,"+");
    }

    public Tensor div (Tensor second){
        Operation op= new div();
        return new Tensor(this.getShape(),List.of(this,second),op,"/");
    }

    public Tensor log (){
        Operation op= new log();
        return new Tensor(this.getShape(),List.of(this),op,"log");
    }

    public Tensor exp (){
        Operation op= new exp();
        return new Tensor(this.getShape(),List.of(this),op,"exp");
    }

    public Tensor pow (double exp){
        Operation op= new pow(exp);
        return new Tensor(this.getShape(),List.of(this),op,"pow");
    }

    public Tensor sum(int dimension){
        Operation op=new sum(dimension);
        int[] newShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != dimension) newShape[j++] = shape[i];
        }
        return new Tensor(newShape,List.of(this),op,"sum");

    }

    public Tensor sum(){
        Operation op=new sum(-1);
        int[] newShape = new int[1];
        newShape[0]=1;
        return new Tensor(newShape,List.of(this),op,"sum");
    }


}
