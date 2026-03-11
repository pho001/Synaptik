package Tensor;

import Backend.ComputeBackend;
import Graph.CompiledGraph;
import Graph.optimizer.GraphOptimizer;
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
    private Runnable backwardFunction;
    private boolean isBackward = false;





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
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();

        this.data = new double[totalSize];

        this.label = label;
        this.shape = dimensions;
        this.strides=computeStrides(dimensions);
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

    }

    public Tensor(int[] shape, List<Tensor> previous, Operation operation, String label) {
        int totalSize=calculateSize(shape);
        //this.data = new double[totalSize];
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.label = label;
        this.shape = shape;
        this.strides=computeStrides(shape);
        this.operation = operation;
        this.data=new double[calculateSize(shape)];
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);
    }


    public Tensor(double[] data, int[] shape, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.label = label;
        this.shape = shape;
        this.strides=computeStrides(shape);
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

    }

    public Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label) {

        this.data = data;
        this.prevTensors = previous != null ? new ArrayList<>(previous) : new ArrayList<>();
        this.label = label;
        this.shape = shape;
        this.strides=strides;
        requiresGrad = previous != null && previous.stream().anyMatch(Tensor::getRequiresGrad);

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
    public void setLabel(String label) {
        this.label = label;
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

    public boolean isBackward() {
        return isBackward;
    }
    public void setBackward(boolean backward) {
        isBackward = backward;
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
        Operation op=new contiguous();
        this.getShape();
        return new Tensor(this.getShape(),List.of(this),op,"contiguous");
    }

    public Tensor add (Tensor second){
        Operation op= new add();
        Tensor out= new Tensor(this.getShape(),List.of(this,second),op,"+");

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (this.getRequiresGrad()) {
                if (this.getGradient() == null) {
                    // Pokud gradient ještě nemá, prostě ho přiřadíme
                    this.setGradient(outGrad);
                } else {
                    // Pokud už gradient má (graf se větvil), tak je SEČTEME
                    // .add() vytvoří nový uzel a ten nastavíme jako nový gradient
                    this.setGradient(this.getGradient().add(outGrad));
                }
            }

            if (second.getRequiresGrad()) {
                if (second.getGradient() == null) {
                    second.setGradient(outGrad);
                } else {
                    second.setGradient(second.getGradient().add(outGrad));
                }
            }
        });
        return out;

    }

    public Tensor sub (Tensor second){
        Operation op= new sub();
        Tensor out= new Tensor(this.getShape(),List.of(this,second),op,"-");
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            // Pro 'a' je to stejné jako u sčítání
            if (this.getRequiresGrad()) {
                if (this.getGradient() == null) {
                    this.setGradient(outGrad);
                } else {
                    this.setGradient(this.getGradient().add(outGrad));
                }
            }

            // Pro 'b' musíme gradient otočit do mínusu (negace)
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = outGrad.neg(); // Nebo outGrad.mul(-1)

                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });
        return out;
    }

    public Tensor mul (Tensor second){
        Operation op= new mul();
        Tensor out= new Tensor(this.getShape(),List.of(this,second),op,"*");

        out.setBackwardFunction(() -> {
            // 1. Získáme gradient, který přitekl z vyšších vrstev sítě
            Tensor outGrad = out.getGradient();

            // 2. Výpočet a akumulace pro první vstup (this)
            if (this.getRequiresGrad()) {
                // Vytvoříme nový uzel grafu: outGrad * b
                Tensor gradForThis = outGrad.mul(second);

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    // Akumulace (součet) s existujícím gradientem
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }

            // 3. Výpočet a akumulace pro druhý vstup (second)
            if (second.getRequiresGrad()) {
                // Vytvoříme nový uzel grafu: outGrad * a
                Tensor gradForSecond = outGrad.mul(this);

                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    // Akumulace (součet) s existujícím gradientem
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });

        return out;

    }

    public Tensor div (Tensor second){
        Operation op= new div();
        Tensor out= new Tensor(this.getShape(),List.of(this,second),op,"/");
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            // Gradient pro čitatele (a): outGrad / b
            if (this.getRequiresGrad()) {
                Tensor gradForThis = outGrad.div(second);

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }

            // Gradient pro jmenovatele (b): -outGrad * a / (b^2)
            if (second.getRequiresGrad()) {
                // Vytváříme graf: -(outGrad) * this / (second * second)
                Tensor gradForSecond = outGrad.neg().mul(this).div(second.pow(2));

                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });
        return out;
    }

    public Tensor neg (){
        Operation op=new neg();
        Tensor out= new Tensor(this.getShape(),List.of(this),op,"neg");
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Gradient otočíme do mínusu.
                // Vytvoří to v grafu nový uzel reprezentující negaci gradientu.
                Tensor gradForThis = outGrad.neg(); // Případně outGrad.mul(-1)

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });
        return out;

    }

    public Tensor log (){
        Operation op= new log();
        Tensor out = new Tensor(this.getShape(),List.of(this),op,"log");
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Derivace ln(x) je 1/x, takže outGrad násobíme 1/x (což je outGrad / x)
                Tensor gradForThis = outGrad.div(this);

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });
        return out;
    }

    public Tensor exp (){
        Operation op= new exp();
        Tensor out = new Tensor(this.getShape(),List.of(this),op,"exp");
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Derivace e^x je e^x. Jelikož 'out' už JE e^x, můžeme ho rovnou použít!
                Tensor gradForThis = outGrad.mul(out);

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });
        return out;
    }

    public Tensor pow(double exp) {
        Operation op = new pow(exp); // Operace si konstantu uloží jako svůj vnitřní stav

        // Pozor! Rodičem v grafu je POUZE 'this'. Konstanta není uzel.
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "pow");

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Matematický vzorec: outGrad * exp * (a^(exp - 1))
                // Zde voláme tvé skalární verze metod!
                Tensor gradForThis = outGrad
                        .mul(exp)
                        .mul(this.pow(exp - 1.0));

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    // Akumulace gradientu
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }

        });

        return out;
    }

    public Tensor mul(double scalar) {
        // Vytvoříš speciální operaci pro násobení skalárem,
        // která si tu konstantu uloží do sebe.
        Operation op = new mulScalar(scalar);

        // V grafu je rodičem pouze 'this'.
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "* constant");

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {

                Tensor gradForThis = outGrad.mul(scalar);

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });

        return out;
    }

    public Tensor inv() {
        Operation op = new inv();
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "inv");

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Derivace 1/x je -1 / (x^2).
                // Triky pro graf:
                // 1. Místo -1 / (x*x) můžeme použít -(out * out), protože out = 1/x.
                // To ušetří v grafu jednu operaci dělení!
                Tensor gradForThis = outGrad.neg().mul(out.mul(out));

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });
        return out;
    }

    public Tensor forwardOutput() {
        Operation op = new noop();
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "noop");
        return out;
    }

    public Tensor sqrt() {
        Operation op = new sqrt();
        Tensor out = new Tensor(this.getShape(), List.of(this), op, "sqrt");

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (this.getRequiresGrad()) {
                // Derivace: outGrad * 0.5 * (1 / sqrt(x))
                // Triky: 1/sqrt(x) je vlastně out.inv()
                // Takže: outGrad * 0.5 * out.inv()
                Tensor gradForThis = outGrad.mul(0.5).mul(out.inv());

                if (this.getGradient() == null) {
                    this.setGradient(gradForThis);
                } else {
                    this.setGradient(this.getGradient().add(gradForThis));
                }
            }
        });
        return out;
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
