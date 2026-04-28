package utils;

import operations.Operation;
import operations.elementwise.unary.exp;
import operations.elementwise.unary.log;
import operations.elementwise.unary.pow;
import org.objectweb.asm.MethodVisitor;

import java.util.ArrayList;
import static org.objectweb.asm.Opcodes.*;


/**
 * Internal bytecode-generation descriptor for a node's value, gradient, operands, and operation.
 */
public class NodeInfo {
    SlotKey valuesArraySlot;
    boolean isIntermediateValue;        //result je vzdycky olozen v poli. Muze byt intermediates
    /** True when this descriptor represents the final cluster output rather than an intermediate. */
    public boolean isFinalOutput;
    int clusterIndex;
    SlotKey gradientSlot;
    ArrayList<OperatorInfo> operators;
    ArrayList<NodeInfo> inputs;
    SlotManager sm;
    int valuesSlotIndex;
    int gradientSlotIndex;
    int reducedIndex;
    int reducedCount;
    Operation operation;

    private NodeInfo(int clusterIndex, SlotManager sm) {
        this.isFinalOutput=true;
        this.clusterIndex=clusterIndex;
        ArrayList<OperatorInfo> operators = new ArrayList<>();
        ArrayList<NodeInfo> inputs = new ArrayList<>();
        valuesArraySlot=SlotKey.CLUSTER_TENSOR_VALUES;
        valuesSlotIndex=sm.get(valuesArraySlot);
        gradientSlot=SlotKey.CLUSTER_TENSOR_GRADS;
        gradientSlotIndex=sm.get(gradientSlot);
        this.sm=sm;
    }

    private NodeInfo(int clusterIndex, SlotManager sm, int reducedIndex, int reducedCount) {
        this.isFinalOutput=false;
        this.clusterIndex=clusterIndex;
        ArrayList<OperatorInfo> operators = new ArrayList<>();
        ArrayList<NodeInfo> inputs = new ArrayList<>();
        valuesArraySlot=SlotKey.CLUSTER_INTERMEDIATES;
        if(!sm.isGroup(valuesArraySlot)){
            valuesSlotIndex=sm.get(valuesArraySlot);
        }
        else{
            valuesSlotIndex=sm.getGroup(valuesArraySlot).get(reducedIndex);
        }
        isIntermediateValue=true;
        gradientSlot=SlotKey.CLUSTER_INNER_GRAD_VALUES;
        gradientSlotIndex=sm.getGroup(gradientSlot).get(clusterIndex);
        this.reducedIndex=reducedIndex;
        this.reducedCount=reducedCount;
        this.sm=sm;
    }

    private NodeInfo(int clusterIndex, SlotManager sm, SlotKey valuesArraySlot, SlotKey gradientSlot) {
        this.isFinalOutput=false;
        this.clusterIndex=clusterIndex;
        ArrayList<OperatorInfo> operators = new ArrayList<>();
        ArrayList<NodeInfo> inputs = new ArrayList<>();
        this.valuesArraySlot=valuesArraySlot;
        valuesSlotIndex=sm.get(valuesArraySlot);
        isIntermediateValue=true;
        this.gradientSlot=gradientSlot;
        gradientSlotIndex=sm.getGroup(gradientSlot).get(clusterIndex);
        this.sm=sm;
    }





    /**
     * Creates a descriptor for the final output of the last cluster.
     *
     * @param clusterIndex cluster-local node index
     * @param sm local slot manager
     * @return final-output node descriptor
     */
    public static NodeInfo fromLastClusterInput(int clusterIndex,SlotManager sm) {
        return new NodeInfo(clusterIndex,sm);
    }

    /**
     * Creates a descriptor for a reduced intermediate input inside a cluster.
     *
     * @param clusterIndex cluster-local node index
     * @param sm local slot manager
     * @param reducedIndex index inside the reduced intermediate layout
     * @param reducedCount number of reduced values stored per loop position
     * @return intermediate node descriptor
     */
    public static NodeInfo fromInnerClusterInput(int clusterIndex,SlotManager sm, int reducedIndex, int reducedCount) {
        return new NodeInfo(clusterIndex,sm,reducedIndex,reducedCount);
    }

    /**
     * Creates a descriptor for an intermediate input addressed directly by cluster index.
     *
     * @param clusterIndex cluster-local node index
     * @param sm local slot manager
     * @return intermediate node descriptor
     */
    public static NodeInfo fromInnerClusterInput(int clusterIndex,SlotManager sm) {
        return new NodeInfo(clusterIndex,sm,SlotKey.CLUSTER_INTERMEDIATES,SlotKey.CLUSTER_INNER_GRAD_VALUES);
    }


    /**
     * Returns the local slot holding this node's value storage.
     *
     * @return value local slot
     */
    public int getValuesSlot() {
        return valuesSlotIndex;
    }

    /**
     * Returns the local slot holding this node's gradient storage.
     *
     * @return gradient local slot
     */
    public int getGradientSlot() {
        return gradientSlotIndex;
    }

    /**
     * Reports whether this node is the final cluster output.
     *
     * @return true for final output descriptors
     */
    public boolean isFinalOutput() {
        return isFinalOutput;
    }

    /**
     * Returns operand descriptors consumed by this node's operation.
     *
     * @return mutable operand descriptor list
     */
    public ArrayList<OperatorInfo> getOperators(){
        return operators;
    }

    /**
     * Replaces operand descriptors for this node's operation.
     *
     * @param operators operand descriptors; at most two are supported
     */
    public void setOperators(ArrayList<OperatorInfo> operators){
        if (operators.size()>2)
            throw new RuntimeException("Maximum of two operands is supported.");
        this.operators=operators;
    }

    /**
     * Adds one operand descriptor to this node's operation.
     *
     * @param operator operand descriptor to append
     */
    public void addOperator(OperatorInfo operator){
        if (operators == null) {
            operators = new ArrayList<>();
        }
        if (operators.size() >= 2) {
            throw new IllegalArgumentException("Maximum of two operands is supported.");
        }
        operators.add(operator);
    }

    /**
     * Emits bytecode that pushes an operand gradient onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param op operand whose gradient should be loaded
     */
    public void emitOperatorGradOnStack(MethodVisitor mv,OperatorInfo op){
        if (op == null) {
            throw new NullPointerException("Operator is null");
        }
        InputType type=op.getInputType();
        switch (type) {
            case CLUSTER_INPUT : {
                mv.visitVarInsn(ALOAD, op.getGradientSlot());               // Stack: [...inputGradArray]
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                mv.visitInsn(DUP2);                                         // Stack: [...array,i,array,i]
                mv.visitInsn(DALOAD);                                       // Stack: [...array,i,value]
                break;
            }
            case CLUSTER_INNER: {
                mv.visitVarInsn(DLOAD, op.getGradientSlot());               // Stack: [...value]
                break;
            }
            default:{
                throw new AssertionError("Only inner and outer operators are supported.");
            }
        }


    }

    /**
     * Emits bytecode that pushes the first operand's gradient onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitFirstOperatorGradOnStack(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(0);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }
        emitOperatorGradOnStack(mv,op);
    }

    /**
     * Emits bytecode that pushes the second operand's gradient onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitSecondOperatorGradOnStack(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(1);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }

        emitOperatorGradOnStack(mv,op);
    }

    /**
     * Emits bytecode that stores the top-of-stack gradient into an operand's gradient location.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param op operand whose gradient should be stored
     */
    public void emitStoreOperatorGrad(MethodVisitor mv, OperatorInfo op){
        if (op == null) {
            throw new NullPointerException("Operator is null");
        }
        InputType type=op.getInputType();
        switch (type){
            case CLUSTER_INPUT:{
                mv.visitInsn(DASTORE);
                break;
            }

            case CLUSTER_INNER:{
                mv.visitVarInsn(DSTORE, op.getGradientSlot());
                break;
            }
            default: {
                throw new AssertionError("Only inner and outer operators are supported.");
            }


        }
    }

    /**
     * Emits bytecode that stores the top-of-stack gradient into the first operand's gradient location.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitStoreFirstOperatorGrad(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(0);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }

        emitStoreOperatorGrad(mv,op);
    }

    /**
     * Emits bytecode that stores the top-of-stack gradient into the second operand's gradient location.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitStoreSecondOperatorGrad(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(1);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }

        emitStoreOperatorGrad(mv,op);
    }





    /**
     * Emits bytecode that pushes an operand value onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param op operand whose value should be loaded
     */
    public void emitOperatorValueOnStack(MethodVisitor mv, OperatorInfo op){
        if (op == null) {
            throw new NullPointerException("Operator is null");
        }
        InputType type=op.getInputType();
        switch (type){
            case CLUSTER_INPUT:{
                mv.visitVarInsn(ALOAD, op.getGradientSlot());               // Stack: [...inputGradArray]
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                mv.visitInsn(DALOAD);
                break;
            }
            case CLUSTER_INNER:{

                mv.visitVarInsn(ALOAD, op.getValuesSlot());
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                pushIntConst(mv,op.getReducedCount());
                mv.visitInsn(IMUL);
                pushIntConst(mv,op.getReducedIndex());
                mv.visitInsn(IADD);
                mv.visitInsn(DALOAD);
                break;

            }
            default: {
                throw new AssertionError("Only inner and outer operators are supported.");
            }


        }

    }

    /**
     * Emits bytecode that pushes the first operand value onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitFirstOperatorValueOnStack(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(0);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }
        emitOperatorValueOnStack(mv,op);
    }

    /**
     * Emits bytecode that pushes the second operand value onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitSecondOperatorValueOnStack(MethodVisitor mv){
        if (operators == null || operators.isEmpty()) {
            throw new IllegalStateException("Operator list is empty");
        }

        OperatorInfo op = operators.get(1);
        if (op == null) {
            throw new IllegalStateException("Second operator is null");
        }
        emitOperatorValueOnStack(mv,op);
    }

    /**
     * Emits bytecode that pushes this node's gradient onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitGradientOnStack(MethodVisitor mv){
        SlotKey.Type type=this.gradientSlot.type;
        switch (type){
            case DOUBLE_ARRAY :{
                mv.visitVarInsn(ALOAD, this.gradientSlotIndex);
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                mv.visitInsn(DALOAD);
                break;
            }
            case DOUBLE :{
                mv.visitVarInsn(DLOAD, this.gradientSlotIndex);
                break;
            }
            default:{
                throw new AssertionError("Type not supported: " + type);
            }
        }

    }

    /**
     * Emits bytecode that pushes this node's value onto the JVM operand stack.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void emitValueOnStack(MethodVisitor mv){
        if (this.isFinalOutput){
            mv.visitVarInsn(ALOAD, this.valuesSlotIndex);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            mv.visitInsn(DALOAD);
        }
        else{
            mv.visitVarInsn(ALOAD, sm.get(SlotKey.CLUSTER_INTERMEDIATES));
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
            pushIntConst(mv,reducedCount);
            mv.visitInsn(IMUL);
            pushIntConst(mv,reducedIndex);
            mv.visitInsn(IADD);
            mv.visitInsn(DALOAD);
        }
    }


    private static void pushIntConst(MethodVisitor mv, int value) {
        if (value >= -1 && value <= 5) {
            mv.visitInsn(ICONST_0 + value);
        } else if (value <= Byte.MAX_VALUE) {
            mv.visitIntInsn(BIPUSH, value);
        } else if (value <= Short.MAX_VALUE) {
            mv.visitIntInsn(SIPUSH, value);
        } else {
            mv.visitLdcInsn(value);
        }
    }

    /**
     * Emits bytecode that stores the top double value into a temporary local register.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param register temporary register key
     */
    public void storeCache(MethodVisitor mv, SlotKey register){
        int slot=sm.get(register);
        mv.visitVarInsn(DSTORE, slot);
    }

    /**
     * Emits bytecode that loads a double value from a temporary local register.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param register temporary register key
     */
    public void loadCache(MethodVisitor mv, SlotKey register){
        int slot=sm.get(register);
        mv.visitVarInsn(DLOAD, slot);
    }

    /**
     * Assigns the operation implemented by this node.
     *
     * @param op operation represented by this descriptor
     */
    public void setOperation(Operation op){
        this.operation=op;
    }

    /**
     * Returns the operation implemented by this node.
     *
     * @return operation represented by this descriptor
     */
    public Operation getOperation(){
        return operation;
    }

    /**
     * Emits bytecode that pushes an operand value using the current intermediate layout.
     *
     * @param mv ASM method visitor receiving bytecode
     * @param op operand whose value should be loaded
     */
    public void emitOperatorValOnStack(MethodVisitor mv,OperatorInfo op){
        // operator muze byt bud vnejsi vstup nebo intermediate. Intermediate muze byt ve forme 1d pole,
        // kde budu muset dopocitat linearni index nebo proste pole.
        if (op == null) {
            throw new NullPointerException("Operator is null");
        }
        InputType inputtype=op.getInputType();
        SlotKey.Type datatype=op.getValuesSlotKey().type;


        switch (inputtype){
            case CLUSTER_INPUT:{
                mv.visitVarInsn(ALOAD, op.getValuesSlot());               // Stack: [...inputGradArray]
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                mv.visitInsn(DALOAD);
                break;
            }
            case CLUSTER_INNER:{

                mv.visitVarInsn(ALOAD, op.getValuesSlot());
                mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
                if (!sm.isGroup(op.getValuesSlotKey())) {
                    pushIntConst(mv,op.getReducedCount());
                    mv.visitInsn(IMUL);
                    pushIntConst(mv,op.getReducedIndex());
                    mv.visitInsn(IADD);
                }
                mv.visitInsn(DALOAD);
                break;

            }
            default: {
                throw new AssertionError("Only inner and outer operators are supported.");
            }


        }

    }

    /**
     * Emits scalar forward bytecode for this node's configured operation.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void performForwardOperation(MethodVisitor mv){
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitVarInsn(ALOAD, this.valuesSlotIndex);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        }


        for(OperatorInfo op:operators){
            emitOperatorValOnStack(mv,op);
        }
        switch (operation) {
            case operations.elementwise.binary.add add -> mv.visitInsn(DADD);        // a[i] + b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sum]
            case operations.elementwise.binary.mul mul -> mv.visitInsn(DMUL);        // a[i] * b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,mul]
            case operations.elementwise.binary.sub sub -> mv.visitInsn(DSUB);        // a[i] - b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sub]
            case operations.elementwise.binary.div div -> mv.visitInsn(DDIV);        // a[i] / b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,div]
            case operations.elementwise.unary.log log -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
            case operations.elementwise.unary.pow pow -> {
                if (pow.getExponent()==0){
                    mv.visitInsn(POP);
                    mv.visitInsn(DCONST_1);
                }
                else if (pow.getExponent()==0.5 || pow.getExponent()==(double)1/2){
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                }

                else if (pow.getExponent()==2){
                    mv.visitInsn(DUP2);
                    mv.visitInsn(DMUL);
                }
                else {
                    mv.visitLdcInsn(pow.getExponent());
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
                }
            }
            case operations.elementwise.unary.exp exp -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                /*
                case sqrt sqrt -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                case sin sin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sin", "(D)D", false);
                case cos cos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "cos", "(D)D", false);
                case tan tan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tan", "(D)D", false);
                case asin asin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "asin", "(D)D", false);
                case acos acos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "acos", "(D)D", false);
                case atan atan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "atan", "(D)D", false);
                case tanh tanh -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tanh", "(D)D", false);

                 */

            default -> throw new UnsupportedOperationException();
        }
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitInsn(DASTORE);

        }
        else{
            mv.visitVarInsn(DSTORE, valuesSlotIndex);
        }


    }

    /**
     * Emits vector forward bytecode for this node's configured operation.
     *
     * @param mv ASM method visitor receiving bytecode
     */
    public void performForwardVectorOperation(MethodVisitor mv){
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitVarInsn(ALOAD, this.valuesSlotIndex);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        }


        for(OperatorInfo op:operators){
            emitOperatorValOnStack(mv,op);
        }
        switch (operation) {
            case operations.elementwise.binary.add add -> mv.visitMethodInsn(INVOKEVIRTUAL,
                                    "jdk/incubator/vector/FloatVector",   // owner
                                    "add",                                // method name
                                    "(Ljdk/incubator/vector/FloatVector;)"
                                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                                    false);         // a[i] + b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sum]
            case operations.elementwise.binary.mul mul -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "mul",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] * b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,mul]
            case operations.elementwise.binary.sub sub -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "div",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] - b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sub]
            case operations.elementwise.binary.div div -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "div",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] / b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,div]
            case operations.elementwise.unary.log log -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
            case operations.elementwise.unary.pow pow -> {
                if (pow.getExponent()==0){
                    mv.visitInsn(POP);
                    mv.visitInsn(DCONST_1);
                }
                else if (pow.getExponent()==0.5 || pow.getExponent()==(double)1/2){
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                }

                else if (pow.getExponent()==2){
                    mv.visitInsn(DUP2);
                    mv.visitInsn(DMUL);
                }
                else {
                    mv.visitLdcInsn(pow.getExponent());
                    mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
                }
            }
            case operations.elementwise.unary.exp exp -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                /*
                case sqrt sqrt -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                case sin sin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sin", "(D)D", false);
                case cos cos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "cos", "(D)D", false);
                case tan tan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tan", "(D)D", false);
                case asin asin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "asin", "(D)D", false);
                case acos acos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "acos", "(D)D", false);
                case atan atan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "atan", "(D)D", false);
                case tanh tanh -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tanh", "(D)D", false);

                 */

            default -> throw new UnsupportedOperationException();
        }
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitInsn(DASTORE);

        }
        else{
            mv.visitVarInsn(DSTORE, valuesSlotIndex);
        }
    }



}
