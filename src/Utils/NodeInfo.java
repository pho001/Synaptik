package Utils;

import Operations.Operation;
import Operations.exp;
import Operations.log;
import Operations.pow;
import org.objectweb.asm.MethodVisitor;

import java.util.ArrayList;
import static org.objectweb.asm.Opcodes.*;


public class NodeInfo {
    SlotKey valuesArraySlot;
    boolean isIntermediateValue;        //result je vzdycky olozen v poli. Muze byt intermediates
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





    public static NodeInfo fromLastClusterInput(int clusterIndex,SlotManager sm) {
        return new NodeInfo(clusterIndex,sm);
    }

    public static NodeInfo fromInnerClusterInput(int clusterIndex,SlotManager sm, int reducedIndex, int reducedCount) {
        return new NodeInfo(clusterIndex,sm,reducedIndex,reducedCount);
    }

    public static NodeInfo fromInnerClusterInput(int clusterIndex,SlotManager sm) {
        return new NodeInfo(clusterIndex,sm,SlotKey.CLUSTER_INTERMEDIATES,SlotKey.CLUSTER_INNER_GRAD_VALUES);
    }


    public int getValuesSlot() {
        return valuesSlotIndex;
    }

    public int getGradientSlot() {
        return gradientSlotIndex;
    }
    public boolean isFinalOutput() {
        return isFinalOutput;
    }

    public ArrayList<OperatorInfo> getOperators(){
        return operators;
    }

    public void setOperators(ArrayList<OperatorInfo> operators){
        if (operators.size()>2)
            throw new RuntimeException("Maximum of two operands is supported.");
        this.operators=operators;
    }

    public void addOperator(OperatorInfo operator){
        if (operators == null) {
            operators = new ArrayList<>();
        }
        if (operators.size() >= 2) {
            throw new IllegalArgumentException("Maximum of two operands is supported.");
        }
        operators.add(operator);
    }

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

    public void storeCache(MethodVisitor mv, SlotKey register){
        int slot=sm.get(register);
        mv.visitVarInsn(DSTORE, slot);
    }

    public void loadCache(MethodVisitor mv, SlotKey register){
        int slot=sm.get(register);
        mv.visitVarInsn(DLOAD, slot);
    }

    public void setOperation(Operation op){
        this.operation=op;
    }

    public Operation getOperation(){
        return operation;
    }

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

    public void performForwardOperation(MethodVisitor mv){
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitVarInsn(ALOAD, this.valuesSlotIndex);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        }


        for(OperatorInfo op:operators){
            emitOperatorValOnStack(mv,op);
        }
        switch (operation) {
            case Operations.add add -> mv.visitInsn(DADD);        // a[i] + b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sum]
            case Operations.mul mul -> mv.visitInsn(DMUL);        // a[i] * b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,mul]
            case Operations.sub sub -> mv.visitInsn(DSUB);        // a[i] - b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sub]
            case Operations.div div -> mv.visitInsn(DDIV);        // a[i] / b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,div]
            case Operations.log log -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
            case Operations.pow pow -> {
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
            case Operations.exp exp -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
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

    public void performForwardVectorOperation(MethodVisitor mv){
        if (valuesArraySlot.type== SlotKey.Type.DOUBLE_ARRAY){
            mv.visitVarInsn(ALOAD, this.valuesSlotIndex);
            mv.visitVarInsn(ILOAD, sm.get(SlotKey.LOOP_COUNTER));
        }


        for(OperatorInfo op:operators){
            emitOperatorValOnStack(mv,op);
        }
        switch (operation) {
            case Operations.add add -> mv.visitMethodInsn(INVOKEVIRTUAL,
                                    "jdk/incubator/vector/FloatVector",   // owner
                                    "add",                                // method name
                                    "(Ljdk/incubator/vector/FloatVector;)"
                                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                                    false);         // a[i] + b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sum]
            case Operations.mul mul -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "mul",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] * b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,mul]
            case Operations.sub sub -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "div",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] - b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sub]
            case Operations.div div -> mv.visitMethodInsn(INVOKEVIRTUAL,
                    "jdk/incubator/vector/FloatVector",   // owner
                    "div",                                // method name
                    "(Ljdk/incubator/vector/FloatVector;)"
                            + "Ljdk/incubator/vector/FloatVector;", // descriptor
                    false);        // a[i] / b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,div]
            case Operations.log log -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
            case Operations.pow pow -> {
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
            case Operations.exp exp -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
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
