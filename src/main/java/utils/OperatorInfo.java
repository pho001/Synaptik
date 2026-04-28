package utils;

import org.objectweb.asm.MethodVisitor;



/**
 * Internal codegen descriptor for an operator operand and its value/gradient local slots.
 */
public class OperatorInfo {

    private int gradientSlot;
    private int valuesSlot;
    private SlotKey gradientSlotKey;
    private SlotKey valuesSlotKey;
    private int index;
    private SlotManager sm;
    private InputType type;
    private int reducedIndex;
    private int reducedCount;

    private OperatorInfo(int valuesSlot, int gradientSlot, SlotKey valuesSlotKey, SlotKey gradientSlotKey, int index,InputType type){
        this.valuesSlot = valuesSlot;
        this.gradientSlot = gradientSlot;
        this.valuesSlotKey = valuesSlotKey;
        this.gradientSlotKey = gradientSlotKey;
        this.index = index;
        this.type=type;
    }

    private OperatorInfo(int valuesSlot, int gradientSlot, SlotKey valuesSlotKey, SlotKey gradientSlotKey,
                         int index,InputType type,int reducedIndex, int reducedCount){
        this.valuesSlot = valuesSlot;
        this.gradientSlot = gradientSlot;
        this.valuesSlotKey = valuesSlotKey;
        this.gradientSlotKey = gradientSlotKey;
        this.index = index;
        this.type=type;
        this.reducedIndex=reducedIndex;
        this.reducedCount=reducedCount;
    }


    /**
     * Creates an operand descriptor for an outer cluster input.
     *
     * @param sm slot manager containing grouped input slots
     * @param inputIndex index of the outer input
     * @return operand descriptor for the input
     */
    public static OperatorInfo fromClusterInput(SlotManager sm, int inputIndex) {
        int valuesSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(inputIndex);
        int gradientSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(inputIndex);
        return new OperatorInfo(valuesSlot, gradientSlot, SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS,
                SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, inputIndex,InputType.CLUSTER_INPUT);
    }


    /**
     * Creates an operand descriptor for a reduced intermediate value.
     *
     * @param sm slot manager containing intermediate slots
     * @param clusterIndex index of the producing cluster node
     * @param reducedIndex index inside the reduced intermediate layout
     * @param reducedCount number of reduced values stored per loop position
     * @return operand descriptor for the intermediate
     */
    public static OperatorInfo fromIntermediate(SlotManager sm, int clusterIndex, int reducedIndex, int reducedCount) {
        int valuesSlot;
        if(!sm.isGroup(SlotKey.CLUSTER_INTERMEDIATES)){
            valuesSlot=sm.get(SlotKey.CLUSTER_INTERMEDIATES);
        }
        else{
            valuesSlot=sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES).get(reducedIndex);
        }
        int gradientSlot = sm.getGroup(SlotKey.CLUSTER_INNER_GRAD_VALUES).get(clusterIndex);
        return new OperatorInfo(valuesSlot, gradientSlot, SlotKey.CLUSTER_INTERMEDIATES, SlotKey.CLUSTER_INNER_GRAD_VALUES,
                clusterIndex, InputType.CLUSTER_INNER, reducedIndex, reducedCount);
    }

    /**
     * Creates an operand descriptor for an intermediate value addressed directly by cluster index.
     *
     * @param sm slot manager containing intermediate slots
     * @param clusterIndex index of the producing cluster node
     * @return operand descriptor for the intermediate
     */
    public static OperatorInfo fromIntermediate(SlotManager sm, int clusterIndex) {

        int valuesSlot;
        if(!sm.isGroup(SlotKey.CLUSTER_INTERMEDIATES)){
            valuesSlot=sm.get(SlotKey.CLUSTER_INTERMEDIATES);
        }
        else{
            valuesSlot=sm.getGroup(SlotKey.CLUSTER_INTERMEDIATES).get(clusterIndex);
        }

        int gradientSlot = sm.getGroup(SlotKey.CLUSTER_INNER_GRAD_VALUES).get(clusterIndex);
        return new OperatorInfo(valuesSlot, gradientSlot, SlotKey.CLUSTER_INTERMEDIATES, SlotKey.CLUSTER_INNER_GRAD_VALUES,
                clusterIndex, InputType.CLUSTER_INNER);
    }



    /**
     * Returns the local slot containing this operand's gradient storage.
     *
     * @return gradient local slot
     */
    public int getGradientSlot(){
        return gradientSlot;
    }

    /**
     * Returns the local slot containing this operand's value storage.
     *
     * @return value local slot
     */
    public int getValuesSlot() {
        return valuesSlot;
    }

    /**
     * Returns whether this operand is an outer input or inner intermediate.
     *
     * @return operand input type
     */
    public InputType getInputType() {
        return type;
    }

    /**
     * Returns the slot key used for value storage.
     *
     * @return value slot key
     */
    public SlotKey getValuesSlotKey() {
        return valuesSlotKey;
    }

    /**
     * Returns the slot key used for gradient storage.
     *
     * @return gradient slot key
     */
    public SlotKey getGradientsSlotKey() {
        return gradientSlotKey;
    }

    /**
     * Returns the reduced layout width for inner intermediate operands.
     *
     * @return number of reduced values stored per loop position
     */
    public int getReducedCount() {
        if (this.type == InputType.CLUSTER_INPUT) {
            throw new RuntimeException("Only supports inner inputs.");
        }
        return reducedCount;
    }

    /**
     * Returns this operand's index in the reduced intermediate layout.
     *
     * @return reduced-layout index
     */
    public int getReducedIndex() {
        if (this.type == InputType.CLUSTER_INPUT) {
            throw new RuntimeException("Only supports inner inputs.");
        }
        return reducedIndex;
    }
}
