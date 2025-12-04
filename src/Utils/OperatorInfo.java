package Utils;

import org.objectweb.asm.MethodVisitor;



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


    public static OperatorInfo fromClusterInput(SlotManager sm, int inputIndex) {
        int valuesSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS).get(inputIndex);
        int gradientSlot = sm.getGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS).get(inputIndex);
        return new OperatorInfo(valuesSlot, gradientSlot, SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS,
                SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, inputIndex,InputType.CLUSTER_INPUT);
    }


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



    public int getGradientSlot(){
        return gradientSlot;
    }

    public int getValuesSlot() {
        return valuesSlot;
    }
    public InputType getInputType() {
        return type;
    }

    public SlotKey getValuesSlotKey() {
        return valuesSlotKey;
    }
    public SlotKey getGradientsSlotKey() {
        return gradientSlotKey;
    }

    public int getReducedCount() {
        if (this.type == InputType.CLUSTER_INPUT) {
            throw new RuntimeException("Only supports inner inputs.");
        }
        return reducedCount;
    }
    public int getReducedIndex() {
        if (this.type == InputType.CLUSTER_INPUT) {
            throw new RuntimeException("Only supports inner inputs.");
        }
        return reducedIndex;
    }
}
