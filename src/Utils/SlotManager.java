package Utils;
import org.objectweb.asm.MethodVisitor;

import java.util.*;

public class SlotManager {

    private int nextSlot = 1;
    private final Map<SlotKey, SlotInfo> allSlots = new LinkedHashMap<>();

    public void define(SlotKey key) {
        checkNotDefined(key);
        allSlots.put(key, new SlotInfo(nextSlot));
        nextSlot += key.type.slotSize;
    }

    public void defineGroup(SlotKey key, int count) {
        checkNotDefined(key);
        List<Integer> group = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            group.add(nextSlot);
            nextSlot += key.type.slotSize;
        }
        allSlots.put(key, new SlotInfo(group));
    }

    public boolean isGroup(SlotKey key) {
        SlotInfo info = allSlots.get(key);
        return info != null && info.isGroup();
    }

    public int get(SlotKey key) {
        SlotInfo info = allSlots.get(key);
        if (info == null || info.isGroup()) {
            throw new IllegalArgumentException("SlotKey is not a single slot or doesn't exist: " + key);
        }
        return info.getSlot();
    }

    public List<Integer> getGroup(SlotKey key) {
        SlotInfo info = allSlots.get(key);
        if (info == null) {
            throw new IllegalArgumentException("SlotKey not defined: " + key);
        }
        return info.getSlots();
    }

    public int getMaxSlot() {
        return nextSlot;
    }

    private void checkNotDefined(SlotKey key) {
        if (allSlots.containsKey(key)) {
            throw new IllegalArgumentException("SlotKey already defined: " + key);
        }
    }

    private void init (int numOuterInputs,int numOperations,int reducedOpsCount,boolean intermediatesAsArrays){
        this.define(SlotKey.CLUSTER_TENSOR_INPUTS);       // musi byt inicializovan jako prvni - jde o prvni parametr metody
        this.define(SlotKey.CLUSTER_TENSOR);              // musi byt inicializovan jako druhy - jde o druhy parametr metody
        this.define(SlotKey.CLUSTER_TENSOR_VALUES);
        this.define(SlotKey.CLUSTER_TENSOR_GRADS);
        this.define(SlotKey.LOOP_COUNTER);
        this.define(SlotKey.SECOND_LOOP_COUNTER);
        if (intermediatesAsArrays) {
            this.define(SlotKey.CLUSTER_INTERMEDIATES);
        }
        else {
            this.defineGroup(SlotKey.CLUSTER_INTERMEDIATES,reducedOpsCount);
        }
        this.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS,numOuterInputs);
        this.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS,numOuterInputs);
        this.defineGroup(SlotKey.CLUSTER_INNER_GRAD_VALUES,numOperations);

    }



}

