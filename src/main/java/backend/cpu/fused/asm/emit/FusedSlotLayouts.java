package backend.cpu.fused.asm.emit;

import utils.SlotKey;
import utils.SlotManager;

final class FusedSlotLayouts {
    private FusedSlotLayouts() {}

    static SlotManager buildRangeSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.CLUSTER_CONTEXT);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_CONTINUATION_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VALUES, nodeCount);
        sm.defineGroup(SlotKey.FUSED_NODE_BOOL_VALUES, nodeCount);
        sm.define(SlotKey.TMP_REGISTER);
        return sm;
    }

    static SlotManager buildVectorSlotLayout(int externalInputCount, int nodeCount) {
        SlotManager sm = new SlotManager();
        sm.define(SlotKey.CLUSTER_TENSOR_INPUTS);
        sm.define(SlotKey.CLUSTER_TENSOR);
        sm.define(SlotKey.CLUSTER_CONTEXT);
        sm.define(SlotKey.RANGE_START);
        sm.define(SlotKey.RANGE_END);
        sm.define(SlotKey.CLUSTER_TENSOR_VALUES);
        sm.define(SlotKey.LOOP_COUNTER);
        sm.define(SlotKey.SECOND_LOOP_COUNTER);
        sm.define(SlotKey.RANGE_UPPER);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_VALUES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_CONTINUATION_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INPUTS_GRAD_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.CLUSTER_INTERMEDIATES_ARRAYS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_NODE_VECTOR_VALUES, nodeCount);
        sm.defineGroup(SlotKey.FUSED_VECTOR_INDEX_MAPS, externalInputCount);
        sm.defineGroup(SlotKey.FUSED_VECTOR_LANE_ARRAYS, externalInputCount);
        sm.define(SlotKey.FUSED_VECTOR_REMAIN);
        sm.define(SlotKey.FUSED_VECTOR_STORAGE_INDEX);
        return sm;
    }
}
