package utils;

/**
 * Internal codegen keys for JVM local-variable slots used by generated kernels.
 */
public enum SlotKey {
    /** Local slot for cluster tensor input metadata. */
    CLUSTER_TENSOR_INPUTS (Type.LIST),
    /** Local slot for the cluster tensor object. */
    CLUSTER_TENSOR (Type.TENSOR),
    /** Local slot for cluster execution context metadata. */
    CLUSTER_CONTEXT (Type.LIST),
    /** Local slot for the output tensor value array. */
    CLUSTER_TENSOR_VALUES (Type.DOUBLE_ARRAY),
    /** Local slot for the output tensor gradient array. */
    CLUSTER_TENSOR_GRADS (Type.DOUBLE_ARRAY),
    /** Primary loop counter local slot. */
    LOOP_COUNTER (Type.INT),
    /** Local slot or grouped slots for intermediate value storage. */
    CLUSTER_INTERMEDIATES (Type.DOUBLE_ARRAY),
    /** Local slot for arrays backing intermediate values. */
    CLUSTER_INTERMEDIATES_ARRAYS (Type.DOUBLE_ARRAY),
    /** Grouped local slots for outer input value arrays. */
    CLUSTER_INPUTS_VALUES_ARRAYS (Type.DOUBLE_ARRAY),
    /** Grouped local slots for continuation value arrays. */
    CLUSTER_INPUTS_CONTINUATION_ARRAYS (Type.DOUBLE_ARRAY),
    /** Grouped local slots for outer input gradient arrays. */
    CLUSTER_INPUTS_GRAD_ARRAYS (Type.DOUBLE_ARRAY),
    /** Grouped local slots for scalar inner gradients. */
    CLUSTER_INNER_GRAD_VALUES (Type.DOUBLE),
    /** Local slot for fused scalar node values. */
    FUSED_NODE_VALUES (Type.DOUBLE),
    /** Local slot for fused boolean node values. */
    FUSED_NODE_BOOL_VALUES (Type.INT),
    /** Local slot for fused vector node values. */
    FUSED_NODE_VECTOR_VALUES (Type.LIST),
    /** Temporary object slot for generated fused vector expressions. */
    FUSED_VECTOR_TEMP (Type.LIST),
    /** Grouped local slots for generated vector gather index maps. */
    FUSED_VECTOR_INDEX_MAPS (Type.LIST),
    /** Grouped local slots for generated segment gather lane scratch arrays. */
    FUSED_VECTOR_LANE_ARRAYS (Type.LIST),
    /** Temporary int register for generated vector gather index math. */
    FUSED_VECTOR_REMAIN (Type.INT),
    /** Temporary int register for generated vector gather storage index math. */
    FUSED_VECTOR_STORAGE_INDEX (Type.INT),
    /** Secondary loop counter local slot. */
    SECOND_LOOP_COUNTER (Type.INT),
    /** Inclusive range-start local slot. */
    RANGE_START (Type.INT),
    /** Exclusive range-end local slot. */
    RANGE_END (Type.INT),
    /** Local slot for fused execution options. */
    FUSED_OPTIONS (Type.LIST),
    /** Local slot for an upper loop bound. */
    RANGE_UPPER (Type.INT),
    /** Temporary double register. */
    TMP_REGISTER(Type.DOUBLE),
    /** Temporary double register. */
    TMP_REGISTER1(Type.DOUBLE),
    /** Temporary double register. */
    TMP_REGISTER2(Type.DOUBLE);

    /**
     * Internal JVM local-slot storage categories and their slot width.
     */
    public enum Type {
        /** Integer local occupying one JVM slot. */
        INT(1),
        /** Object/list local occupying one JVM slot. */
        LIST(1),
        /** Tensor object local occupying one JVM slot. */
        TENSOR(1),
        /** Double-array object local occupying one JVM slot. */
        DOUBLE_ARRAY(1),
        /** Double primitive local occupying two JVM slots. */
        DOUBLE(2),
        /** Long primitive local occupying two JVM slots. */
        LONG(2);

        /** JVM local-variable slot width for values of this type. */
        public final int slotSize;

        Type(int slotSize) {
            this.slotSize = slotSize;
        }
    }

    /** Storage category associated with this key. */
    public final Type type;

    SlotKey(Type type) {
        this.type = type;
    }
}
