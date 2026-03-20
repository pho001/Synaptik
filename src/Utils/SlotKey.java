package Utils;

public enum SlotKey {
    CLUSTER_TENSOR_INPUTS (Type.LIST),
    CLUSTER_TENSOR (Type.TENSOR),
    CLUSTER_TENSOR_VALUES (Type.DOUBLE_ARRAY),
    CLUSTER_TENSOR_GRADS (Type.DOUBLE_ARRAY),
    LOOP_COUNTER (Type.INT),
    CLUSTER_INTERMEDIATES (Type.DOUBLE_ARRAY),
    CLUSTER_INTERMEDIATES_ARRAYS (Type.DOUBLE_ARRAY),
    CLUSTER_INPUTS_VALUES_ARRAYS (Type.DOUBLE_ARRAY),
    CLUSTER_INPUTS_GRAD_ARRAYS (Type.DOUBLE_ARRAY),
    CLUSTER_INNER_GRAD_VALUES (Type.DOUBLE),
    FUSED_NODE_VALUES (Type.DOUBLE),
    FUSED_NODE_VECTOR_VALUES (Type.LIST),
    SECOND_LOOP_COUNTER (Type.INT),
    RANGE_START (Type.INT),
    RANGE_END (Type.INT),
    RANGE_UPPER (Type.INT),
    TMP_REGISTER(Type.DOUBLE),
    TMP_REGISTER1(Type.DOUBLE),
    TMP_REGISTER2(Type.DOUBLE);

    public enum Type {
        INT(1),
        LIST(1),
        TENSOR(1),
        DOUBLE_ARRAY(1),
        DOUBLE(2),
        LONG(2);

        public final int slotSize;

        Type(int slotSize) {
            this.slotSize = slotSize;
        }
    }

    public final Type type;

    SlotKey(Type type) {
        this.type = type;
    }
}
