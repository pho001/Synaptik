package operations;

public interface Operation {
    enum OpArityClass {
        ELEMENT_WISE,
        REDUCTION,
        LAYOUT,
        LINEAR_ALGEBRA,
        SPECIAL,
        FUSED
    };
    enum OpType {
        ADD(OpArityClass.ELEMENT_WISE, true),
        SUB(OpArityClass.ELEMENT_WISE, true),
        MUL(OpArityClass.ELEMENT_WISE, true),
        DIV(OpArityClass.ELEMENT_WISE, true),
        MIN(OpArityClass.ELEMENT_WISE, true),
        MAX(OpArityClass.ELEMENT_WISE, true),
        GT(OpArityClass.ELEMENT_WISE, false),
        GE(OpArityClass.ELEMENT_WISE, false),
        LT(OpArityClass.ELEMENT_WISE, false),
        LE(OpArityClass.ELEMENT_WISE, false),
        EQ(OpArityClass.ELEMENT_WISE, false),
        NE(OpArityClass.ELEMENT_WISE, false),
        LOGICAL_AND(OpArityClass.ELEMENT_WISE, false),
        LOGICAL_OR(OpArityClass.ELEMENT_WISE, false),
        LOGICAL_NOT(OpArityClass.ELEMENT_WISE, false),
        MIN_GRAD(OpArityClass.SPECIAL, false),
        MAX_GRAD(OpArityClass.SPECIAL, false),
        REDUCE_MIN(OpArityClass.REDUCTION, false),
        REDUCE_MAX(OpArityClass.REDUCTION, false),
        REDUCE_ALL(OpArityClass.REDUCTION, false),
        REDUCE_ANY(OpArityClass.REDUCTION, false),
        SOFTMAX(OpArityClass.SPECIAL, false),
        LOG_SOFTMAX(OpArityClass.SPECIAL, false),
        NLL_LOSS(OpArityClass.SPECIAL, false),
        CROSS_ENTROPY_LOSS(OpArityClass.SPECIAL, false),
        REDUCE_MIN_GRAD(OpArityClass.SPECIAL, false),
        REDUCE_MAX_GRAD(OpArityClass.SPECIAL, false),
        GATHER(OpArityClass.SPECIAL, false),
        GATHER_GRAD(OpArityClass.SPECIAL, false),
        TAKE_ALONG_AXIS(OpArityClass.SPECIAL, false),
        TAKE_ALONG_AXIS_GRAD(OpArityClass.SPECIAL, false),
        SCATTER_ADD(OpArityClass.SPECIAL, false),
        LINEAR(OpArityClass.SPECIAL, false),
        CONV2D(OpArityClass.SPECIAL, false),
        CONV2D_GEMM(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_WEIGHT(OpArityClass.SPECIAL, false),
        MAX_POOL2D(OpArityClass.SPECIAL, false),
        MAX_POOL2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        AVG_POOL2D(OpArityClass.SPECIAL, false),
        AVG_POOL2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        WHERE(OpArityClass.ELEMENT_WISE, false),
        MATMUL(OpArityClass.LINEAR_ALGEBRA, false),
        NEG(OpArityClass.ELEMENT_WISE, true),
        INV(OpArityClass.ELEMENT_WISE, true),
        LOG(OpArityClass.ELEMENT_WISE, true),
        EXP(OpArityClass.ELEMENT_WISE, true),
        FAST_EXP(OpArityClass.ELEMENT_WISE, true),
        TANH(OpArityClass.ELEMENT_WISE, true),
        FAST_TANH(OpArityClass.ELEMENT_WISE, true),
        POW(OpArityClass.ELEMENT_WISE, true),
        SQRT(OpArityClass.ELEMENT_WISE, true),
        ABS(OpArityClass.ELEMENT_WISE, false),
        MUL_SCALAR(OpArityClass.ELEMENT_WISE, true),
        SUM(OpArityClass.REDUCTION, false),
        MEAN(OpArityClass.REDUCTION, false),
        RELU(OpArityClass.ELEMENT_WISE, true),
        CLAMP_MIN(OpArityClass.ELEMENT_WISE, false),
        CLAMP_MAX(OpArityClass.ELEMENT_WISE, false),
        SIGMOID(OpArityClass.ELEMENT_WISE, false),
        CONTIGUOUS(OpArityClass.LAYOUT, false),
        RESHAPE(OpArityClass.LAYOUT, false),
        EXPAND(OpArityClass.LAYOUT, false),
        SELECT(OpArityClass.LAYOUT, false),
        PERMUTE(OpArityClass.LAYOUT, false),
        EXPAND_DIMS(OpArityClass.LAYOUT, false),
        SQUEEZE(OpArityClass.LAYOUT, false),
        NOOP(OpArityClass.SPECIAL, false),
        FUSED(OpArityClass.FUSED, false),
        UNKNOWN(OpArityClass.SPECIAL, false);

        private final OpArityClass category;
        private final boolean fusable;

        OpType(OpArityClass category, boolean fusable) {
            this.category = category;
            this.fusable = fusable;
        }

        public OpArityClass category() {
            return category;
        }

        public boolean isFusable() {
            return fusable;
        }
    }

    OpType opType();

    String getExpression();

    default boolean isCheap() { return false; }

}
