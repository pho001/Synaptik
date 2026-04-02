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
        MIN_GRAD(OpArityClass.SPECIAL, false),
        MAX_GRAD(OpArityClass.SPECIAL, false),
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
        MUL_SCALAR(OpArityClass.ELEMENT_WISE, true),
        SUM(OpArityClass.REDUCTION, false),
        RELU(OpArityClass.ELEMENT_WISE, true),
        SIGMOID(OpArityClass.ELEMENT_WISE, false),
        CONTIGUOUS(OpArityClass.LAYOUT, false),
        RESHAPE(OpArityClass.LAYOUT, false),
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
