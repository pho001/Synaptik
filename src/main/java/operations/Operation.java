package operations;

/**
 * Immutable descriptor for a tensor graph operation.
 *
 * <p>Operation instances carry semantic metadata such as the operation kind,
 * display expression, and any shape, broadcast, or gradient parameters needed by
 * compilation and execution. Implementations are expected to be side-effect
 * free and must not own tensor storage.</p>
 */
public interface Operation {
    /**
     * Broad execution category used by optimizers and backend selection.
     */
    enum OpArityClass {
        ELEMENT_WISE,
        REDUCTION,
        LAYOUT,
        LINEAR_ALGEBRA,
        SPECIAL,
        FUSED
    };
    /**
     * Stable operation identifier used throughout graph construction,
     * optimization, lowering, and backend dispatch.
     */
    enum OpType {
        ADD(OpArityClass.ELEMENT_WISE, true),
        SUB(OpArityClass.ELEMENT_WISE, true),
        MUL(OpArityClass.ELEMENT_WISE, true),
        DIV(OpArityClass.ELEMENT_WISE, true),
        MIN(OpArityClass.ELEMENT_WISE, true),
        MAX(OpArityClass.ELEMENT_WISE, true),
        GT(OpArityClass.ELEMENT_WISE, true),
        GE(OpArityClass.ELEMENT_WISE, true),
        LT(OpArityClass.ELEMENT_WISE, true),
        LE(OpArityClass.ELEMENT_WISE, true),
        EQ(OpArityClass.ELEMENT_WISE, true),
        NE(OpArityClass.ELEMENT_WISE, true),
        LOGICAL_AND(OpArityClass.ELEMENT_WISE, true),
        LOGICAL_OR(OpArityClass.ELEMENT_WISE, true),
        LOGICAL_NOT(OpArityClass.ELEMENT_WISE, true),
        MIN_GRAD(OpArityClass.SPECIAL, false),
        MAX_GRAD(OpArityClass.SPECIAL, false),
        REDUCE_MIN(OpArityClass.REDUCTION, false),
        REDUCE_MAX(OpArityClass.REDUCTION, false),
        REDUCE_ALL(OpArityClass.REDUCTION, false),
        REDUCE_ANY(OpArityClass.REDUCTION, false),
        SOFTMAX(OpArityClass.SPECIAL, false),
        SOFTMAX_GRAD(OpArityClass.SPECIAL, false),
        LOG_SOFTMAX(OpArityClass.SPECIAL, false),
        LOG_SOFTMAX_GRAD(OpArityClass.SPECIAL, false),
        NLL_LOSS(OpArityClass.SPECIAL, false),
        CROSS_ENTROPY_LOSS(OpArityClass.SPECIAL, false),
        CROSS_ENTROPY_LOSS_INDICES(OpArityClass.SPECIAL, false),
        CROSS_ENTROPY_LOSS_INDICES_GRAD(OpArityClass.SPECIAL, false),
        REDUCE_MIN_GRAD(OpArityClass.SPECIAL, false),
        REDUCE_MAX_GRAD(OpArityClass.SPECIAL, false),
        GATHER(OpArityClass.SPECIAL, false),
        GATHER_GRAD(OpArityClass.SPECIAL, false),
        GATHER_AXIS(OpArityClass.SPECIAL, false),
        GATHER_AXIS_GRAD(OpArityClass.SPECIAL, false),
        GATHER_ND(OpArityClass.SPECIAL, false),
        GATHER_ND_GRAD(OpArityClass.SPECIAL, false),
        TAKE_ALONG_AXIS(OpArityClass.SPECIAL, false),
        TAKE_ALONG_AXIS_GRAD(OpArityClass.SPECIAL, false),
        SCATTER_ADD(OpArityClass.SPECIAL, false),
        SCATTER_ELEMENTS(OpArityClass.SPECIAL, false),
        SCATTER_ND(OpArityClass.SPECIAL, false),
        SCALED_DOT_PRODUCT_ATTENTION(OpArityClass.SPECIAL, false),
        SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(OpArityClass.SPECIAL, false),
        SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS(OpArityClass.SPECIAL, false),
        LINEAR(OpArityClass.SPECIAL, false),
        CONV2D(OpArityClass.SPECIAL, false),
        CONV2D_GEMM(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_WEIGHT(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_INPUT_GEMM(OpArityClass.SPECIAL, false),
        CONV2D_BACKWARD_WEIGHT_GEMM(OpArityClass.SPECIAL, false),
        MAX_POOL2D(OpArityClass.SPECIAL, false),
        MAX_POOL2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        AVG_POOL2D(OpArityClass.SPECIAL, false),
        AVG_POOL2D_BACKWARD_INPUT(OpArityClass.SPECIAL, false),
        LAYER_NORM(OpArityClass.SPECIAL, false),
        RMS_NORM(OpArityClass.SPECIAL, false),
        WHERE(OpArityClass.ELEMENT_WISE, true),
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
        ABS(OpArityClass.ELEMENT_WISE, true),
        MUL_SCALAR(OpArityClass.ELEMENT_WISE, true),
        SUM(OpArityClass.REDUCTION, false),
        MEAN(OpArityClass.REDUCTION, false),
        RELU(OpArityClass.ELEMENT_WISE, true),
        CLAMP_MIN(OpArityClass.ELEMENT_WISE, true),
        CLAMP_MAX(OpArityClass.ELEMENT_WISE, true),
        SIGMOID(OpArityClass.ELEMENT_WISE, true),
        CONTIGUOUS(OpArityClass.LAYOUT, false),
        RESHAPE(OpArityClass.LAYOUT, false),
        EXPAND(OpArityClass.LAYOUT, false),
        SELECT(OpArityClass.LAYOUT, false),
        SLICE(OpArityClass.LAYOUT, false),
        SLICE_GRAD(OpArityClass.SPECIAL, false),
        CONCAT(OpArityClass.LAYOUT, false),
        PERMUTE(OpArityClass.LAYOUT, false),
        EXPAND_DIMS(OpArityClass.LAYOUT, false),
        SQUEEZE(OpArityClass.LAYOUT, false),
        CAST(OpArityClass.SPECIAL, false),
        CONST_SCALAR(OpArityClass.SPECIAL, false),
        NOOP(OpArityClass.SPECIAL, false),
        FUSED(OpArityClass.FUSED, false),
        UNKNOWN(OpArityClass.SPECIAL, false);

        private final OpArityClass category;
        private final boolean fusable;

        OpType(OpArityClass category, boolean fusable) {
            this.category = category;
            this.fusable = fusable;
        }

        /**
         * Returns the broad execution category for this operation type.
         *
         * @return operation category used for optimization and dispatch policy
         */
        public OpArityClass category() {
            return category;
        }

        /**
         * Indicates whether operations of this type are eligible for generic
         * elementwise fusion.
         *
         * @return {@code true} when the type can participate in elementwise
         *         fusion
         */
        public boolean isFusable() {
            return fusable;
        }
    }

    /**
     * Returns the stable operation type for this descriptor.
     *
     * @return operation type used by optimizers and backend dispatch
     */
    OpType opType();

    /**
     * Returns a short symbolic expression used in graph dumps and diagnostics.
     *
     * @return human-readable operation expression
     */
    String getExpression();

    /**
     * Indicates whether this operation is cheap enough for optimizers to
     * duplicate or inline when profitable.
     *
     * @return {@code true} for inexpensive scalar or simple elementwise
     *         descriptors; {@code false} by default
     */
    default boolean isCheap() { return false; }

}
