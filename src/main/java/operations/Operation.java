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

    enum OpSemanticFamily {
        ARITHMETIC,
        TRANSCENDENTAL,
        COMPARISON,
        LOGICAL,
        SELECTION,
        REDUCTION,
        LAYOUT,
        LINEAR_ALGEBRA,
        SPECIAL,
        FUSED,
        UNKNOWN
    }

    enum OpComputationalCost {
        TRIVIAL,
        CHEAP,
        MEDIUM,
        EXPENSIVE,
        UNKNOWN
    }

    enum OpControlTrait {
        NONE,
        BRANCHLESS,
        SELECT_MASK,
        BOOL_LOGIC,
        UNKNOWN
    }

    enum OpResultKind {
        NUMERIC,
        BOOLEAN,
        SHAPE_VIEW,
        INDEX,
        UNKNOWN
    }

    /**
     * Stable operation identifier used throughout graph construction,
     * optimization, lowering, and backend dispatch.
     */
    enum OpType {
        ADD(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SUB(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        MUL(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        DIV(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        MIN(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        MAX(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        GT(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        GE(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        LT(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        LE(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        EQ(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        NE(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.COMPARISON, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.BOOLEAN),
        LOGICAL_AND(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.LOGICAL, OpComputationalCost.CHEAP, OpControlTrait.BOOL_LOGIC, OpResultKind.BOOLEAN),
        LOGICAL_OR(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.LOGICAL, OpComputationalCost.CHEAP, OpControlTrait.BOOL_LOGIC, OpResultKind.BOOLEAN),
        LOGICAL_NOT(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.LOGICAL, OpComputationalCost.CHEAP, OpControlTrait.BOOL_LOGIC, OpResultKind.BOOLEAN),
        MIN_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.UNKNOWN, OpControlTrait.NONE, OpResultKind.NUMERIC),
        MAX_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.UNKNOWN, OpControlTrait.NONE, OpResultKind.NUMERIC),
        REDUCE_MIN(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        REDUCE_MAX(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        REDUCE_PROD(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CUMSUM(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        ARGMAX(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.INDEX),
        REDUCE_ALL(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.BOOL_LOGIC, OpResultKind.BOOLEAN),
        REDUCE_ANY(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.BOOL_LOGIC, OpResultKind.BOOLEAN),
        SOFTMAX(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SOFTMAX_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        LOG_SOFTMAX(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        LOG_SOFTMAX_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        NLL_LOSS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CROSS_ENTROPY_LOSS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CROSS_ENTROPY_LOSS_INDICES(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CROSS_ENTROPY_LOSS_INDICES_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        REDUCE_MIN_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.UNKNOWN, OpControlTrait.NONE, OpResultKind.NUMERIC),
        REDUCE_MAX_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.UNKNOWN, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER_AXIS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER_AXIS_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER_ND(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        GATHER_ND_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        TAKE_ALONG_AXIS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        TAKE_ALONG_AXIS_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCATTER_ADD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCATTER_AXIS_ADD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCATTER_ELEMENTS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCATTER_ND(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCALED_DOT_PRODUCT_ATTENTION(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        LINEAR(OpArityClass.SPECIAL, false, OpSemanticFamily.LINEAR_ALGEBRA, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CONV2D(OpArityClass.SPECIAL, false, OpSemanticFamily.LINEAR_ALGEBRA, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        MAX_POOL2D(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        AVG_POOL2D(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        LAYER_NORM(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        RMS_NORM(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        WHERE(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.SELECTION, OpComputationalCost.CHEAP, OpControlTrait.SELECT_MASK, OpResultKind.NUMERIC),
        MATMUL(OpArityClass.LINEAR_ALGEBRA, false, OpSemanticFamily.LINEAR_ALGEBRA, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        NEG(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        INV(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        LOG(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        EXP(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        FAST_EXP(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        ERF(OpArityClass.ELEMENT_WISE, false, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        TANH(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        FAST_TANH(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        POW(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        POW_TENSOR(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SQRT(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        ABS(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        FLOOR(OpArityClass.ELEMENT_WISE, false, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CEIL(OpArityClass.ELEMENT_WISE, false, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SIGN(OpArityClass.ELEMENT_WISE, false, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        MUL_SCALAR(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SUM(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        MEAN(OpArityClass.REDUCTION, false, OpSemanticFamily.REDUCTION, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        RELU(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        CLAMP_MIN(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        CLAMP_MAX(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.ARITHMETIC, OpComputationalCost.CHEAP, OpControlTrait.BRANCHLESS, OpResultKind.NUMERIC),
        SIGMOID(OpArityClass.ELEMENT_WISE, true, OpSemanticFamily.TRANSCENDENTAL, OpComputationalCost.EXPENSIVE, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CONTIGUOUS(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        RESHAPE(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        EXPAND(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        SELECT(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        SLICE(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        SLICE_GRAD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        SLICE_SCATTER_ADD(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CONCAT(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        PAD(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        TILE(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        UNFOLD_AXIS(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        UNFOLD2D(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        FOLD2D(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.MEDIUM, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        PERMUTE(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        EXPAND_DIMS(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        SQUEEZE(OpArityClass.LAYOUT, false, OpSemanticFamily.LAYOUT, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.SHAPE_VIEW),
        CAST(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.CHEAP, OpControlTrait.NONE, OpResultKind.NUMERIC),
        CONST_SCALAR(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.NUMERIC),
        NOOP(OpArityClass.SPECIAL, false, OpSemanticFamily.SPECIAL, OpComputationalCost.TRIVIAL, OpControlTrait.NONE, OpResultKind.UNKNOWN),
        FUSED(OpArityClass.FUSED, false, OpSemanticFamily.FUSED, OpComputationalCost.UNKNOWN, OpControlTrait.UNKNOWN, OpResultKind.UNKNOWN),
        UNKNOWN(OpArityClass.SPECIAL, false, OpSemanticFamily.UNKNOWN, OpComputationalCost.UNKNOWN, OpControlTrait.UNKNOWN, OpResultKind.UNKNOWN);

        private final OpArityClass category;
        private final boolean fusable;
        private final OpSemanticFamily semanticFamily;
        private final OpComputationalCost computationalCost;
        private final OpControlTrait controlTrait;
        private final OpResultKind resultKind;

        OpType(
                OpArityClass category,
                boolean fusable,
                OpSemanticFamily semanticFamily,
                OpComputationalCost computationalCost,
                OpControlTrait controlTrait,
                OpResultKind resultKind
        ) {
            this.category = category;
            this.fusable = fusable;
            this.semanticFamily = semanticFamily;
            this.computationalCost = computationalCost;
            this.controlTrait = controlTrait;
            this.resultKind = resultKind;
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

        public OpSemanticFamily semanticFamily() {
            return semanticFamily;
        }

        public OpComputationalCost computationalCost() {
            return computationalCost;
        }

        public OpControlTrait controlTrait() {
            return controlTrait;
        }

        public OpResultKind resultKind() {
            return resultKind;
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
