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
        ADD,
        SUB,
        MUL,
        DIV,
        MIN,
        MAX,
        GT,
        GE,
        LT,
        LE,
        EQ,
        NE,
        LOGICAL_AND,
        LOGICAL_OR,
        LOGICAL_NOT,
        MIN_GRAD,
        MAX_GRAD,
        REDUCE_MIN,
        REDUCE_MAX,
        REDUCE_PROD,
        CUMSUM,
        ARGMAX,
        REDUCE_ALL,
        REDUCE_ANY,
        SOFTMAX,
        SOFTMAX_GRAD,
        LOG_SOFTMAX,
        LOG_SOFTMAX_GRAD,
        NLL_LOSS,
        CROSS_ENTROPY_LOSS,
        CROSS_ENTROPY_LOSS_INDICES,
        CROSS_ENTROPY_LOSS_INDICES_GRAD,
        REDUCE_MIN_GRAD,
        REDUCE_MAX_GRAD,
        GATHER,
        GATHER_GRAD,
        GATHER_AXIS,
        GATHER_AXIS_GRAD,
        GATHER_ND,
        GATHER_ND_GRAD,
        TAKE_ALONG_AXIS,
        TAKE_ALONG_AXIS_GRAD,
        SCATTER_ADD,
        SCATTER_AXIS_ADD,
        SCATTER_ELEMENTS,
        SCATTER_ND,
        SCALED_DOT_PRODUCT_ATTENTION,
        SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
        SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
        LINEAR,
        CONV2D,
        MAX_POOL2D,
        AVG_POOL2D,
        LAYER_NORM,
        RMS_NORM,
        WHERE,
        MATMUL,
        NEG,
        INV,
        LOG,
        EXP,
        FAST_EXP,
        ERF,
        TANH,
        FAST_TANH,
        POW,
        POW_TENSOR,
        SQRT,
        ABS,
        FLOOR,
        CEIL,
        SIGN,
        MUL_SCALAR,
        SUM,
        MEAN,
        RELU,
        CLAMP_MIN,
        CLAMP_MAX,
        SIGMOID,
        CONTIGUOUS,
        RESHAPE,
        EXPAND,
        SELECT,
        SLICE,
        SLICE_BACKWARD,
        CONCAT,
        PAD,
        TILE,
        UNFOLD_AXIS,
        UNFOLD2D,
        FOLD2D,
        PERMUTE,
        EXPAND_DIMS,
        SQUEEZE,
        CAST,
        CONST_SCALAR,
        NOOP,
        FUSED,
        UNKNOWN;
    }

    /**
     * Returns the stable operation type for this descriptor.
     *
     * @return operation type used by optimizers and backend dispatch
     */
    OpType opType();

    /**
     * Returns the broad execution category for this operation descriptor.
     *
     * @return operation category used for optimization and dispatch policy
     */
    OpArityClass arityClass();

    /**
     * Indicates whether this operation is eligible for generic elementwise fusion.
     *
     * @return {@code true} when this operation can participate in elementwise fusion
     */
    boolean isFusable();

    /**
     * Returns the semantic family used by prepare-time planning policy.
     *
     * @return backend-neutral operation semantic family
     */
    OpSemanticFamily semanticFamily();

    /**
     * Returns the approximate computational cost used by prepare-time planning policy.
     *
     * @return backend-neutral operation cost class
     */
    OpComputationalCost computationalCost();

    /**
     * Returns the control-flow shape used by prepare-time planning policy.
     *
     * @return backend-neutral operation control trait
     */
    OpControlTrait controlTrait();

    /**
     * Returns the logical result kind produced by this operation.
     *
     * @return backend-neutral operation result kind
     */
    OpResultKind resultKind();

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
    default boolean isCheap() {
        OpComputationalCost cost = computationalCost();
        return cost == OpComputationalCost.TRIVIAL || cost == OpComputationalCost.CHEAP;
    }

}
