package io.github.pho001.synaptik.backend.metal;

import java.util.Objects;

/**
 * Describes the current structural route state of every schema-thirteen Metal operation kind.
 *
 * <p>This package-private catalog is cold descriptive metadata. It neither admits an occurrence
 * nor approves an implementation route: {@link MetalCapabilityProvider} remains the capability
 * authority and the partition preparer separately enforces the current route freeze. Lookup is an
 * exhaustive enum switch returning shared enum constants, so it performs no allocation, map
 * lookup, reflection, wire-number duplication, or string dispatch.</p>
 */
final class MetalOperationRouteCatalog {
    /** Structural MPSGraph realization state, independent of correctness approval. */
    enum MpsGraphState {
        DIRECT,
        COMPOSED,
        UNAVAILABLE
    }

    /** Current custom-kernel implementation state, independent of capability admission. */
    enum CustomKernelState {
        AVAILABLE,
        PENDING,
        UNAVAILABLE_WITH_PROOF
    }

    /** Closed structural reason identities for the MPSGraph state. */
    enum MpsGraphReason {
        MD_ARITH,
        MD_CAST,
        MD_PRED,
        MD_SHAPE,
        MD_REDUCE,
        MD_INDEX,
        MD_MATMUL,
        MD_ATTENTION_BASE,
        MD_CONV_BASE,
        MD_DENSE_CE,
        MD_BN_INFER,
        MD_SOFTMAX,
        MD_SORT,
        MD_POOL2D,
        MD_IM2COL,
        MC_SCALAR,
        MC_UNARY,
        MC_SELECT,
        MC_STACK,
        MC_UNFOLD_AXIS,
        MC_FOLD_AXIS,
        MC_WINDOW3D,
        MC_MSE,
        MC_INDEX_CE,
        MC_BN_TRAIN,
        MC_LAYER_NORM,
        MC_RMS_NORM,
        MC_LOG_SOFTMAX,
        MC_INDEX64,
        MC_LOGSUMEXP,
        MC_STDDEV,
        MC_NORM,
        MU_RNG,
        MU_RECURRENT
    }

    /** Closed current-source reason identities for the custom-kernel state. */
    enum CustomKernelReason {
        CA_NEG,
        CA_0052,
        CP_POINT,
        CP_MOVE,
        CP_CONTRACT,
        CP_AGGREGATE,
        CP_STATE
    }

    /**
     * Shared immutable combinations returned by exhaustive lookup.
     *
     * <p>Constants may describe several operation kinds that have the same four catalog facts.
     * They are metadata only and intentionally contain no selector, capability, or route choice.</p>
     */
    enum Entry {
        DIRECT_ARITH_CUSTOM_NEG(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_NEG),
        DIRECT_ARITH_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        DIRECT_CAST_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_CAST,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        DIRECT_SHAPE_PENDING_MOVE(MpsGraphState.DIRECT, MpsGraphReason.MD_SHAPE,
                CustomKernelState.PENDING, CustomKernelReason.CP_MOVE),
        DIRECT_REDUCE_PENDING_AGGREGATE(MpsGraphState.DIRECT, MpsGraphReason.MD_REDUCE,
                CustomKernelState.PENDING, CustomKernelReason.CP_AGGREGATE),
        DIRECT_MATMUL_PENDING_CONTRACT(MpsGraphState.DIRECT, MpsGraphReason.MD_MATMUL,
                CustomKernelState.PENDING, CustomKernelReason.CP_CONTRACT),
        DIRECT_INDEX_PENDING_MOVE(MpsGraphState.DIRECT, MpsGraphReason.MD_INDEX,
                CustomKernelState.PENDING, CustomKernelReason.CP_MOVE),
        COMPOSED_UNFOLD_AXIS_PENDING_MOVE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_UNFOLD_AXIS, CustomKernelState.PENDING,
                CustomKernelReason.CP_MOVE),
        DIRECT_PRED_CUSTOM_0052(MpsGraphState.DIRECT, MpsGraphReason.MD_PRED,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0052),
        DIRECT_ARITH_CUSTOM_0052(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0052),
        COMPOSED_SCALAR_CUSTOM_0052(MpsGraphState.COMPOSED, MpsGraphReason.MC_SCALAR,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0052),
        DIRECT_REDUCE_CUSTOM_0052(MpsGraphState.DIRECT, MpsGraphReason.MD_REDUCE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0052),
        DIRECT_ATTENTION_PENDING_CONTRACT(MpsGraphState.DIRECT,
                MpsGraphReason.MD_ATTENTION_BASE, CustomKernelState.PENDING,
                CustomKernelReason.CP_CONTRACT),
        DIRECT_CONV_PENDING_CONTRACT(MpsGraphState.DIRECT, MpsGraphReason.MD_CONV_BASE,
                CustomKernelState.PENDING, CustomKernelReason.CP_CONTRACT),
        DIRECT_PRED_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_PRED,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        COMPOSED_SCALAR_PENDING_POINT(MpsGraphState.COMPOSED, MpsGraphReason.MC_SCALAR,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        COMPOSED_UNARY_PENDING_POINT(MpsGraphState.COMPOSED, MpsGraphReason.MC_UNARY,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        COMPOSED_SELECT_PENDING_MOVE(MpsGraphState.COMPOSED, MpsGraphReason.MC_SELECT,
                CustomKernelState.PENDING, CustomKernelReason.CP_MOVE),
        COMPOSED_STACK_PENDING_MOVE(MpsGraphState.COMPOSED, MpsGraphReason.MC_STACK,
                CustomKernelState.PENDING, CustomKernelReason.CP_MOVE),
        COMPOSED_FOLD_AXIS_PENDING_MOVE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_FOLD_AXIS, CustomKernelState.PENDING,
                CustomKernelReason.CP_MOVE),
        DIRECT_IM2COL_PENDING_MOVE(MpsGraphState.DIRECT, MpsGraphReason.MD_IM2COL,
                CustomKernelState.PENDING, CustomKernelReason.CP_MOVE),
        COMPOSED_WINDOW3D_PENDING_MOVE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.PENDING,
                CustomKernelReason.CP_MOVE),
        COMPOSED_MSE_PENDING_AGGREGATE(MpsGraphState.COMPOSED, MpsGraphReason.MC_MSE,
                CustomKernelState.PENDING, CustomKernelReason.CP_AGGREGATE),
        DIRECT_DENSE_CE_PENDING_AGGREGATE(MpsGraphState.DIRECT,
                MpsGraphReason.MD_DENSE_CE, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_INDEX_CE_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_INDEX_CE, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        DIRECT_BN_PENDING_AGGREGATE(MpsGraphState.DIRECT, MpsGraphReason.MD_BN_INFER,
                CustomKernelState.PENDING, CustomKernelReason.CP_AGGREGATE),
        COMPOSED_BN_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_BN_TRAIN, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_LAYER_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_LAYER_NORM, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_RMS_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_RMS_NORM, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        DIRECT_SOFTMAX_PENDING_AGGREGATE(MpsGraphState.DIRECT,
                MpsGraphReason.MD_SOFTMAX, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_LOG_SOFTMAX_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_LOG_SOFTMAX, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        DIRECT_SORT_PENDING_AGGREGATE(MpsGraphState.DIRECT, MpsGraphReason.MD_SORT,
                CustomKernelState.PENDING, CustomKernelReason.CP_AGGREGATE),
        COMPOSED_INDEX64_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_INDEX64, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        DIRECT_POOL2D_PENDING_AGGREGATE(MpsGraphState.DIRECT,
                MpsGraphReason.MD_POOL2D, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_WINDOW3D_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        UNAVAILABLE_RNG_PENDING_STATE(MpsGraphState.UNAVAILABLE, MpsGraphReason.MU_RNG,
                CustomKernelState.PENDING, CustomKernelReason.CP_STATE),
        UNAVAILABLE_RECURRENT_PENDING_STATE(MpsGraphState.UNAVAILABLE,
                MpsGraphReason.MU_RECURRENT, CustomKernelState.PENDING,
                CustomKernelReason.CP_STATE),
        COMPOSED_LOGSUMEXP_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_LOGSUMEXP, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_STDDEV_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_STDDEV, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_NORM_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_NORM, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE);

        private final MpsGraphState mpsGraphState;
        private final MpsGraphReason mpsGraphReason;
        private final CustomKernelState customKernelState;
        private final CustomKernelReason customKernelReason;

        Entry(MpsGraphState mpsGraphState, MpsGraphReason mpsGraphReason,
                CustomKernelState customKernelState, CustomKernelReason customKernelReason) {
            this.mpsGraphState = Objects.requireNonNull(mpsGraphState, "mpsGraphState");
            this.mpsGraphReason = Objects.requireNonNull(mpsGraphReason, "mpsGraphReason");
            this.customKernelState = Objects.requireNonNull(customKernelState, "customKernelState");
            this.customKernelReason = Objects.requireNonNull(customKernelReason,
                    "customKernelReason");
        }

        MpsGraphState mpsGraphState() {
            return mpsGraphState;
        }

        MpsGraphReason mpsGraphReason() {
            return mpsGraphReason;
        }

        CustomKernelState customKernelState() {
            return customKernelState;
        }

        CustomKernelReason customKernelReason() {
            return customKernelReason;
        }
    }

    private MetalOperationRouteCatalog() {}

    /**
     * Returns the shared descriptive entry for one exact registered operation kind.
     *
     * @param kind non-null schema-thirteen node kind
     * @return non-null shared immutable catalog entry
     * @throws NullPointerException if {@code kind} is {@code null}
     */
    static Entry entry(MetalMpsGraphProgram.NodeKind kind) {
        return switch (Objects.requireNonNull(kind, "kind")) {
            case NEG -> Entry.DIRECT_ARITH_CUSTOM_NEG;
            case ADD, SUB, MUL, DIV, TENSOR_POW -> Entry.DIRECT_ARITH_PENDING_POINT;
            case CAST -> Entry.DIRECT_CAST_PENDING_POINT;
            case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS,
                    PAD, SLICE, SLICE_UPDATE, CONCAT, TILE -> Entry.DIRECT_SHAPE_PENDING_MOVE;
            case ABS, RECIPROCAL, LOG, EXP, ERF, SQRT, RSQRT, FLOOR, CEIL, SIGN,
                    RELU, SIGMOID, TANH -> Entry.DIRECT_ARITH_PENDING_POINT;
            case SUM, MEAN, PROD, ALL, ANY, VARIANCE ->
                    Entry.DIRECT_REDUCE_PENDING_AGGREGATE;
            case MATMUL -> Entry.DIRECT_MATMUL_PENDING_CONTRACT;
            case GATHER, ONE_HOT, SCATTER_ELEMENTS, GATHER_ELEMENTS, SCATTER_ADD,
                    GATHER_ND, SCATTER_ND -> Entry.DIRECT_INDEX_PENDING_MOVE;
            case UNFOLD_AXIS -> Entry.COMPOSED_UNFOLD_AXIS_PENDING_MOVE;
            case GT, GE, LT, LE, EQ, NE -> Entry.DIRECT_PRED_CUSTOM_0052;
            case TENSOR_MIN, TENSOR_MAX, CLAMP -> Entry.DIRECT_ARITH_CUSTOM_0052;
            case SCALAR_MIN, SCALAR_MAX -> Entry.COMPOSED_SCALAR_CUSTOM_0052;
            case REDUCTION_MIN, REDUCTION_MAX, CUM_SUM, CUM_PROD ->
                    Entry.DIRECT_REDUCE_CUSTOM_0052;
            case SCALED_DOT_PRODUCT_ATTENTION -> Entry.DIRECT_ATTENTION_PENDING_CONTRACT;
            case CONV2D, CONV3D -> Entry.DIRECT_CONV_PENDING_CONTRACT;
            case IS_FINITE, IS_NAN, IS_INF, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE ->
                    Entry.DIRECT_PRED_PENDING_POINT;
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW ->
                    Entry.COMPOSED_SCALAR_PENDING_POINT;
            case LOG1P, EXPM1, GELU, GELU_TANH_APPROXIMATION, SILU ->
                    Entry.COMPOSED_UNARY_PENDING_POINT;
            case SELECT -> Entry.COMPOSED_SELECT_PENDING_MOVE;
            case STACK -> Entry.COMPOSED_STACK_PENDING_MOVE;
            case FOLD_AXIS -> Entry.COMPOSED_FOLD_AXIS_PENDING_MOVE;
            case UNFOLD2D, FOLD2D -> Entry.DIRECT_IM2COL_PENDING_MOVE;
            case UNFOLD3D, FOLD3D -> Entry.COMPOSED_WINDOW3D_PENDING_MOVE;
            case MEAN_SQUARED_ERROR -> Entry.COMPOSED_MSE_PENDING_AGGREGATE;
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    Entry.DIRECT_DENSE_CE_PENDING_AGGREGATE;
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    Entry.COMPOSED_INDEX_CE_PENDING_AGGREGATE;
            case BATCH_NORM_INFERENCE -> Entry.DIRECT_BN_PENDING_AGGREGATE;
            case BATCH_NORM_TRAINING -> Entry.COMPOSED_BN_PENDING_AGGREGATE;
            case LAYER_NORM -> Entry.COMPOSED_LAYER_PENDING_AGGREGATE;
            case RMS_NORM -> Entry.COMPOSED_RMS_PENDING_AGGREGATE;
            case SOFTMAX -> Entry.DIRECT_SOFTMAX_PENDING_AGGREGATE;
            case LOG_SOFTMAX -> Entry.COMPOSED_LOG_SOFTMAX_PENDING_AGGREGATE;
            case SORT -> Entry.DIRECT_SORT_PENDING_AGGREGATE;
            case ARGSORT, TOP_K, ARG_MAX, ARG_MIN ->
                    Entry.COMPOSED_INDEX64_PENDING_AGGREGATE;
            case MAX_POOL2D, AVERAGE_POOL2D -> Entry.DIRECT_POOL2D_PENDING_AGGREGATE;
            case MAX_POOL3D, AVERAGE_POOL3D -> Entry.COMPOSED_WINDOW3D_PENDING_AGGREGATE;
            case DROPOUT, INITIAL_STATE -> Entry.UNAVAILABLE_RNG_PENDING_STATE;
            case RNN_TANH, GRU_RESET_AFTER, LSTM -> Entry.UNAVAILABLE_RECURRENT_PENDING_STATE;
            case LOG_SUM_EXP -> Entry.COMPOSED_LOGSUMEXP_PENDING_AGGREGATE;
            case STANDARD_DEVIATION -> Entry.COMPOSED_STDDEV_PENDING_AGGREGATE;
            case L1_NORM, L2_NORM -> Entry.COMPOSED_NORM_PENDING_AGGREGATE;
        };
    }
}
