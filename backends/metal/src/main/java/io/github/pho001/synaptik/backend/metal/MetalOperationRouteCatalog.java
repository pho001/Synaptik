package io.github.pho001.synaptik.backend.metal;

import java.util.Objects;

/**
 * Describes the current structural route state of every schema-twenty Metal operation kind.
 *
 * <p>This package-private catalog is cold descriptive metadata. It neither admits an occurrence nor
 * approves an implementation route: {@link MetalCapabilityProvider} remains the capability
 * authority and the partition preparer separately enforces the current route freeze. Lookup is an
 * exhaustive enum switch returning shared enum constants, so it performs no allocation, map lookup,
 * reflection, wire-number duplication, or string dispatch. The EXP entry records its direct
 * FLOAT32 MPSGraph construction and still-pending FLOAT32 custom pointwise route. Its separately
 * qualified BFLOAT16/FLOAT16 custom kernels are occurrence-specific and do not change this
 * operation-wide FLOAT32 catalog state.
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
        CA_0057,
        CA_0058,
        CA_0059,
        CA_0060,
        CA_0061,
        CA_0063,
        CA_0064,
        CA_0065,
        CA_0066,
        CA_0069,
        CP_POINT,
        CP_POWER,
        CP_ELEMENTARY,
        CP_RECURSIVE_SITES,
        CP_CONTRACT,
        CP_AGGREGATE,
        CP_STATE
    }

    /**
     * Shared immutable combinations returned by exhaustive lookup.
     *
     * <p>Constants may describe several operation kinds that have the same four catalog facts. They
     * are metadata only and intentionally contain no selector, capability, or route choice.
     */
    enum Entry {
        DIRECT_ARITH_CUSTOM_NEG(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_NEG),
        DIRECT_ARITH_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        DIRECT_ARITH_CUSTOM_0058(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0058),
        DIRECT_ARITH_PENDING_POWER(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.PENDING, CustomKernelReason.CP_POWER),
        DIRECT_ARITH_PENDING_ELEMENTARY(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.PENDING, CustomKernelReason.CP_ELEMENTARY),
        DIRECT_ARITH_PENDING_RECURSIVE(MpsGraphState.DIRECT, MpsGraphReason.MD_ARITH,
                CustomKernelState.PENDING, CustomKernelReason.CP_RECURSIVE_SITES),
        DIRECT_CAST_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_CAST,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        DIRECT_CAST_CUSTOM_0059(MpsGraphState.DIRECT, MpsGraphReason.MD_CAST,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        DIRECT_SHAPE_CUSTOM_0059(MpsGraphState.DIRECT, MpsGraphReason.MD_SHAPE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        DIRECT_SHAPE_CUSTOM_0060(MpsGraphState.DIRECT, MpsGraphReason.MD_SHAPE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0060),
    DIRECT_SHAPE_CUSTOM_0066(
        MpsGraphState.DIRECT,
        MpsGraphReason.MD_SHAPE,
        CustomKernelState.AVAILABLE,
        CustomKernelReason.CA_0066),
        DIRECT_REDUCE_CUSTOM_0069(MpsGraphState.DIRECT, MpsGraphReason.MD_REDUCE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0069),
        DIRECT_REDUCE_PENDING_AGGREGATE(MpsGraphState.DIRECT, MpsGraphReason.MD_REDUCE,
                CustomKernelState.PENDING, CustomKernelReason.CP_AGGREGATE),
        DIRECT_REDUCE_CUSTOM_0060(MpsGraphState.DIRECT, MpsGraphReason.MD_REDUCE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0060),
        DIRECT_MATMUL_CUSTOM_0061(MpsGraphState.DIRECT, MpsGraphReason.MD_MATMUL,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0061),
        DIRECT_INDEX_CUSTOM_0059(MpsGraphState.DIRECT, MpsGraphReason.MD_INDEX,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        DIRECT_INDEX_CUSTOM_0060(MpsGraphState.DIRECT, MpsGraphReason.MD_INDEX,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0060),
    DIRECT_INDEX_CUSTOM_0066(
        MpsGraphState.DIRECT,
        MpsGraphReason.MD_INDEX,
        CustomKernelState.AVAILABLE,
        CustomKernelReason.CA_0066),
        DIRECT_INDEX_CUSTOM_0069(MpsGraphState.DIRECT, MpsGraphReason.MD_INDEX,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0069),
    COMPOSED_UNFOLD_AXIS_CUSTOM_0066(
        MpsGraphState.COMPOSED,
        MpsGraphReason.MC_UNFOLD_AXIS,
        CustomKernelState.AVAILABLE,
        CustomKernelReason.CA_0066),
        DIRECT_PRED_CUSTOM_0052(MpsGraphState.DIRECT, MpsGraphReason.MD_PRED,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0052),
        DIRECT_PRED_CUSTOM_0057(MpsGraphState.DIRECT, MpsGraphReason.MD_PRED,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0057),
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
        DIRECT_CONV_CUSTOM_0064(MpsGraphState.DIRECT, MpsGraphReason.MD_CONV_BASE,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0064),
        DIRECT_PRED_PENDING_POINT(MpsGraphState.DIRECT, MpsGraphReason.MD_PRED,
                CustomKernelState.PENDING, CustomKernelReason.CP_POINT),
        COMPOSED_SCALAR_PENDING_POINT(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_SCALAR, CustomKernelState.PENDING,
                CustomKernelReason.CP_POINT),
        COMPOSED_SCALAR_PENDING_POWER(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_SCALAR, CustomKernelState.PENDING,
                CustomKernelReason.CP_POWER),
        COMPOSED_UNARY_PENDING_POINT(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_UNARY, CustomKernelState.PENDING,
                CustomKernelReason.CP_POINT),
        COMPOSED_UNARY_PENDING_ELEMENTARY(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_UNARY, CustomKernelState.PENDING,
                CustomKernelReason.CP_ELEMENTARY),
        COMPOSED_UNARY_PENDING_RECURSIVE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_UNARY, CustomKernelState.PENDING,
                CustomKernelReason.CP_RECURSIVE_SITES),
        COMPOSED_SELECT_CUSTOM_0059(MpsGraphState.COMPOSED, MpsGraphReason.MC_SELECT,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        COMPOSED_STACK_CUSTOM_0059(MpsGraphState.COMPOSED, MpsGraphReason.MC_STACK,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        COMPOSED_FOLD_AXIS_CUSTOM_0060(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_FOLD_AXIS, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0060),
        DIRECT_IM2COL_CUSTOM_0059(MpsGraphState.DIRECT, MpsGraphReason.MD_IM2COL,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0059),
        DIRECT_IM2COL_CUSTOM_0060(MpsGraphState.DIRECT, MpsGraphReason.MD_IM2COL,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0060),
        COMPOSED_WINDOW3D_CUSTOM_0059(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0059),
        COMPOSED_WINDOW3D_CUSTOM_0060(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0060),
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
        DIRECT_SORT_CUSTOM_0063(MpsGraphState.DIRECT, MpsGraphReason.MD_SORT,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0063),
        COMPOSED_INDEX64_CUSTOM_0063(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_INDEX64, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0063),
        DIRECT_POOL2D_PENDING_AGGREGATE(MpsGraphState.DIRECT,
                MpsGraphReason.MD_POOL2D, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        DIRECT_POOL2D_CUSTOM_0064(MpsGraphState.DIRECT,
                MpsGraphReason.MD_POOL2D, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0064),
        COMPOSED_WINDOW3D_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_WINDOW3D_CUSTOM_0064(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_WINDOW3D, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0064),
        UNAVAILABLE_RNG_CUSTOM_0065(MpsGraphState.UNAVAILABLE, MpsGraphReason.MU_RNG,
                CustomKernelState.AVAILABLE, CustomKernelReason.CA_0065),
        UNAVAILABLE_RECURRENT_PENDING_STATE(MpsGraphState.UNAVAILABLE,
                MpsGraphReason.MU_RECURRENT, CustomKernelState.PENDING,
                CustomKernelReason.CP_STATE),
        COMPOSED_LOGSUMEXP_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_LOGSUMEXP, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_STDDEV_PENDING_AGGREGATE(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_STDDEV, CustomKernelState.PENDING,
                CustomKernelReason.CP_AGGREGATE),
        COMPOSED_NORM_CUSTOM_0069(MpsGraphState.COMPOSED,
                MpsGraphReason.MC_NORM, CustomKernelState.AVAILABLE,
                CustomKernelReason.CA_0069),
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
     * @param kind non-null schema-twenty node kind
     * @return non-null shared immutable catalog entry
     * @throws NullPointerException if {@code kind} is {@code null}
     */
    static Entry entry(MetalMpsGraphProgram.NodeKind kind) {
        return switch (Objects.requireNonNull(kind, "kind")) {
            case NEG -> Entry.DIRECT_ARITH_CUSTOM_NEG;
            case ADD, SUB, MUL, DIV -> Entry.DIRECT_ARITH_PENDING_POINT;
            case TENSOR_POW -> Entry.DIRECT_ARITH_PENDING_POWER;
            case CAST -> Entry.DIRECT_CAST_CUSTOM_0059;
            case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS ->
                    Entry.DIRECT_SHAPE_CUSTOM_0066;
            case SLICE_UPDATE -> Entry.DIRECT_SHAPE_CUSTOM_0060;
            case PAD, SLICE, CONCAT, TILE -> Entry.DIRECT_SHAPE_CUSTOM_0059;
            case ABS, EXP, SIGMOID -> Entry.DIRECT_ARITH_PENDING_POINT;
            case FLOOR, CEIL, SIGN, RELU -> Entry.DIRECT_ARITH_CUSTOM_0058;
            case LOG, ERF, SQRT, TANH -> Entry.DIRECT_ARITH_PENDING_ELEMENTARY;
            case RECIPROCAL -> Entry.COMPOSED_UNARY_PENDING_POINT;
            case RSQRT -> Entry.DIRECT_ARITH_PENDING_RECURSIVE;
            case SUM, MEAN -> Entry.DIRECT_REDUCE_PENDING_AGGREGATE;
            case VARIANCE -> Entry.DIRECT_REDUCE_CUSTOM_0069;
            case PROD, ALL, ANY -> Entry.DIRECT_REDUCE_CUSTOM_0060;
            case MATMUL -> Entry.DIRECT_MATMUL_CUSTOM_0061;
            case GATHER, ONE_HOT, SCATTER_ELEMENTS -> Entry.DIRECT_INDEX_CUSTOM_0066;
            case SCATTER_ADD -> Entry.DIRECT_INDEX_CUSTOM_0069;
            case SCATTER_ND -> Entry.DIRECT_INDEX_CUSTOM_0060;
            case GATHER_ELEMENTS, GATHER_ND -> Entry.DIRECT_INDEX_CUSTOM_0059;
            case UNFOLD_AXIS -> Entry.COMPOSED_UNFOLD_AXIS_CUSTOM_0066;
            case GT, GE, LT, LE, EQ, NE -> Entry.DIRECT_PRED_CUSTOM_0052;
            case TENSOR_MIN, TENSOR_MAX, CLAMP -> Entry.DIRECT_ARITH_CUSTOM_0052;
            case SCALAR_MIN, SCALAR_MAX -> Entry.COMPOSED_SCALAR_CUSTOM_0052;
            case REDUCTION_MIN, REDUCTION_MAX, CUM_SUM, CUM_PROD ->
                    Entry.DIRECT_REDUCE_CUSTOM_0052;
            case SCALED_DOT_PRODUCT_ATTENTION -> Entry.DIRECT_ATTENTION_PENDING_CONTRACT;
            case CONV2D, CONV3D -> Entry.DIRECT_CONV_CUSTOM_0064;
            case IS_FINITE, IS_NAN, IS_INF, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE ->
                    Entry.DIRECT_PRED_CUSTOM_0057;
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV ->
                    Entry.COMPOSED_SCALAR_PENDING_POINT;
            case SCALAR_POW -> Entry.COMPOSED_SCALAR_PENDING_POWER;
            case LOG1P, EXPM1 -> Entry.COMPOSED_UNARY_PENDING_ELEMENTARY;
            case GELU, GELU_TANH_APPROXIMATION, SILU ->
                    Entry.COMPOSED_UNARY_PENDING_RECURSIVE;
            case SELECT -> Entry.COMPOSED_SELECT_CUSTOM_0059;
            case STACK -> Entry.COMPOSED_STACK_CUSTOM_0059;
            case FOLD_AXIS -> Entry.COMPOSED_FOLD_AXIS_CUSTOM_0060;
            case FOLD2D -> Entry.DIRECT_IM2COL_CUSTOM_0060;
            case UNFOLD2D -> Entry.DIRECT_IM2COL_CUSTOM_0059;
            case FOLD3D -> Entry.COMPOSED_WINDOW3D_CUSTOM_0060;
            case UNFOLD3D -> Entry.COMPOSED_WINDOW3D_CUSTOM_0059;
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
            case SORT -> Entry.DIRECT_SORT_CUSTOM_0063;
            case ARGSORT, TOP_K, ARG_MAX, ARG_MIN ->
                    Entry.COMPOSED_INDEX64_CUSTOM_0063;
            case MAX_POOL2D, AVERAGE_POOL2D -> Entry.DIRECT_POOL2D_CUSTOM_0064;
            case MAX_POOL3D, AVERAGE_POOL3D -> Entry.COMPOSED_WINDOW3D_CUSTOM_0064;
            case DROPOUT, INITIAL_STATE -> Entry.UNAVAILABLE_RNG_CUSTOM_0065;
            case RNN_TANH, GRU_RESET_AFTER, LSTM -> Entry.UNAVAILABLE_RECURRENT_PENDING_STATE;
            case LOG_SUM_EXP -> Entry.COMPOSED_LOGSUMEXP_PENDING_AGGREGATE;
            case STANDARD_DEVIATION -> Entry.COMPOSED_STDDEV_PENDING_AGGREGATE;
            case L1_NORM -> Entry.COMPOSED_NORM_CUSTOM_0069;
            case L2_NORM -> Entry.COMPOSED_NORM_PENDING_AGGREGATE;
        };
    }
}
