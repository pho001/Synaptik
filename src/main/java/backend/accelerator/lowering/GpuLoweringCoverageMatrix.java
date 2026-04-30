package backend.accelerator.lowering;

import backend.ComputeBackend;
import operations.Operation;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Backend-neutral GPU lowering coverage source of truth for Metal and CUDA.
 */
public final class GpuLoweringCoverageMatrix {
    private static final List<GpuLoweringCoverageEntry> ENTRIES = buildEntries();
    private static final Map<ComputeBackend, Map<Operation.OpType, GpuLoweringCoverageEntry>> BY_BACKEND_AND_OP =
            indexEntries(ENTRIES);

    private GpuLoweringCoverageMatrix() {
    }

    /**
     * Returns all checked-in Metal and CUDA coverage rows.
     */
    public static List<GpuLoweringCoverageEntry> entries() {
        return ENTRIES;
    }

    /**
     * Returns checked-in coverage rows for one backend.
     */
    public static List<GpuLoweringCoverageEntry> entriesFor(ComputeBackend backend) {
        if (backend == null) {
            return List.of();
        }
        return ENTRIES.stream()
                .filter(entry -> entry.backend() == backend)
                .toList();
    }

    /**
     * Returns the coverage row for one backend operation, or an explicit unsupported row when unlisted.
     */
    public static GpuLoweringCoverageEntry entryFor(ComputeBackend backend, Operation.OpType opType) {
        if (backend == null || opType == null) {
            return unsupported(backend, opType, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                    "missing backend or operation metadata");
        }
        Map<Operation.OpType, GpuLoweringCoverageEntry> byOp = BY_BACKEND_AND_OP.get(backend);
        GpuLoweringCoverageEntry entry = byOp == null ? null : byOp.get(opType);
        if (entry != null) {
            return entry;
        }
        return unsupported(backend, opType, familyFor(opType), GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "operation is not in the checked-in GPU lowering coverage matrix");
    }

    /**
     * Returns true when the source-level coverage matrix marks the backend operation as supported.
     */
    public static boolean isSupported(ComputeBackend backend, Operation.OpType opType) {
        return entryFor(backend, opType).status() == GpuLoweringCoverageStatus.SUPPORTED;
    }

    private static List<GpuLoweringCoverageEntry> buildEntries() {
        ArrayList<GpuLoweringCoverageEntry> entries = new ArrayList<>();
        addBackend(entries, ComputeBackend.GPU_METAL);
        addBackend(entries, ComputeBackend.GPU_CUDA);
        return List.copyOf(entries);
    }

    private static void addBackend(List<GpuLoweringCoverageEntry> entries, ComputeBackend backend) {
        addSupported(entries, backend, GpuLoweringOperationFamily.MATMUL_LINEAR,
                "native accelerator DAG matmul/linear path",
                Operation.OpType.MATMUL,
                Operation.OpType.LINEAR);
        addSupported(entries, backend, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                "native accelerator DAG elementwise path",
                Operation.OpType.ADD,
                Operation.OpType.SUB,
                Operation.OpType.MUL,
                Operation.OpType.DIV,
                Operation.OpType.RELU,
                Operation.OpType.TANH,
                Operation.OpType.FAST_TANH,
                Operation.OpType.SIGMOID,
                Operation.OpType.ABS,
                Operation.OpType.EXP,
                Operation.OpType.FAST_EXP,
                Operation.OpType.LOG,
                Operation.OpType.NEG,
                Operation.OpType.SQRT,
                Operation.OpType.INV,
                Operation.OpType.MUL_SCALAR,
                Operation.OpType.WHERE,
                Operation.OpType.CLAMP_MIN,
                Operation.OpType.CLAMP_MAX);
        addSupported(entries, backend, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                "layout/view-adjacent accelerator DAG metadata or materialization path",
                Operation.OpType.RESHAPE,
                Operation.OpType.CONTIGUOUS,
                Operation.OpType.NOOP,
                Operation.OpType.PERMUTE,
                Operation.OpType.EXPAND_DIMS,
                Operation.OpType.SQUEEZE);
        addSupported(entries, backend, GpuLoweringOperationFamily.SOFTMAX_LIKE,
                "native accelerator DAG softmax path",
                Operation.OpType.SOFTMAX);
        add(entries, backend, Operation.OpType.LOG_SOFTMAX, GpuLoweringOperationFamily.SOFTMAX_LIKE,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "Phase 11 expands this through SOFTMAX plus LOG lowering");
        add(entries, backend, Operation.OpType.SUM, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "forward reductions are matrix-first until native coverage is added");
        add(entries, backend, Operation.OpType.MEAN, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "mean requires reduction plus scale lowering coverage");
        add(entries, backend, Operation.OpType.REDUCE_MIN, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "forward reduce-min is not in the tested accelerator planner allowlist");
        add(entries, backend, Operation.OpType.REDUCE_MAX, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "forward reduce-max is not in the tested accelerator planner allowlist");
        add(entries, backend, Operation.OpType.LAYER_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.DEFERRED_FUSED_REGION,
                "normalization requires compound reduction-adjacent GPU region execution");
        add(entries, backend, Operation.OpType.RMS_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.DEFERRED_FUSED_REGION,
                "normalization requires compound reduction-adjacent GPU region execution");
        add(entries, backend, Operation.OpType.NLL_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "loss-adjacent operation remains explicit CPU fallback");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "loss-adjacent operation remains explicit CPU fallback");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                "index-target loss uses INT32 targets outside the current accelerator DAG dtype contract");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                "index-target loss gradient uses INT32 targets outside the current accelerator DAG dtype contract");
        add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, GpuLoweringOperationFamily.ATTENTION,
                GpuLoweringCoverageStatus.FALLBACK,
                backend == ComputeBackend.GPU_METAL
                        ? GpuLoweringUnsupportedReason.CAPABILITY_MISSING
                        : GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                backend == ComputeBackend.GPU_METAL
                        ? "direct forward SDPA waits for verified native scale and mask semantics"
                        : "CUDA planner does not yet admit direct forward SDPA regions");
        if (backend == ComputeBackend.GPU_METAL) {
            addSupported(entries, backend, GpuLoweringOperationFamily.ATTENTION,
                    "Metal backward SDPA is represented by the shared accelerator DAG",
                    Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD);
        } else {
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA backward SDPA native path is not currently capability-gated as supported");
        }
        add(entries, backend, Operation.OpType.CONV2D, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "conv/pool coverage is outside the current tested accelerator planner allowlist");
        add(entries, backend, Operation.OpType.MAX_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "conv/pool coverage is outside the current tested accelerator planner allowlist");
        add(entries, backend, Operation.OpType.GATHER, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "index/scatter/gather kernels remain CPU-owned");
        add(entries, backend, Operation.OpType.SCATTER_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "index/scatter/gather kernels remain CPU-owned");
        add(entries, backend, Operation.OpType.GT, GpuLoweringOperationFamily.COMPARE_BOOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                "BOOL-producing compare nodes are outside current accelerator output dtype support");
        add(entries, backend, Operation.OpType.EQ, GpuLoweringOperationFamily.COMPARE_BOOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                "BOOL-producing compare nodes are outside current accelerator output dtype support");
        addSupported(entries, backend, GpuLoweringOperationFamily.BACKWARD_ADJACENT,
                "native accelerator DAG backward-adjacent path",
                Operation.OpType.SOFTMAX_GRAD,
                Operation.OpType.LOG_SOFTMAX_GRAD,
                Operation.OpType.REDUCE_MIN_GRAD,
                Operation.OpType.REDUCE_MAX_GRAD,
                Operation.OpType.MIN_GRAD,
                Operation.OpType.MAX_GRAD);
        add(entries, backend, Operation.OpType.FUSED, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                GpuLoweringCoverageStatus.FALLBACK,
                GpuLoweringUnsupportedReason.DEFERRED_FUSED_REGION,
                "Phase 12 owns compound fused GPU region execution");
    }

    private static void addSupported(
            List<GpuLoweringCoverageEntry> entries,
            ComputeBackend backend,
            GpuLoweringOperationFamily family,
            String note,
            Operation.OpType... opTypes
    ) {
        for (Operation.OpType opType : opTypes) {
            add(entries, backend, opType, family, GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED, note);
        }
    }

    private static void add(
            List<GpuLoweringCoverageEntry> entries,
            ComputeBackend backend,
            Operation.OpType opType,
            GpuLoweringOperationFamily family,
            GpuLoweringCoverageStatus status,
            GpuLoweringUnsupportedReason reason,
            String note
    ) {
        entries.add(new GpuLoweringCoverageEntry(backend, opType, family, status, reason, note));
    }

    private static GpuLoweringCoverageEntry unsupported(
            ComputeBackend backend,
            Operation.OpType opType,
            GpuLoweringOperationFamily family,
            GpuLoweringUnsupportedReason reason,
            String note
    ) {
        return new GpuLoweringCoverageEntry(
                backend == null ? ComputeBackend.CPU : backend,
                opType == null ? Operation.OpType.UNKNOWN : opType,
                family,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                reason,
                note
        );
    }

    private static Map<ComputeBackend, Map<Operation.OpType, GpuLoweringCoverageEntry>> indexEntries(
            List<GpuLoweringCoverageEntry> entries
    ) {
        EnumMap<ComputeBackend, Map<Operation.OpType, GpuLoweringCoverageEntry>> byBackend = new EnumMap<>(ComputeBackend.class);
        for (GpuLoweringCoverageEntry entry : entries) {
            byBackend.computeIfAbsent(entry.backend(), ignored -> new EnumMap<>(Operation.OpType.class))
                    .putIfAbsent(entry.opType(), entry);
        }
        return Map.copyOf(byBackend);
    }

    private static GpuLoweringOperationFamily familyFor(Operation.OpType opType) {
        if (opType == null) {
            return GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
        }
        return switch (opType.category()) {
            case LINEAR_ALGEBRA -> GpuLoweringOperationFamily.MATMUL_LINEAR;
            case ELEMENT_WISE -> opType == Operation.OpType.GT
                    || opType == Operation.OpType.GE
                    || opType == Operation.OpType.LT
                    || opType == Operation.OpType.LE
                    || opType == Operation.OpType.EQ
                    || opType == Operation.OpType.NE
                    || opType == Operation.OpType.LOGICAL_AND
                    || opType == Operation.OpType.LOGICAL_OR
                    || opType == Operation.OpType.LOGICAL_NOT
                    ? GpuLoweringOperationFamily.COMPARE_BOOL
                    : GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
            case LAYOUT -> GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT;
            case REDUCTION -> GpuLoweringOperationFamily.REDUCTION;
            case FUSED -> GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
            case SPECIAL -> specialFamilyFor(opType);
        };
    }

    private static GpuLoweringOperationFamily specialFamilyFor(Operation.OpType opType) {
        return switch (opType) {
            case SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD -> GpuLoweringOperationFamily.SOFTMAX_LIKE;
            case LAYER_NORM, RMS_NORM -> GpuLoweringOperationFamily.NORMALIZATION;
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD -> GpuLoweringOperationFamily.LOSS_ADJACENT;
            case SCALED_DOT_PRODUCT_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS -> GpuLoweringOperationFamily.ATTENTION;
            case CONV2D, CONV2D_GEMM, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT, CONV2D_BACKWARD_INPUT_GEMM,
                    CONV2D_BACKWARD_WEIGHT_GEMM, MAX_POOL2D, MAX_POOL2D_BACKWARD_INPUT, AVG_POOL2D,
                    AVG_POOL2D_BACKWARD_INPUT -> GpuLoweringOperationFamily.CONV_POOL;
            case GATHER, GATHER_GRAD, TAKE_ALONG_AXIS, TAKE_ALONG_AXIS_GRAD, SCATTER_ADD -> GpuLoweringOperationFamily.INDEX_SCATTER_GATHER;
            case REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, MIN_GRAD, MAX_GRAD -> GpuLoweringOperationFamily.BACKWARD_ADJACENT;
            default -> GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
        };
    }
}
