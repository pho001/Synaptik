package backend.accelerator.lowering;

import backend.ComputeBackend;
import operations.Operation;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
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
     * Returns checked-in coverage rows for one backend and semantic operation family.
     */
    public static List<GpuLoweringCoverageEntry> entriesForFamily(ComputeBackend backend, GpuLoweringOperationFamily family) {
        if (backend == null || family == null) {
            return List.of();
        }
        return ENTRIES.stream()
                .filter(entry -> entry.backend() == backend && entry.family() == family)
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

    /**
     * Returns a stable planner-facing detail string for a matrix-backed unsupported operation.
     */
    public static String plannerUnsupportedDetail(ComputeBackend backend, Operation.OpType opType) {
        GpuLoweringCoverageEntry entry = entryFor(backend, opType);
        return entry.reason().name()
                + ": operation " + entry.opType() + " is not supported by " + entry.backend() + " lowering"
                + " family=" + entry.family()
                + " status=" + entry.status().name().toLowerCase(Locale.ROOT)
                + " note=" + entry.note();
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
                "native accelerator DAG softmax path; target=transformer_block_hot_path",
                Operation.OpType.SOFTMAX);
        add(entries, backend, Operation.OpType.LOG_SOFTMAX, GpuLoweringOperationFamily.SOFTMAX_LIKE,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "lowered as SOFTMAX followed by LOG using existing accelerator DAG primitives; target=transformer_block_hot_path");
        add(entries, backend, Operation.OpType.SUM, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "native accelerator DAG forward sum reduction path; target=reduction_chain_small");
        add(entries, backend, Operation.OpType.MEAN, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "native accelerator DAG forward mean reduction path; target=reduction_chain_small");
        add(entries, backend, Operation.OpType.REDUCE_MIN, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "native accelerator DAG forward reduce-min path; target=reduction_chain_small");
        add(entries, backend, Operation.OpType.REDUCE_MAX, GpuLoweringOperationFamily.REDUCTION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "native accelerator DAG forward reduce-max path; target=reduction_chain_small");
        add(entries, backend, Operation.OpType.LAYER_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "lowered as repeated MEAN plus elementwise normalization DAG with epsilon scalar; target=layer_norm_small");
        add(entries, backend, Operation.OpType.RMS_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "lowered as repeated MEAN plus elementwise RMS normalization DAG with epsilon scalar; target=rms_norm_small");
        add(entries, backend, Operation.OpType.NLL_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                "loss-adjacent operation remains explicit CPU fallback because no native loss primitive exists; reduction semantics must match CPU; target=transformer_block_hot_path");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                "dense-target cross-entropy remains explicit CPU fallback because no native loss primitive exists; reduction semantics must match CPU; target=transformer_block_hot_path");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                "index-target loss uses INT32 targets plus bounds, ignore-index, and reduction-denominator semantics outside the current accelerator DAG contract; target=transformer_block_hot_path");
        add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, GpuLoweringOperationFamily.LOSS_ADJACENT,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                "index-target loss gradient uses INT32 targets and scatter-like per-class gradient semantics outside the current accelerator DAG contract; target=transformer_block_hot_path");
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "direct FLOAT32 rank-3/4 native MPSGraph primitive SDPA DAG supports unmasked, dense external BOOL masked, causal, and external+causal effective mask modes; target=transformer_block_hot_path target=masked_sdpa_small");
        } else {
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.FALLBACK,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA direct forward SDPA native/lowered path is not implemented; target=transformer_block_hot_path");
        }
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
        addConvPoolRows(entries, backend);
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.GATHER, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "forward gather lowers to Metal gatherAlongAxis with expanded INT32 indices; scoped to dense FLOAT32 value input and static in-bounds INT32 index input; target=gather_take_small");
        } else {
            add(entries, backend, Operation.OpType.GATHER, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA forward gather native/lowered path is not implemented yet");
        }
        add(entries, backend, Operation.OpType.GATHER_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                "gather gradient requires duplicate-index accumulation parity before GPU support");
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "take-along-axis lowers to Metal gatherAlongAxis with INT32 indices; scoped to dense FLOAT32 value input and static in-bounds INT32 index input; target=gather_take_small");
        } else {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA take-along-axis native/lowered path is not implemented yet");
        }
        add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                "take-along-axis gradient requires duplicate-index accumulation parity before GPU support");
        add(entries, backend, Operation.OpType.SCATTER_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                "scatter-add remains CPU-owned until duplicate-index accumulation semantics are proven for native GPU execution");
        addBoolOutputRows(entries, backend);
        addSupported(entries, backend, GpuLoweringOperationFamily.BACKWARD_ADJACENT,
                "native accelerator DAG backward-adjacent path",
                Operation.OpType.SOFTMAX_GRAD,
                Operation.OpType.LOG_SOFTMAX_GRAD,
                Operation.OpType.REDUCE_MIN_GRAD,
                Operation.OpType.REDUCE_MAX_GRAD,
                Operation.OpType.MIN_GRAD,
                Operation.OpType.MAX_GRAD);
        add(entries, backend, Operation.OpType.FUSED, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED,
                "CPU Operation.OpType.FUSED remains CPU-only for Phase 12; GPU compound regions lower from normal graph operations");
    }

    private static void addConvPoolRows(List<GpuLoweringCoverageEntry> entries, ComputeBackend backend) {
        String backendLabel = backend.name();
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.CONV2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal direct FLOAT32 dense NCHW/OIHW Conv2D forward lowers to MPSGraph convolution2D; scoped to groups=1, dilation=1, stride/padding, and optional bias; target=conv2d_resnet_3x3");
        } else {
            add(entries, backend, Operation.OpType.CONV2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    backendLabel + " conv2d NCHW rank-4 native/lowered path is not implemented; stride/padding/dilation/groups must be proven before support; target=conv2d_resnet_3x3");
        }
        add(entries, backend, Operation.OpType.CONV2D_GEMM, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " lowered conv2d GEMM path is CPU-owned until im2col/GEMM/output-layout semantics are represented in the accelerator DAG; target=conv2d_resnet_3x3");
        add(entries, backend, Operation.OpType.CONV2D_BACKWARD_INPUT, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " conv2d backward-input native/lowered path is not implemented; gradient shape, padding, dilation, and groups require parity evidence");
        add(entries, backend, Operation.OpType.CONV2D_BACKWARD_WEIGHT, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " conv2d backward-weight native/lowered path is not implemented; accumulation and grouped weight layout require parity evidence");
        add(entries, backend, Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " lowered conv2d backward-input GEMM path is CPU-owned until accelerator DAG primitives cover the full layout contract");
        add(entries, backend, Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " lowered conv2d backward-weight GEMM path is CPU-owned until accelerator DAG primitives cover accumulation and layout semantics");
        add(entries, backend, Operation.OpType.MAX_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " max-pool native/lowered path is not implemented; kernel/stride/padding and tie behavior must match CPU; target=max_pool2d_small");
        add(entries, backend, Operation.OpType.MAX_POOL2D_BACKWARD_INPUT, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " max-pool backward-input path is not implemented; first-max tie routing must match CPU");
        add(entries, backend, Operation.OpType.AVG_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " avg-pool native/lowered path is not implemented; countIncludePad divisor semantics must match CPU; target=max_pool2d_small");
        add(entries, backend, Operation.OpType.AVG_POOL2D_BACKWARD_INPUT, GpuLoweringOperationFamily.CONV_POOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                backendLabel + " avg-pool backward-input path is not implemented; divisor and padding semantics must match CPU");
    }

    private static void addBoolOutputRows(List<GpuLoweringCoverageEntry> entries, ComputeBackend backend) {
        addBoolOutputRow(entries, backend, Operation.OpType.GT, "greater-than compare");
        addBoolOutputRow(entries, backend, Operation.OpType.GE, "greater-or-equal compare");
        addBoolOutputRow(entries, backend, Operation.OpType.LT, "less-than compare");
        addBoolOutputRow(entries, backend, Operation.OpType.LE, "less-or-equal compare");
        addBoolOutputRow(entries, backend, Operation.OpType.EQ, "equal compare");
        addBoolOutputRow(entries, backend, Operation.OpType.NE, "not-equal compare");
        addBoolOutputRow(entries, backend, Operation.OpType.LOGICAL_AND, "logical AND");
        addBoolOutputRow(entries, backend, Operation.OpType.LOGICAL_OR, "logical OR");
        addBoolOutputRow(entries, backend, Operation.OpType.LOGICAL_NOT, "logical NOT");
        addBoolOutputRow(entries, backend, Operation.OpType.REDUCE_ALL, "BOOL all reduction");
        addBoolOutputRow(entries, backend, Operation.OpType.REDUCE_ANY, "BOOL any reduction");
    }

    private static void addBoolOutputRow(
            List<GpuLoweringCoverageEntry> entries,
            ComputeBackend backend,
            Operation.OpType opType,
            String label
    ) {
        if (backend == ComputeBackend.GPU_METAL && isBoolMetalSupported(opType)) {
            add(entries, backend, opType, GpuLoweringOperationFamily.COMPARE_BOOL,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    label + " has native Metal BOOL output DAG execution and one-byte BOOL buffer residency; external BOOL predicate input residency for WHERE is separate");
            return;
        }
        add(entries, backend, opType, GpuLoweringOperationFamily.COMPARE_BOOL,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                label + " produces BOOL output, which is outside current native accelerator output dtype support; external BOOL predicate input residency for WHERE is separate");
    }

    private static boolean isBoolMetalSupported(Operation.OpType opType) {
        return switch (opType) {
            case GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, REDUCE_ALL, REDUCE_ANY -> true;
            default -> false;
        };
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
