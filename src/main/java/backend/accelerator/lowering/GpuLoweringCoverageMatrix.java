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
        if (backend == ComputeBackend.GPU_METAL) {
            addSupported(entries, backend, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    "MPSGraph-first elementwise parity gap closed through native accelerator DAG mapping",
                    Operation.OpType.MIN,
                    Operation.OpType.MAX,
                    Operation.OpType.POW,
                    Operation.OpType.ERF,
                    Operation.OpType.FLOOR,
                    Operation.OpType.CEIL,
                    Operation.OpType.SIGN);
            add(entries, backend, Operation.OpType.POW_TENSOR, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "Metal tensor-exponent POW_TENSOR remains unsupported until native accelerator DAG execution maps binary power semantics");
        } else {
            add(entries, backend, Operation.OpType.MIN, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA MIN remains unsupported until native accelerator DAG execution maps elementwise minimum");
            add(entries, backend, Operation.OpType.MAX, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA MAX remains unsupported until native accelerator DAG execution maps elementwise maximum");
            add(entries, backend, Operation.OpType.POW, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA POW remains unsupported until scalar exponent power is mapped in native accelerator DAG execution");
            add(entries, backend, Operation.OpType.POW_TENSOR, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA POW_TENSOR remains unsupported until native accelerator DAG execution maps binary power semantics");
            for (Operation.OpType opType : List.of(Operation.OpType.ERF, Operation.OpType.FLOOR, Operation.OpType.CEIL, Operation.OpType.SIGN)) {
                add(entries, backend, opType, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                        GpuLoweringCoverageStatus.UNSUPPORTED,
                        GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                        "CUDA " + opType + " remains unsupported until native accelerator DAG execution maps the unary math primitive");
            }
        }
        addSupported(entries, backend, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                "layout/view-adjacent accelerator DAG metadata or materialization path",
                Operation.OpType.RESHAPE,
                Operation.OpType.CONTIGUOUS,
                Operation.OpType.NOOP,
                Operation.OpType.PERMUTE,
                Operation.OpType.EXPAND_DIMS,
                Operation.OpType.SQUEEZE);
        add(entries, backend, Operation.OpType.UNFOLD2D, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                backend.name() + " UNFOLD2D supports scoped dense "
                        + (backend == ComputeBackend.GPU_METAL ? "FLOAT32/BFLOAT16" : "FLOAT32")
                        + " native im2col lowering with "
                        + (backend == ComputeBackend.GPU_METAL ? "stride=1 and dilation=1" : "dilation=1")
                        + "; wider geometry is rejected by planner semantics");
        add(entries, backend, Operation.OpType.FOLD2D, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                backend.name() + " FOLD2D supports scoped dense "
                        + (backend == ComputeBackend.GPU_METAL ? "FLOAT32/BFLOAT16" : "FLOAT32")
                        + " native col2im accumulation lowering with "
                        + (backend == ComputeBackend.GPU_METAL ? "stride=1 and dilation=1" : "dilation=1")
                        + "; wider geometry is rejected by planner semantics");
        add(entries, backend, Operation.OpType.UNFOLD_AXIS, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                backend.name() + " UNFOLD_AXIS supports scoped dense "
                        + (backend == ComputeBackend.GPU_METAL ? "FLOAT32/BFLOAT16" : "FLOAT32")
                        + " native axis sliding-window materialization"
                        + (backend == ComputeBackend.GPU_METAL ? " with step=1" : "")
                        + "; wider dtype/rank coverage is rejected by planner semantics");
        if (backend == ComputeBackend.GPU_METAL) {
            addSupported(entries, backend, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                    "Metal MPSGraph layout path maps broadcast EXPAND and single-index SELECT into native accelerator DAG shape ops",
                    Operation.OpType.EXPAND,
                    Operation.OpType.SELECT);
            addSupported(entries, backend, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                    "Metal MPSGraph layout path supports dense FLOAT32/BFLOAT16 static slice, concat, constant pad, and tile descriptors",
                    Operation.OpType.SLICE,
                    Operation.OpType.CONCAT,
                    Operation.OpType.PAD,
                    Operation.OpType.TILE);
            add(entries, backend, Operation.OpType.SLICE_BACKWARD, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal SLICE_BACKWARD supports static dense FLOAT32/BFLOAT16 step=1 backward layout writes by lowering to zero-fill pad with explicit before/after attributes");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.CAST, GpuLoweringOperationFamily.DTYPE_CONVERSION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal CAST supports scoped identity casts plus FLOAT32 <-> BFLOAT16 conversion through explicit cast-pair policy; FLOAT64, runtime INT64, and general BOOL/INT32 numeric casts remain unsupported");
        } else {
            add(entries, backend, Operation.OpType.CAST, GpuLoweringOperationFamily.DTYPE_CONVERSION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA CAST remains unsupported until a CUDA cast-pair policy and native/lowered dtype conversion path exist");
        }
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
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.REDUCE_PROD, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "native Metal MPSGraph product reduction path for dense FLOAT32/BFLOAT16 inputs; target=reduction_chain_small");
            add(entries, backend, Operation.OpType.ARGMAX, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "native Metal MPSGraph argmax path for dense FLOAT32/BFLOAT16 inputs with public INT64 index outputs");
            add(entries, backend, Operation.OpType.CUMSUM, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "native Metal MPSGraph cumulative sum path for dense FLOAT32/BFLOAT16 inputs with static axis/exclusive/reverse metadata");
        } else {
            add(entries, backend, Operation.OpType.REDUCE_PROD, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA product reduction remains unsupported until native accelerator DAG execution maps REDUCE_PROD");
            add(entries, backend, Operation.OpType.ARGMAX, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA argmax remains unsupported until native index-output reduction execution maps ARGMAX");
            add(entries, backend, Operation.OpType.CUMSUM, GpuLoweringOperationFamily.REDUCTION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA cumulative sum remains unsupported until native scan execution maps CUMSUM");
        }
        add(entries, backend, Operation.OpType.LAYER_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "lowered as repeated MEAN plus elementwise normalization DAG with epsilon scalar; target=layer_norm_small");
        add(entries, backend, Operation.OpType.RMS_NORM, GpuLoweringOperationFamily.NORMALIZATION,
                GpuLoweringCoverageStatus.SUPPORTED,
                GpuLoweringUnsupportedReason.SUPPORTED,
                "lowered as repeated MEAN plus elementwise RMS normalization DAG with epsilon scalar; target=rms_norm_small");
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.NLL_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "dense FLOAT32/BFLOAT16 rank 1..4 NLL loss lowers to target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path");
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "dense FLOAT32/BFLOAT16 rank 1..4 cross entropy lowers to SOFTMAX, LOG, target multiply, all-axis SUM, and scalar mean scaling; target=loss_dense_small target=transformer_block_hot_path");
        } else {
            add(entries, backend, Operation.OpType.NLL_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                    "loss-adjacent operation remains explicit CPU fallback because no native loss primitive exists; reduction semantics must match CPU; target=transformer_block_hot_path");
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                    "dense-target cross-entropy remains explicit CPU fallback because no native loss primitive exists; reduction semantics must match CPU; target=transformer_block_hot_path");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal index-target cross entropy lowers to MPSGraph softmax/log/gather/reduction with dense FLOAT32/BFLOAT16 logits, dense INT32 in-bounds targets, ignore-index masking, and NONE/SUM/MEAN reductions; target=transformer_block_hot_path");
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal index-target cross entropy gradient lowers to MPSGraph softmax plus scatterAlongAxis sample-scale subtraction with dense FLOAT32/BFLOAT16 logits/sampleScale and dense INT32 in-bounds targets; target=transformer_block_hot_path");
        } else {
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "index-target loss uses INT32 targets plus bounds, ignore-index, and reduction-denominator semantics outside the current accelerator DAG contract; target=transformer_block_hot_path");
            add(entries, backend, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, GpuLoweringOperationFamily.LOSS_ADJACENT,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "index-target loss gradient uses INT32 targets and scatter-like per-class gradient semantics outside the current accelerator DAG contract; target=transformer_block_hot_path");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "direct FLOAT32/BFLOAT16 rank-3/4 native MPSGraph primitive SDPA DAG supports unmasked, dense external BOOL masked, causal, and external+causal effective mask modes; target=transformer_block_hot_path target=masked_sdpa_small");
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "attention weights publication lowers from the producer SDPA descriptor to a native Metal softmax(QK^T * scale + mask) DAG without CPU cache materialization; target=masked_sdpa_small");
        } else {
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.FALLBACK,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA direct forward SDPA native/lowered path is not implemented; validated unmasked, dense external BOOL masked, causal, and external+causal mask modes reject with visible capability blockers; target=transformer_block_hot_path target=masked_sdpa_small");
            add(entries, backend, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, GpuLoweringOperationFamily.ATTENTION,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA attention weights publication is unsupported until CUDA can lower or recompute softmax(QK^T * scale + mask) without CPU cache materialization");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            addSupported(entries, backend, GpuLoweringOperationFamily.ATTENTION,
                    "Metal backward SDPA is represented by the shared accelerator DAG for unmasked, dense external BOOL masked, causal, and external+causal 3/4-input SDPA producers",
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
                    "forward gather lowers to Metal gatherAlongAxis with expanded INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small");
            add(entries, backend, Operation.OpType.GATHER_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "ONNX-style gatherAxis lowers to Metal gatherAlongAxis with broadcast INT32 indices for dense FLOAT32/BFLOAT16 value input and static in-bounds 1-D index input; target=gather_take_small");
            add(entries, backend, Operation.OpType.GATHER_ND, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal gather-nd lowers to MPSGraph gatherNDWithUpdatesTensor for dense FLOAT32/BFLOAT16 values, dense static non-negative in-bounds INT32 tuple indices, slice suffix outputs, and validated batch_dims");
            add(entries, backend, Operation.OpType.GATHER_ND_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "gather-nd gradient remains CPU-owned until tuple-index duplicate accumulation and batch_dims semantics are proven for Metal");
        } else {
            add(entries, backend, Operation.OpType.GATHER, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA forward gather native/lowered path is not implemented yet");
            add(entries, backend, Operation.OpType.GATHER_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA ONNX-style gatherAxis native/lowered path is not implemented yet");
            add(entries, backend, Operation.OpType.GATHER_ND, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "gather-nd remains CPU-owned until tuple-index read, slice suffix addressing, and static bounds checks are proven for CUDA");
            add(entries, backend, Operation.OpType.GATHER_ND_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "gather-nd gradient remains CPU-owned until tuple-index duplicate accumulation and batch_dims semantics are proven for CUDA");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.GATHER_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal gather gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small");
            add(entries, backend, Operation.OpType.GATHER_AXIS_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal gatherAxis gradient lowers to MPSGraph scatterAlongAxis add with broadcast INT32 indices, dense FLOAT32/BFLOAT16 gradients, and duplicate accumulation on device; target=scatter_index_gradient_small");
        } else {
            add(entries, backend, Operation.OpType.GATHER_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    "gather gradient remains CPU-owned until Phase 36 proves duplicate-index accumulation parity, static bounds checks, and gradient scatter residency");
            add(entries, backend, Operation.OpType.GATHER_AXIS_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    "gatherAxis gradient remains CPU-owned until CUDA proves duplicate-index accumulation parity, static bounds checks, and gradient scatter residency");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "take-along-axis lowers to Metal gatherAlongAxis with INT32 indices; scoped to dense FLOAT32/BFLOAT16 value input and static in-bounds INT32 index input; target=gather_take_small");
        } else {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    "CUDA take-along-axis native/lowered path is not implemented yet");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal take-along-axis gradient lowers to MPSGraph scatterAlongAxis add with dense FLOAT32/BFLOAT16 gradients, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small");
            add(entries, backend, Operation.OpType.SCATTER_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal scatter-add lowers to MPSGraph scatterAlongAxis add onto the base tensor with dense FLOAT32/BFLOAT16 values, static in-bounds INT32 indices, and duplicate accumulation on device; target=scatter_index_gradient_small");
            add(entries, backend, Operation.OpType.SCATTER_AXIS_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "scatter-axis-add remains CPU-owned until rank-changing gather inverse writes and duplicate-index accumulation are proven for Metal");
            add(entries, backend, Operation.OpType.SCATTER_ELEMENTS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal scatter-elements lowers to MPSGraph scatterAlongAxis with dense FLOAT32/BFLOAT16 data/updates, static non-negative in-bounds INT32 indices, and NONE/ADD/MUL/MAX/MIN reduction parity");
            add(entries, backend, Operation.OpType.SCATTER_ND, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal scatter-nd lowers to MPSGraph scatterNDWithDataTensor with dense FLOAT32/BFLOAT16 data/updates, static non-negative in-bounds INT32 tuple indices, slice suffix updates, validated batch_dims, and NONE/ADD/MUL/MAX/MIN reduction parity");
        } else {
            add(entries, backend, Operation.OpType.TAKE_ALONG_AXIS_GRAD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    "take-along-axis gradient remains CPU-owned until Phase 36 proves duplicate-index accumulation parity, rank-preserving static bounds checks, and gradient scatter residency");
            add(entries, backend, Operation.OpType.SCATTER_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_DUPLICATE_INDEX,
                    "scatter-add remains CPU-owned until Phase 36 proves duplicate-index accumulation order/tolerance, static bounds checks, and native write-add semantics");
            add(entries, backend, Operation.OpType.SCATTER_AXIS_ADD, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "scatter-axis-add remains CPU-owned until rank-changing gather inverse writes and duplicate-index accumulation are proven for CUDA");
            add(entries, backend, Operation.OpType.SCATTER_ELEMENTS, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "scatter-elements remains CPU-owned until rank-preserving write, reduction, and duplicate-index policies are proven for CUDA");
            add(entries, backend, Operation.OpType.SCATTER_ND, GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                    "scatter-nd remains CPU-owned until tuple-index write, slice update, reduction, and duplicate-index policies are proven for CUDA");
        }
        if (backend == ComputeBackend.GPU_CUDA) {
            add(entries, backend, Operation.OpType.SLICE_BACKWARD, GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                    "slice backward remains CPU-owned until stepped slice sparse writes are proven for CUDA lowering");
        }
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
        add(entries, backend, Operation.OpType.CONST_SCALAR, GpuLoweringOperationFamily.ELEMENTWISE_CHAIN,
                GpuLoweringCoverageStatus.UNSUPPORTED,
                GpuLoweringUnsupportedReason.UNSUPPORTED_OPERATION,
                "CONST_SCALAR is an internal CPU fused-plan scalar node, not a public standalone GPU compute op; GPU DAG lowering carries scalar values as primitive metadata");
    }

    private static void addConvPoolRows(List<GpuLoweringCoverageEntry> entries, ComputeBackend backend) {
        String backendLabel = backend.name();
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.CONV2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal direct FLOAT32/BFLOAT16 dense NCHW/OIHW Conv2D forward lowers to MPSGraph convolution2D; scoped to groups=1, dilation=1, stride/padding, and optional bias; target=conv2d_resnet_3x3");
        } else {
            add(entries, backend, Operation.OpType.CONV2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    backendLabel + " conv2d NCHW rank-4 native/lowered path is not implemented; stride/padding/dilation/groups must be proven before support; target=conv2d_resnet_3x3");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.MAX_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal direct FLOAT32/BFLOAT16 dense NCHW MAX_POOL2D forward lowers to MPSGraph maxPooling2D; scoped to compact kernel/stride/padding metadata and CPU tie behavior parity; target=max_pool2d_small");
        } else {
            add(entries, backend, Operation.OpType.MAX_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    backendLabel + " max-pool native/lowered path is not implemented; kernel/stride/padding and tie behavior must match CPU; target=max_pool2d_small");
        }
        if (backend == ComputeBackend.GPU_METAL) {
            add(entries, backend, Operation.OpType.AVG_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.SUPPORTED,
                    GpuLoweringUnsupportedReason.SUPPORTED,
                    "Metal direct FLOAT32/BFLOAT16 dense NCHW AVG_POOL2D forward lowers to MPSGraph avgPooling2D; scoped to countIncludePad=false and compact kernel/stride/padding metadata; target=avg_pool2d_small");
        } else {
            add(entries, backend, Operation.OpType.AVG_POOL2D, GpuLoweringOperationFamily.CONV_POOL,
                    GpuLoweringCoverageStatus.UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CAPABILITY_MISSING,
                    backendLabel + " avg-pool native/lowered path is not implemented; countIncludePad divisor semantics must match CPU; target=avg_pool2d_small");
        }
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
        return switch (opType) {
            case MATMUL -> GpuLoweringOperationFamily.MATMUL_LINEAR;
            case GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT ->
                    GpuLoweringOperationFamily.COMPARE_BOOL;
            case ADD, SUB, MUL, DIV, MIN, MAX, WHERE, NEG, INV, LOG, EXP, FAST_EXP, ERF,
                 TANH, FAST_TANH, POW, POW_TENSOR, SQRT, ABS, FLOOR, CEIL, SIGN, MUL_SCALAR,
                 RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, FUSED ->
                    GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, SLICE, CONCAT, PAD, TILE, UNFOLD_AXIS,
                 UNFOLD2D, FOLD2D, PERMUTE, EXPAND_DIMS, SQUEEZE ->
                    GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT;
            case REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY, SUM, MEAN ->
                    GpuLoweringOperationFamily.REDUCTION;
            default -> specialFamilyFor(opType);
        };
    }

    private static GpuLoweringOperationFamily specialFamilyFor(Operation.OpType opType) {
        return switch (opType) {
            case SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD -> GpuLoweringOperationFamily.SOFTMAX_LIKE;
            case LAYER_NORM, RMS_NORM -> GpuLoweringOperationFamily.NORMALIZATION;
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD -> GpuLoweringOperationFamily.LOSS_ADJACENT;
            case SCALED_DOT_PRODUCT_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS -> GpuLoweringOperationFamily.ATTENTION;
            case CONV2D, MAX_POOL2D, AVG_POOL2D -> GpuLoweringOperationFamily.CONV_POOL;
            case GATHER, GATHER_GRAD, GATHER_AXIS, GATHER_AXIS_GRAD, GATHER_ND, GATHER_ND_GRAD,
                 TAKE_ALONG_AXIS, TAKE_ALONG_AXIS_GRAD, SCATTER_ADD, SCATTER_AXIS_ADD,
                 SCATTER_ELEMENTS, SCATTER_ND -> GpuLoweringOperationFamily.INDEX_SCATTER_GATHER;
            case SLICE_BACKWARD -> GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT;
            case REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, MIN_GRAD, MAX_GRAD -> GpuLoweringOperationFamily.BACKWARD_ADJACENT;
            default -> GpuLoweringOperationFamily.ELEMENTWISE_CHAIN;
        };
    }
}
