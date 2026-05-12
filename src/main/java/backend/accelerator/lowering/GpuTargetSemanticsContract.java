package backend.accelerator.lowering;

import operations.Operation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Semantics contract for v1.4 GPU target operation families.
 *
 * <p>These contracts describe the semantic facts a backend implementation must preserve before
 * planner admission. They are deliberately backend-neutral; Metal/CUDA capability differences are
 * still handled by legality adapters and prepared executables.</p>
 */
public record GpuTargetSemanticsContract(
        Operation.OpType opType,
        GpuLoweringOperationFamily family,
        String dtypeContract,
        String rankContract,
        String layoutContract,
        String shapeContract,
        String parameterContract,
        String numericalContract,
        boolean plannerAdmissionBlocked,
        String blockerReason
) {
    private static final List<GpuTargetSemanticsContract> CONTRACTS = buildContracts();
    private static final Map<Operation.OpType, GpuTargetSemanticsContract> BY_OP = CONTRACTS.stream()
            .collect(Collectors.toUnmodifiableMap(GpuTargetSemanticsContract::opType, Function.identity()));

    public GpuTargetSemanticsContract {
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (family == null) {
            throw new IllegalArgumentException("family cannot be null");
        }
        dtypeContract = normalize(dtypeContract);
        rankContract = normalize(rankContract);
        layoutContract = normalize(layoutContract);
        shapeContract = normalize(shapeContract);
        parameterContract = normalize(parameterContract);
        numericalContract = normalize(numericalContract);
        blockerReason = normalize(blockerReason);
    }

    /**
     * Returns all v1.4 target semantics contracts.
     */
    public static List<GpuTargetSemanticsContract> contracts() {
        return CONTRACTS;
    }

    /**
     * Returns a contract for one operation type, or {@code null} for non-target operations.
     */
    public static GpuTargetSemanticsContract forOp(Operation.OpType opType) {
        return BY_OP.get(opType);
    }

    private static List<GpuTargetSemanticsContract> buildContracts() {
        ArrayList<GpuTargetSemanticsContract> out = new ArrayList<>();
        addReduction(out, Operation.OpType.SUM, "sum reduction accumulates along descriptor axis");
        addReduction(out, Operation.OpType.MEAN, "mean reduction accumulates then divides by reduced extent");
        addReduction(out, Operation.OpType.REDUCE_MIN, "minimum reduction preserves CPU tie/NaN behavior within tolerance policy");
        addReduction(out, Operation.OpType.REDUCE_MAX, "maximum reduction preserves CPU tie/NaN behavior within tolerance policy");
        out.add(new GpuTargetSemanticsContract(
                Operation.OpType.LAYER_NORM,
                GpuLoweringOperationFamily.NORMALIZATION,
                "floating compute dtype only until backend-specific widening is proven",
                "input rank >= 1; normalized trailing dimensions must match gamma/beta contracts",
                "contiguous and legal view layouts only; unsupported strided/storage-offset cases reject explicitly",
                "output shape equals input shape; gamma and beta broadcast across normalized axes",
                "epsilon is applied inside variance normalization exactly once",
                "CPU parity tolerance must cover variance accumulation order without hiding shape/dtype errors",
                false,
                ""
        ));
        out.add(new GpuTargetSemanticsContract(
                Operation.OpType.RMS_NORM,
                GpuLoweringOperationFamily.NORMALIZATION,
                "floating compute dtype only until backend-specific widening is proven",
                "input rank >= 1; normalized trailing dimensions must match gamma contract",
                "contiguous and legal view layouts only; unsupported strided/storage-offset cases reject explicitly",
                "output shape equals input shape; gamma broadcasts across normalized axes",
                "epsilon is applied inside root-mean-square normalization exactly once",
                "CPU parity tolerance must cover reduction order without hiding shape/dtype errors",
                false,
                ""
        ));
        out.add(new GpuTargetSemanticsContract(
                Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
                GpuLoweringOperationFamily.ATTENTION,
                "GPU admission is dtype-matched FLOAT32/BFLOAT16 query/key/value; dense effective BOOL masks are legal only where backend mask semantics are verified",
                "Phase 25 GPU admission is rank 3 or 4 attention tensors with softmax over the key axis",
                "supported dense/view layouts only; unsupported layout families reject before native admission",
                "query/key head dimensions must match, key/value sequence dimensions must match, output follows broadcast batch/head, query length, and value channel dimensions",
                "scale must match AttentionOptions resolved scale; unmasked, causal, and external BOOL mask paths are separate verified cases",
                "CPU parity tolerance must cover softmax stability, scale application order, mask fill behavior, and backend SDPA math differences",
                false,
                "Metal admits scale-verified unmasked, dense external BOOL masked, causal, and external+causal FLOAT32/BFLOAT16 rank-3/4 forward SDPA after parity evidence; unsupported backend variants remain capability-gated"
        ));
        addDenseLoss(out, Operation.OpType.NLL_LOSS, "dense target tensor multiplies log-probabilities along the class axis; public dense loss is mean-reduced by non-class sample count");
        addDenseLoss(out, Operation.OpType.CROSS_ENTROPY_LOSS, "dense target tensor multiplies log-softmax logits along the class axis; public dense loss is mean-reduced by non-class sample count");
        addIndexTargetLoss(out, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES, "INT32 target indices, bounds checks, ignore-index, and reduction mode must match CPU");
        addIndexTargetLoss(out, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, "INT32 target indices and gradient scatter semantics must match CPU");
        addConvPool(out, Operation.OpType.CONV2D, "conv2d NCHW-style shape, stride, padding, dilation, groups, and dtype contract");
        addConvPool(out, Operation.OpType.CONV2D_GEMM, "lowered conv2d GEMM shape contract must preserve CPU conv semantics");
        addConvPool(out, Operation.OpType.CONV2D_BACKWARD_INPUT, "conv2d input-gradient shape, stride, padding, dilation, and groups must match CPU");
        addConvPool(out, Operation.OpType.CONV2D_BACKWARD_WEIGHT, "conv2d weight-gradient accumulation, stride, padding, dilation, and groups must match CPU");
        addConvPool(out, Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM, "lowered conv2d input-gradient GEMM contract must preserve CPU gradient semantics");
        addConvPool(out, Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM, "lowered conv2d weight-gradient GEMM contract must preserve CPU accumulation semantics");
        addConvPool(out, Operation.OpType.MAX_POOL2D, "max-pool kernel/stride/padding shape and tie semantics must match CPU");
        addConvPool(out, Operation.OpType.MAX_POOL2D_BACKWARD_INPUT, "max-pool backward routes gradient to the first maximal element in CPU scan order");
        addConvPool(out, Operation.OpType.AVG_POOL2D, "avg-pool kernel/stride/padding and divisor semantics must match CPU");
        addConvPool(out, Operation.OpType.AVG_POOL2D_BACKWARD_INPUT, "avg-pool backward divisor and countIncludePad semantics must match CPU");
        addIndex(out, Operation.OpType.GATHER, "INT32 indices; bounds and axis behavior must match CPU gather");
        addIndex(out, Operation.OpType.GATHER_GRAD, "duplicate-index accumulation must match CPU gather gradient");
        addIndex(out, Operation.OpType.TAKE_ALONG_AXIS, "INT32 indices; axis-aligned take semantics must match CPU");
        addIndex(out, Operation.OpType.TAKE_ALONG_AXIS_GRAD, "duplicate-index accumulation must match CPU take-along-axis gradient");
        addIndex(out, Operation.OpType.SCATTER_ADD, "duplicate-index accumulation order/tolerance must match CPU scatter-add");
        addIndex(out, Operation.OpType.SCATTER_ELEMENTS, "rank-preserving write, reduction, and duplicate-index policy must match CPU scatter-elements");
        addCompare(out, Operation.OpType.GT);
        addCompare(out, Operation.OpType.GE);
        addCompare(out, Operation.OpType.LT);
        addCompare(out, Operation.OpType.LE);
        addCompare(out, Operation.OpType.EQ);
        addCompare(out, Operation.OpType.NE);
        addBoolLogical(out, Operation.OpType.LOGICAL_AND, "BOOL logical AND with broadcast semantics");
        addBoolLogical(out, Operation.OpType.LOGICAL_OR, "BOOL logical OR with broadcast semantics");
        addBoolLogical(out, Operation.OpType.LOGICAL_NOT, "BOOL logical NOT preserves input shape");
        addBoolReduction(out, Operation.OpType.REDUCE_ALL, "all reduction returns true only when every selected BOOL input is true");
        addBoolReduction(out, Operation.OpType.REDUCE_ANY, "any reduction returns true when at least one selected BOOL input is true");
        return List.copyOf(out);
    }

    private static void addReduction(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String numerical) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.REDUCTION,
                "floating compute dtype first; unsupported non-floating reductions reject explicitly",
                "rank 1-4 until native ABI rank expansion",
                "contiguous and legal view layouts only; unsupported strided/storage-offset cases reject explicitly",
                "axis is removed unless keepDims is true; keepDims retains extent 1 on the reduced axis",
                "descriptor axis and keepDims are part of the lowered primitive contract",
                numerical,
                false,
                ""
        ));
    }

    private static void addDenseLoss(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.LOSS_ADJACENT,
                "dense dtype-matched FLOAT32/BFLOAT16 scores/log-probabilities and dense targets for the Metal candidate scope; other dtypes reject explicitly",
                "rank 1-4 with a descriptor class axis inside input rank",
                "dense zero-offset input and target layouts only until GPU-side loss layout repair is proven",
                "dense target shape equals input shape; output shape is [1] for the current public mean-reduced dense loss contract",
                parameter,
                "CPU parity must cover class-axis reduction, sample-count denominator, distribution targets, and numerically stable log-softmax behavior for cross entropy",
                false,
                "Metal admits the scoped dense FLOAT32/BFLOAT16 loss subset after lowering/parity evidence; other backends and unsupported variants remain capability-gated"
        ));
    }

    private static void addIndexTargetLoss(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.LOSS_ADJACENT,
                "dtype-matched FLOAT32/BFLOAT16 logits/probabilities plus INT32 index targets for admitted Metal loss candidates",
                "rank follows the CPU loss descriptor contract",
                "supported dense/view layouts only",
                "scalar or reduced output shape follows loss reduction mode",
                parameter,
                "CPU parity must cover reduction mode, ignore-index denominator behavior, bounds checks, and numerically stable log/softmax behavior",
                true,
                "UNSUPPORTED_INDEX_SEMANTICS until INT32 targets, ignore-index, bounds, class weights, denominator rules, and Phase 36 scatter/index-gradient blockers are proven"
        ));
    }

    private static void addConvPool(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.CONV_POOL,
                "floating input/filter/output dtype first",
                "rank 4 image tensors until contract is expanded",
                "explicit supported layout only; unsupported layout rejects before backend selection",
                "output shape must match CPU shape inference for stride, padding, dilation, and kernel",
                parameter,
                "CPU parity must cover boundary/padding behavior",
                false,
                ""
        ));
    }

    private static void addIndex(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        boolean writeOrGradient = opType == Operation.OpType.GATHER_GRAD
                || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD
                || opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.SCATTER_ELEMENTS;
        String blockerReason = "";
        if (writeOrGradient) {
            blockerReason = opType == Operation.OpType.SCATTER_ELEMENTS
                    ? "UNSUPPORTED_INDEX_SEMANTICS until backend proves rank-preserving write reductions, duplicate policy, and static bounds checks"
                    : "UNSUPPORTED_DUPLICATE_INDEX until backend proves CPU-compatible duplicate-index accumulation and static bounds checks";
        }
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.INDEX_SCATTER_GATHER,
                "INT32 index tensors plus dtype-matched FLOAT32/BFLOAT16 value tensors; native compute support is operation-specific",
                "rank follows operation descriptor and selected axis",
                "supported dense/view layouts only; Phase 36 native write/gradient candidates require dense zero-offset inputs and outputs",
                "output shape must match CPU indexing shape inference; gather-grad and take-along-axis-grad output the original input shape, scatter-add outputs the base shape",
                parameter,
                "CPU parity must cover duplicate indices, logical-index accumulation order/tolerance, repeated writes to one destination, and bounds behavior",
                writeOrGradient,
                blockerReason
        ));
    }

    private static void addCompare(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.COMPARE_BOOL,
                "floating or comparable input dtype; BOOL output storage is one byte per element",
                "rank follows broadcasted input shape",
                "supported dense/view layouts only",
                "output shape is the broadcasted input shape",
                "BOOL output may feed WHERE or mask consumers without CPU materialization when backend supports it",
                "CPU parity must match comparison predicate exactly",
                false,
                ""
        ));
    }

    private static void addBoolLogical(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.COMPARE_BOOL,
                "BOOL input and BOOL output storage are one byte per element; current native GPU compute/output support must reject until implemented",
                "rank follows broadcasted input shape for binary logical ops or input shape for unary logical ops",
                "supported dense/view layouts only",
                "output shape follows broadcast or unary input shape",
                parameter,
                "CPU parity must match boolean truth-table semantics exactly",
                false,
                "external BOOL predicate input residency for WHERE is separate from native BOOL-producing GPU compute"
        ));
    }

    private static void addBoolReduction(ArrayList<GpuTargetSemanticsContract> out, Operation.OpType opType, String parameter) {
        out.add(new GpuTargetSemanticsContract(
                opType,
                GpuLoweringOperationFamily.COMPARE_BOOL,
                "BOOL input and BOOL output storage are one byte per element; current native GPU compute/output support must reject until implemented",
                "rank follows BOOL reduction axis; keepDims retains extent 1 on the reduced axis",
                "supported dense/view layouts only",
                "axis is removed unless keepDims is true; scalar all/any uses the CPU reduction contract",
                parameter,
                "CPU parity must match boolean reduction identity and axis behavior exactly",
                false,
                "BOOL reductions are not numeric reductions and require explicit native BOOL output support before planner admission"
        ));
    }

    private static String normalize(String value) {
        return value == null ? "" : value.strip();
    }
}
