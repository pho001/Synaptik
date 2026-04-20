package backend.kernels.cpu.plan;

import backend.kernels.cpu.CpuExecutionBackend;
import backend.kernels.cpu.elementwise.plan.ElementwiseDispatchPlanner;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.fused.plan.FusedDispatchPlanner;
import backend.kernels.cpu.fused.plan.PreparedFusedDispatch;
import backend.kernels.cpu.linalg.attention.plan.ResolvedAttentionHints;
import backend.kernels.cpu.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.kernels.cpu.linalg.attention.plan.ScaledDotProductAttentionPlanner;
import backend.kernels.cpu.linalg.matmul.plan.MatMulPlanner;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.nn.conv2d.plan.Conv2dPlanner;
import backend.kernels.cpu.nn.conv2d.plan.ResolvedConv2dHints;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.reduction.plan.ReductionPlanner;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.CpuMatMulMicroKernel;
import config.backend.SumAccuracyMode;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import operations.fused.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public final class CpuExecutionPlanner {
    public static final int DEFAULT_MATMUL_TILE_M = CpuPlanningPolicy.DEFAULT_MATMUL_TILE_M;
    public static final int DEFAULT_MATMUL_TILE_N = CpuPlanningPolicy.DEFAULT_MATMUL_TILE_N;
    public static final int DEFAULT_MATMUL_TILE_K = CpuPlanningPolicy.DEFAULT_MATMUL_TILE_K;

    private final CpuPlanningPolicy policy;
    private final ElementwiseDispatchPlanner elementwiseDispatchPlanner;
    private final FusedDispatchPlanner fusedDispatchPlanner;
    private final ReductionPlanner reductionPlanner;
    private final MatMulPlanner matMulPlanner;
    private final Conv2dPlanner conv2dPlanner;
    private final ScaledDotProductAttentionPlanner attentionPlanner;
    private final CpuComputeContractResolver computeContractResolver;

    public CpuExecutionPlanner(
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int matMulParallelMinSize,
            int contiguousMaterializeThreshold,
            int cheapF64MaterializeThreshold,
            int cheapF32MaterializeThreshold,
            int cheapBF16MaterializeThreshold,
            int whereMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel
    ) {
        this(
                matMulTileM,
                matMulTileN,
                matMulTileK,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize,
                reductionVectorMinSize,
                attentionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                attentionParallelMinSize,
                matMulParallelMinSize,
                contiguousMaterializeThreshold,
                cheapF64MaterializeThreshold,
                cheapF32MaterializeThreshold,
                cheapBF16MaterializeThreshold,
                whereMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                attentionMatMulPolicy,
                matMulMicroKernel,
                attentionMatMulMicroKernel,
                matMulTileM,
                matMulTileN,
                matMulTileK
        );
    }

    public CpuExecutionPlanner(
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int matMulParallelMinSize,
            int contiguousMaterializeThreshold,
            int cheapF64MaterializeThreshold,
            int cheapF32MaterializeThreshold,
            int cheapBF16MaterializeThreshold,
            int whereMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel,
            int attentionMatMulTileM,
            int attentionMatMulTileN,
            int attentionMatMulTileK
    ) {
        this.policy = new CpuPlanningPolicy(
                matMulTileM,
                matMulTileN,
                matMulTileK,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize,
                reductionVectorMinSize,
                attentionVectorMinSize,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                reductionParallelMinSize,
                attentionParallelMinSize,
                matMulParallelMinSize,
                contiguousMaterializeThreshold,
                cheapF64MaterializeThreshold,
                cheapF32MaterializeThreshold,
                cheapBF16MaterializeThreshold,
                whereMaterializeThreshold,
                lowCostTargetChunksPerWorker,
                mediumCostTargetChunksPerWorker,
                highCostTargetChunksPerWorker,
                minScalarChunkSize,
                minVectorChunkSize,
                minReductionChunkSize,
                commonPoolLowCostMaxWorkPerWorker,
                fusedCheapContiguousAsmVectorWidth,
                fusedCheapStridedAsmVectorWidth,
                fusedNonCheapContiguousAsmVectorWidth,
                fusedNonCheapStridedAsmVectorWidth,
                sumAccuracyMode,
                attentionMatMulPolicy,
                matMulMicroKernel,
                attentionMatMulMicroKernel,
                attentionMatMulTileM,
                attentionMatMulTileN,
                attentionMatMulTileK
        );
        this.elementwiseDispatchPlanner = new ElementwiseDispatchPlanner(policy);
        this.fusedDispatchPlanner = new FusedDispatchPlanner(policy);
        this.reductionPlanner = new ReductionPlanner(policy);
        this.matMulPlanner = new MatMulPlanner(policy);
        this.conv2dPlanner = new Conv2dPlanner();
        this.attentionPlanner = new ScaledDotProductAttentionPlanner(reductionPlanner, matMulPlanner);
        this.computeContractResolver = new CpuComputeContractResolver();
    }

    public static CpuExecutionPlanner from(CpuKernelConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new CpuExecutionPlanner(
                config.matMulTileM(),
                config.matMulTileN(),
                config.matMulTileK(),
                config.cheapVectorMinSize(),
                config.transcendentalVectorMinSize(),
                config.fusedCheapVectorMinSize(),
                config.fusedTranscendentalVectorMinSize(),
                config.reductionVectorMinSize(),
                config.attentionVectorMinSize(),
                config.cheapParallelMinSize(),
                config.transcendentalParallelMinSize(),
                config.fusedCheapParallelMinSize(),
                config.fusedTranscendentalParallelMinSize(),
                config.reductionParallelMinSize(),
                config.attentionParallelMinSize(),
                config.matMulParallelMinSize(),
                config.contiguousMaterializeThreshold(),
                config.cheapF64MaterializeThreshold(),
                config.cheapF32MaterializeThreshold(),
                config.cheapBF16MaterializeThreshold(),
                config.whereMaterializeThreshold(),
                config.lowCostTargetChunksPerWorker(),
                config.mediumCostTargetChunksPerWorker(),
                config.highCostTargetChunksPerWorker(),
                config.minScalarChunkSize(),
                config.minVectorChunkSize(),
                config.minReductionChunkSize(),
                config.commonPoolLowCostMaxWorkPerWorker(),
                config.fusedCheapContiguousAsmVectorWidth(),
                config.fusedCheapStridedAsmVectorWidth(),
                config.fusedNonCheapContiguousAsmVectorWidth(),
                config.fusedNonCheapStridedAsmVectorWidth(),
                config.sumAccuracyMode(),
                config.attentionMatMulPolicy(),
                config.matMulMicroKernel(),
                config.attentionMatMulMicroKernel(),
                config.attentionMatMulTileM(),
                config.attentionMatMulTileN(),
                config.attentionMatMulTileK()
        );
    }

    public int contiguousMaterializeThreshold() {
        return policy.contiguousMaterializeThreshold();
    }

    public SumAccuracyMode sumAccuracyMode() {
        return policy.sumAccuracyMode();
    }

    public int matMulTileM() {
        return policy.matMulTileM();
    }

    public int matMulTileN() {
        return policy.matMulTileN();
    }

    public int matMulTileK() {
        return policy.matMulTileK();
    }

    public int matMulParallelMinSize() {
        return policy.matMulParallelMinSize();
    }

    public int plannedWorkers() {
        return policy.plannedWorkers();
    }

    public boolean shouldMaterializeNonContiguous(int logicalSize) {
        return policy.shouldMaterializeNonContiguous(logicalSize);
    }

    public boolean shouldMaterializeCheapStridedElementwise(Operation op, DataType targetType, int logicalSize) {
        return policy.shouldMaterializeCheapStridedElementwise(op, targetType, logicalSize);
    }

    public int preferredVectorWidth(ResolvedCpuComputeContract contract) {
        return policy.preferredVectorWidth(contract);
    }

    public int resolvedFusedAsmVectorWidth(ResolvedCpuComputeContract contract, FusedOperation fused) {
        return policy.resolvedFusedAsmVectorWidth(contract, fused);
    }

    public ResolvedDispatchHints resolveDispatchHints(Operation op, Tensor node, ResolvedCpuComputeContract contract) {
        return elementwiseDispatchPlanner.resolve(op, node, contract);
    }

    public PreparedFusedDispatch resolveFusedDispatch(FusedOperation fused, Tensor node, ResolvedCpuComputeContract contract) {
        return fusedDispatchPlanner.resolve(fused, node, contract);
    }

    public ResolvedReductionHints resolveReductionHints(int logicalSize, ResolvedCpuComputeContract contract) {
        return reductionPlanner.resolve(logicalSize, contract);
    }

    public ResolvedAttentionHints resolveAttentionHints(
            int independentTasks,
            int workPerTask,
            int vectorSpan,
            ResolvedCpuComputeContract contract
    ) {
        return reductionPlanner.resolveAttentionHints(independentTasks, workPerTask, vectorSpan, contract);
    }

    public ResolvedCpuComputeContract resolveComputeContract(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            BlasConfig blasConfig,
            ResolvedMatMulHints matMulHints,
            ResolvedConv2dHints conv2dHints
    ) {
        return computeContractResolver.resolve(op, node, matMulHints, conv2dHints);
    }

    public ResolvedMatMulHints resolveMatMulHints(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig) {
        return matMulPlanner.resolve(a, b, out, blasConfig);
    }

    public ResolvedMatMulHints resolveJavaMatMulHints(int[] aShape, int[] bShape, int[] outShape, DataType outDataType) {
        return matMulPlanner.resolveJava(aShape, bShape, outShape, outDataType);
    }

    public ResolvedMatMulHints resolveAttentionMatMulHints(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig) {
        return matMulPlanner.resolveAttention(a, b, out, blasConfig);
    }

    public ResolvedMatMulHints resolveAttentionMatMulHints(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            BlasConfig blasConfig
    ) {
        return matMulPlanner.resolveAttention(aShape, aContiguous, bShape, bContiguous, outShape, outDataType, outContiguous, blasConfig);
    }

    public ResolvedMatMulHints resolveAttentionJavaMatMulHints(int[] aShape, int[] bShape, int[] outShape, DataType outDataType) {
        return matMulPlanner.resolveAttentionJava(aShape, bShape, outShape, outDataType);
    }

    public ResolvedConv2dHints resolveConv2dHints(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            Conv2dConfig conv2dConfig
    ) {
        return conv2dPlanner.resolve(op, inputs, node, conv2dConfig);
    }

    public ResolvedScaledDotProductAttentionPlan resolveScaledDotProductAttentionPlan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            BlasConfig blasConfig
    ) {
        return attentionPlanner.resolve(op, inputs, node, blasConfig);
    }

    public int computeChunkSize(int totalLength, int alignment, int targetChunksPerWorker, int minChunkSize) {
        return policy.computeChunkSize(totalLength, alignment, targetChunksPerWorker, minChunkSize);
    }

    public int fusedDirectVectorMinSize(FusedOperation operation) {
        return policy.fusedDirectVectorMinSize(operation);
    }
}
