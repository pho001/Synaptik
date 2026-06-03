package tuning.candidate;

import config.profile.ExecutionProfile;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

public final class ExecutableProfileFingerprint {
    private ExecutableProfileFingerprint() {
    }

    public static String of(Candidate candidate) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }
        return of(candidate.profile());
    }

    public static String of(ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return sha256(canonicalSpec(profile));
    }

    private static String canonicalSpec(ExecutionProfile profile) {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("profileName=").append(profile.profileName()).append('|');
        sb.append("candidateName=").append(profile.candidateName()).append('|');
        sb.append("dataType=").append(profile.dataType().name()).append('|');
        sb.append("mode=").append(profile.mode().name()).append('|');

        var compile = profile.compile();
        var graph = compile.graphOptimization();
        var backend = compile.backendPlanning();
        var region = compile.regionOptimization();
        var memoryPlanning = compile.memoryPlanning();
        sb.append("graph.ar=").append(graph.algebraicRewrite()).append('|');
        sb.append("graph.cf=").append(graph.constantFolding()).append('|');
        sb.append("graph.cse=").append(graph.commonSubexpressionElimination()).append('|');
        sb.append("graph.dce=").append(graph.deadCodeElimination()).append('|');
        sb.append("graph.lower=").append(graph.optionalLowering()).append('|');
        sb.append("conv2dLowering=").append(graph.rewrite().conv2dLowering().mode().name()).append('|');
        sb.append("cse.strict=").append(graph.cse().strictSafety()).append('|');
        sb.append("backend.discovery=").append(backend.discoveryMode().name()).append('|');
        sb.append("backend.failure=").append(backend.failurePolicy().name()).append('|');
        sb.append("backend.requirementScope=").append(backend.requirementScope().name()).append('|');
        sb.append("backend.targets=");
        backend.targets().stream()
                .map(Enum::name)
                .sorted()
                .forEach(target -> sb.append(target).append(','));
        sb.append('|');
        sb.append("backend.ownershipPlanner=").append(backend.ownershipPlanner().name()).append('|');
        sb.append("backend.search.maxSearchNodes=").append(backend.search().maxSearchNodes()).append('|');
        sb.append("backend.search.maxVisitedCandidates=").append(backend.search().maxVisitedCandidates()).append('|');
        sb.append("backend.cost.transferCostPreset=")
                .append(backend.cost().planningCostProfile().transferCostPreset().name())
                .append('|');
        sb.append("cpuRegion.policy=").append(backend.cpuRegions().policy().name()).append('|');
        sb.append("cpuRegion.maxRegionNodes=").append(backend.cpuRegions().maxRegionNodes()).append('|');
        sb.append("cpuRegion.fanout=").append(backend.cpuRegions().fanoutPolicy().name()).append('|');
        sb.append("cpuRegion.boundary=").append(backend.cpuRegions().boundaryPolicy().name()).append('|');
        sb.append("region.enabled=").append(region.enabled()).append('|');
        sb.append("fuse.maxClusterNodes=").append(region.fuse().maxClusterNodes()).append('|');
        sb.append("fuse.scoreThreshold=").append(fmt(region.fuse().scoreThreshold())).append('|');
        sb.append("fuse.internalEdgeBonus=").append(fmt(region.fuse().internalEdgeBonus())).append('|');
        sb.append("fuse.externalInputPenalty=").append(fmt(region.fuse().externalInputPenalty())).append('|');
        sb.append("fuse.sharedExpensivePenalty=").append(fmt(region.fuse().sharedExpensivePenalty())).append('|');
        sb.append("fuse.nonCheapBonus=").append(fmt(region.fuse().nonCheapBonus())).append('|');
        sb.append("fuse.preserveSharedExpensiveNodes=").append(region.fuse().preserveSharedExpensiveNodes()).append('|');
        sb.append("cpuFusion.mode=").append(region.cpuFusion().mode().name()).append('|');
        sb.append("cpuFusion.maxChainNodes=").append(region.cpuFusion().maxChainNodes()).append('|');
        sb.append("cpuFusion.fanout=").append(region.cpuFusion().fanoutPolicy().name()).append('|');
        sb.append("cpuFusion.layout=").append(region.cpuFusion().layoutPolicy().name()).append('|');
        sb.append("cpuFusion.cheapProducer=").append(region.cpuFusion().cheapProducerPolicy().name()).append('|');
        sb.append("memoryPlanning.enabled=").append(memoryPlanning.enabled()).append('|');
        var memory = memoryPlanning.memory();
        sb.append('|');
        sb.append("memory.separatePools=").append(memory.separateForwardBackwardPools()).append('|');
        sb.append("memory.crossPhase=").append(memory.allowCrossPhaseReuse()).append('|');
        sb.append("memory.allowLarger=").append(memory.allowLargerBufferReuse()).append('|');
        sb.append("memory.minReusable=").append(memory.minReusableBufferSize()).append('|');

        var runtime = profile.runtime();
        var cpu = runtime.kernel().cpu();
        sb.append("cpu.unroll=").append(cpu.loopUnrollFactor()).append('|');
        sb.append("cpu.tileM=").append(cpu.matMulTileM()).append('|');
        sb.append("cpu.tileN=").append(cpu.matMulTileN()).append('|');
        sb.append("cpu.tileK=").append(cpu.matMulTileK()).append('|');
        sb.append("cpu.attentionTileM=").append(cpu.attentionMatMulTileM()).append('|');
        sb.append("cpu.attentionTileN=").append(cpu.attentionMatMulTileN()).append('|');
        sb.append("cpu.attentionTileK=").append(cpu.attentionMatMulTileK()).append('|');
        sb.append("cpu.cheapVectorMin=").append(cpu.cheapVectorMinSize()).append('|');
        sb.append("cpu.transVectorMin=").append(cpu.transcendentalVectorMinSize()).append('|');
        sb.append("cpu.redVectorMin=").append(cpu.reductionVectorMinSize()).append('|');
        sb.append("cpu.cheapParallelMin=").append(cpu.cheapParallelMinSize()).append('|');
        sb.append("cpu.transParallelMin=").append(cpu.transcendentalParallelMinSize()).append('|');
        sb.append("cpu.redParallelMin=").append(cpu.reductionParallelMinSize()).append('|');
        sb.append("cpu.matmulParallelMin=").append(cpu.matMulParallelMinSize()).append('|');
        sb.append("cpu.contiguousThreshold=").append(cpu.contiguousMaterializeThreshold()).append('|');
        sb.append("cpu.lowChunksPerWorker=").append(cpu.lowCostTargetChunksPerWorker()).append('|');
        sb.append("cpu.mediumChunksPerWorker=").append(cpu.mediumCostTargetChunksPerWorker()).append('|');
        sb.append("cpu.highChunksPerWorker=").append(cpu.highCostTargetChunksPerWorker()).append('|');
        sb.append("cpu.minScalarChunk=").append(cpu.minScalarChunkSize()).append('|');
        sb.append("cpu.minVectorChunk=").append(cpu.minVectorChunkSize()).append('|');
        sb.append("cpu.minReductionChunk=").append(cpu.minReductionChunkSize()).append('|');
        sb.append("cpu.commonPoolLowCostMaxWorkPerWorker=").append(cpu.commonPoolLowCostMaxWorkPerWorker()).append('|');
        sb.append("cpu.fusedAsmVectorWidth=").append(cpu.fusedAsmVectorWidth()).append('|');
        sb.append("cpu.sumAccuracy=").append(cpu.sumAccuracyMode()).append('|');
        sb.append("cpu.attnMatMul=").append(cpu.attentionMatMulPolicy()).append('|');
        sb.append("cpu.matMulMicroKernel=").append(cpu.matMulMicroKernel()).append('|');
        sb.append("cpu.attentionMatMulMicroKernel=").append(cpu.attentionMatMulMicroKernel()).append('|');
        sb.append("approx.mode=").append(runtime.approximation().approxMode()).append('|');
        sb.append("approx.forceExact=").append(runtime.approximation().forceExactTranscendentals()).append('|');
        sb.append("blas.provider=").append(runtime.blas().provider()).append('|');
        sb.append("blas.minWork=").append(runtime.blas().matmulMinWork()).append('|');
        sb.append("blas.f32Req=").append(runtime.blas().f32RequireMgeK()).append('|');
        sb.append("blas.f32MaxNOverK=").append(fmt(runtime.blas().f32MaxNOverK())).append('|');
        sb.append("blas.threads=").append(runtime.blas().threads()).append('|');
        sb.append("blas.openBlasArrayCopyThreads=").append(runtime.blas().openBlasArrayCopyThreads()).append('|');
        sb.append("blas.openBlasNativeSegmentThreads=").append(runtime.blas().openBlasNativeSegmentThreads()).append('|');
        sb.append("conv2d.provider=").append(runtime.conv2d().provider()).append('|');
        sb.append("conv2d.f64MinWork=").append(runtime.conv2d().f64MinWork()).append('|');
        sb.append("conv2d.f32MinWork=").append(runtime.conv2d().f32MinWork()).append('|');
        sb.append("conv2d.f32Req=").append(runtime.conv2d().f32RequireMgeK()).append('|');
        sb.append("conv2d.f32MaxNOverK=").append(fmt(runtime.conv2d().f32MaxNOverK())).append('|');
        sb.append("conv2d.bf16MinWork=").append(runtime.conv2d().bf16MinWork()).append('|');
        sb.append("conv2d.bf16Req=").append(runtime.conv2d().bf16RequireMgeK()).append('|');
        sb.append("conv2d.bf16MaxNOverK=").append(fmt(runtime.conv2d().bf16MaxNOverK())).append('|');
        sb.append("fused.allowFallback=").append(runtime.fused().allowBackendFallback()).append('|');
        return sb.toString();
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.12f", v);
    }

    static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
