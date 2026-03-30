package benchmark.autotune;

import benchmark.OptimizationStage;
import benchmark.OptimizerCandidate;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Locale;

public final class AutoTuneResult {
    private final OptimizerCandidate candidate;
    private final int graphInfSize;
    private final int graphTrnSize;
    private final double forwardMs;
    private final double trainMs;
    private final double broadcastMs;
    private final double score;

    public AutoTuneResult(
            OptimizerCandidate candidate,
            int graphInfSize,
            int graphTrnSize,
            double forwardMs,
            double trainMs,
            double broadcastMs,
            double score
    ) {
        this.candidate = candidate;
        this.graphInfSize = graphInfSize;
        this.graphTrnSize = graphTrnSize;
        this.forwardMs = forwardMs;
        this.trainMs = trainMs;
        this.broadcastMs = broadcastMs;
        this.score = score;
    }

    public static AutoTuneResult forTraining(CandidatePerf perf) {
        return new AutoTuneResult(
                perf.candidate(),
                perf.graphInfSize(),
                perf.graphTrnSize(),
                perf.forwardMs(),
                perf.trainMs(),
                perf.broadcastMs(),
                perf.trainingScore()
        );
    }

    public static AutoTuneResult forInference(CandidatePerf perf) {
        return new AutoTuneResult(
                perf.candidate(),
                perf.graphInfSize(),
                perf.graphTrnSize(),
                perf.forwardMs(),
                perf.trainMs(),
                perf.broadcastMs(),
                perf.inferenceScore()
        );
    }

    public OptimizerCandidate candidate() {
        return candidate;
    }

    public int graphInfSize() {
        return graphInfSize;
    }

    public int graphTrnSize() {
        return graphTrnSize;
    }

    public double forwardMs() {
        return forwardMs;
    }

    public double trainMs() {
        return trainMs;
    }

    public double broadcastMs() {
        return broadcastMs;
    }

    public double score() {
        return score;
    }

    public String toJson(int validCount, int mismatchCount) {
        var knobs = candidate.knobs();
        var fuse = knobs.fuseConfig();
        var kernels = knobs.kernelConfig();
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"timestamp\": \"").append(OffsetDateTime.now()).append("\",\n");
        sb.append("  \"candidateName\": \"").append(candidate.name()).append("\",\n");
        sb.append("  \"stageOrder\": ").append(stageOrderJson(candidate.stageOrder())).append(",\n");
        sb.append("  \"knobs\": {\n");
        sb.append("    \"strictCseSafety\": ").append(knobs.strictCseSafety()).append(",\n");
        sb.append("    \"kernel\": {\n");
        sb.append("      \"cpu\": {\n");
        sb.append("        \"cpuLoopUnrollFactor\": ").append(kernels.cpu().loopUnrollFactor()).append(",\n");
        sb.append("        \"cpuMatMulTileM\": ").append(kernels.cpu().matMulTileM()).append(",\n");
        sb.append("        \"cpuMatMulTileN\": ").append(kernels.cpu().matMulTileN()).append(",\n");
        sb.append("        \"cpuMatMulTileK\": ").append(kernels.cpu().matMulTileK()).append(",\n");
        sb.append("        \"cpuVectorMinSize\": ").append(kernels.cpu().vectorMinSize()).append(",\n");
        sb.append("        \"cpuParallelMinSize\": ").append(kernels.cpu().parallelMinSize()).append(",\n");
        sb.append("        \"cpuMatMulParallelMinSize\": ").append(kernels.cpu().matMulParallelMinSize()).append(",\n");
        sb.append("        \"cpuParallelism\": ").append(kernels.cpu().parallelism()).append(",\n");
        sb.append("        \"cpuChunksPerWorker\": ").append(kernels.cpu().chunksPerWorker()).append(",\n");
        sb.append("        \"cpuMinChunkSize\": ").append(kernels.cpu().minChunkSize()).append(",\n");
        sb.append("        \"cpuContiguousMaterializeThreshold\": ").append(kernels.cpu().contiguousMaterializeThreshold()).append(",\n");
        sb.append("        \"cpuLowCostNsPerElementThreshold\": ").append(String.format(Locale.US, "%.8f", kernels.cpu().lowCostNsPerElementThreshold())).append(",\n");
        sb.append("        \"cpuVectorPolicyCheap\": \"").append(kernels.cpu().vectorPolicyCheap().name()).append("\",\n");
        sb.append("        \"cpuVectorPolicyTranscendental\": \"").append(kernels.cpu().vectorPolicyTranscendental().name()).append("\",\n");
        sb.append("        \"cpuVectorPolicyReduction\": \"").append(kernels.cpu().vectorPolicyReduction().name()).append("\"\n");
        sb.append("      },\n");
        sb.append("      \"cuda\": {\n");
        sb.append("        \"cudaLoopUnrollFactor\": ").append(kernels.cuda().loopUnrollFactor()).append(",\n");
        sb.append("        \"cudaMatMulTileM\": ").append(kernels.cuda().matMulTileM()).append(",\n");
        sb.append("        \"cudaMatMulTileN\": ").append(kernels.cuda().matMulTileN()).append(",\n");
        sb.append("        \"cudaMatMulTileK\": ").append(kernels.cuda().matMulTileK()).append("\n");
        sb.append("      },\n");
        sb.append("      \"opencl\": {\n");
        sb.append("        \"openclLoopUnrollFactor\": ").append(kernels.opencl().loopUnrollFactor()).append(",\n");
        sb.append("        \"openclMatMulTileM\": ").append(kernels.opencl().matMulTileM()).append(",\n");
        sb.append("        \"openclMatMulTileN\": ").append(kernels.opencl().matMulTileN()).append(",\n");
        sb.append("        \"openclMatMulTileK\": ").append(kernels.opencl().matMulTileK()).append("\n");
        sb.append("      }\n");
        sb.append("    },\n");
        sb.append("    \"blas\": {\n");
        sb.append("      \"provider\": \"").append(knobs.blasProvider()).append("\",\n");
        sb.append("      \"matmulMinWork\": ").append(knobs.blasMatMulMinWork()).append(",\n");
        sb.append("      \"f32RequireMgeK\": ").append(knobs.blasF32RequireMgeK()).append(",\n");
        sb.append("      \"f32MaxNOverK\": ").append(String.format(Locale.US, "%.8f", knobs.blasF32MaxNOverK())).append("\n");
        sb.append("    },\n");
        sb.append("    \"fuse\": {\n");
        sb.append("      \"maxClusterNodes\": ").append(fuse.maxClusterNodes()).append(",\n");
        sb.append("      \"scoreThreshold\": ").append(String.format(Locale.US, "%.8f", fuse.scoreThreshold())).append(",\n");
        sb.append("      \"internalEdgeBonus\": ").append(String.format(Locale.US, "%.8f", fuse.internalEdgeBonus())).append(",\n");
        sb.append("      \"externalInputPenalty\": ").append(String.format(Locale.US, "%.8f", fuse.externalInputPenalty())).append(",\n");
        sb.append("      \"sharedExpensivePenalty\": ").append(String.format(Locale.US, "%.8f", fuse.sharedExpensivePenalty())).append(",\n");
        sb.append("      \"nonCheapBonus\": ").append(String.format(Locale.US, "%.8f", fuse.nonCheapBonus())).append(",\n");
        sb.append("      \"preserveSharedExpensiveNodes\": ").append(fuse.preserveSharedExpensiveNodes()).append("\n");
        sb.append("    }\n");
        sb.append("  },\n");
        sb.append("  \"metrics\": {\n");
        sb.append("    \"graphInfSize\": ").append(graphInfSize).append(",\n");
        sb.append("    \"graphTrnSize\": ").append(graphTrnSize).append(",\n");
        sb.append("    \"forwardMs\": ").append(String.format(Locale.US, "%.8f", forwardMs)).append(",\n");
        sb.append("    \"trainMs\": ").append(String.format(Locale.US, "%.8f", trainMs)).append(",\n");
        sb.append("    \"broadcastMs\": ").append(String.format(Locale.US, "%.8f", broadcastMs)).append(",\n");
        sb.append("    \"score\": ").append(String.format(Locale.US, "%.8f", score)).append(",\n");
        sb.append("    \"validCandidates\": ").append(validCount).append(",\n");
        sb.append("    \"mismatchedCandidates\": ").append(mismatchCount).append("\n");
        sb.append("  }\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String stageOrderJson(List<OptimizationStage> stages) {
        if (stages.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < stages.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append("\"").append(stages.get(i).name()).append("\"");
        }
        sb.append("]");
        return sb.toString();
    }
}
