package Benchmark.autotune;

import Benchmark.OptimizerCandidate;
import Benchmark.TuningKnobs;

import java.util.Locale;

public final class CoarseKnobSignature {
    private final boolean strictCseSafety;
    private final String fuseProfileKey;
    private final String kernelProfileKey;
    private final String blasProfileKey;

    public CoarseKnobSignature(boolean strictCseSafety, String fuseProfileKey, String kernelProfileKey, String blasProfileKey) {
        this.strictCseSafety = strictCseSafety;
        this.fuseProfileKey = fuseProfileKey;
        this.kernelProfileKey = kernelProfileKey;
        this.blasProfileKey = blasProfileKey;
    }

    public static CoarseKnobSignature of(OptimizerCandidate candidate) {
        TuningKnobs knobs = candidate.knobs();
        var fuse = knobs.fuseConfig();
        var kernel = knobs.kernelConfig();
        var cpu = kernel.cpu();
        String fuseKey = fuse.maxClusterNodes() + "|"
                + fmtDouble(fuse.scoreThreshold()) + "|"
                + fmtDouble(fuse.internalEdgeBonus()) + "|"
                + fmtDouble(fuse.externalInputPenalty()) + "|"
                + fmtDouble(fuse.sharedExpensivePenalty()) + "|"
                + fmtDouble(fuse.nonCheapBonus()) + "|"
                + fuse.preserveSharedExpensiveNodes();
        String kernelKey = cpu.loopUnrollFactor() + "|"
                + cpu.matMulTileM() + "|"
                + cpu.matMulTileN() + "|"
                + cpu.matMulTileK() + "|"
                + cpu.vectorMinSize() + "|"
                + cpu.parallelMinSize() + "|"
                + cpu.matMulParallelMinSize() + "|"
                + cpu.parallelism() + "|"
                + cpu.chunksPerWorker() + "|"
                + cpu.minChunkSize() + "|"
                + cpu.contiguousMaterializeThreshold() + "|"
                + cpu.sumAccuracyMode() + "|"
                + fmtDouble(cpu.lowCostNsPerElementThreshold()) + "|"
                + cpu.vectorPolicyCheap() + "|"
                + cpu.vectorPolicyTranscendental() + "|"
                + cpu.vectorPolicyReduction();
        String blasKey = knobs.blasProvider() + "|"
                + knobs.blasMatMulMinWork() + "|"
                + knobs.blasF32RequireMgeK() + "|"
                + fmtDouble(knobs.blasF32MaxNOverK());
        return new CoarseKnobSignature(knobs.strictCseSafety(), fuseKey, kernelKey, blasKey);
    }

    public int distance(CoarseKnobSignature other) {
        int diff = 0;
        if (strictCseSafety != other.strictCseSafety) diff++;
        if (!fuseProfileKey.equals(other.fuseProfileKey)) diff++;
        if (!kernelProfileKey.equals(other.kernelProfileKey)) diff++;
        if (!blasProfileKey.equals(other.blasProfileKey)) diff++;
        return diff;
    }

    public boolean strictCseSafety() {
        return strictCseSafety;
    }

    public String fuseProfileKey() {
        return fuseProfileKey;
    }

    public String kernelProfileKey() {
        return kernelProfileKey;
    }

    public String blasProfileKey() {
        return blasProfileKey;
    }

    private static String fmtDouble(double v) {
        return String.format(Locale.US, "%.12f", v);
    }
}
