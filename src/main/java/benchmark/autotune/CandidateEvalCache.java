package benchmark.autotune;

import benchmark.OptimizerCandidate;
import benchmark.measure.CandidateMeasurementCachePort;
import benchmark.measure.CandidateMeasurementResult;
import tensor.DataType;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public final class CandidateEvalCache implements CandidateMeasurementCachePort {
    private final Map<String, CandidateMeasurementResult> byKey = new HashMap<>();
    private final String baseContext;
    private final Function<OptimizerCandidate, String> fingerprintFn;

    public CandidateEvalCache(String baseContext, Function<OptimizerCandidate, String> fingerprintFn) {
        this.baseContext = baseContext;
        this.fingerprintFn = fingerprintFn;
    }

    public CandidateEvalCache(
            DataType dtype,
            int graphBlocks,
            int b0,
            int b1,
            int f,
            int size,
            int broadcastASize,
            int broadcastBSize,
            int broadcastCSize,
            Function<OptimizerCandidate, String> fingerprintFn
    ) {
        this(
                "dtype=" + dtype
                        + "|graphBlocks=" + graphBlocks
                        + "|bshape=" + b0 + "x" + b1 + "x" + f
                        + "|size=" + size
                        + "|bcastA=" + broadcastASize
                        + "|bcastB=" + broadcastBSize
                        + "|bcastC=" + broadcastCSize,
                fingerprintFn
        );
    }

    @Override
    public CandidateMeasurementResult get(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters) {
        return byKey.get(cacheKey(candidate, tier, warmupIters, measureIters));
    }

    @Override
    public void put(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters, CandidateMeasurementResult result) {
        byKey.put(cacheKey(candidate, tier, warmupIters, measureIters), result);
    }

    private String cacheKey(OptimizerCandidate candidate, String tier, int warmupIters, int measureIters) {
        return baseContext
                + "|tier=" + tier
                + "|warmup=" + warmupIters
                + "|measure=" + measureIters
                + "|fp=" + fingerprintFn.apply(candidate);
    }
}
