package numerics;

import config.profile.ExecutionProfile;
import tensor.DataType;

/**
 * User-facing command-line entry point for numerics diagnostics.
 */
public final class NumericsCli {
    private NumericsCli() {}

    /**
     * Runs the numerics harness using {@code numerics.*} system properties.
     *
     * @param args ignored command-line arguments
     */
    public static void main(String[] args) {
        NumericsHarness.Config cfg = new NumericsHarness.Config();
        cfg.dtype = resolveDtype();
        cfg.size = Integer.getInteger("numerics.size", cfg.size);
        cfg.graphBlocks = Integer.getInteger("numerics.graphBlocks", cfg.graphBlocks);
        cfg.b0 = Integer.getInteger("numerics.broadcastB0", cfg.b0);
        cfg.b1 = Integer.getInteger("numerics.broadcastB1", cfg.b1);
        cfg.f = Integer.getInteger("numerics.broadcastF", cfg.f);
        cfg.seed = Long.getLong("numerics.seed", cfg.seed);

        String stageA = System.getProperty("numerics.stageA", "NONE");
        String stageB = System.getProperty("numerics.stageB", "AR");
        String nameA = System.getProperty("numerics.nameA", "A");
        String nameB = System.getProperty("numerics.nameB", "B");

        NumericsHarness harness = new NumericsHarness(cfg);
        ExecutionProfile a = harness.profile(nameA, NumericsHarness.parseStages(stageA));
        ExecutionProfile b = harness.profile(nameB, NumericsHarness.parseStages(stageB));

        NumericsPolicy policy = NumericsPolicy.defaultsFor(cfg.dtype);
        NumericsReport report = harness.run(a, b, policy);
        System.out.print(report.toPrettyString());
    }

    private static DataType resolveDtype() {
        String raw = System.getProperty("numerics.dtype", DataType.FLOAT32.name()).trim().toUpperCase();
        return DataType.valueOf(raw);
    }
}
