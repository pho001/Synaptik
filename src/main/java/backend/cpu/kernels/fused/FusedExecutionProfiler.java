package backend.cpu.kernels.fused;

import backend.cpu.kernels.*;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

public final class FusedExecutionProfiler {
    private static final boolean ENABLED = Boolean.getBoolean("cg.cpu.fused.profile");
    private static final ConcurrentHashMap<String, Entry> ENTRIES = new ConcurrentHashMap<>();

    private FusedExecutionProfiler() {}

    public static boolean enabled() {
        return ENABLED;
    }

    public static void recordCompile(
            String signature,
            String expression,
            int clusterSize,
            int externalInputCount,
            int precisionMode,
            boolean lowCostHint,
            long compileNs
    ) {
        if (!ENABLED || signature == null) {
            return;
        }
        Entry entry = ENTRIES.computeIfAbsent(signature, k -> new Entry(signature));
        entry.expression = expression;
        entry.clusterSize = clusterSize;
        entry.externalInputCount = externalInputCount;
        entry.precisionMode = precisionMode;
        entry.lowCostHint = lowCostHint;
        entry.compileCount.increment();
        entry.compileNsTotal.add(Math.max(0L, compileNs));
        updateMax(entry.compileNsMax, compileNs);
    }

    public static void recordRun(
            String signature,
            CpuExecutionMode mode,
            int length,
            int chunks,
            boolean useCommonPool,
            boolean preferVector,
            long elapsedNs
    ) {
        if (!ENABLED || signature == null) {
            return;
        }
        Entry entry = ENTRIES.computeIfAbsent(signature, k -> new Entry(signature));
        entry.runCount.increment();
        entry.runNsTotal.add(Math.max(0L, elapsedNs));
        updateMax(entry.runNsMax, elapsedNs);
        entry.totalLength.add(Math.max(0, length));
        entry.totalChunks.add(Math.max(1, chunks));
        if (useCommonPool) {
            entry.commonPoolRuns.increment();
        }
        if (preferVector) {
            entry.vectorPreferredRuns.increment();
        }
        if (mode != null) {
            entry.modeCounts.computeIfAbsent(mode, ignored -> new LongAdder()).increment();
        }
    }

    public static void reset() {
        ENTRIES.clear();
    }

    public static String dumpSummary() {
        if (!ENABLED) {
            return "Fused profiler disabled.";
        }
        List<Entry> entries = new ArrayList<>(ENTRIES.values());
        entries.sort(Comparator.comparingLong(Entry::runNsTotalValue).reversed());
        StringBuilder sb = new StringBuilder(4096);
        sb.append("Fused profiler entries=").append(entries.size()).append('\n');
        for (Entry entry : entries) {
            long runs = entry.runCount.longValue();
            long compileCount = entry.compileCount.longValue();
            double avgRunMs = runs == 0 ? 0.0 : (entry.runNsTotal.longValue() / 1_000_000.0) / runs;
            double avgCompileMs = compileCount == 0 ? 0.0 : (entry.compileNsTotal.longValue() / 1_000_000.0) / compileCount;
            double avgLen = runs == 0 ? 0.0 : (double) entry.totalLength.longValue() / runs;
            double avgChunks = runs == 0 ? 0.0 : (double) entry.totalChunks.longValue() / runs;
            sb.append("signature=").append(entry.signature).append('\n');
            sb.append("  expr=").append(entry.expression).append('\n');
            sb.append("  clusterSize=").append(entry.clusterSize)
                    .append(", externalInputs=").append(entry.externalInputCount)
                    .append(", precisionMode=").append(entry.precisionMode)
                    .append(", lowCostHint=").append(entry.lowCostHint).append('\n');
            sb.append("  compileCount=").append(compileCount)
                    .append(", avgCompileMs=").append(fmt(avgCompileMs))
                    .append(", maxCompileMs=").append(fmt(entry.compileNsMax.longValue() / 1_000_000.0)).append('\n');
            sb.append("  runCount=").append(runs)
                    .append(", avgRunMs=").append(fmt(avgRunMs))
                    .append(", maxRunMs=").append(fmt(entry.runNsMax.longValue() / 1_000_000.0))
                    .append(", avgLen=").append(fmt(avgLen))
                    .append(", avgChunks=").append(fmt(avgChunks)).append('\n');
            sb.append("  commonPoolRuns=").append(entry.commonPoolRuns.longValue())
                    .append(", vectorPreferredRuns=").append(entry.vectorPreferredRuns.longValue()).append('\n');
            sb.append("  modes=");
            boolean first = true;
            for (Map.Entry<CpuExecutionMode, LongAdder> modeEntry : entry.modeCounts.entrySet()) {
                if (!first) {
                    sb.append(", ");
                }
                first = false;
                sb.append(modeEntry.getKey()).append('=').append(modeEntry.getValue().longValue());
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    private static void updateMax(LongAdder target, long value) {
        long current = target.longValue();
        if (value <= current) {
            return;
        }
        synchronized (target) {
            if (value > target.longValue()) {
                target.reset();
                target.add(value);
            }
        }
    }

    private static String fmt(double value) {
        return String.format(Locale.US, "%.3f", value);
    }

    private static final class Entry {
        private final String signature;
        private volatile String expression = "";
        private volatile int clusterSize = -1;
        private volatile int externalInputCount = -1;
        private volatile int precisionMode = -1;
        private volatile boolean lowCostHint;
        private final LongAdder compileCount = new LongAdder();
        private final LongAdder compileNsTotal = new LongAdder();
        private final LongAdder compileNsMax = new LongAdder();
        private final LongAdder runCount = new LongAdder();
        private final LongAdder runNsTotal = new LongAdder();
        private final LongAdder runNsMax = new LongAdder();
        private final LongAdder totalLength = new LongAdder();
        private final LongAdder totalChunks = new LongAdder();
        private final LongAdder commonPoolRuns = new LongAdder();
        private final LongAdder vectorPreferredRuns = new LongAdder();
        private final EnumMap<CpuExecutionMode, LongAdder> modeCounts = new EnumMap<>(CpuExecutionMode.class);

        private Entry(String signature) {
            this.signature = signature;
        }

        private long runNsTotalValue() {
            return runNsTotal.longValue();
        }
    }
}
