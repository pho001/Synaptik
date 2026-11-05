package io.github.pho001.synaptik.tools.cli;

import java.util.function.LongSupplier;

/**
 * Bounded, synchronized scalar telemetry. The caller records only committed steps; no model,
 * tensor or native storage crosses into this object. Step recording allocates no objects.
 */
final class TrainingProgressStats {
    static final int LOSS_CAPACITY = 120;
    static final int EPOCH_CAPACITY = 8;

    private final LongSupplier clock;
    private final long started;
    private final double[] losses = new double[LOSS_CAPACITY];
    private final int[] epochNumbers = new int[EPOCH_CAPACITY];
    private final long[] epochDurations = new long[EPOCH_CAPACITY];
    private final long[] epochStepCounts = new long[EPOCH_CAPACITY];
    private final long[] epochByteCounts = new long[EPOCH_CAPACITY];
    private final double[] epochMeans = new double[EPOCH_CAPACITY];
    private long steps;
    private long examples;
    private long bytes;
    private long epochStarted;
    private long epochEnded;
    private long epochSteps;
    private long epochBytes;
    private double epochMean;
    private double lastLoss = Double.NaN;
    private int epoch;
    private boolean epochActive;
    private int lossHead;
    private int lossSize;
    private int epochHead;
    private int epochSize;
    private long sampledSteps;
    private long rateTime;
    private long rateSteps;
    private long rateBytes;

    /** @param clock monotonic nanosecond clock, retained for scalar measurements */
    TrainingProgressStats(LongSupplier clock) {
        this.clock = clock;
        started = clock.getAsLong();
        rateTime = started;
    }

    /** @param number one-based epoch being entered, after the previous epoch ended */
    synchronized void beginEpoch(int number) {
        if (epochActive || number != epoch + 1) {
            throw new IllegalStateException("Epochs must begin once, in order");
        }
        epoch = number;
        epochStarted = clock.getAsLong();
        epochEnded = epochStarted;
        epochSteps = 0;
        epochBytes = 0;
        epochMean = Double.NaN;
        epochActive = true;
    }

    /**
     * Records a successful update, including legal nonfinite detached objectives.
     * @param loss actual pre-update detached objective, finite or nonfinite
     * @param count actual positive number of examples consumed
     * @param payload actual nonnegative decoded input-plus-target bytes consumed
     */
    synchronized void step(double loss, long count, long payload) {
        if (!epochActive) {
            throw new IllegalStateException("No active epoch");
        }
        if (count <= 0 || payload < 0) {
            throw new IllegalArgumentException("Invalid successful step payload");
        }
        long nextSteps = Math.addExact(steps, 1);
        long nextExamples = Math.addExact(examples, count);
        long nextBytes = Math.addExact(bytes, payload);
        long nextEpochBytes = Math.addExact(epochBytes, payload);
        long nextEpochSteps = Math.addExact(epochSteps, 1);
        epochMean = mean(epochMean, loss, nextEpochSteps);
        steps = nextSteps;
        examples = nextExamples;
        bytes = nextBytes;
        epochBytes = nextEpochBytes;
        epochSteps = nextEpochSteps;
        lastLoss = loss;
    }

    /** Ends the current epoch and retains only the most recent bounded epoch summaries. */
    synchronized void endEpoch() {
        if (!epochActive) {
            throw new IllegalStateException("No active epoch");
        }
        epochEnded = clock.getAsLong();
        epochNumbers[epochHead] = epoch;
        epochDurations[epochHead] = duration(epochEnded, epochStarted);
        epochStepCounts[epochHead] = epochSteps;
        epochByteCounts[epochHead] = epochBytes;
        epochMeans[epochHead] = epochMean;
        epochHead = (epochHead + 1) % EPOCH_CAPACITY;
        epochSize = Math.min(epochSize + 1, EPOCH_CAPACITY);
        epochActive = false;
    }

    /**
     * Samples the latest actual completed step at refresh/final cadence. Both rate numerators
     * use precisely the same elapsed interval; a zero interval has unavailable rates.
     * @return detached bounded snapshot, safe to render without this object's lock
     */
    synchronized Snapshot snapshot() {
        long now = clock.getAsLong();
        long interval = duration(now, rateTime);
        double stepRate = rate(steps - rateSteps, interval);
        double mbitRate = mbitRate(bytes - rateBytes, interval);
        // Do not consume payload at a zero-time sample: the next positive interval owns it.
        if (interval > 0) {
            rateTime = now;
            rateSteps = steps;
            rateBytes = bytes;
        }
        if (steps != sampledSteps) {
            losses[lossHead] = lastLoss;
            lossHead = (lossHead + 1) % LOSS_CAPACITY;
            lossSize = Math.min(lossSize + 1, LOSS_CAPACITY);
            sampledSteps = steps;
        }
        double[] history = new double[lossSize];
        for (int i = 0; i < lossSize; i++) {
            history[i] = losses[(lossHead - lossSize + i + LOSS_CAPACITY) % LOSS_CAPACITY];
        }
        Epoch[] recent = new Epoch[epochSize];
        for (int i = 0; i < epochSize; i++) {
            int index = (epochHead - epochSize + i + EPOCH_CAPACITY) % EPOCH_CAPACITY;
            recent[i] = new Epoch(epochNumbers[index], epochDurations[index],
                    epochStepCounts[index], epochByteCounts[index], epochMeans[index]);
        }
        return new Snapshot(steps, examples, bytes, epoch, epochSteps,
                duration(now, started), epoch == 0 ? 0 : duration(epochActive ? now : epochEnded, epochStarted),
                lastLoss, epochMean, stepRate, mbitRate, history, recent);
    }

    private static double mean(double previous, double value, long count) {
        if (count == 1) {
            return value;
        }
        if (!Double.isFinite(previous) || !Double.isFinite(value)) {
            return previous + value;
        }
        // Weighted terms cannot overflow for a finite mean, even with opposing extrema.
        return previous * ((count - 1.0) / count) + value / count;
    }

    static long duration(long end, long start) {
        return Math.max(0, end - start);
    }

    static double rate(long count, long nanos) {
        return nanos > 0 ? count * (1_000_000_000.0 / nanos) : Double.NaN;
    }

    static double mbitRate(long bytes, long nanos) {
        return nanos > 0 ? bytes * (8_000.0 / nanos) : Double.NaN;
    }

    /** Detached scalar state and bounded copies; loss values are never normalized or repaired. */
    record Snapshot(long steps, long examples, long bytes, int epoch, long epochSteps,
            long elapsedNanos, long epochNanos, double loss, double epochMean,
            double stepsPerSecond, double payloadMbitPerSecond, double[] losses, Epoch[] recentEpochs) {}

    /** Actual completed epoch measurements, not a final-model evaluation. */
    record Epoch(int number, long nanos, long steps, long bytes, double meanLoss) {}
}
