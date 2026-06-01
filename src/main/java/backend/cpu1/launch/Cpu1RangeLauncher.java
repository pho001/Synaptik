package backend.cpu1.launch;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * Shared range launcher for cpu1 prepared units.
 */
public final class Cpu1RangeLauncher {
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();

    private Cpu1RangeLauncher() {
    }

    public static void launch(int elementCount, Cpu1LaunchConfig launchConfig, RangeBody body) {
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
        if (elementCount <= 0) {
            return;
        }
        int workerCount = launchConfig.workerCount();
        if (workerCount == 1 || elementCount == 1) {
            body.compute(0, elementCount);
            return;
        }
        int chunk = chunkSize(elementCount, launchConfig);
        int taskCount = slotCount(elementCount, launchConfig);
        List<RecursiveAction> tasks = new ArrayList<>(taskCount);
        for (int start = 0; start < elementCount; start += chunk) {
            int rangeStart = start;
            int rangeEnd = Math.min(elementCount, rangeStart + chunk);
            tasks.add(new RecursiveAction() {
                @Override
                protected void compute() {
                    body.compute(rangeStart, rangeEnd);
                }
            });
        }
        invoke(workerCount, tasks);
    }

    public static void launchIndexed(int elementCount, Cpu1LaunchConfig launchConfig, IndexedRangeBody body) {
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        Objects.requireNonNull(body, "body cannot be null");
        if (elementCount <= 0) {
            return;
        }
        int workerCount = launchConfig.workerCount();
        if (workerCount == 1 || elementCount == 1) {
            body.compute(0, 0, elementCount);
            return;
        }
        int chunk = chunkSize(elementCount, launchConfig);
        int taskCount = slotCount(elementCount, launchConfig);
        List<RecursiveAction> tasks = new ArrayList<>(taskCount);
        for (int taskIndex = 0, start = 0; start < elementCount; taskIndex++, start += chunk) {
            int slotIndex = taskIndex;
            int rangeStart = start;
            int rangeEnd = Math.min(elementCount, rangeStart + chunk);
            tasks.add(new RecursiveAction() {
                @Override
                protected void compute() {
                    body.compute(slotIndex, rangeStart, rangeEnd);
                }
            });
        }
        invoke(workerCount, tasks);
    }

    public static int slotCount(int elementCount, Cpu1LaunchConfig launchConfig) {
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        if (elementCount <= 1 || launchConfig.workerCount() == 1) {
            return 1;
        }
        int chunk = chunkSize(elementCount, launchConfig);
        return (elementCount + chunk - 1) / chunk;
    }

    public static int chunkSize(int elementCount, Cpu1LaunchConfig launchConfig) {
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        if (launchConfig.hasResolvedChunkSize()) {
            return launchConfig.chunkSize();
        }
        int rangeWorkers = Math.min(launchConfig.workerCount(), Math.max(1, elementCount));
        return (Math.max(1, elementCount) + rangeWorkers - 1) / rangeWorkers;
    }

    private static void invoke(int workerCount, List<RecursiveAction> tasks) {
        ForkJoinPool pool = POOLS.computeIfAbsent(workerCount, ForkJoinPool::new);
        try {
            pool.submit(() -> ForkJoinTask.invokeAll(tasks)).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("cpu1 range execution interrupted", e);
        } catch (Exception e) {
            throw new IllegalStateException("cpu1 range execution failed", e);
        }
    }

    @FunctionalInterface
    public interface RangeBody {
        void compute(int startInclusive, int endExclusive);
    }

    @FunctionalInterface
    public interface IndexedRangeBody {
        void compute(int slotIndex, int startInclusive, int endExclusive);
    }
}
