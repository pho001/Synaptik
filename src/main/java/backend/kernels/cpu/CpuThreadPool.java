package backend.kernels.cpu;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public final class CpuThreadPool {
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();

    private CpuThreadPool() {}

    public static void runChunks(int chunks, int parallelism, IntConsumer chunkBody) {
        runChunks(chunks, parallelism, chunkBody, false);
    }

    public static void runChunks(int chunks, int parallelism, IntConsumer chunkBody, boolean preferCommonPool) {
        if (chunks <= 0) {
            return;
        }
        if (chunks == 1 || parallelism <= 1) {
            for (int i = 0; i < chunks; i++) {
                chunkBody.accept(i);
            }
            return;
        }
        if (preferCommonPool) {
            IntStream.range(0, chunks).parallel().forEach(chunkBody);
            return;
        }

        ForkJoinPool pool = POOLS.computeIfAbsent(parallelism, ForkJoinPool::new);
        try {
            pool.submit(() -> IntStream.range(0, chunks).parallel().forEach(chunkBody)).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel chunk execution failed", e);
        }
    }
}
