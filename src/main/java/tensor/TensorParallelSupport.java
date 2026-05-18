package tensor;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

final class TensorParallelSupport {
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();

    private TensorParallelSupport() {
    }

    static void runChunks(int chunks, int parallelism, IntConsumer chunkBody) {
        if (chunks <= 0) {
            return;
        }
        if (chunks == 1 || parallelism <= 1) {
            for (int i = 0; i < chunks; i++) {
                chunkBody.accept(i);
            }
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
