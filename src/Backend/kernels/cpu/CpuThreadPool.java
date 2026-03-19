package Backend.kernels.cpu;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.IntConsumer;

public final class CpuThreadPool {
    private static final ConcurrentHashMap<Integer, ForkJoinPool> POOLS = new ConcurrentHashMap<>();

    private CpuThreadPool() {}

    public static void runChunks(int chunks, int parallelism, IntConsumer chunkBody) {
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
        int grain = Math.max(1, (chunks + (parallelism * 4) - 1) / (parallelism * 4));
        pool.invoke(new ChunkRangeTask(0, chunks, grain, chunkBody));
    }

    private static final class ChunkRangeTask extends RecursiveAction {
        private final int from;
        private final int to;
        private final int grain;
        private final IntConsumer chunkBody;

        private ChunkRangeTask(int from, int to, int grain, IntConsumer chunkBody) {
            this.from = from;
            this.to = to;
            this.grain = grain;
            this.chunkBody = chunkBody;
        }

        @Override
        protected void compute() {
            if (to - from <= grain) {
                for (int i = from; i < to; i++) {
                    chunkBody.accept(i);
                }
                return;
            }
            int mid = (from + to) >>> 1;
            invokeAll(
                    new ChunkRangeTask(from, mid, grain, chunkBody),
                    new ChunkRangeTask(mid, to, grain, chunkBody)
            );
        }
    }
}
