package backend.cpu.nativecpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;

/**
 * Run-local native CPU block pool.
 */
final class NativeCpuMemoryPool {
    private record Key(long bytes, long alignment) {
    }

    static final class Block {
        private final Arena arena;
        private final MemorySegment segment;
        private final long allocatedBytes;
        private final long alignment;

        Block(Arena arena, MemorySegment segment, long allocatedBytes, long alignment) {
            this.arena = arena;
            this.segment = segment;
            this.allocatedBytes = allocatedBytes;
            this.alignment = alignment;
        }

        MemorySegment segment() {
            return segment;
        }

        long allocatedBytes() {
            return allocatedBytes;
        }

        long alignment() {
            return alignment;
        }

        void close() {
            arena.close();
        }
    }

    private final long maxPoolBytes;
    private final Map<Key, ArrayDeque<Block>> available = new HashMap<>();
    private long pooledBytes;

    NativeCpuMemoryPool(long maxPoolBytes) {
        this.maxPoolBytes = Math.max(0L, maxPoolBytes);
    }

    synchronized Block acquire(long allocatedBytes, long alignment) {
        ArrayDeque<Block> blocks = available.get(new Key(allocatedBytes, alignment));
        if (blocks == null) {
            return null;
        }
        Block block = blocks.pollFirst();
        if (block == null) {
            return null;
        }
        pooledBytes = Math.max(0L, pooledBytes - block.allocatedBytes());
        if (blocks.isEmpty()) {
            available.remove(new Key(allocatedBytes, alignment));
        }
        return block;
    }

    synchronized boolean release(Block block) {
        if (block == null || maxPoolBytes <= 0L || block.allocatedBytes() > maxPoolBytes
                || pooledBytes + block.allocatedBytes() > maxPoolBytes) {
            return false;
        }
        available.computeIfAbsent(new Key(block.allocatedBytes(), block.alignment()), ignored -> new ArrayDeque<>())
                .addFirst(block);
        pooledBytes += block.allocatedBytes();
        return true;
    }

    synchronized long drain() {
        long drained = 0L;
        for (ArrayDeque<Block> blocks : available.values()) {
            while (!blocks.isEmpty()) {
                Block block = blocks.pollFirst();
                drained += block.allocatedBytes();
                block.close();
            }
        }
        available.clear();
        pooledBytes = 0L;
        return drained;
    }
}
