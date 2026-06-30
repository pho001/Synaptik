package runtime.memory.nativecpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;

/**
 * Native CPU block pool keyed by size and alignment.
 */
public final class NativeCpuMemoryPool implements AutoCloseable {
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
    private boolean closed;

    public NativeCpuMemoryPool(long maxPoolBytes) {
        this.maxPoolBytes = Math.max(0L, maxPoolBytes);
    }

    synchronized Block acquire(long allocatedBytes, long alignment) {
        if (closed) {
            return null;
        }
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
        if (closed || block == null || maxPoolBytes <= 0L || block.allocatedBytes() > maxPoolBytes
                || pooledBytes + block.allocatedBytes() > maxPoolBytes) {
            return false;
        }
        available.computeIfAbsent(new Key(block.allocatedBytes(), block.alignment()), ignored -> new ArrayDeque<>())
                .addFirst(block);
        pooledBytes += block.allocatedBytes();
        return true;
    }

    public synchronized long drain() {
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

    public synchronized boolean closed() {
        return closed;
    }

    @Override
    public synchronized void close() {
        closed = true;
        drain();
    }
}
