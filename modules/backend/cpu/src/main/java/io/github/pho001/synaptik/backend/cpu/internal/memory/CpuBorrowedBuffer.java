package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import java.util.Objects;

/**
 * Non-owning CPU representation of one exact caller-owned host-storage segment.
 *
 * <p>The wrapper strongly retains the storage while borrowed, but it neither extends the JDK
 * segment scope nor closes, slices, reinterprets, or copies the segment. Its lifecycle therefore
 * follows the caller-owned storage rather than the wrapper's no-op {@link #close()} method.</p>
 */
public final class CpuBorrowedBuffer extends CpuBufferRepresentation {
    private final HostTensorStorage storage;

    private CpuBorrowedBuffer(HostTensorStorage storage) {
        super(storage.dataType(), storage.byteSize(), storage.segment());
        this.storage = storage;
    }

    /**
     * Borrows exact live storage without copying or acquiring cleanup ownership.
     *
     * @param storage non-null live host storage whose exact segment is retained
     * @return a new non-owning CPU wrapper
     * @throws NullPointerException if {@code storage} is null
     * @throws IllegalStateException if its segment scope is not alive
     */
    public static CpuBorrowedBuffer borrow(HostTensorStorage storage) {
        Objects.requireNonNull(storage, "storage");
        if (!storage.isAlive()) throw new IllegalStateException("segment scope is not alive");
        return new CpuBorrowedBuffer(storage);
    }

    /** Returns the caller-owned storage.
     * @return the exact non-null borrowed storage reference */
    public HostTensorStorage storage() { return storage; }

    /** @return {@code true} exactly when the caller-owned storage is no longer alive */
    @Override protected boolean isClosed() { return !storage.isAlive(); }

    /** Performs no action because the caller retains storage ownership. */
    @Override public void close() { }
}
