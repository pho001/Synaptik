package tensor;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

abstract class AbstractNativeTensorStorage implements NativeTensorStorage {
    private final DataType type;
    private final int size;
    private final long elementSizeBytes;
    private final NativeMemoryAllocation allocation;
    private final MemorySegment segment;
    private final long byteOffset;
    private final long byteSize;
    private final boolean ownsSegment;
    private final AtomicLong version = new AtomicLong();

    AbstractNativeTensorStorage(
            DataType type,
            int size,
            long elementSizeBytes,
            NativeMemoryAllocation allocation,
            long byteOffset,
            boolean ownsSegment
    ) {
        this.type = Objects.requireNonNull(type, "type cannot be null");
        if (size < 0) {
            throw new IllegalArgumentException("size cannot be negative: " + size);
        }
        if (elementSizeBytes <= 0L) {
            throw new IllegalArgumentException("elementSizeBytes must be positive: " + elementSizeBytes);
        }
        this.size = size;
        this.elementSizeBytes = elementSizeBytes;
        this.allocation = Objects.requireNonNull(allocation, "allocation cannot be null");
        if (byteOffset < 0L) {
            throw new IllegalArgumentException("byteOffset cannot be negative: " + byteOffset);
        }
        this.byteOffset = byteOffset;
        this.byteSize = Math.multiplyExact((long) size, elementSizeBytes);
        this.ownsSegment = ownsSegment;
        MemorySegment base = allocation.segment();
        long requiredBytes;
        try {
            requiredBytes = Math.addExact(this.byteOffset, this.byteSize);
        } catch (ArithmeticException ex) {
            throw new IllegalArgumentException("Native storage view byte range overflows long. byteOffset="
                    + this.byteOffset + ", byteSize=" + this.byteSize, ex);
        }
        if (requiredBytes > allocation.byteSize()) {
            throw new IllegalArgumentException("Native storage view exceeds allocation byteSize. required="
                    + requiredBytes + ", allocation=" + allocation.byteSize());
        }
        this.segment = base.asSlice(this.byteOffset, this.byteSize);
    }

    @Override
    public final DataType getType() {
        return type;
    }

    @Override
    public final int getSize() {
        return size;
    }

    @Override
    public final long version() {
        return version.get();
    }

    @Override
    public void markModified() {
        version.incrementAndGet();
    }

    @Override
    public final MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    @Override
    public final long byteOffset() {
        return byteOffset;
    }

    @Override
    public final long byteSize() {
        return byteSize;
    }

    @Override
    public final long elementSizeBytes() {
        return elementSizeBytes;
    }

    @Override
    public final boolean ownsSegment() {
        return ownsSegment;
    }

    @Override
    public final NativeMemoryAllocation allocation() {
        return allocation;
    }

    @Override
    public final boolean closed() {
        return allocation.closed();
    }

    @Override
    public final void ensureOpen() {
        allocation.ensureOpen();
    }

    protected final long byteOffsetForIndex(int flatIndex) {
        ensureOpen();
        if (flatIndex < 0 || flatIndex >= size) {
            throw new IndexOutOfBoundsException("Native storage index out of bounds. index="
                    + flatIndex + ", size=" + size);
        }
        return (long) flatIndex * elementSizeBytes;
    }

    @Override
    public void close() {
        if (ownsSegment) {
            allocation.close();
        }
    }
}
