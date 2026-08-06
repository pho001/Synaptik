package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * A non-owning, direct buffer argument produced once during CPU cold binding.
 *
 * <p>Array variants expose an observable primitive heap carrier and express the selected region
 * with a carrier-relative byte offset. {@link Segment} retains the exact selected
 * {@link MemorySegment} when no matching carrier can be observed, including genuine native
 * segments and JDK 26 read-only heap segments whose {@link MemorySegment#heapBase()} is empty.
 * No variant copies bytes, extends the segment lifetime, or permits access after the owner or
 * borrower closes the underlying segment.</p>
 */
public sealed interface CpuBufferArgument {
    /** Returns the selected-region base.
     * @return the carrier byte offset, or zero for a segment */
    long byteOffset();

    /** Returns the selected-region size.
     * @return the exact non-negative byte size */
    long byteSize();

    /** Reports selected-region mutability.
     * @return whether writes are forbidden */
    boolean readOnly();

    /**
     * Direct region of an observable {@code double[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset aligned to {@link Double#BYTES}
     * @param byteSize non-negative region size in bytes aligned to {@link Double#BYTES}
     * @param readOnly whether the originating segment forbids writes
     */
    record Doubles(double[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable double-array region. */
        public Doubles { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Double.BYTES); }
    }

    /**
     * Direct region of an observable {@code float[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset aligned to {@link Float#BYTES}
     * @param byteSize non-negative region size in bytes aligned to {@link Float#BYTES}
     * @param readOnly whether the originating segment forbids writes
     */
    record Floats(float[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable float-array region. */
        public Floats { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Float.BYTES); }
    }

    /**
     * Direct region of an observable {@code short[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset aligned to {@link Short#BYTES}
     * @param byteSize non-negative region size in bytes aligned to {@link Short#BYTES}
     * @param readOnly whether the originating segment forbids writes
     */
    record Shorts(short[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable short-array region. */
        public Shorts { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Short.BYTES); }
    }

    /**
     * Direct region of an observable {@code int[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset aligned to {@link Integer#BYTES}
     * @param byteSize non-negative region size in bytes aligned to {@link Integer#BYTES}
     * @param readOnly whether the originating segment forbids writes
     */
    record Ints(int[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable int-array region. */
        public Ints { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Integer.BYTES); }
    }

    /**
     * Direct region of an observable {@code long[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset aligned to {@link Long#BYTES}
     * @param byteSize non-negative region size in bytes aligned to {@link Long#BYTES}
     * @param readOnly whether the originating segment forbids writes
     */
    record Longs(long[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable long-array region. */
        public Longs { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Long.BYTES); }
    }

    /**
     * Direct region of an observable {@code byte[]} carrier.
     *
     * @param carrier exact non-null heap carrier; retained by reference and never copied
     * @param byteOffset non-negative carrier-relative byte offset
     * @param byteSize non-negative region size in bytes
     * @param readOnly whether the originating segment forbids writes
     */
    record Bytes(byte[] carrier, long byteOffset, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates a directly addressable byte-array region. */
        public Bytes { validateHeap(carrier, carrier == null ? 0 : carrier.length,
                byteOffset, byteSize, Byte.BYTES); }
    }

    /**
     * Direct access through one exact segment or slice whose matching primitive heap carrier is
     * not observable.
     *
     * <p>This name describes the access form, not the segment's provenance: the retained segment
     * may be native or heap-backed. In particular, JDK 26 does not expose a heap base for a
     * read-only heap segment, so such a segment remains exact and copy-free in this variant.</p>
     *
     * @param dataType exact non-null logical data type used by the later typed route
     * @param segment exact non-null selected segment or slice; retained without reinterpretation
     * @param byteSize non-negative size in bytes, exactly equal to {@code segment.byteSize()}
     * @param readOnly whether writes through {@code segment} are forbidden
     */
    record Segment(DataType dataType, MemorySegment segment, long byteSize, boolean readOnly)
            implements CpuBufferArgument {
        /** Validates one exact accessible segment region. */
        public Segment {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(segment, "segment");
            if (byteSize < 0) throw new IllegalArgumentException("byteSize must be non-negative");
            if (segment.byteSize() != byteSize) {
                throw new IllegalArgumentException("segment byte size must equal byteSize");
            }
            if (!segment.scope().isAlive() || !segment.isAccessibleBy(Thread.currentThread())) {
                throw new IllegalStateException("segment is not accessible");
            }
        }

        /**
         * Returns the offset relative to the retained exact selected segment.
         *
         * @return always zero because {@link #segment()} is the selected region or slice
         */
        @Override public long byteOffset() { return 0L; }
    }

    /** Validates one carrier-relative region shared by all observable-array variants. */
    private static void validateHeap(
            Object carrier, int carrierLength, long offset, long size, int width) {
        Objects.requireNonNull(carrier, "carrier");
        if (offset < 0) throw new IllegalArgumentException("byteOffset must be non-negative");
        if (size < 0) throw new IllegalArgumentException("byteSize must be non-negative");
        if (offset % width != 0 || size % width != 0) {
            throw new IllegalArgumentException("heap region must be carrier-width aligned");
        }
        long capacity = Math.multiplyExact((long) carrierLength, width);
        if (offset > capacity || size > capacity - offset) {
            throw new IllegalArgumentException("heap region exceeds carrier capacity");
        }
    }
}
