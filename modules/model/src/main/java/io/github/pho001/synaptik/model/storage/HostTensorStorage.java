package io.github.pho001.synaptik.model.storage;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;

/**
 * Describes an exact region of physical tensor-element capacity that is visible in host memory.
 *
 * <p>This sealed boundary exposes raw storage facts only: logical element type, physical element
 * capacity, exact byte size, the underlying memory segment, raw mutability, and current liveness.
 * The only permitted implementation is {@link MemorySegmentStorage}, so every instance obeys the
 * same exact-sizing and borrowed-lifetime contract.</p>
 *
 * <p>Host storage is not a public mutable tensor, tensor descriptor, layout, graph value, runtime
 * residency record, prepared-memory slot, device buffer, or backend-specific storage object. It
 * does not allocate or close memory, validate tensor geometry, or define typed element access,
 * alignment, or byte order.</p>
 */
public sealed interface HostTensorStorage permits MemorySegmentStorage {
    /**
     * Returns the logical type of every complete physical element in this storage.
     *
     * <p>The type supplies the element byte width used to establish exact sizing. It does not
     * select a Java carrier, byte order, alignment, or backend representation.</p>
     *
     * @return the exact non-null data type supplied when this storage was created
     */
    DataType dataType();

    /**
     * Returns the physical number of complete elements represented by the exact byte region.
     *
     * <p>This non-negative capacity is independent of a tensor's logical element count, layout
     * offset, or referenced element span.</p>
     *
     * @return the non-negative physical element capacity
     */
    long elementCapacity();

    /**
     * Returns the exact size of the underlying raw-memory region in bytes.
     *
     * <p>The result is the checked product of {@link #elementCapacity()} and
     * {@link DataType#byteWidth()}, and is exactly equal to {@code segment().byteSize()}. Zero
     * capacity therefore has zero byte size.</p>
     *
     * @return the non-negative exact byte size
     */
    long byteSize();

    /**
     * Returns the exact memory-segment object supplied to this storage.
     *
     * <p>No copy, slice, reinterpretation, read-only view, or replacement lifetime is created.
     * For writable segments, callers can mutate raw bytes subject to the JDK segment's scope and
     * thread-access rules. The caller remains responsible for the segment's lifetime; this
     * storage does not own or close an arena. The returned segment may have a scope that is no
     * longer alive, in which case JDK memory-access operations enforce closed-scope failures.</p>
     *
     * @return the exact non-null supplied memory-segment reference
     */
    MemorySegment segment();

    /**
     * Reports the read-only property of the supplied memory segment.
     *
     * <p>A false result describes a writable raw segment but does not add a mutation API,
     * synchronization, ownership, or version tracking. JDK scope and thread-access rules still
     * apply.</p>
     *
     * @return {@code true} when the supplied segment is read-only; otherwise {@code false}
     */
    boolean isReadOnly();

    /**
     * Reports whether the supplied segment's scope is alive at the instant of the query.
     *
     * <p>The result is a point-in-time observation and does not guarantee that the scope remains
     * alive for a later access. It also does not establish thread accessibility or replace the
     * JDK's memory-access checks.</p>
     *
     * @return {@code true} when {@code segment().scope().isAlive()} is currently true; otherwise
     *     {@code false}
     */
    boolean isAlive();
}
