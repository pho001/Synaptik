package io.github.pho001.synaptik.runtime.memory;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Describes the final ordered byte geometry of buffer and workspace slots in one prepared plan.
 *
 * <p>The two lists are immutable snapshots that preserve caller-supplied order and retain the
 * exact immutable entry and slot references. Buffer slots must be unique among buffer entries,
 * and workspace slots must be unique among workspace entries. These are separate nominal identity
 * domains, so equal numeric values across a {@link BufferSlot} and a {@link WorkspaceSlot} are
 * valid. Either list may be empty. The supplied list containers are not retained.
 *
 * <p>This plan contains only reusable slot geometry. It does not retain Prepare analysis
 * requirements or their source-to-slot associations, and it does not sort, derive, allocate,
 * bind, address, own, or release physical storage. Physical resource allocation and per-run slot
 * binding belong to later Runtime and backend contracts. Ordinary record equality and hashing use
 * the immutable list contents; record text is diagnostic, not a serialization format.
 *
 * @param buffers the ordered buffer-slot geometry entries to snapshot; must be non-null, contain
 *     no null entries, and use each buffer slot at most once
 * @param workspaces the ordered workspace-slot geometry entries to snapshot; must be non-null,
 *     contain no null entries, and use each workspace slot at most once
 * @param requiresBatchSize whether every run must supply an explicit non-negative batch size,
 *     independently of whether individual declarations have constant byte counts
 */
public record PreparedMemoryPlan(
        List<PreparedMemoryPlan.BufferEntry> buffers,
        List<PreparedMemoryPlan.WorkspaceEntry> workspaces,
        boolean requiresBatchSize) {
    /**
     * Creates a static-mode immutable snapshot of final slot geometry.
     * This overload always sets {@code requiresBatchSize=false} and rejects run-bound declarations,
     * even when their evaluated bytes happen to be constant. No recipe is evaluated here.
     *
     * <p>Top-level lists are validated before their contents. Buffer entries are then validated
     * and snapshotted before workspace entries are validated and snapshotted.
     *
     * @param buffers the ordered buffer entries to snapshot; must be non-null, contain no null
     *     entries, and use each buffer slot at most once
     * @param workspaces the ordered workspace entries to snapshot; must be non-null, contain no
     *     null entries, and use each workspace slot at most once
     * @throws NullPointerException if {@code buffers}, {@code workspaces}, or an entry is
     *     {@code null}
     * @throws IllegalArgumentException if a buffer slot or workspace slot occurs more than once
     *     in its respective list, or a declaration is run-bound in this static-only overload
     */
    public PreparedMemoryPlan(
            List<PreparedMemoryPlan.BufferEntry> buffers,
            List<PreparedMemoryPlan.WorkspaceEntry> workspaces) {
        this(buffers, workspaces, false);
    }

    /**
     * Snapshots structural declarations without evaluating run-bound recipes.
     *
     * @param buffers non-null ordered buffer declarations without nulls or duplicate slots
     * @param workspaces non-null ordered workspace declarations without nulls or duplicate slots
     * @param requiresBatchSize explicit run mode; false permits only fixed declarations
     * @throws NullPointerException if a list or entry is null
     * @throws IllegalArgumentException if slots repeat or a static plan has a run-bound declaration
     */
    public PreparedMemoryPlan(
            List<PreparedMemoryPlan.BufferEntry> buffers,
            List<PreparedMemoryPlan.WorkspaceEntry> workspaces,
            boolean requiresBatchSize) {
        Objects.requireNonNull(buffers, "buffers");
        Objects.requireNonNull(workspaces, "workspaces");

        var bufferSlots = new HashSet<BufferSlot>();
        for (int index = 0; index < buffers.size(); index++) {
            BufferEntry entry =
                    Objects.requireNonNull(buffers.get(index), "buffers[" + index + "]");
            if (!bufferSlots.add(entry.slot())) {
                throw new IllegalArgumentException(
                        "buffers[" + index + "].slot duplicates " + entry.slot());
            }
            if (!requiresBatchSize && entry.size().fixedByteSize().isEmpty()) {
                throw new IllegalArgumentException("static plan requires fixed buffer sizes");
            }
        }
        this.buffers = List.copyOf(buffers);

        var workspaceSlots = new HashSet<WorkspaceSlot>();
        for (int index = 0; index < workspaces.size(); index++) {
            WorkspaceEntry entry =
                    Objects.requireNonNull(workspaces.get(index), "workspaces[" + index + "]");
            if (!workspaceSlots.add(entry.slot())) {
                throw new IllegalArgumentException(
                        "workspaces[" + index + "].slot duplicates " + entry.slot());
            }
            if (!requiresBatchSize && entry.size().fixedByteSize().isEmpty()) {
                throw new IllegalArgumentException("static plan requires fixed workspace sizes");
            }
        }
        this.workspaces = List.copyOf(workspaces);
        this.requiresBatchSize = requiresBatchSize;
    }

    /**
     * Storage-free, immutable exact byte declaration evaluated once during cold run admission.
     * Implementations must be thread-safe and perform checked numeric arithmetic only.
     */
    public interface ByteSize {
        /**
         * Reports structural byte independence, not the result at any representative batch size.
         * @return exact non-negative fixed bytes, or empty for every run-bound recipe
         */
        OptionalLong fixedByteSize();

        /**
         * Resolves the exact usable span, excluding allocator padding.
         * @param batchSize non-negative actual batch size
         * @return non-negative exact byte count
         * @throws IllegalArgumentException if the batch size is negative
         * @throws ArithmeticException if checked sizing overflows
         */
        long byteSize(long batchSize);
    }

    /**
     * An exact fixed usable span, including zero, independent of a run's batch size.
     * @param bytes non-negative usable byte count
     */
    public record FixedByteSize(long bytes) implements ByteSize {
        /**
         * Validates a fixed byte declaration without allocating storage.
         * @param bytes non-negative usable bytes
         * @throws IllegalArgumentException if bytes is negative
         */
        public FixedByteSize {
            if (bytes < 0) throw new IllegalArgumentException("byteSize must be non-negative");
        }
        /** @return the exact non-negative fixed byte count */
        @Override public OptionalLong fixedByteSize() { return OptionalLong.of(bytes); }
        /**
         * Returns the fixed span after validating the numeric argument.
         * @param batchSize non-negative batch size, not otherwise used
         * @return exact fixed usable bytes
         * @throws IllegalArgumentException if batchSize is negative
         */
        @Override public long byteSize(long batchSize) {
            if (batchSize < 0) throw new IllegalArgumentException("batchSize must be non-negative");
            return bytes;
        }
    }

    /**
     * Returns the immutable ordered buffer-entry snapshot.
     *
     * @return the immutable list in exact supplied order, retaining the supplied non-null entry
     *     and slot references; never {@code null}
     */
    public List<PreparedMemoryPlan.BufferEntry> buffers() {
        return buffers;
    }

    /**
     * Returns the immutable ordered workspace-entry snapshot.
     *
     * @return the immutable list in exact supplied order, retaining the supplied non-null entry
     *     and slot references; never {@code null}
     */
    public List<PreparedMemoryPlan.WorkspaceEntry> workspaces() {
        return workspaces;
    }

    /**
     * Describes exact byte geometry for one buffer slot.
     *
     * <p>The entry retains the exact slot reference. Its geometry is declarative and does not
     * allocate, own, bind, address, or release bytes. Ordinary record equality and hashing use the
     * retained slot and two geometry values; record text is diagnostic only.
     *
     * @param slot the non-null buffer-slot identity retained exactly
     * @param size non-null immutable exact byte declaration
     * @param byteAlignment the exact required alignment in bytes; must be a positive power of two
     */
    public record BufferEntry(BufferSlot slot, ByteSize size, long byteAlignment) {
        /**
         * Creates exact geometry for one buffer slot.
         *
         * @param slot the buffer-slot identity to retain exactly; must be non-null
         * @param byteSize the exact required byte count; must be non-negative and may be zero
         * @param byteAlignment the exact required alignment in bytes; must be a positive power of
         *     two from {@code 1} through {@code 1L << 62}
         * @throws NullPointerException if {@code slot} is {@code null}
         * @throws IllegalArgumentException if {@code byteSize} is negative or
         *     {@code byteAlignment} is not a positive power of two
         */
        public BufferEntry(BufferSlot slot, long byteSize, long byteAlignment) {
            this(Objects.requireNonNull(slot, "slot"), new FixedByteSize(byteSize), byteAlignment);
        }

        /**
         * Retains one structural byte declaration, without evaluating it.
         * @param slot non-null exact buffer identity
         * @param size non-null immutable byte declaration
         * @param byteAlignment power-of-two bytes from 1 through 1L&lt;&lt;62
         * @throws NullPointerException if slot or size is null
         * @throws IllegalArgumentException if alignment is invalid
         */
        public BufferEntry(BufferSlot slot, ByteSize size, long byteAlignment) {
            this.slot = Objects.requireNonNull(slot, "slot");
            this.size = Objects.requireNonNull(size, "size");
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
            this.byteAlignment = byteAlignment;
        }

        /**
         * Returns the exact buffer-slot reference supplied at construction.
         *
         * @return the retained non-null buffer slot
         */
        public BufferSlot slot() {
            return slot;
        }

        /**
         * Returns the exact required byte count without allocating or owning storage.
         *
         * @return the non-negative byte count; zero is valid
         * @throws IllegalStateException if this declaration is run-bound
         */
        public long byteSize() {
            return size.fixedByteSize().orElseThrow(
                    () -> new IllegalStateException("byte size requires run geometry"));
        }

        /**
         * Returns the exact required byte alignment without allocating or binding storage.
         *
         * @return the positive power-of-two byte alignment from {@code 1} through
         *     {@code 1L << 62}
         */
        public long byteAlignment() {
            return byteAlignment;
        }
    }

    /**
     * Describes exact byte geometry for one workspace slot.
     *
     * <p>The entry retains the exact slot reference. Its geometry is declarative and does not
     * allocate, own, bind, address, or release bytes. Ordinary record equality and hashing use the
     * retained slot and two geometry values; record text is diagnostic only.
     *
     * @param slot the non-null workspace-slot identity retained exactly
     * @param size non-null immutable exact byte declaration
     * @param byteAlignment the exact required alignment in bytes; must be a positive power of two
     */
    public record WorkspaceEntry(WorkspaceSlot slot, ByteSize size, long byteAlignment) {
        /**
         * Creates exact geometry for one workspace slot.
         *
         * @param slot the workspace-slot identity to retain exactly; must be non-null
         * @param byteSize the exact required byte count; must be non-negative and may be zero
         * @param byteAlignment the exact required alignment in bytes; must be a positive power of
         *     two from {@code 1} through {@code 1L << 62}
         * @throws NullPointerException if {@code slot} is {@code null}
         * @throws IllegalArgumentException if {@code byteSize} is negative or
         *     {@code byteAlignment} is not a positive power of two
         */
        public WorkspaceEntry(WorkspaceSlot slot, long byteSize, long byteAlignment) {
            this(Objects.requireNonNull(slot, "slot"), new FixedByteSize(byteSize), byteAlignment);
        }

        /**
         * Retains one structural scratch declaration, without evaluating it.
         * @param slot non-null exact workspace identity
         * @param size non-null immutable byte declaration
         * @param byteAlignment power-of-two bytes from 1 through 1L&lt;&lt;62
         * @throws NullPointerException if slot or size is null
         * @throws IllegalArgumentException if alignment is invalid
         */
        public WorkspaceEntry(WorkspaceSlot slot, ByteSize size, long byteAlignment) {
            this.slot = Objects.requireNonNull(slot, "slot");
            this.size = Objects.requireNonNull(size, "size");
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
            this.byteAlignment = byteAlignment;
        }

        /**
         * Returns the exact workspace-slot reference supplied at construction.
         *
         * @return the retained non-null workspace slot
         */
        public WorkspaceSlot slot() {
            return slot;
        }

        /**
         * Returns the exact required byte count without allocating or owning storage.
         *
         * @return the non-negative byte count; zero is valid
         * @throws IllegalStateException if this declaration is run-bound
         */
        public long byteSize() {
            return size.fixedByteSize().orElseThrow(
                    () -> new IllegalStateException("byte size requires run geometry"));
        }

        /**
         * Returns the exact required byte alignment without allocating or binding storage.
         *
         * @return the positive power-of-two byte alignment from {@code 1} through
         *     {@code 1L << 62}
         */
        public long byteAlignment() {
            return byteAlignment;
        }
    }
}
