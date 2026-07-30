package io.github.pho001.synaptik.runtime.memory;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;

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
 */
public record PreparedMemoryPlan(
        List<PreparedMemoryPlan.BufferEntry> buffers,
        List<PreparedMemoryPlan.WorkspaceEntry> workspaces) {
    /**
     * Creates an immutable snapshot of final slot geometry.
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
     *     in its respective list
     */
    public PreparedMemoryPlan(
            List<PreparedMemoryPlan.BufferEntry> buffers,
            List<PreparedMemoryPlan.WorkspaceEntry> workspaces) {
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
        }
        this.workspaces = List.copyOf(workspaces);
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
     * @param byteSize the exact required byte count; must be non-negative and may be zero
     * @param byteAlignment the exact required alignment in bytes; must be a positive power of two
     */
    public record BufferEntry(BufferSlot slot, long byteSize, long byteAlignment) {
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
            this.slot = Objects.requireNonNull(slot, "slot");
            if (byteSize < 0) {
                throw new IllegalArgumentException("byteSize must be non-negative");
            }
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
            this.byteSize = byteSize;
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
         */
        public long byteSize() {
            return byteSize;
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
     * @param byteSize the exact required byte count; must be non-negative and may be zero
     * @param byteAlignment the exact required alignment in bytes; must be a positive power of two
     */
    public record WorkspaceEntry(WorkspaceSlot slot, long byteSize, long byteAlignment) {
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
            this.slot = Objects.requireNonNull(slot, "slot");
            if (byteSize < 0) {
                throw new IllegalArgumentException("byteSize must be non-negative");
            }
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
            this.byteSize = byteSize;
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
         */
        public long byteSize() {
            return byteSize;
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
