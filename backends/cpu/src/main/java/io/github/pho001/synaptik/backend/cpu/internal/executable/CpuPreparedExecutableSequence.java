package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

/**
 * Exact two-unit Runtime recipe for one materialized Conv2d pointwise suffix.
 *
 * <p>The outer selections contain each external input and the intermediate and final outputs
 * exactly once. Runtime therefore performs one atomic executable validity transition. Cold
 * binding validates every selected CPU representation and all cross-unit read/write and
 * write/write aliases before binding both children. Hot execution invokes the direct Conv2d
 * child to completion, including any worker join, before invoking the direct pointwise child.
 * This type owns no scheduling, publication, validity mutation, resource, or general unit list.</p>
 */
public final class CpuPreparedExecutableSequence extends PreparedExecutable {
    private final CpuPreparedExecutable conv2d;
    private final CpuPreparedExecutable suffix;

    /**
     * Creates the sole admitted CPU-private two-unit composite.
     *
     * @param memoryPlan exact immutable shared plan
     * @param selections deduplicated external-read then intermediate/final-write selections
     * @param accesses aligned Runtime-facing access declarations
     * @param conv2d exact first child
     * @param suffix exact second child
     * @throws NullPointerException if a reference or list element is {@code null}
     * @throws IllegalArgumentException if plans, access counts, or child order disagree
     */
    public CpuPreparedExecutableSequence(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> selections, List<BufferAccess> accesses,
            CpuPreparedExecutable conv2d, CpuPreparedExecutable suffix) {
        super(memoryPlan, selections, List.of(), accesses);
        this.conv2d = Objects.requireNonNull(conv2d, "conv2d");
        this.suffix = Objects.requireNonNull(suffix, "suffix");
        var outer = new java.util.LinkedHashSet<BufferSelection>(selections);
        var children = new java.util.LinkedHashSet<BufferSelection>();
        var writes = new java.util.HashSet<BufferSelection>();
        collect(conv2d, children, writes);
        collect(suffix, children, writes);
        if (conv2d.memoryPlan() != memoryPlan || suffix.memoryPlan() != memoryPlan
                || outer.size() != selections.size() || !outer.equals(children)
                || java.util.stream.IntStream.range(0, selections.size()).anyMatch(index ->
                    (accesses.get(index) == BufferAccess.WRITE_ONLY)
                        != writes.contains(selections.get(index)))
                || writes.size() < 2
                || accesses.stream().anyMatch(access -> access == BufferAccess.READ_WRITE)) {
            throw new IllegalArgumentException("Conv2d sequence facts disagree");
        }
    }

    private static void collect(CpuPreparedExecutable child,
            java.util.Set<BufferSelection> selections, java.util.Set<BufferSelection> writes) {
        for (int index = 0; index < child.bufferSelectionCount(); index++) {
            BufferSelection selection = child.bufferSelection(index);
            selections.add(selection);
            if (child.accessBindings().get(index).plan().accessKind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.WRITE) {
                writes.add(selection);
            }
        }
    }

    /** Returns the first child without transferring ownership.
     * @return the non-null exact direct-Conv2d child retained by this sequence
     */
    public CpuPreparedExecutable conv2d() { return conv2d; }

    /** Returns the second child without transferring ownership.
     * @return the non-null exact materialized pointwise-suffix child retained by this sequence
     */
    public CpuPreparedExecutable suffix() { return suffix; }

    @Override protected boolean acceptsBufferRepresentation(int index,
            BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()) {
            return false;
        }
        boolean write = bufferAccess(index) == BufferAccess.WRITE_ONLY;
        try {
            return !write || !cpu.argument().readOnly();
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        return false;
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (workspaces.length != 0) {
            throw new IllegalArgumentException("Conv2d sequence has no workspace");
        }
        var arguments = new CpuBufferArgument[buffers.length];
        for (int i = 0; i < buffers.length; i++) {
            arguments[i] = ((CpuBufferRepresentation) buffers[i]).argument();
        }
        for (int left = 0; left < arguments.length; left++) {
            for (int right = left + 1; right < arguments.length; right++) {
                if (bufferAccess(left) == BufferAccess.READ_ONLY
                        && bufferAccess(right) == BufferAccess.READ_ONLY) continue;
                if (overlaps(arguments[left], arguments[right])) {
                    throw new IllegalArgumentException(
                            "Conv2d sequence outputs must not overlap another selected span");
                }
            }
        }
        BoundInvocation first = conv2d.bind(state);
        BoundInvocation second = suffix.bind(state);
        return new SequenceInvocation(state, first, second);
    }

    private static boolean overlaps(CpuBufferArgument left, CpuBufferArgument right) {
        if (left.byteSize() == 0 || right.byteSize() == 0) return false;
        MemorySegment a = segment(left);
        MemorySegment b = segment(right);
        return a.asOverlappingSlice(b).isPresent();
    }

    private static MemorySegment segment(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Floats value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Shorts value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Ints value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Longs value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Bytes value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }

    private static final class SequenceInvocation extends BoundInvocation {
        private final BoundInvocation first;
        private final BoundInvocation second;

        private SequenceInvocation(RunState state, BoundInvocation first,
                BoundInvocation second) {
            super(state);
            this.first = first;
            this.second = second;
        }

        @Override protected void executeBound() {
            first.execute();
            second.execute();
        }
    }
}
