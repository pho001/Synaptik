package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering.Facts;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuRecurrentScanKernel;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import java.util.Objects;

/**
 * Reusable direct scan recipe, with one immutable kernel and no retained mutable storage.
 * Cold binding checks exact typed spans, access and disjoint writes without reading values.
 * Bound execution repeats access preflight and validates every length after preceding producers,
 * before even padding or raw initial-state copies can write any scan result.
 */
public final class CpuRecurrentScanPreparedExecutable extends PreparedExecutable {
    private final Facts facts;
    private final CpuRecurrentScanKernel kernel;
    private final CpuConcurrencyBudget budget;

    /**
     * Constructs one post-assignment direct recipe; resources remain owned by each run.
     * @param plan non-null exact shared memory plan
     * @param buffers unique reads then every sibling output, in lowering order
     * @param workspace exact optional run-owned row scratch selection
     * @param facts non-null immutable checked occurrence facts
     * @param budget borrowed inline concurrency admission, or null when not configured
     * @throws IllegalArgumentException if selection counts disagree with exact scan requirements
     */
    public CpuRecurrentScanPreparedExecutable(PreparedMemoryPlan plan, List<BufferSelection> buffers,
            Optional<WorkspaceSelection> workspace, Facts facts, CpuConcurrencyBudget budget) {
        super(plan, buffers, workspace.stream().toList(), java.util.stream.IntStream.range(0, buffers.size())
                .mapToObj(p -> p < facts.readCount() ? BufferAccess.READ_ONLY : BufferAccess.WRITE_ONLY).toList());
        this.facts = Objects.requireNonNull(facts, "facts");
        this.budget = budget;
        kernel = new CpuRecurrentScanKernel(facts.geometry());
        if (buffers.size() != facts.boundaryValues().size()
                || workspace.isPresent() != (facts.geometry().scratchBytes() > 0))
            throw new IllegalArgumentException("recurrent selections disagree");
    }

    /**
     * Checks typed accessible aligned capacity and write permission, without reading values.
     * @param p boundary position in unique-read then sibling-output order
     * @param representation non-null selected run representation
     * @return whether the exact selected representation meets this boundary's immutable metadata
     */
    @Override protected boolean acceptsBufferRepresentation(int p, BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != facts.boundaryTypes().get(p)) return false;
        try {
            var s = cpu.segmentArgument().segment();
            int width = cpu.dataType().byteWidth();
            return s.byteSize() >= Math.multiplyExact(facts.spans().get(p), width)
                    && s.address() % width == 0 && (p < facts.readCount() || !s.isReadOnly());
        } catch (IllegalArgumentException | IllegalStateException failure) { return false; }
    }

    /**
     * Checks the exact row workspace without allocating or reading a gate value.
     * @param p sole workspace position, zero
     * @param representation non-null selected run workspace
     * @return whether size, alignment, access and write permission match the declared scratch
     */
    @Override protected boolean acceptsWorkspaceRepresentation(int p, WorkspaceRepresentation representation) {
        return representation instanceof CpuContiguousWorkspace cpu && cpu.isAccessible()
                && cpu.byteSize() == facts.geometry().scratchBytes() && cpu.byteAlignment() >= Float.BYTES
                && cpu.writableSegment().address() % Float.BYTES == 0 && !cpu.writableSegment().isReadOnly();
    }

    /**
     * Binds exact CPU segments and chooses one family call after metadata validation.
     * @param state exact open run owner retained by the invocation
     * @param buffers fresh compatible selected boundaries; never retained as a nominal hot array
     * @param workspaces fresh compatible optional row workspace
     * @return invocation retaining direct typed bindings, not runtime length values
     * @throws IllegalArgumentException if any write or scratch overlaps another selected span
     */
    @Override protected BoundInvocation bindCompatible(RunState state, BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        var typed = new CpuBufferRepresentation[buffers.length];
        var segments = new MemorySegment[buffers.length];
        for (int p = 0; p < buffers.length; p++) {
            typed[p] = (CpuBufferRepresentation) buffers[p];
            segments[p] = typed[p].segmentArgument().segment();
        }
        var scratchOwner = workspaces.length == 0 ? null : (CpuContiguousWorkspace) workspaces[0];
        MemorySegment scratch = scratchOwner == null ? null : scratchOwner.writableSegment();
        for (int left = 0; left < segments.length; left++) {
            for (int right = left + 1; right < segments.length; right++)
                if ((left >= facts.readCount() || right >= facts.readCount())
                        && overlaps(segments[left], segments[right]))
                    throw new IllegalArgumentException("recurrent writes must not overlap another boundary");
            if (scratch != null && overlaps(segments[left], scratch))
                throw new IllegalArgumentException("recurrent scratch must not overlap any boundary");
        }
        // Retain the caller's physical base, but never let initialization touch capacity tails.
        for (int p = 0; p < segments.length; p++) segments[p] = segments[p].asSlice(0,
                Math.multiplyExact(facts.spans().get(p), facts.boundaryTypes().get(p).byteWidth()));
        var g = facts.geometry();
        var roles = facts.rolePositions();
        MemorySegment x = segments[roles.get(0)], lengths = segments[roles.get(1)], h0 = segments[roles.get(2)];
        boolean cell = g.gates() == 4;
        int weight = cell ? 4 : 3, output = facts.readCount();
        MemorySegment c0 = cell ? segments[roles.get(3)] : null;
        MemorySegment w = segments[roles.get(weight)], u = segments[roles.get(weight + 1)];
        MemorySegment bias = g.bias() ? segments[roles.get(weight + 2)] : null;
        MemorySegment y = segments[output], h = segments[output + 1], c = cell ? segments[output + 2] : null;
        Runnable call = switch (g.kind()) {
            case RNN_TANH -> () -> kernel.rnn(x, lengths, h0, w, u, bias, y, h, scratch);
            case GRU_RESET_AFTER -> () -> kernel.gru(x, lengths, h0, w, u, bias, y, h, scratch);
            case LSTM -> () -> kernel.lstm(x, lengths, h0, c0, w, u, bias, y, h, c, scratch);
        };
        return new Invocation(state, typed, scratchOwner, lengths, call);
    }

    /**
     * Checks physical overlap while allowing empty spans.
     * @param a first accessible segment, never null
     * @param b second accessible segment, never null
     * @return whether the nonempty spans share physical bytes
     */
    private static boolean overlaps(MemorySegment a, MemorySegment b) {
        return a.byteSize() != 0 && b.byteSize() != 0 && a.asOverlappingSlice(b).isPresent();
    }

    /** Invocation-local direct bindings; only the enclosing immutable recipe is shared across runs. */
    private final class Invocation extends BoundInvocation {
        private final CpuBufferRepresentation[] buffers;
        private final CpuContiguousWorkspace scratch;
        private final MemorySegment lengths;
        private final Runnable call;
        /**
         * Retains invocation-local typed references without owning caller memory.
         * @param state exact open run state
         * @param buffers typed boundary owners used only for per-invocation access recheck
         * @param scratch run-owned row workspace, or null for an empty domain
         * @param lengths exact logical INT64 segment; values are read only during execution
         * @param call cold-selected direct family call over logical-size segments
         */
        private Invocation(RunState state, CpuBufferRepresentation[] buffers,
                CpuContiguousWorkspace scratch, MemorySegment lengths, Runnable call) {
            super(state); this.buffers = buffers; this.scratch = scratch; this.lengths = lengths; this.call = call;
        }
        /**
         * Acquires ordinary inline admission, rechecks access and all lengths, then performs writes.
         * @throws IllegalArgumentException if access changed or any row length lies outside [0,T]
         */
        @Override protected void executeBound() {
            CpuConcurrencyBudget.Lease lease = budget == null ? null : budget.acquire(1);
            try {
                for (int p = 0; p < buffers.length; p++)
                    if (!buffers[p].isAccessible())
                        throw new IllegalArgumentException("recurrent representation is no longer accessible: " + p);
                // Type, size, alignment, writability and overlap are immutable cold-bound facts.
                if (scratch != null && !scratch.isAccessible())
                    throw new IllegalArgumentException("recurrent scratch is no longer accessible");
                int time = facts.geometry().time();
                for (int row = 0; row < facts.geometry().batch(); row++) {
                    long value = lengths.getAtIndex(ValueLayout.JAVA_LONG, row);
                    if (value < 0 || value > time) throw new IllegalArgumentException(
                            "invalid recurrent length row=" + row + " value=" + value + " T=" + time);
                }
                call.run();
            } finally { if (lease != null) lease.close(); }
        }
    }
}
