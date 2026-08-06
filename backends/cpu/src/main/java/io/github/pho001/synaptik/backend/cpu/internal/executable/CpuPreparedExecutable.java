package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
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
 * One immutable partition-level CPU executable that strongly owns one generated artifact.
 * Cold binding validates and copies four direct segment references plus primitive start/end;
 * the hot call performs one state guard and one exact direct-handle invocation.
 */
public final class CpuPreparedExecutable extends PreparedExecutable {
    private final CpuGeneratedKernel artifact;
    private final CpuAccessPlan.Binding binding;

    /**
     * Creates the exact four-buffer partition recipe.
     * @param memoryPlan non-null exact prepared memory plan retained by the recipe
     * @param bufferSelections non-null ordered {@code a}, {@code b}, {@code c}, and output
     *     selections; copied by the superclass
     * @param artifact non-null generated unit artifact retained strongly for the executable lifetime
     * @param binding non-null checked compatible extent and primitive-bound binding
     * @throws NullPointerException if a required argument is {@code null}
     * @throws IllegalArgumentException if selections are invalid or do not contain exactly four
     *     buffers
     */
    public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections, CpuGeneratedKernel artifact,
            CpuAccessPlan.Binding binding) {
        super(memoryPlan, bufferSelections, List.of(), List.of(BufferAccess.READ_ONLY,
                BufferAccess.READ_ONLY, BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        if (bufferSelections.size() != 4) throw new IllegalArgumentException(
                "CPU fused executable requires exactly four buffer selections");
        this.artifact = Objects.requireNonNull(artifact, "artifact");
        this.binding = Objects.requireNonNull(binding, "binding");
    }

    /** Returns owned generated code.
     * @return the non-null strongly owned artifact */
    public CpuGeneratedKernel artifact() { return artifact; }
    /** Returns invocation geometry.
     * @return the non-null immutable cold-bound binding */
    public CpuAccessPlan.Binding binding() { return binding; }

    @Override protected boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != io.github.pho001.synaptik.model.datatype.DataType.FLOAT64) return false;
        var entry = memoryPlan().buffers().get(bufferSelection(selectionIndex).bufferIndex());
        if (cpu.byteSize() != entry.byteSize()) return false;
        MemorySegment segment = cpu.segment();
        return segment.address() % entry.byteAlignment() == 0
                && (selectionIndex < 3 || !segment.isReadOnly());
    }

    @Override protected boolean acceptsWorkspaceRepresentation(
            int selectionIndex, WorkspaceRepresentation representation) { return false; }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (workspaces.length != 0) throw new IllegalArgumentException("CPU fused unit uses no workspace");
        MemorySegment a = ((CpuBufferRepresentation) buffers[0]).segment();
        MemorySegment b = ((CpuBufferRepresentation) buffers[1]).segment();
        MemorySegment c = ((CpuBufferRepresentation) buffers[2]).segment();
        MemorySegment output = ((CpuBufferRepresentation) buffers[3]).segment();
        if (overlaps(a, output) || overlaps(b, output) || overlaps(c, output)) {
            throw new IllegalArgumentException("output must not overlap an input");
        }
        return new Invocation(state, artifact, a, b, c, output, binding.start(), binding.end());
    }

    private static boolean overlaps(MemorySegment left, MemorySegment right) {
        return left.asOverlappingSlice(right).isPresent();
    }

    private static final class Invocation extends BoundInvocation {
        private final CpuGeneratedKernel artifact;
        private final MemorySegment a;
        private final MemorySegment b;
        private final MemorySegment c;
        private final MemorySegment output;
        private final long start;
        private final long end;
        Invocation(RunState state, CpuGeneratedKernel artifact, MemorySegment a, MemorySegment b,
                MemorySegment c, MemorySegment output, long start, long end) {
            super(state);
            this.artifact = artifact;
            this.a = a; this.b = b; this.c = c; this.output = output;
            this.start = start; this.end = end;
        }
        @Override protected void executeBound() {
            try { artifact.entryPoint().invokeExact(a, b, c, output, start, end); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new IllegalStateException(
                    "generated CPU kernel invocation failed", failure); }
        }
    }
}
