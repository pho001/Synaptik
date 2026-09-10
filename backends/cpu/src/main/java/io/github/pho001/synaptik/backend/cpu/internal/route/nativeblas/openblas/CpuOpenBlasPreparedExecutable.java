package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.model.datatype.DataType;
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
 * Immutable reusable prepared recipe for the narrow borrowed-provider OpenBLAS route.
 * Cold binding rechecks the borrowed provider, exact data types, native storage, alignment,
 * mutability, sizes, and non-overlap before retaining direct segments. A copied input is replaced
 * only by the already-declared run-owned native workspace. Hot execution performs exactly one
 * pre-typed SGEMM or DGEMM call with {@code alpha=1} and {@code beta=0}; it performs no route,
 * provider, representation, graph, slot, or cost decision and owns no resource.
 */
public final class CpuOpenBlasPreparedExecutable extends PreparedExecutable {
    private final CpuOpenBlasRoutePlan plan;
    private final CpuOpenBlasInvocation invocation;

    /**
     * Creates one exact native recipe from finalized dense selections.
     *
     * @param memoryPlan non-null exact shared memory plan retained by this recipe
     * @param buffers exactly three ordered left, right, and output selections
     * @param workspaces empty for direct execution or the one copy-workspace selection
     * @param plan non-null immutable route plan selected during cold analysis
     * @param invocation non-null borrowed invocation whose owner must outlive use
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if resource cardinality disagrees with the route plan
     */
    public CpuOpenBlasPreparedExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            CpuOpenBlasRoutePlan plan, CpuOpenBlasInvocation invocation) {
        super(memoryPlan, buffers, workspaces, List.of(BufferAccess.READ_ONLY,
                BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        this.plan = Objects.requireNonNull(plan, "plan");
        this.invocation = Objects.requireNonNull(invocation, "invocation");
        if (buffers.size() != 3 || workspaces.size() != (plan.materialization().isPresent() ? 1 : 0)) {
            throw new IllegalArgumentException("OpenBLAS resource cardinality disagrees");
        }
    }

    /** Returns the immutable selected route facts.
     * @return the retained non-null route plan */
    public CpuOpenBlasRoutePlan routePlan() { return plan; }

    @Override protected boolean acceptsBufferRepresentation(int index,
            BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != plan.dataType()) return false;
        try {
            CpuBufferArgument argument = cpu.argument();
            if (index == plan.copiedBoundaryPosition()) return true;
            return argument instanceof CpuBufferArgument.Segment segment
                    && segment.segment().isNative()
                    && (index != plan.outputBoundaryPosition() || !segment.readOnly());
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        return representation instanceof CpuContiguousWorkspace workspace
                && workspace.isAccessible();
    }

    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (!invocation.isOpen()) {
            throw new IllegalStateException("borrowed OpenBLAS provider is closed");
        }
        if (invocation.threadCount() != 1) {
            throw new IllegalStateException("borrowed OpenBLAS provider must remain single-threaded");
        }
        var arguments = new CpuBufferArgument[3];
        for (int index = 0; index < arguments.length; index++) {
            CpuBufferRepresentation cpu = (CpuBufferRepresentation) buffers[index];
            arguments[index] = cpu.argument();
            if (cpu.dataType() != plan.dataType()) {
                throw new IllegalArgumentException("OpenBLAS bound data type changed");
            }
        }
        MemorySegment left = effectiveSegment(arguments, workspaces,
                plan.leftBoundaryPosition());
        MemorySegment right = effectiveSegment(arguments, workspaces,
                plan.rightBoundaryPosition());
        MemorySegment output = effectiveSegment(arguments, workspaces,
                plan.outputBoundaryPosition());
        if (output.isReadOnly()) throw new IllegalArgumentException(
                "OpenBLAS output segment must be writable");
        validateSegment(left, matrixBytes(plan.m(), plan.k(), plan.dataType()), plan.dataType());
        validateSegment(right, matrixBytes(plan.k(), plan.n(), plan.dataType()), plan.dataType());
        validateSegment(output, matrixBytes(plan.m(), plan.n(), plan.dataType()), plan.dataType());
        if (overlaps(output, left) || overlaps(output, right)) {
            throw new IllegalArgumentException("OpenBLAS output must not overlap an input");
        }
        if (workspaces.length == 1) {
            MemorySegment workspace = ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
            for (CpuBufferArgument argument : arguments) {
                if (overlaps(workspace, segment(argument))) throw new IllegalArgumentException(
                        "OpenBLAS copy workspace must not overlap a selected buffer");
            }
        }
        return plan.dataType() == DataType.FLOAT32
                ? floatInvocation(state, left, right, output)
                : doubleInvocation(state, left, right, output);
    }

    private BoundInvocation floatInvocation(RunState state, MemorySegment left,
            MemorySegment right, MemorySegment output) {
        return new BoundInvocation(state) {
            @Override protected void executeBound() {
                invocation.sgemm(plan.m(), plan.n(), plan.k(), 1.0f,
                        left, right, 0.0f, output);
            }
        };
    }

    private BoundInvocation doubleInvocation(RunState state, MemorySegment left,
            MemorySegment right, MemorySegment output) {
        return new BoundInvocation(state) {
            @Override protected void executeBound() {
                invocation.dgemm(plan.m(), plan.n(), plan.k(), 1.0d,
                        left, right, 0.0d, output);
            }
        };
    }

    private MemorySegment effectiveSegment(CpuBufferArgument[] arguments,
            WorkspaceRepresentation[] workspaces, int position) {
        if (position == plan.copiedBoundaryPosition()) {
            return ((CpuContiguousWorkspace) workspaces[0]).writableSegment();
        }
        CpuBufferArgument argument = arguments[position];
        if (!(argument instanceof CpuBufferArgument.Segment direct)
                || !direct.segment().isNative()) {
            throw new IllegalArgumentException("OpenBLAS direct boundary must remain native");
        }
        return direct.segment();
    }

    private static void validateSegment(MemorySegment segment, long requiredBytes, DataType type) {
        if (!segment.isNative() || !segment.scope().isAlive()
                || !segment.isAccessibleBy(Thread.currentThread())
                || segment.byteSize() < requiredBytes
                || Math.floorMod(segment.address(), type.byteWidth()) != 0) {
            throw new IllegalArgumentException("OpenBLAS native segment geometry disagrees");
        }
    }

    private static long matrixBytes(long rows, long columns, DataType type) {
        return Math.multiplyExact(Math.multiplyExact(rows, columns), type.byteWidth());
    }

    private static boolean overlaps(MemorySegment left, MemorySegment right) {
        return left.byteSize() != 0 && right.byteSize() != 0
                && left.asOverlappingSlice(right).isPresent();
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
}
