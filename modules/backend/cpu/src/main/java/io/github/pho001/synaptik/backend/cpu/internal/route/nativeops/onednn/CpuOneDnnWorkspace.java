package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Run-owned scratch and native invocation-control objects acquired before cold binding.
 * A workspace is not shared by concurrent runs. Its concrete implementation retains the
 * persistent kernel/library lease and destroys invocation handles before releasing that lease.
 */
public abstract class CpuOneDnnWorkspace implements WorkspaceRepresentation {
    private final CpuOneDnnKernel kernel;
    private final CpuContiguousWorkspace scratchpad;
    private boolean closed;

    /**
     * Allocates the declared user scratch; concrete subclasses acquire invocation handles.
     *
     * @param kernel non-null persistent kernel borrowed under the subclass's retained lease
     * @throws NullPointerException if kernel is null
     * @throws IllegalArgumentException if its scratch geometry is invalid
     */
    protected CpuOneDnnWorkspace(CpuOneDnnKernel kernel) {
        this.kernel = Objects.requireNonNull(kernel, "kernel");
        scratchpad = CpuContiguousWorkspace.allocate(kernel.scratchpadBytes(), 64);
    }

    /**
     * Identifies the exact persistent kernel this invocation belongs to.
     *
     * @return non-null borrowed kernel, never a transferable close duty
     */
    public final CpuOneDnnKernel kernel() { return kernel; }
    /**
     * Reports declared user scratch, excluding native control objects and ingress copies.
     *
     * @return non-negative scratch byte count
     */
    public final long byteSize() { return scratchpad.byteSize(); }
    /**
     * Reports whether this run's scratch is live and accessible to the current thread.
     *
     * @return false after close or when scratch access is unavailable
     */
    public final boolean isAccessible() { return !closed && scratchpad.isAccessible(); }
    /**
     * Borrows the exact user-scratch span for cold native binding.
     *
     * @return non-null writable scratch segment owned by this workspace
     * @throws IllegalStateException if scratch is closed or inaccessible
     */
    public final MemorySegment writableSegment() { return scratchpad.writableSegment(); }

    /**
     * Updates already-created native data handles; allocates no native resources or data copies.
     *
     * @param inputs non-null descriptor-order array of live native F32 spans; borrowed until
     *     execution completes, with descriptor-relative origins already applied once
     * @param output non-null live writable native F32 output, disjoint from inputs and scratch
     * @throws NullPointerException if an array or segment is null
     * @throws IllegalArgumentException if counts, spans, alignment, mutability or aliases disagree
     * @throws IllegalStateException if closed/inaccessible or native data-handle binding fails
     */
    public abstract void bind(MemorySegment[] inputs, MemorySegment output);

    /**
     * Executes fixed native handles and waits synchronously without allocation or selection.
     * The caller holds the shared CPU budget reservation and prevents concurrent mutation/closure.
     *
     * @throws IllegalStateException if unbound, closed/inaccessible, or native execute/wait fails
     */
    public abstract void execute();

    /**
     * Destroys invocation handles and releases the concrete workspace's retained kernel lease.
     *
     * @throws IllegalStateException if native destruction fails
     */
    protected abstract void closeNative();

    /**
     * Checks complete data/scratch disjointness before changing any native data handle.
     *
     * @param inputs non-null descriptor-order live input spans, borrowing caller storage
     * @param output non-null writable output span, borrowing caller storage
     * @throws NullPointerException if an array or segment is null
     * @throws IllegalArgumentException if counts, native span geometry, access or aliasing is invalid
     */
    protected final void validateBindings(MemorySegment[] inputs, MemorySegment output) {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(output, "output");
        var spec = kernel.spec();
        if (inputs.length != spec.inputs().size()) {
            throw new IllegalArgumentException("oneDNN input count disagrees");
        }
        requireData(output, spec.output().byteSize(), true);
        MemorySegment scratch = writableSegment();
        if (scratch.asOverlappingSlice(output).isPresent()) {
            throw new IllegalArgumentException("oneDNN output must not overlap scratch");
        }
        for (int index = 0; index < inputs.length; index++) {
            MemorySegment input = Objects.requireNonNull(inputs[index], "input");
            requireData(input, spec.inputs().get(index).byteSize(), false);
            if (output.asOverlappingSlice(input).isPresent()
                    || scratch.asOverlappingSlice(input).isPresent()) {
                throw new IllegalArgumentException("oneDNN writes must not overlap an input");
            }
        }
    }

    private static void requireData(MemorySegment segment, long bytes, boolean writable) {
        if (!segment.isNative() || !segment.scope().isAlive()
                || !segment.isAccessibleBy(Thread.currentThread()) || segment.byteSize() < bytes
                || segment.address() % Float.BYTES != 0 || writable && segment.isReadOnly()) {
            throw new IllegalArgumentException("incompatible oneDNN native data span");
        }
    }

    /**
     * Guards workspace operations without allocating or reacquiring resources.
     *
     * @throws IllegalStateException if closed or inaccessible to the current thread
     */
    protected final void requireOpen() {
        if (!isAccessible()) throw new IllegalStateException("oneDNN workspace is closed or inaccessible");
    }

    /**
     * Idempotently releases invocation-native resources, retained leases and user scratch.
     * Scratch cleanup still runs after a native cleanup failure.
     *
     * @throws IllegalStateException if native destruction or scratch cleanup fails
     */
    @Override public final void close() {
        if (closed) return;
        closed = true;
        Throwable failure = null;
        try { closeNative(); } catch (RuntimeException | Error problem) { failure = problem; }
        try { scratchpad.close(); } catch (RuntimeException | Error problem) {
            if (failure == null) failure = problem;
            else if (problem != failure) failure.addSuppressed(problem);
        }
        if (failure != null) throw CpuOneDnnLibrary.propagate(failure);
    }
}
