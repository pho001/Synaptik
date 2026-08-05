package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

/**
 * Guard-free direct generated call for one pointwise {@code ADD} node.
 *
 * <p>The containing partition invocation performs the only Runtime state guard. This child keeps
 * exact direct segment and method-handle fields and performs no graph, slot, storage, route, or
 * artifact lookup.</p>
 */
final class CpuPointwiseAddInvocation implements CpuPortableKernelInvocation {
    private final MethodHandle entryPoint;
    private final MemorySegment left;
    private final MemorySegment right;
    private final MemorySegment output;
    private final long elementCount;

    /**
     * @param entryPoint exact non-null direct generated handle with the pointwise ADD signature
     * @param left exact non-null readable left segment, borrowed for the containing run
     * @param right exact non-null readable right segment, borrowed for the containing run
     * @param output exact non-null writable output segment, borrowed for the containing run
     * @param elementCount non-negative number of flat elements
     * @throws NullPointerException if a reference is {@code null}, in declaration order
     * @throws IllegalArgumentException if {@code elementCount} is negative
     */
    CpuPointwiseAddInvocation(MethodHandle entryPoint, MemorySegment left, MemorySegment right,
            MemorySegment output, long elementCount) {
        this.entryPoint = Objects.requireNonNull(entryPoint, "entryPoint");
        this.left = Objects.requireNonNull(left, "left");
        this.right = Objects.requireNonNull(right, "right");
        this.output = Objects.requireNonNull(output, "output");
        if (elementCount < 0) throw new IllegalArgumentException(
                "elementCount must be non-negative");
        this.elementCount = elementCount;
    }

    /**
     * Invokes the exact generated segment signature without adaptation or dispatch.
     *
     * @throws RuntimeException if the generated call reports an unchecked failure
     * @throws Error if the generated call reports an error
     * @throws IllegalStateException if the exact method-handle invocation reports a checked
     *     throwable
     */
    @Override public void execute() {
        try {
            entryPoint.invokeExact(left, right, output, elementCount);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("generated pointwise ADD invocation failed", failure);
        }
    }
}
