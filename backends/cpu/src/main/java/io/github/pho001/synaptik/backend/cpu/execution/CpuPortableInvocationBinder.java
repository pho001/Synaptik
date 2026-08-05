package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.invoke.MethodHandle;

/**
 * Cold family-owned construction seam for one signature-specific direct node call.
 *
 * <p>A family implementation copies the exact handle and required direct carrier, segment,
 * workspace, worker, and primitive range fields into its node call. It must not adapt the
 * handle, retain either supplied array, acquire resource ownership, or defer representation and
 * argument classification to the hot call.</p>
 */
@FunctionalInterface
interface CpuPortableInvocationBinder {
    /**
     * Copies already checked direct fields into one guard-free node call. The supplied run state
     * identifies the containing partition invocation but is not guarded again by the child call.
     *
     * @param runState exact non-null open state being bound
     * @param entryPoint exact non-null direct generated entry handle
     * @param specialization exact non-null immutable generated signature
     * @param parallelConfiguration exact non-null prepared range recipe
     * @param workerGroup borrowed non-null open worker group
     * @param bufferArguments fresh non-null direct arguments; the returned invocation must not
     *     retain this array
     * @param workspaces fresh non-null direct workspaces; the returned invocation must not retain
     *     this array
     * @return non-null signature-specific guard-free node call retaining direct typed fields
     */
    CpuPortableKernelInvocation bind(
            RunState runState,
            MethodHandle entryPoint,
            CpuKernelSpecialization specialization,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);
}

/**
 * Guard-free direct node call executed only by one partition-level bound invocation.
 *
 * <p>The containing Runtime {@code BoundInvocation} performs the sole run-state-open check before
 * calling the ordered children. A child therefore owns no Runtime lifecycle, lookup, or resource
 * state.</p>
 */
@FunctionalInterface
interface CpuPortableKernelInvocation {
    /**
     * Executes one already-bound generated node kernel.
     *
     * @throws RuntimeException if generated execution reports an unchecked failure
     * @throws Error if generated execution reports an error
     */
    void execute();
}
