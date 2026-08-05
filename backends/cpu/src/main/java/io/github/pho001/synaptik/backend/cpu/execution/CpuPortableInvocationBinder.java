package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.invoke.MethodHandle;

/**
 * Cold family-owned construction seam for one signature-specific direct invocation.
 *
 * <p>A family implementation copies the exact handle and required direct carrier, segment,
 * workspace, worker, and primitive range fields into its invocation. It must not adapt the
 * handle, retain either supplied array, acquire resource ownership, or defer representation and
 * argument classification to the hot call.</p>
 */
@FunctionalInterface
interface CpuPortableInvocationBinder {
    /**
     * Copies already checked direct fields into one invocation associated with {@code runState}.
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
     * @return non-null signature-specific invocation associated with the exact supplied run state
     *     and retaining direct typed fields
     */
    BoundInvocation bind(
            RunState runState,
            MethodHandle entryPoint,
            CpuKernelSpecialization specialization,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);
}
