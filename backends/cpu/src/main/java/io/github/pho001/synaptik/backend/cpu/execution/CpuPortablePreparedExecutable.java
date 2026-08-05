package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Objects;

/**
 * Immutable portable CPU Runtime recipe retaining a generated artifact and typed cold binder.
 *
 * <p>The recipe strongly retains its generated artifact and exact direct entry handle, while the
 * worker group is borrowed from CPU composition and must outlive binding and invocation. One
 * instance may bind concurrently to distinct open {@link RunState} objects. Each family-owned
 * bound invocation remains associated with one state and is not safe to race with that state's
 * closure. The recipe owns no physical representation or close lifecycle.</p>
 */
final class CpuPortablePreparedExecutable extends CpuPreparedExecutable {
    private final CpuGeneratedKernel generatedKernel;
    private final MethodHandle entryPoint;
    private final CpuKernelSpecialization specialization;
    private final CpuPreparedParallelConfiguration parallelConfiguration;
    private final CpuWorkerGroup workerGroup;
    private final CpuPortableInvocationBinder invocationBinder;

    /**
     * Creates one reusable portable CPU recipe after shared slot assignment.
     *
     * @param memoryPlan exact non-null shared plan
     * @param bufferSelections non-null ordered assigned buffer recipes
     * @param workspaceSelections non-null ordered assigned workspace recipes
     * @param bufferAccesses non-null accesses aligned with buffer selections
     * @param bufferDataTypes non-null data types aligned with buffer selections
     * @param generatedKernel exact non-null strongly retained generated artifact
     * @param parallelConfiguration exact non-null prepared range configuration
     * @param workerGroup borrowed non-null open worker group
     * @param invocationBinder non-null family-owned typed cold binder
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if specialization, selection, or worker facts disagree
     * @throws IllegalStateException if the worker group is closed
     */
    CpuPortablePreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections,
            List<WorkspaceSelection> workspaceSelections,
            List<BufferAccess> bufferAccesses,
            List<DataType> bufferDataTypes,
            CpuGeneratedKernel generatedKernel,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup,
            CpuPortableInvocationBinder invocationBinder) {
        super(memoryPlan, bufferSelections, workspaceSelections, bufferAccesses, bufferDataTypes);
        this.generatedKernel = Objects.requireNonNull(generatedKernel, "generatedKernel");
        this.entryPoint = Objects.requireNonNull(generatedKernel.entryPoint(), "entryPoint");
        if (entryPoint != generatedKernel.entryPoint()) {
            throw new IllegalArgumentException("generated kernel entry point must be stable");
        }
        this.specialization = generatedKernel.specialization();
        if (specialization.arguments().size() != bufferSelections.size()) {
            throw new IllegalArgumentException(
                    "specialization argument count must equal buffer selection count "
                            + bufferSelections.size());
        }
        this.parallelConfiguration =
                Objects.requireNonNull(parallelConfiguration, "parallelConfiguration");
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        if (workerGroup.workerCount() != parallelConfiguration.workerCount()) {
            throw new IllegalArgumentException(
                    "worker group count does not match prepared parallel configuration");
        }
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
        this.invocationBinder = Objects.requireNonNull(invocationBinder, "invocationBinder");
    }

    /**
     * Returns the artifact whose reachability keeps the hidden generated class available.
     *
     * @return exact non-null strongly retained generated artifact
     */
    CpuGeneratedKernel generatedKernel() { return generatedKernel; }
    /**
     * Returns the direct entry handle used by the family-owned binder.
     *
     * @return exact non-null direct entry handle retained independently from the artifact
     */
    MethodHandle entryPoint() { return entryPoint; }

    /**
     * Validates specialization carriers and parallel accessibility before one binder call.
     *
     * @param runState exact open state already checked by Runtime
     * @param bufferArguments fresh direct CPU arguments
     * @param workspaces fresh direct native workspaces
     * @return non-null family-owned signature-specific invocation
     * @throws IllegalArgumentException if argument count, selected carrier, baked array offset, or
     *     parallel segment/workspace accessibility disagrees with the specialization
     * @throws IllegalStateException if the borrowed worker group is closed
     */
    @Override
    protected BoundInvocation bindCpu(
            RunState runState,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces) {
        if (bufferArguments.length != specialization.arguments().size()) {
            throw new IllegalArgumentException("buffer argument count does not match specialization");
        }
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
        for (int index = 0; index < bufferArguments.length; index++) {
            validateCarrier(index, specialization.arguments().get(index), bufferArguments[index]);
        }
        if (specialization.executionMode().parallel()) {
            for (int index = 0; index < bufferArguments.length; index++) {
                if (bufferArguments[index] instanceof CpuBufferArgument.Segment segment
                        && !workerGroup.isAccessibleByEveryWorker(segment.segment())) {
                    throw new IllegalArgumentException(
                            "bufferArguments[" + index + "] is not accessible by every CPU worker");
                }
            }
            for (int index = 0; index < workspaces.length; index++) {
                if (!workerGroup.isAccessibleByEveryWorker(workspaces[index].segment())) {
                    throw new IllegalArgumentException(
                            "workspaces[" + index + "] is not accessible by every CPU worker");
                }
            }
        }
        return invocationBinder.bind(runState, entryPoint, specialization, parallelConfiguration,
                workerGroup, bufferArguments, workspaces);
    }

    private static void validateCarrier(
            int index, CpuKernelSpecialization.Argument expected, CpuBufferArgument actual) {
        boolean carrierMatches = switch (expected.carrier()) {
            case DOUBLE_ARRAY -> actual instanceof CpuBufferArgument.Doubles;
            case FLOAT_ARRAY -> actual instanceof CpuBufferArgument.Floats;
            case SHORT_ARRAY -> actual instanceof CpuBufferArgument.Shorts;
            case INT_ARRAY -> actual instanceof CpuBufferArgument.Ints;
            case LONG_ARRAY -> actual instanceof CpuBufferArgument.Longs;
            case BYTE_ARRAY -> actual instanceof CpuBufferArgument.Bytes;
            case MEMORY_SEGMENT -> actual instanceof CpuBufferArgument.Segment segment
                    && segment.dataType() == expected.dataType() && segment.byteOffset() == 0;
        };
        if (!carrierMatches) {
            throw new IllegalArgumentException(
                    "bufferArguments[" + index + "] does not match specialization carrier");
        }
        if (expected.carrier() != CpuKernelSpecialization.Carrier.MEMORY_SEGMENT
                && expected.byteOffsetBaked()
                && actual.byteOffset() != expected.bakedByteOffset()) {
            throw new IllegalArgumentException(
                    "bufferArguments[" + index + "] byte offset does not match specialization");
        }
    }
}
