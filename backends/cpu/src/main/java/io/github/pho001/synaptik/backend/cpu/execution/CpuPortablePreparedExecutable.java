package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable portable CPU Runtime recipe for an ordered generated node-kernel sequence.
 *
 * <p>Shared partition buffers are selected once. Each kernel recipe strongly retains its exact
 * generated artifact and maps its direct argument positions into those partition selections.
 * Cold binding creates guard-free direct node calls; one partition-level bound invocation owns
 * the sole run-state guard and invokes them in node order.</p>
 */
final class CpuPortablePreparedExecutable extends CpuPreparedExecutable {
    /**
     * One finalized generated node recipe in partition order.
     *
     * @param generatedKernel exact non-null generated artifact retained strongly
     * @param bufferArgumentIndices non-null mapping from kernel argument positions to
     *     partition-level selected-buffer positions; copied defensively
     * @param workspaceIndices non-null mapping from kernel workspace positions to
     *     partition-level selected-workspace positions; copied defensively
     * @param invocationBinder non-null family-owned direct child-call binder
     */
    record KernelRecipe(
            CpuGeneratedKernel generatedKernel,
            int[] bufferArgumentIndices,
            int[] workspaceIndices,
            CpuPortableInvocationBinder invocationBinder) {
        /**
         * Validates and snapshots one finalized node recipe.
         *
         * @param generatedKernel exact non-null generated artifact retained strongly
         * @param bufferArgumentIndices non-null mapping from kernel argument positions to
         *     partition-level selected-buffer positions; copied defensively
         * @param workspaceIndices non-null mapping from kernel workspace positions to
         *     partition-level selected-workspace positions; copied defensively
         * @param invocationBinder non-null family-owned direct child-call binder
         * @throws NullPointerException if a reference is {@code null}, in declaration order
         * @throws IllegalArgumentException if the buffer mapping does not match the generated
         *     signature or either mapping contains a negative position
         */
        KernelRecipe {
            Objects.requireNonNull(generatedKernel, "generatedKernel");
            Objects.requireNonNull(bufferArgumentIndices, "bufferArgumentIndices");
            Objects.requireNonNull(workspaceIndices, "workspaceIndices");
            Objects.requireNonNull(invocationBinder, "invocationBinder");
            bufferArgumentIndices = bufferArgumentIndices.clone();
            workspaceIndices = workspaceIndices.clone();
            if (bufferArgumentIndices.length != generatedKernel.specialization().arguments().size()) {
                throw new IllegalArgumentException(
                        "buffer argument index count must equal specialization argument count");
            }
            for (int index : bufferArgumentIndices) {
                if (index < 0) throw new IllegalArgumentException(
                        "buffer argument indices must be non-negative");
            }
            for (int index : workspaceIndices) {
                if (index < 0) throw new IllegalArgumentException(
                        "workspace indices must be non-negative");
            }
        }

        /**
         * Returns an isolated snapshot of the buffer-position mapping.
         *
         * @return non-null cloned buffer-position mapping
         */
        @Override public int[] bufferArgumentIndices() { return bufferArgumentIndices.clone(); }

        /**
         * Returns an isolated snapshot of the workspace-position mapping.
         *
         * @return non-null cloned workspace-position mapping
         */
        @Override public int[] workspaceIndices() { return workspaceIndices.clone(); }
    }

    private final List<KernelRecipe> kernelRecipes;
    private final CpuPreparedParallelConfiguration parallelConfiguration;
    private final CpuWorkerGroup workerGroup;

    /**
     * Creates one reusable ordered partition recipe after shared slot assignment.
     *
     * @param memoryPlan exact non-null shared plan
     * @param bufferSelections non-null unique ordered partition buffer selections
     * @param workspaceSelections non-null unique ordered partition workspace selections
     * @param bufferAccesses non-null union accesses aligned with buffer selections
     * @param bufferDataTypes non-null data types aligned with buffer selections
     * @param kernelRecipes non-null non-empty ordered finalized node recipes
     * @param parallelConfiguration exact non-null prepared range configuration
     * @param workerGroup borrowed non-null open worker group
     * @throws NullPointerException if a required reference or indexed recipe is null
     * @throws IllegalArgumentException if recipe indices or worker facts disagree
     * @throws IllegalStateException if the borrowed worker group is closed
     */
    CpuPortablePreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections,
            List<WorkspaceSelection> workspaceSelections,
            List<BufferAccess> bufferAccesses,
            List<DataType> bufferDataTypes,
            List<KernelRecipe> kernelRecipes,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup) {
        super(memoryPlan, bufferSelections, workspaceSelections, bufferAccesses, bufferDataTypes);
        Objects.requireNonNull(kernelRecipes, "kernelRecipes");
        var copied = new ArrayList<KernelRecipe>(kernelRecipes.size());
        for (int index = 0; index < kernelRecipes.size(); index++) {
            KernelRecipe recipe = Objects.requireNonNull(
                    kernelRecipes.get(index), "kernelRecipes[" + index + "]");
            for (int bufferIndex : recipe.bufferArgumentIndices) {
                if (bufferIndex >= bufferSelections.size()) throw new IllegalArgumentException(
                        "kernelRecipes[" + index + "] buffer index is out of range");
            }
            for (int workspaceIndex : recipe.workspaceIndices) {
                if (workspaceIndex >= workspaceSelections.size()) throw new IllegalArgumentException(
                        "kernelRecipes[" + index + "] workspace index is out of range");
            }
            copied.add(recipe);
        }
        if (copied.isEmpty()) throw new IllegalArgumentException("kernelRecipes must not be empty");
        this.kernelRecipes = List.copyOf(copied);
        this.parallelConfiguration =
                Objects.requireNonNull(parallelConfiguration, "parallelConfiguration");
        this.workerGroup = Objects.requireNonNull(workerGroup, "workerGroup");
        if (workerGroup.workerCount() != parallelConfiguration.workerCount()) {
            throw new IllegalArgumentException(
                    "worker group count does not match prepared parallel configuration");
        }
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
    }

    /**
     * Creates a one-kernel compatibility recipe for the synthetic CPU-0004 path.
     *
     * @param memoryPlan exact non-null shared plan
     * @param bufferSelections non-null ordered buffer selections
     * @param workspaceSelections non-null ordered workspace selections
     * @param bufferAccesses non-null accesses aligned with buffer selections
     * @param bufferDataTypes non-null data types aligned with buffer selections
     * @param generatedKernel exact non-null generated artifact retained strongly
     * @param parallelConfiguration exact non-null prepared range configuration
     * @param workerGroup borrowed non-null open worker group
     * @param invocationBinder non-null family-owned direct binder
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if the shared selections, generated signature, or worker
     *     facts disagree
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
        this(memoryPlan, bufferSelections, workspaceSelections, bufferAccesses, bufferDataTypes,
                List.of(new KernelRecipe(generatedKernel,
                        sequential(bufferSelections.size()), sequential(workspaceSelections.size()),
                        invocationBinder)), parallelConfiguration, workerGroup);
    }

    private static int[] sequential(int size) {
        int[] values = new int[size];
        for (int index = 0; index < size; index++) values[index] = index;
        return values;
    }

    /**
     * Returns the finalized node recipes in execution order.
     *
     * @return immutable non-empty node-recipe list in partition order
     */
    List<KernelRecipe> kernelRecipes() { return kernelRecipes; }

    /**
     * Returns the first generated kernel for the one-kernel compatibility path.
     *
     * @return exact non-null first generated artifact
     */
    CpuGeneratedKernel generatedKernel() { return kernelRecipes.getFirst().generatedKernel(); }

    /**
     * Returns the first generated kernel's direct entry point.
     *
     * @return exact non-null direct method handle
     */
    java.lang.invoke.MethodHandle entryPoint() { return generatedKernel().entryPoint(); }

    /**
     * Cold-binds every node recipe from checked partition-level direct arguments.
     *
     * @param runState exact non-null open run state supplied by the Runtime binding boundary
     * @param bufferArguments non-null fresh direct partition buffer arguments
     * @param workspaces non-null fresh direct partition workspaces
     * @return non-null partition invocation with one Runtime guard and ordered guard-free child
     *     calls
     * @throws NullPointerException if a family binder returns {@code null}
     * @throws IllegalArgumentException if a mapped carrier, offset, or parallel-access fact is
     *     incompatible with its node specialization
     * @throws IllegalStateException if the borrowed worker group is closed
     */
    @Override
    protected BoundInvocation bindCpu(
            RunState runState,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces) {
        if (workerGroup.isClosed()) throw new IllegalStateException("CPU worker group is closed");
        var invocations = new ArrayList<CpuPortableKernelInvocation>(kernelRecipes.size());
        for (int kernelIndex = 0; kernelIndex < kernelRecipes.size(); kernelIndex++) {
            KernelRecipe recipe = kernelRecipes.get(kernelIndex);
            var specialization = recipe.generatedKernel().specialization();
            int[] bufferIndices = recipe.bufferArgumentIndices;
            var kernelBuffers = new CpuBufferArgument[bufferIndices.length];
            for (int index = 0; index < bufferIndices.length; index++) {
                CpuBufferArgument argument = bufferArguments[bufferIndices[index]];
                validateCarrier(kernelIndex, index, specialization.arguments().get(index), argument);
                if (specialization.executionMode().parallel()
                        && argument instanceof CpuBufferArgument.Segment segment
                        && !workerGroup.isAccessibleByEveryWorker(segment.segment())) {
                    throw new IllegalArgumentException(kernelIndex == 0
                            ? "bufferArguments[" + index
                                    + "] is not accessible by every CPU worker"
                            : "kernelRecipes[" + kernelIndex + "].bufferArguments[" + index
                                    + "] is not accessible by every CPU worker");
                }
                kernelBuffers[index] = argument;
            }
            int[] workspaceIndices = recipe.workspaceIndices;
            var kernelWorkspaces = new CpuNativeWorkspace[workspaceIndices.length];
            for (int index = 0; index < workspaceIndices.length; index++) {
                CpuNativeWorkspace workspace = workspaces[workspaceIndices[index]];
                if (specialization.executionMode().parallel()
                        && !workerGroup.isAccessibleByEveryWorker(workspace.segment())) {
                    throw new IllegalArgumentException(kernelIndex == 0
                            ? "workspaces[" + index + "] is not accessible by every CPU worker"
                            : "kernelRecipes[" + kernelIndex + "].workspaces[" + index
                                    + "] is not accessible by every CPU worker");
                }
                kernelWorkspaces[index] = workspace;
            }
            invocations.add(Objects.requireNonNull(recipe.invocationBinder().bind(
                    runState, recipe.generatedKernel().entryPoint(), specialization,
                    parallelConfiguration, workerGroup, kernelBuffers, kernelWorkspaces),
                    "boundInvocation"));
        }
        if (invocations.size() == 1 && invocations.getFirst() instanceof BoundInvocation legacy) {
            return legacy;
        }
        return new PartitionInvocation(runState, invocations);
    }

    private static void validateCarrier(int kernelIndex, int argumentIndex,
            CpuKernelSpecialization.Argument expected, CpuBufferArgument actual) {
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
        String prefix = kernelIndex == 0
                ? "bufferArguments[" + argumentIndex + "]"
                : "kernelRecipes[" + kernelIndex + "].bufferArguments[" + argumentIndex + "]";
        if (!carrierMatches) throw new IllegalArgumentException(
                prefix + " does not match specialization carrier");
        if (expected.carrier() != CpuKernelSpecialization.Carrier.MEMORY_SEGMENT
                && expected.byteOffsetBaked()
                && actual.byteOffset() != expected.bakedByteOffset()) {
            throw new IllegalArgumentException(prefix + " byte offset does not match specialization");
        }
    }

    private static final class PartitionInvocation extends BoundInvocation {
        private final CpuPortableKernelInvocation[] invocations;

        PartitionInvocation(RunState runState, List<CpuPortableKernelInvocation> invocations) {
            super(runState);
            this.invocations = invocations.toArray(CpuPortableKernelInvocation[]::new);
        }

        @Override protected void executeBound() {
            for (CpuPortableKernelInvocation invocation : invocations) invocation.execute();
        }
    }
}
