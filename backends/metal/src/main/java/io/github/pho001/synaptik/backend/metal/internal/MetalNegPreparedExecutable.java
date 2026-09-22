package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * Immutable Runtime recipe for one selected shape-specialized Metal NEG route.
 *
 * <p>Selections are feeds in stable order followed by targets in stable order. Cold binding
 * validates live context-local buffer representations and byte extents, then creates a
 * route-specific bound invocation. MPSGraph retains direct slices of its run-owned native-address
 * workspace; the custom singleton retains direct typed input and output references and has no
 * workspace. Hot execution makes exactly one matching native call and performs no lookup, graph
 * inspection, route selection, cast, address marshalling, or collection allocation. The custom
 * native route writes the assigned output buffer directly without an explicit host-staging or
 * intermediate-copy step.</p>
 */
final class MetalNegPreparedExecutable extends PreparedExecutable {
    private final MetalMpsGraphExecutableResource mpsGraphResource;
    private final MetalNegKernelPipelineResource customResource;
    private final int inputCount;
    private final long[] requiredBytes;

    /**
     * Creates the immutable recipe from assigned feed and target plan positions.
     *
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param feedPlanIndices non-null stable feed buffer positions
     * @param targetPlanIndices non-null stable target buffer positions
     * @param resource non-null borrowed persistent executable resource owned by PreparedExecution
     * @param feedRequiredBytes non-null byte extents aligned with feeds
     * @param targetRequiredBytes non-null byte extents aligned with targets
     * @param workspacePlanIndex dense position of the exact assigned address workspace
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selection and geometry cardinalities disagree
     */
    MetalNegPreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            int[] feedPlanIndices,
            int[] targetPlanIndices,
            MetalMpsGraphExecutableResource resource,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes,
            int workspacePlanIndex) {
        super(memoryPlan, selections(feedPlanIndices, targetPlanIndices),
                List.of(new WorkspaceSelection(workspacePlanIndex)),
                accesses(feedPlanIndices.length, targetPlanIndices.length));
        this.mpsGraphResource = Objects.requireNonNull(resource, "resource");
        this.customResource = null;
        this.inputCount = feedPlanIndices.length;
        this.requiredBytes = new long[feedRequiredBytes.length + targetRequiredBytes.length];
        System.arraycopy(feedRequiredBytes, 0, requiredBytes, 0, feedRequiredBytes.length);
        System.arraycopy(targetRequiredBytes, 0, requiredBytes,
                feedRequiredBytes.length, targetRequiredBytes.length);
        if (feedPlanIndices.length != feedRequiredBytes.length
                || targetPlanIndices.length != targetRequiredBytes.length) {
            throw new IllegalArgumentException("Metal NEG selection geometry disagrees");
        }
    }

    /**
     * Creates the workspace-free custom singleton recipe.
     *
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param feedPlanIndex assigned input position
     * @param targetPlanIndex assigned output position
     * @param resource non-null borrowed custom pipeline owned by PreparedExecution
     * @throws NullPointerException if {@code memoryPlan} or {@code resource} is {@code null}
     * @throws IllegalArgumentException if either plan index is outside the memory plan
     */
    MetalNegPreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            int feedPlanIndex,
            int targetPlanIndex,
            MetalNegKernelPipelineResource resource) {
        super(memoryPlan,
                List.of(new BufferSelection(feedPlanIndex, 0),
                        new BufferSelection(targetPlanIndex, 0)),
                List.of(),
                List.of(BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        this.mpsGraphResource = null;
        this.customResource = Objects.requireNonNull(resource, "resource");
        this.inputCount = 1;
        this.requiredBytes = new long[] {resource.requiredBytes(), resource.requiredBytes()};
    }

    @Override
    protected boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation) {
        MetalDeviceContext context = customResource == null
                ? mpsGraphResource.context() : customResource.context();
        return representation instanceof MetalBufferRepresentation metal
                && metal.belongsTo(context)
                && metal.byteSize() >= requiredBytes[selectionIndex];
    }

    @Override
    protected boolean acceptsWorkspaceRepresentation(
            int selectionIndex, WorkspaceRepresentation representation) {
        return mpsGraphResource != null
                && representation instanceof AddressWorkspace workspace
                && workspace.belongsTo(mpsGraphResource.context())
                && workspace.pointerCount() == requiredBytes.length;
    }

    @Override
    protected BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations) {
        if (customResource != null) {
            var input = (MetalBufferRepresentation) bufferRepresentations[0];
            var output = (MetalBufferRepresentation) bufferRepresentations[1];
            if (input == output) {
                throw new IllegalArgumentException(
                        "Metal NEG input and output buffers must not alias");
            }
            return new CustomBoundInvocation(runState, customResource, input, output);
        }
        var workspace = (AddressWorkspace) workspaceRepresentations[0];
        for (int index = 0; index < inputCount; index++) {
            workspace.set(index, ((MetalBufferRepresentation) bufferRepresentations[index])
                    .executionHandle());
        }
        int outputCount = bufferRepresentations.length - inputCount;
        for (int index = 0; index < outputCount; index++) {
            MetalBufferRepresentation output = (MetalBufferRepresentation)
                    bufferRepresentations[inputCount + index];
            for (int input = 0; input < inputCount; input++) {
                if (bufferRepresentations[input] == output) {
                    throw new IllegalArgumentException(
                            "Metal NEG input and output buffers must not alias");
                }
            }
            workspace.set(inputCount + index, output.executionHandle());
        }
        MemorySegment inputs = workspace.segment().asSlice(0L, (long) inputCount * ADDRESS.byteSize());
        MemorySegment outputs = workspace.segment().asSlice(
                (long) inputCount * ADDRESS.byteSize(), (long) outputCount * ADDRESS.byteSize());
        return new MpsGraphBoundInvocation(
                runState, mpsGraphResource, inputCount, inputs, outputCount, outputs);
    }

    private static final class MpsGraphBoundInvocation extends BoundInvocation {
        private final MetalMpsGraphExecutableResource resource;
        private final int inputCount;
        private final MemorySegment inputs;
        private final int outputCount;
        private final MemorySegment outputs;

        private MpsGraphBoundInvocation(
                RunState runState,
                MetalMpsGraphExecutableResource resource,
                int inputCount,
                MemorySegment inputs,
                int outputCount,
                MemorySegment outputs) {
            super(runState);
            this.resource = resource;
            this.inputCount = inputCount;
            this.inputs = inputs;
            this.outputCount = outputCount;
            this.outputs = outputs;
        }

        @Override
        protected void executeBound() {
            resource.run(inputCount, inputs, outputCount, outputs);
        }
    }

    private static final class CustomBoundInvocation extends BoundInvocation {
        private final MetalNegKernelPipelineResource resource;
        @SuppressWarnings("unused")
        private final MetalBufferRepresentation input;
        @SuppressWarnings("unused")
        private final MetalBufferRepresentation output;
        private final MetalNativeApi.Handle inputHandle;
        private final MetalNativeApi.Handle outputHandle;

        private CustomBoundInvocation(
                RunState runState,
                MetalNegKernelPipelineResource resource,
                MetalBufferRepresentation input,
                MetalBufferRepresentation output) {
            super(runState);
            this.resource = resource;
            this.input = input;
            this.output = output;
            this.inputHandle = input.executionHandle();
            this.outputHandle = output.executionHandle();
        }

        @Override
        protected void executeBound() {
            resource.run(inputHandle, outputHandle);
        }
    }

    /** Per-run native-address workspace populated only during cold binding. */
    static final class AddressWorkspace implements WorkspaceRepresentation {
        private final MetalDeviceContext context;
        private final int pointerCount;
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment;
        private boolean closed;

        AddressWorkspace(MetalDeviceContext context, int pointerCount) {
            this.context = Objects.requireNonNull(context, "context");
            if (pointerCount <= 0) throw new IllegalArgumentException("pointerCount must be positive");
            this.pointerCount = pointerCount;
            this.segment = arena.allocate(ADDRESS, pointerCount);
        }

        synchronized boolean belongsTo(MetalDeviceContext expected) {
            return !closed && context == expected;
        }

        int pointerCount() { return pointerCount; }
        MemorySegment segment() { return segment; }

        synchronized void set(int index, MetalNativeApi.Handle handle) {
            if (closed) throw new IllegalStateException("Metal address workspace is closed");
            segment.setAtIndex(ADDRESS, index, handle.carrier());
        }

        @Override
        public synchronized void close() {
            if (closed) return;
            closed = true;
            arena.close();
        }
    }

    private static List<BufferSelection> selections(int[] feeds, int[] targets) {
        Objects.requireNonNull(feeds, "feedPlanIndices");
        Objects.requireNonNull(targets, "targetPlanIndices");
        var result = new ArrayList<BufferSelection>(feeds.length + targets.length);
        for (int value : feeds) result.add(new BufferSelection(value, 0));
        for (int value : targets) result.add(new BufferSelection(value, 0));
        return result;
    }

    private static List<BufferAccess> accesses(int inputs, int outputs) {
        var result = new ArrayList<BufferAccess>(inputs + outputs);
        for (int index = 0; index < inputs; index++) result.add(BufferAccess.READ_ONLY);
        for (int index = 0; index < outputs; index++) result.add(BufferAccess.WRITE_ONLY);
        return result;
    }
}
