package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.trace.id.TraceInvocationId;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Objects;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import static java.lang.foreign.ValueLayout.ADDRESS;


/**
 * Immutable Runtime recipe for one selected shape-specialized Metal supported-operation route.
 *
 * <p>Selections are feeds in stable order followed by targets in stable order. Cold binding
 * validates live context-local buffer representations and byte extents, then creates a
 * route-specific bound invocation. MPSGraph retains direct slices of its run-owned native-address
 * workspace and writes full dense represented-order targets, including authenticated composed
 * affine-view publications; canonical {@code CONTIGUOUS} targets use the ordinary path. The custom
 * singleton retains direct typed input and output references and
 * has no workspace. Hot execution makes exactly one matching native call and performs no lookup,
 * graph inspection, route selection, cast, address marshalling, or collection allocation. The
 * custom resource and MPSGraph resource are nominally distinct and cannot be interchanged.</p>
 */
final class MetalNegPreparedExecutable extends PreparedExecutable {
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private final MetalNegPreparationPlan preparationPlan;
    private final MetalMpsGraphExecutableResource mpsGraphResource;
    private final MetalNegKernelPipelineResource customResource;
    private final List<Optional<MetalPreparedSplatResource>> splatResources;
    private final int inputCount;
    private final long[] requiredBytes;

    /**
     * Creates the immutable recipe from assigned feed and target plan positions.
     *
     * @param preparationPlan exact non-null immutable backend analysis plan
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param feedPlanIndices non-null stable feed buffer positions
     * @param feedRepresentationIndices non-null owner positions aligned with feed buffers
     * @param targetPlanIndices non-null stable target buffer positions
     * @param targetRepresentationIndices non-null owner positions aligned with target buffers
     * @param resource non-null borrowed persistent executable resource owned by PreparedExecution
     * @param splatResources immutable source-owned splat resources aligned with feeds
     * @param feedRequiredBytes non-null byte extents aligned with feeds
     * @param targetRequiredBytes non-null byte extents aligned with targets
     * @param workspacePlanIndex dense position of the exact assigned address workspace
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if selection and geometry cardinalities disagree
     */
    MetalNegPreparedExecutable(
            MetalNegPreparationPlan preparationPlan,
            PreparedMemoryPlan memoryPlan,
            int[] feedPlanIndices,
            int[] feedRepresentationIndices,
            int[] targetPlanIndices,
            int[] targetRepresentationIndices,
            MetalMpsGraphExecutableResource resource,
            List<Optional<MetalPreparedSplatResource>> splatResources,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes,
            int workspacePlanIndex) {
        super(
                memoryPlan,
                selections(
                        feedPlanIndices,
                        feedRepresentationIndices,
                        targetPlanIndices,
                        targetRepresentationIndices),
                List.of(new WorkspaceSelection(workspacePlanIndex)),
                accesses(feedPlanIndices.length, targetPlanIndices.length));
        this.preparationPlan = Objects.requireNonNull(preparationPlan, "preparationPlan");
        this.mpsGraphResource = Objects.requireNonNull(resource, "resource");
        this.customResource = null;
        this.splatResources = List.copyOf(splatResources);
        this.inputCount = feedPlanIndices.length;
        this.requiredBytes = new long[feedRequiredBytes.length + targetRequiredBytes.length];
        System.arraycopy(feedRequiredBytes, 0, requiredBytes, 0, feedRequiredBytes.length);
        System.arraycopy(targetRequiredBytes, 0, requiredBytes,
                feedRequiredBytes.length, targetRequiredBytes.length);
        if (feedPlanIndices.length != feedRequiredBytes.length
                || feedPlanIndices.length != feedRepresentationIndices.length
                || feedPlanIndices.length != this.splatResources.size()
                || targetPlanIndices.length != targetRequiredBytes.length
                || targetPlanIndices.length != targetRepresentationIndices.length) {
            throw new IllegalArgumentException("Metal NEG selection geometry disagrees");
        }
        validateSplatResources();
    }

    /**
     * Creates the workspace-free custom singleton recipe.
     *
     * @param preparationPlan exact non-null immutable backend analysis plan
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param feedPlanIndex assigned input position
     * @param feedRepresentationIndex assigned Metal input representation position
     * @param targetPlanIndex assigned output position
     * @param targetRepresentationIndex assigned Metal output representation position
     * @param resource non-null borrowed custom pipeline owned by PreparedExecution
     * @param splatResources immutable source-owned splat resources aligned with the singleton feed
     * @throws NullPointerException if {@code memoryPlan} or {@code resource} is {@code null}
     * @throws IllegalArgumentException if either plan index is outside the memory plan
     */
    MetalNegPreparedExecutable(
            MetalNegPreparationPlan preparationPlan,
            PreparedMemoryPlan memoryPlan,
            int feedPlanIndex,
            int feedRepresentationIndex,
            int targetPlanIndex,
            int targetRepresentationIndex,
            MetalNegKernelPipelineResource resource,
            List<Optional<MetalPreparedSplatResource>> splatResources) {
        super(
                memoryPlan,
                List.of(
                        new BufferSelection(feedPlanIndex, feedRepresentationIndex),
                        new BufferSelection(targetPlanIndex, targetRepresentationIndex)),
                List.of(),
                List.of(BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        this.preparationPlan = Objects.requireNonNull(preparationPlan, "preparationPlan");
        this.mpsGraphResource = null;
        this.customResource = Objects.requireNonNull(resource, "resource");
        this.splatResources = List.copyOf(splatResources);
        this.inputCount = 1;
        this.requiredBytes = new long[] {resource.requiredBytes(), resource.requiredBytes()};
        if (this.splatResources.size() != 1) {
            throw new IllegalArgumentException("Metal custom splat-resource cardinality disagrees");
        }
        validateSplatResources();
    }

    /** @return the exact immutable backend plan retained for cold schedule assembly */
    MetalNegPreparationPlan preparationPlan() {
        return preparationPlan;
    }

    Optional<MetalPreparedSplatResource> splatResource(ValueId valueId) {
        Objects.requireNonNull(valueId, "valueId");
        int feed = preparationPlan.feedValueIds().indexOf(valueId);
        return feed < 0 ? Optional.empty() : splatResources.get(feed);
    }

    Optional<MetalBufferRepresentation.DenseAffinePublication> denseAffinePublication(
            ValueId valueId) {
        Objects.requireNonNull(valueId, "valueId");
        int targetPosition = preparationPlan.targetValueIds().indexOf(valueId);
        if (targetPosition < 0) {
            return Optional.empty();
        }
        int value = preparationPlan.targetValueIndices()[targetPosition];
        var descriptor = preparationPlan.descriptors().get(value);
        long byteSize = preparationPlan.targetRequiredBytes()[targetPosition];
        return preparationPlan.denseAffineProducerKind(
                        targetPosition, valueId, descriptor, byteSize)
                .map(producerKind -> new MetalBufferRepresentation.DenseAffinePublication(
                        this,
                        targetPosition,
                        valueId,
                        producerKind,
                        descriptor,
                        byteSize));
    }

    @Override
    protected boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation) {
        MetalDeviceContext context = customResource == null
                ? mpsGraphResource.context() : customResource.context();
        if (selectionIndex < inputCount
                && MetalPreparedSplatResource.isBinding(representation)) {
            Optional<io.github.pho001.synaptik.model.datatype.ScalarValue> splat =
                    preparationPlan.feedSplats().get(selectionIndex);
            return splat.isPresent()
                    && MetalPreparedSplatResource.exactReadBuffer(
                            representation,
                            context,
                            requiredBytes[selectionIndex],
                            splat.orElseThrow()) != null;
        }
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
            MetalBufferRepresentation input = readInput(0, bufferRepresentations[0]);
            var output = (MetalBufferRepresentation) bufferRepresentations[1];
            if (input == output) {
                throw new IllegalArgumentException(
                        "Metal NEG input and output buffers must not alias");
            }
            return new CustomBoundInvocation(
                    runState, preparationPlan, customResource, input, output);
        }
        var workspace = (AddressWorkspace) workspaceRepresentations[0];
        var inputBuffers = new MetalBufferRepresentation[inputCount];
        for (int index = 0; index < inputCount; index++) {
            inputBuffers[index] = readInput(index, bufferRepresentations[index]);
            workspace.set(index, inputBuffers[index].executionHandle());
        }
        int outputCount = bufferRepresentations.length - inputCount;
        for (int index = 0; index < outputCount; index++) {
            MetalBufferRepresentation output = (MetalBufferRepresentation)
                    bufferRepresentations[inputCount + index];
            for (MetalBufferRepresentation input : inputBuffers) {
                if (input == output) {
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
                runState, preparationPlan, mpsGraphResource, inputBuffers,
                inputCount, inputs, outputCount, outputs);
    }

    private MetalBufferRepresentation readInput(
            int index, BufferRepresentation representation) {
        if (representation instanceof MetalBufferRepresentation metal) {
            return metal;
        }
        return MetalPreparedSplatResource.exactReadBuffer(
                representation,
                customResource == null ? mpsGraphResource.context() : customResource.context(),
                requiredBytes[index],
                preparationPlan.feedSplats().get(index).orElseThrow());
    }

    private void validateSplatResources() {
        boolean[] sources = preparationPlan.feedSplatSources();
        if (sources.length != splatResources.size()) {
            throw new IllegalArgumentException("Metal splat-resource cardinality disagrees");
        }
        for (int index = 0; index < sources.length; index++) {
            if (sources[index] != splatResources.get(index).isPresent()) {
                throw new IllegalArgumentException(
                        "Metal splat-resource source ownership disagrees");
            }
        }
    }

    private static final class MpsGraphBoundInvocation extends BoundInvocation {
        private final MetalNegPreparationPlan preparationPlan;
        private final MetalMpsGraphExecutableResource resource;
        private final MetalBufferRepresentation[] inputBuffers;
        private final int inputCount;
        private final MemorySegment inputs;
        private final int outputCount;
        private final MemorySegment outputs;

        private MpsGraphBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalMpsGraphExecutableResource resource,
                MetalBufferRepresentation[] inputBuffers,
                int inputCount,
                MemorySegment inputs,
                int outputCount,
                MemorySegment outputs) {
            super(runState);
            this.preparationPlan = preparationPlan;
            this.resource = resource;
            this.inputBuffers = inputBuffers;
            this.inputCount = inputCount;
            this.inputs = inputs;
            this.outputCount = outputCount;
            this.outputs = outputs;
        }

        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId =
                    traceUnit == null ? null : traceUnit.beginInvocation();
            try {
                resource.run(inputCount, inputs, outputCount, outputs);
            } catch (MetalNativeApi.NativeFailure failure) {
                Error observerFailure = null;
                if (traceUnit != null) {
                    observerFailure = traceUnit.invocationFailed(invocationId, failure);
                }
                if (failure.status() != MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS) {
                    throw failure;
                }
                RuntimeException reproduced;
                try {
                    reproduced = reproduceIndexFailure();
                } catch (RuntimeException rescanFailure) {
                    failure.addSuppressed(rescanFailure);
                    throw failure;
                }
                if (reproduced != null) {
                    if (observerFailure != null) {
                        MetalTraceProducer.suppressObserverError(reproduced, observerFailure);
                    }
                    throw reproduced;
                }
                throw failure;
            } catch (RuntimeException | Error failure) {
                if (traceUnit != null) {
                    traceUnit.invocationFailed(invocationId, failure);
                }
                throw failure;
            }
            if (traceUnit != null) {
                traceUnit.invocationSucceeded(invocationId);
            }
        }

        private RuntimeException reproduceIndexFailure() {
            int[] feedIndices = preparationPlan.feedValueIndices();
            List<MetalMpsGraphProgram.Node> nodes = preparationPlan.graphProgram().nodes();
            for (MetalMpsGraphProgram.Node node : nodes) {
                int indexValue;
                long bound;
                String family;
                if (node.kind() == MetalMpsGraphProgram.NodeKind.GATHER) {
                    family = "GATHER";
                    indexValue = node.secondInputIndex();
                    bound = dataAxisExtent(node);
                } else if (node.kind() == MetalMpsGraphProgram.NodeKind.ONE_HOT) {
                    family = "ONE_HOT";
                    indexValue = node.firstInputIndex();
                    bound = node.attributeValues()[0];
                } else if (node.kind()
                        == MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS) {
                    family = "SCATTER_ELEMENTS";
                    indexValue = node.secondInputIndex();
                    bound = dataAxisExtent(node);
                } else {
                    continue;
                }
                int feedPosition = feedPosition(feedIndices, indexValue);
                if (feedPosition < 0) {
                    throw new IllegalStateException(
                            "Metal index value is not a stable executable feed");
                }
                long[] indexShape = preparationPlan.descriptors().get(indexValue)
                        .shape().toLongArray();
                long elements = preparationPlan.descriptors().get(indexValue)
                        .shape().knownElementCount().orElseThrow();
                long bytes = Math.multiplyExact(elements, Integer.BYTES);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment indices = arena.allocate(bytes, Integer.BYTES);
                    inputBuffers[feedPosition].download(0L, indices, 0L, bytes);
                    for (long ordinal = 0L; ordinal < elements; ordinal++) {
                        int value = indices.getAtIndex(NATIVE_INT, ordinal);
                        if (value < 0 || (long) value >= bound) {
                            String message = switch (family) {
                                case "GATHER" ->
                                    "GATHER index at logical position " + ordinal
                                            + " for data axis " + node.axis()
                                            + " is out of bounds: value=" + value
                                            + ", extent=" + bound;
                                case "ONE_HOT" ->
                                    "ONE_HOT index at logical position " + ordinal
                                            + " is out of bounds: value=" + value
                                            + ", depth=" + bound;
                                default ->
                                    "SCATTER_ELEMENTS index at logical position " + ordinal
                                            + " for data axis " + node.axis()
                                            + " is out of bounds: value=" + value
                                            + ", extent=" + bound;
                            };
                            return new IndexOutOfBoundsException(message);
                        }
                    }
                    if (node.kind()
                            == MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS) {
                        for (long later = 1L; later < elements; later++) {
                            for (long first = 0L; first < later; first++) {
                                if (sameScatterTarget(
                                        indices, later, first, indexShape, node.axis())) {
                                    return new IllegalArgumentException(
                                            "SCATTER_ELEMENTS duplicate target at logical"
                                                    + " update position " + later
                                                    + "; first addressed at logical update"
                                                    + " position " + first);
                                }
                            }
                        }
                    }
                }
            }
            return null;
        }

        private long dataAxisExtent(MetalMpsGraphProgram.Node node) {
            long[] dataShape = preparationPlan.descriptors()
                    .get(node.firstInputIndex()).shape().toLongArray();
            return dataShape[node.axis()];
        }

        private static boolean sameScatterTarget(
                MemorySegment indices,
                long left,
                long right,
                long[] shape,
                int selectedAxis) {
            for (int axis = 0; axis < shape.length; axis++) {
                if (axis == selectedAxis) {
                    if (indices.getAtIndex(NATIVE_INT, left)
                            != indices.getAtIndex(NATIVE_INT, right)) {
                        return false;
                    }
                } else if (coordinate(left, shape, axis)
                        != coordinate(right, shape, axis)) {
                    return false;
                }
            }
            return true;
        }

        private static long coordinate(long ordinal, long[] shape, int axis) {
            long stride = 1L;
            for (int dimension = shape.length; dimension-- > axis + 1;) {
                stride = Math.multiplyExact(stride, shape[dimension]);
            }
            return ordinal / stride % shape[axis];
        }

        private static int feedPosition(int[] feedIndices, int valueIndex) {
            for (int position = 0; position < feedIndices.length; position++) {
                if (feedIndices[position] == valueIndex) return position;
            }
            return -1;
        }
    }

    private static final class CustomBoundInvocation extends BoundInvocation {
        private final MetalNegPreparationPlan preparationPlan;
        private final MetalNegKernelPipelineResource resource;
        @SuppressWarnings("unused")
        private final MetalBufferRepresentation input;
        @SuppressWarnings("unused")
        private final MetalBufferRepresentation output;
        private final MetalNativeApi.Handle inputHandle;
        private final MetalNativeApi.Handle outputHandle;

        private CustomBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalNegKernelPipelineResource resource,
                MetalBufferRepresentation input,
                MetalBufferRepresentation output) {
            super(runState);
            this.preparationPlan =
                    Objects.requireNonNull(preparationPlan, "preparationPlan");
            this.resource = resource;
            this.input = input;
            this.output = output;
            this.inputHandle = input.executionHandle();
            this.outputHandle = output.executionHandle();
        }

        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId =
                    traceUnit == null ? null : traceUnit.beginInvocation();
            try {
                resource.run(inputHandle, outputHandle);
            } catch (RuntimeException | Error failure) {
                if (traceUnit != null) {
                    traceUnit.invocationFailed(invocationId, failure);
                }
                throw failure;
            }
            if (traceUnit != null) {
                traceUnit.invocationSucceeded(invocationId);
            }
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

    private static List<BufferSelection> selections(
            int[] feeds,
            int[] feedRepresentations,
            int[] targets,
            int[] targetRepresentations) {
        Objects.requireNonNull(feeds, "feedPlanIndices");
        Objects.requireNonNull(feedRepresentations, "feedRepresentationIndices");
        Objects.requireNonNull(targets, "targetPlanIndices");
        Objects.requireNonNull(targetRepresentations, "targetRepresentationIndices");
        if (feeds.length != feedRepresentations.length
                || targets.length != targetRepresentations.length) {
            throw new IllegalArgumentException("Metal NEG selection positions disagree");
        }
        var result = new ArrayList<BufferSelection>(feeds.length + targets.length);
        for (int index = 0; index < feeds.length; index++) {
            result.add(new BufferSelection(feeds[index], feedRepresentations[index]));
        }
        for (int index = 0; index < targets.length; index++) {
            result.add(new BufferSelection(targets[index], targetRepresentations[index]));
        }
        return result;
    }

    private static List<BufferAccess> accesses(int inputs, int outputs) {
        var result = new ArrayList<BufferAccess>(inputs + outputs);
        for (int index = 0; index < inputs; index++) result.add(BufferAccess.READ_ONLY);
        for (int index = 0; index < outputs; index++) result.add(BufferAccess.WRITE_ONLY);
        return result;
    }
}
