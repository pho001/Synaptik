package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
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
 * <p>Selections are the authenticated compact materialized values in stable slot order. Cold
 * binding validates context-local buffer representations and byte extents, then creates one
 * route-specific bound invocation. Ordinary MPSGraph and the shared exact custom whole-program
 * route retain direct slices of a run-owned native-address workspace and write direct assigned
 * targets. The custom-program workspace carries the stable compact materialized-slot table plus
 * target aliases; generated pointwise interiors have no declared selection. The dedicated
 * singleton-NEG resource retains direct typed input/output references and has no workspace. Hot
 * Java execution makes exactly one matching native call and performs no lookup, graph inspection,
 * address marshalling, collection allocation, retry, or fallback.</p>
 */
final class MetalNegPreparedExecutable extends PreparedExecutable {
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong NATIVE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private final MetalNegPreparationPlan preparationPlan;
    private final MetalMpsGraphExecutableResource mpsGraphResource;
    private final MetalNegKernelPipelineResource customResource;
    private final List<Optional<MetalPreparedSplatResource>> splatResources;
    private final int inputCount;
    private final int targetCount;
    private final int internalCount;
    private final long[] requiredBytes;
    private final InvocationByteTotals byteTotals;
    private final int splatCount;
    private final int workspaceCount;

    /**
     * Creates the immutable recipe from assigned feed and target plan positions.
     *
     * @param preparationPlan exact non-null immutable backend analysis plan
     * @param memoryPlan exact non-null shared prepared memory plan
     * @param feedPlanIndices non-null stable feed buffer positions
     * @param feedRepresentationIndices non-null owner positions aligned with feed buffers
     * @param targetPlanIndices non-null stable target buffer positions
     * @param targetRepresentationIndices non-null owner positions aligned with target buffers
     * @param internalPlanIndices non-null stable internal-value buffer positions
     * @param internalRepresentationIndices non-null owner positions aligned with internal values
     * @param resource non-null borrowed persistent executable resource owned by PreparedExecution
     * @param splatResources immutable source-owned splat resources aligned with feeds
     * @param feedRequiredBytes non-null byte extents aligned with feeds
     * @param targetRequiredBytes non-null byte extents aligned with targets
     * @param internalRequiredBytes non-null byte extents aligned with internal values
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
            int[] internalPlanIndices,
            int[] internalRepresentationIndices,
            MetalMpsGraphExecutableResource resource,
            List<Optional<MetalPreparedSplatResource>> splatResources,
            long[] feedRequiredBytes,
            long[] targetRequiredBytes,
            long[] internalRequiredBytes,
            int workspacePlanIndex) {
        super(
                memoryPlan,
                selections(
                        feedPlanIndices,
                        feedRepresentationIndices,
                        targetPlanIndices,
                        targetRepresentationIndices,
                        internalPlanIndices,
                        internalRepresentationIndices),
                List.of(new WorkspaceSelection(workspacePlanIndex)),
                accesses(
                        feedPlanIndices.length,
                        targetPlanIndices.length,
                        internalPlanIndices.length));
        this.preparationPlan = Objects.requireNonNull(preparationPlan, "preparationPlan");
        this.mpsGraphResource = Objects.requireNonNull(resource, "resource");
        this.customResource = null;
        this.splatResources = List.copyOf(splatResources);
        this.inputCount = feedPlanIndices.length;
        this.targetCount = targetPlanIndices.length;
        this.internalCount = internalPlanIndices.length;
        this.requiredBytes = new long[
                feedRequiredBytes.length
                        + targetRequiredBytes.length
                        + internalRequiredBytes.length];
        System.arraycopy(feedRequiredBytes, 0, requiredBytes, 0, feedRequiredBytes.length);
        System.arraycopy(targetRequiredBytes, 0, requiredBytes,
                feedRequiredBytes.length, targetRequiredBytes.length);
        System.arraycopy(internalRequiredBytes, 0, requiredBytes,
                feedRequiredBytes.length + targetRequiredBytes.length,
                internalRequiredBytes.length);
        if (feedPlanIndices.length != feedRequiredBytes.length
                || feedPlanIndices.length != feedRepresentationIndices.length
                || feedPlanIndices.length != this.splatResources.size()
                || targetPlanIndices.length != targetRequiredBytes.length
                || targetPlanIndices.length != targetRepresentationIndices.length
                || internalPlanIndices.length != internalRequiredBytes.length
                || internalPlanIndices.length != internalRepresentationIndices.length) {
            throw new IllegalArgumentException("Metal NEG selection geometry disagrees");
        }
        validateSplatResources();
        if (preparationPlan.traceUnit() == null) {
            this.byteTotals = null;
        } else {
            long pointerCount = preparationPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM
                    ? Math.addExact(
                            (long) preparationPlan.pointwiseFusionPlan()
                                    .materializedProgramValueIndices().length,
                            targetCount)
                    : requiredBytes.length;
            long workspaceBytes = Math.multiplyExact(pointerCount, ADDRESS.byteSize());
            this.byteTotals = byteTotals(
                    feedRequiredBytes,
                    targetRequiredBytes,
                    internalRequiredBytes,
                    this.splatResources,
                    workspaceBytes);
        }
        this.splatCount = presentCount(this.splatResources);
        this.workspaceCount = 1;
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
        this.targetCount = 1;
        this.internalCount = 0;
        this.requiredBytes = new long[] {resource.requiredBytes(), resource.requiredBytes()};
        if (this.splatResources.size() != 1) {
            throw new IllegalArgumentException("Metal custom splat-resource cardinality disagrees");
        }
        validateSplatResources();
        this.byteTotals = preparationPlan.traceUnit() == null
                ? null
                : byteTotals(
                        new long[] {resource.requiredBytes()},
                        new long[] {resource.requiredBytes()},
                        new long[0],
                        this.splatResources,
                        0L);
        this.splatCount = presentCount(this.splatResources);
        this.workspaceCount = 0;
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
        return preparationPlan.publicationProducerKind(
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
        int pointerCount = preparationPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM
                ? Math.addExact(
                        preparationPlan.pointwiseFusionPlan()
                                .materializedProgramValueIndices().length,
                        targetCount)
                : requiredBytes.length;
        return mpsGraphResource != null
                && representation instanceof AddressWorkspace workspace
                && workspace.belongsTo(mpsGraphResource.context())
                && workspace.pointerCount() == pointerCount;
    }

    @Override
    protected BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations) {
        if (customResource != null) {
            MetalBufferRepresentation input = readInput(0, bufferRepresentations[0]);
            var output = (MetalBufferRepresentation) bufferRepresentations[1];
            if (input.executionHandle().carrier().address()
                    == output.executionHandle().carrier().address()) {
                throw new IllegalArgumentException(
                        "Metal NEG input and output buffers must not alias");
            }
            return new CustomBoundInvocation(
                    runState,
                    preparationPlan,
                    customResource,
                    input,
                    output,
                    byteTotals,
                    splatCount,
                    workspaceCount);
        }
        if (preparationPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM) {
            return bindCustomProgram(runState, bufferRepresentations, workspaceRepresentations);
        }
        var workspace = (AddressWorkspace) workspaceRepresentations[0];
        var inputBuffers = new MetalBufferRepresentation[inputCount];
        for (int index = 0; index < inputCount; index++) {
            inputBuffers[index] = readInput(index, bufferRepresentations[index]);
            workspace.set(index, inputBuffers[index].executionHandle());
        }
        int outputCount = targetCount;
        var outputBuffers = new MetalBufferRepresentation[outputCount];
        for (int index = 0; index < outputCount; index++) {
            MetalBufferRepresentation output = (MetalBufferRepresentation)
                    bufferRepresentations[inputCount + index];
            long outputHandle = output.executionHandle().carrier().address();
            for (MetalBufferRepresentation input : inputBuffers) {
                if (input.executionHandle().carrier().address() == outputHandle) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph input and output buffers must not alias");
                }
            }
            for (int previous = 0; previous < index; previous++) {
                if (outputBuffers[previous].executionHandle().carrier().address()
                        == outputHandle) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph output buffers must not alias");
                }
            }
            outputBuffers[index] = output;
            workspace.set(inputCount + index, output.executionHandle());
        }
        MemorySegment inputs = workspace.segment().asSlice(0L, (long) inputCount * ADDRESS.byteSize());
        MemorySegment outputs = workspace.segment().asSlice(
                (long) inputCount * ADDRESS.byteSize(), (long) outputCount * ADDRESS.byteSize());
        var replayBuffers =
                new MetalBufferRepresentation[preparationPlan.descriptors().size()];
        int[] feedValues = preparationPlan.feedValueIndices();
        for (int index = 0; index < inputBuffers.length; index++) {
            replayBuffers[feedValues[index]] = inputBuffers[index];
        }
        return new MpsGraphBoundInvocation(
                runState, preparationPlan, mpsGraphResource, replayBuffers,
                inputCount, inputs, outputCount, outputs,
                byteTotals, splatCount, workspaceCount);
    }

    private BoundInvocation bindCustomProgram(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations) {
        var workspace = (AddressWorkspace) workspaceRepresentations[0];
        int[] programToSlot = preparationPlan.pointwiseFusionPlan().programToMaterializedSlot();
        int materializedCount =
                preparationPlan.pointwiseFusionPlan().materializedProgramValueIndices().length;
        var materializedBuffers = new MetalBufferRepresentation[materializedCount];
        var replayBuffers =
                new MetalBufferRepresentation[preparationPlan.valueIds().size()];
        int[] feedValues = preparationPlan.feedValueIndices();
        for (int index = 0; index < inputCount; index++) {
            MetalBufferRepresentation input = readInput(index, bufferRepresentations[index]);
            int programValue = feedValues[index];
            materializedBuffers[programToSlot[programValue]] = input;
            replayBuffers[programValue] = input;
        }
        int[] targetValues = preparationPlan.targetValueIndices();
        for (int index = 0; index < targetCount; index++) {
            int programValue = targetValues[index];
            MetalBufferRepresentation output = (MetalBufferRepresentation)
                    bufferRepresentations[inputCount + index];
            materializedBuffers[programToSlot[programValue]] = output;
            replayBuffers[programValue] = output;
        }
        int[] internalValues = preparationPlan.internalValueIndices();
        for (int index = 0; index < internalCount; index++) {
            int programValue = internalValues[index];
            MetalBufferRepresentation internal = (MetalBufferRepresentation)
                    bufferRepresentations[inputCount + targetCount + index];
            materializedBuffers[programToSlot[programValue]] = internal;
            replayBuffers[programValue] = internal;
        }
        for (int slot = 0; slot < materializedBuffers.length; slot++) {
            MetalBufferRepresentation buffer = Objects.requireNonNull(
                    materializedBuffers[slot], "materialized value buffer");
            long handle = buffer.executionHandle().carrier().address();
            for (int previous = 0; previous < slot; previous++) {
                if (materializedBuffers[previous].executionHandle().carrier().address() == handle) {
                    throw new IllegalArgumentException(
                            "Metal custom-program materialized value buffers must not alias");
                }
            }
            workspace.set(slot, buffer.executionHandle());
        }
        for (int target = 0; target < targetCount; target++) {
            workspace.set(materializedCount + target,
                    materializedBuffers[programToSlot[targetValues[target]]].executionHandle());
        }
        MemorySegment values = workspace.segment().asSlice(
                0L, (long) materializedCount * ADDRESS.byteSize());
        MemorySegment outputs = workspace.segment().asSlice(
                (long) materializedCount * ADDRESS.byteSize(),
                (long) targetCount * ADDRESS.byteSize());
        return new MpsGraphBoundInvocation(
                runState,
                preparationPlan,
                mpsGraphResource,
                replayBuffers,
                materializedCount,
                values,
                targetCount,
                outputs,
                byteTotals,
                splatCount,
                workspaceCount);

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

    private static InvocationByteTotals byteTotals(
            long[] inputBytes,
            long[] outputBytes,
            long[] internalBytes,
            List<? extends Optional<?>> splats,
            long workspaceBytes) {
        if (splats.size() != inputBytes.length || workspaceBytes < 0L) {
            throw new IllegalArgumentException("Metal invocation byte facts disagree");
        }
        long aggregateInput = aggregateBytes(inputBytes);
        long aggregateOutput = aggregateBytes(outputBytes);
        long aggregateInternal = aggregateBytes(internalBytes);
        long aggregateSplat = 0L;
        for (int index = 0; index < inputBytes.length; index++) {
            if (splats.get(index).isPresent()) {
                aggregateSplat = Math.addExact(aggregateSplat, inputBytes[index]);
            }
        }
        long aggregateRequired = Math.addExact(
                Math.addExact(aggregateInput, aggregateOutput),
                Math.addExact(aggregateInternal, workspaceBytes));
        return new InvocationByteTotals(
                aggregateInput,
                aggregateOutput,
                aggregateInternal,
                aggregateSplat,
                workspaceBytes,
                aggregateRequired);
    }

    private static long aggregateBytes(long[] requiredBytes) {
        long total = 0L;
        for (long bytes : requiredBytes) total = Math.addExact(total, bytes);
        return total;
    }

    private static int presentCount(List<? extends Optional<?>> values) {
        int count = 0;
        for (Optional<?> value : values) if (value.isPresent()) count++;
        return count;
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

    private record InvocationByteTotals(
            long input,
            long output,
            long internal,
            long splat,
            long workspace,
            long required) {
        private InvocationByteTotals {
            if (input < 0L || output < 0L || internal < 0L || splat < 0L
                    || workspace < 0L || required < 0L || splat > input
                    || required != Math.addExact(
                            Math.addExact(input, output),
                            Math.addExact(internal, workspace))) {
                throw new IllegalArgumentException("Metal invocation byte totals disagree");
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
        private final InvocationByteTotals byteTotals;
        private final int splatCount;
        private final int workspaceCount;

        private MpsGraphBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalMpsGraphExecutableResource resource,
                MetalBufferRepresentation[] inputBuffers,
                int inputCount,
                MemorySegment inputs,
                int outputCount,
                MemorySegment outputs,
                InvocationByteTotals byteTotals,
                int splatCount,
                int workspaceCount) {
            super(runState);
            this.preparationPlan = preparationPlan;
            this.resource = resource;
            this.inputBuffers = inputBuffers;
            this.inputCount = inputCount;
            this.inputs = inputs;
            this.outputCount = outputCount;
            this.outputs = outputs;
            this.byteTotals = byteTotals;
            this.splatCount = splatCount;
            this.workspaceCount = workspaceCount;
        }

        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId =
                    traceUnit == null ? null : traceUnit.beginInvocation();
            if (traceUnit != null) {
                traceUnit.invocationPlanned(
                        invocationId,
                        preparationPlan.feedValueIds().size(),
                        preparationPlan.targetValueIds().size(),
                        preparationPlan.internalValueIds().size(),
                        byteTotals.input(),
                        byteTotals.output(),
                        byteTotals.internal(),
                        byteTotals.splat(),
                        byteTotals.workspace(),
                        byteTotals.required(),
                        splatCount,
                        workspaceCount);
            }
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
            List<MetalMpsGraphProgram.Node> nodes = preparationPlan.graphProgram().nodes();
            for (MetalMpsGraphProgram.Node node : nodes) {
                int indexValue;
                String family;
                if (node.kind() == MetalMpsGraphProgram.NodeKind.GATHER) {
                    family = "GATHER";
                    indexValue = node.secondInputIndex();
                } else if (node.kind() == MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS) {
                    family = "GATHER_ELEMENTS";
                    indexValue = node.secondInputIndex();
                } else if (node.kind() == MetalMpsGraphProgram.NodeKind.GATHER_ND) {
                    family = "GATHER_ND";
                    indexValue = node.secondInputIndex();
                } else if (node.kind() == MetalMpsGraphProgram.NodeKind.ONE_HOT) {
                    family = "ONE_HOT";
                    indexValue = node.firstInputIndex();
                } else if (node.kind()
                        == MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS) {
                    family = "SCATTER_ELEMENTS";
                    indexValue = node.secondInputIndex();
                } else {
                    continue;
                }
                var indexDescriptor = preparationPlan.descriptors().get(indexValue);
                DataType indexType = indexDescriptor.dataType();
                int width = indexType.byteWidth();
                long[] indexShape = indexDescriptor.shape().toLongArray();
                long elements = indexDescriptor.shape().knownElementCount().orElseThrow();
                long bytes = Math.multiplyExact(elements, width);
                MetalBufferRepresentation indexBuffer = inputBuffers[indexValue];
                if (indexBuffer == null) {
                    throw new IllegalStateException(
                            "Metal index value has no stable executable buffer");
                }
                long[] dataShape = preparationPlan.descriptors()
                        .get(node.firstInputIndex()).shape().toLongArray();
                int batch = family.equals("GATHER_ND")
                        ? Math.toIntExact(node.attributeWords()[0]) : 0;
                long tuple = family.equals("GATHER_ND")
                        ? indexShape[indexShape.length - 1] : 0L;
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment indices = arena.allocate(bytes, width);
                    indexBuffer.download(0L, indices, 0L, bytes);
                    for (long ordinal = 0L; ordinal < elements; ordinal++) {
                        long value = readIndex(indices, ordinal, indexType);
                        int selectedAxis = family.equals("GATHER_ND")
                                ? Math.addExact(batch, Math.toIntExact(ordinal % tuple))
                                : node.axis();
                        long bound = family.equals("ONE_HOT")
                                ? node.attributeValues()[0] : dataShape[selectedAxis];
                        if (value < 0L || value >= bound) {
                            String message = family.equals("ONE_HOT")
                                    ? "ONE_HOT index at logical position " + ordinal
                                            + " is out of bounds: value=" + value
                                            + ", depth=" + bound
                                    : family + " index at logical position " + ordinal
                                            + " for data axis " + selectedAxis
                                            + " is out of bounds: value=" + value
                                            + ", extent=" + bound;
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

        private static long readIndex(
                MemorySegment indices, long ordinal, DataType dataType) {
            return switch (dataType) {
                case INT32 -> indices.getAtIndex(NATIVE_INT, ordinal);
                case INT64 -> indices.getAtIndex(NATIVE_LONG, ordinal);
                default -> throw new IllegalStateException(
                        "Metal index replay requires INT32 or INT64");
            };
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

        private final InvocationByteTotals byteTotals;
        private final int splatCount;
        private final int workspaceCount;
        private CustomBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalNegKernelPipelineResource resource,
                MetalBufferRepresentation input,
                MetalBufferRepresentation output,
                InvocationByteTotals byteTotals,
                int splatCount,
                int workspaceCount) {
            super(runState);
            this.preparationPlan =
                    Objects.requireNonNull(preparationPlan, "preparationPlan");
            this.resource = resource;
            this.input = input;
            this.output = output;
            this.inputHandle = input.executionHandle();
            this.outputHandle = output.executionHandle();
            this.byteTotals = byteTotals;
            this.splatCount = splatCount;
            this.workspaceCount = workspaceCount;
        }
        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId =
                    traceUnit == null ? null : traceUnit.beginInvocation();
            if (traceUnit != null) {
                traceUnit.invocationPlanned(
                        invocationId,
                        1,
                        1,
                        0,
                        byteTotals.input(),
                        byteTotals.output(),
                        byteTotals.internal(),
                        byteTotals.splat(),
                        byteTotals.workspace(),
                        byteTotals.required(),
                        splatCount,
                        workspaceCount);
            }
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
            int[] targetRepresentations,
            int[] internals,
            int[] internalRepresentations) {
        Objects.requireNonNull(feeds, "feedPlanIndices");
        Objects.requireNonNull(feedRepresentations, "feedRepresentationIndices");
        Objects.requireNonNull(targets, "targetPlanIndices");
        Objects.requireNonNull(targetRepresentations, "targetRepresentationIndices");
        Objects.requireNonNull(internals, "internalPlanIndices");
        Objects.requireNonNull(internalRepresentations, "internalRepresentationIndices");
        if (feeds.length != feedRepresentations.length
                || targets.length != targetRepresentations.length
                || internals.length != internalRepresentations.length) {
            throw new IllegalArgumentException("Metal selection positions disagree");
        }
        var result = new ArrayList<BufferSelection>(
                feeds.length + targets.length + internals.length);
        for (int index = 0; index < feeds.length; index++) {
            result.add(new BufferSelection(feeds[index], feedRepresentations[index]));
        }
        for (int index = 0; index < targets.length; index++) {
            result.add(new BufferSelection(targets[index], targetRepresentations[index]));
        }
        for (int index = 0; index < internals.length; index++) {
            result.add(new BufferSelection(internals[index], internalRepresentations[index]));
        }
        return result;
    }

    private static List<BufferAccess> accesses(int inputs, int outputs, int internals) {
        var result = new ArrayList<BufferAccess>(inputs + outputs + internals);
        for (int index = 0; index < inputs; index++) result.add(BufferAccess.READ_ONLY);
        for (int index = 0; index < outputs; index++) result.add(BufferAccess.WRITE_ONLY);
        for (int index = 0; index < internals; index++) result.add(BufferAccess.WRITE_ONLY);
        return result;
    }
}
