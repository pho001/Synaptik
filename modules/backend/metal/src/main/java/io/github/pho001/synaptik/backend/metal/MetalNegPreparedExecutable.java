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
 * Java execution makes exactly one matching native call. Only synchronous failure reads a bounded
 * Scan validation record; it never replays a Scan body, inspects inactive payload or retries.</p>
 */
final class MetalNegPreparedExecutable extends PreparedExecutable {
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong NATIVE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private final MetalNegPreparationPlan preparationPlan;
    private final MetalProgramExecutableResource programResource;
    private final MetalNegKernelPipelineResource customResource;
    private final List<Optional<MetalPreparedSplatResource>> splatResources;
    private final int inputCount;
    private final int targetCount;
    private final int internalCount;
    private final long[] requiredBytes;
    private final InvocationByteTotals byteTotals;
    private final int workspaceCount;
    private final long[] numericalWorkspaceBytes;
    private final int workspacePointerCount;

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
            MetalProgramExecutableResource resource,
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
        this.programResource = Objects.requireNonNull(resource, "resource");
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
        this.numericalWorkspaceBytes = preparationPlan.numericalWorkspaceBytes();
        this.workspacePointerCount = Math.addExact(
                preparationPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM
                        ? Math.addExact(preparationPlan.pointwiseFusionPlan()
                                .materializedProgramValueIndices().length, targetCount)
                        : requiredBytes.length,
                numericalWorkspaceBytes.length);
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
        if (preparationPlan.traceUnit() == null
                || !preparationPlan.traceUnit().enabled()) {
            this.byteTotals = null;
        } else {
            long workspaceBytes = MetalProgramWorkspace.totalBytes(
                    workspacePointerCount, numericalWorkspaceBytes);
            this.byteTotals = byteTotals(
                    feedRequiredBytes,
                    targetRequiredBytes,
                    internalRequiredBytes,
                    workspaceBytes);
        }
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
        this.programResource = null;
        this.customResource = Objects.requireNonNull(resource, "resource");
        this.splatResources = List.copyOf(splatResources);
        this.inputCount = 1;
        this.targetCount = 1;
        this.internalCount = 0;
        this.requiredBytes = new long[] {resource.requiredBytes(), resource.requiredBytes()};
        this.numericalWorkspaceBytes = preparationPlan.numericalWorkspaceBytes();
        this.workspacePointerCount = 0;
        if (this.splatResources.size() != 1) {
            throw new IllegalArgumentException("Metal custom splat-resource cardinality disagrees");
        }
        validateSplatResources();
        this.byteTotals = preparationPlan.traceUnit() == null
                        || !preparationPlan.traceUnit().enabled()
                ? null
                : byteTotals(
                        new long[] {resource.requiredBytes()},
                        new long[] {resource.requiredBytes()},
                        new long[0],
                        0L);
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
                ? programResource.context() : customResource.context();
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
        return programResource != null
                && representation instanceof AddressWorkspace workspace
                && workspace.belongsTo(programResource.context())
                && workspace.pointerCount() == workspacePointerCount
                && workspace.matchesScratch(numericalWorkspaceBytes);
    }

    @Override
    protected BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations) {
        int boundSplatCount = 0;
        long boundSplatBytes = 0L;
        MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
        if (byteTotals != null && traceUnit != null && traceUnit.enabled()) {
            for (int index = 0; index < inputCount; index++) {
                if (MetalPreparedSplatResource.isBinding(bufferRepresentations[index])) {
                    boundSplatCount++;
                    boundSplatBytes = Math.addExact(boundSplatBytes, requiredBytes[index]);
                }
            }
        }
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
                    boundSplatBytes,
                    boundSplatCount,
                    workspaceCount);
        }
        if (preparationPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM) {
            return bindCustomProgram(
                    runState,
                    bufferRepresentations,
                    workspaceRepresentations,
                    boundSplatBytes,
                    boundSplatCount);
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
        return new ProgramBoundInvocation(
                runState, preparationPlan, programResource, replayBuffers,
                inputCount, inputs, outputCount, outputs,
                byteTotals, boundSplatBytes, boundSplatCount, workspaceCount, new ScanFailureBinding[0]);
    }

    private BoundInvocation bindCustomProgram(
            RunState runState,
            BufferRepresentation[] bufferRepresentations,
            WorkspaceRepresentation[] workspaceRepresentations,
            long boundSplatBytes,
            int boundSplatCount) {
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
        int valueCount = Math.addExact(materializedCount, workspace.scratchCount());
        for (int scratch = 0; scratch < workspace.scratchCount(); scratch++)
            workspace.set(materializedCount + scratch, workspace.scratchHandle(scratch));
        for (int target = 0; target < targetCount; target++) {
            workspace.set(valueCount + target,
                    materializedBuffers[programToSlot[targetValues[target]]].executionHandle());
        }
        MemorySegment values = workspace.segment().asSlice(
                0L, (long) valueCount * ADDRESS.byteSize());
        MemorySegment outputs = workspace.segment().asSlice(
                (long) valueCount * ADDRESS.byteSize(),
                (long) targetCount * ADDRESS.byteSize());
        return new ProgramBoundInvocation(
                runState,
                preparationPlan,
                programResource,
                replayBuffers,
                valueCount,
                values,
                targetCount,
                outputs,
                byteTotals,
                boundSplatBytes,
                boundSplatCount,
                workspaceCount, workspace.scanFailures(preparationPlan));

    }

    private MetalBufferRepresentation readInput(
            int index, BufferRepresentation representation) {
        if (representation instanceof MetalBufferRepresentation metal) {
            return metal;
        }
        return MetalPreparedSplatResource.exactReadBuffer(
                representation,
                customResource == null ? programResource.context() : customResource.context(),
                requiredBytes[index],
                preparationPlan.feedSplats().get(index).orElseThrow());
    }

    private static InvocationByteTotals byteTotals(
            long[] inputBytes,
            long[] outputBytes,
            long[] internalBytes,
            long workspaceBytes) {
        if (workspaceBytes < 0L) {
            throw new IllegalArgumentException("Metal invocation byte facts disagree");
        }
        long aggregateInput = aggregateBytes(inputBytes);
        long aggregateOutput = aggregateBytes(outputBytes);
        long aggregateInternal = aggregateBytes(internalBytes);
        long aggregateRequired = Math.addExact(
                Math.addExact(aggregateInput, aggregateOutput),
                Math.addExact(aggregateInternal, workspaceBytes));
        return new InvocationByteTotals(
                aggregateInput,
                aggregateOutput,
                aggregateInternal,
                workspaceBytes,
                aggregateRequired);
    }

    private static long aggregateBytes(long[] requiredBytes) {
        long total = 0L;
        for (long bytes : requiredBytes) total = Math.addExact(total, bytes);
        return total;
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
            long workspace,
            long required) {
        private InvocationByteTotals {
            if (input < 0L || output < 0L || internal < 0L
                    || workspace < 0L || required < 0L
                    || required != Math.addExact(
                            Math.addExact(input, output),
                            Math.addExact(internal, workspace))) {
                throw new IllegalArgumentException("Metal invocation byte totals disagree");
            }
        }
    }

    private static final class ProgramBoundInvocation extends BoundInvocation {
        private final MetalNegPreparationPlan preparationPlan;
        private final MetalProgramExecutableResource resource;
        private final MetalBufferRepresentation[] inputBuffers;
        private final int inputCount;
        private final MemorySegment inputs;
        private final int outputCount;
        private final MemorySegment outputs;
        private final InvocationByteTotals byteTotals;
        private final long splatBytes;
        private final int splatCount;
        private final int workspaceCount;
        private final ScanFailureBinding[] scanFailures;

        private ProgramBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalProgramExecutableResource resource,
                MetalBufferRepresentation[] inputBuffers,
                int inputCount,
                MemorySegment inputs,
                int outputCount,
                MemorySegment outputs,
                InvocationByteTotals byteTotals,
                long splatBytes,
                int splatCount,
                int workspaceCount, ScanFailureBinding[] scanFailures) {
            super(runState);
            this.preparationPlan = preparationPlan;
            this.resource = resource;
            this.inputBuffers = inputBuffers;
            this.inputCount = inputCount;
            this.inputs = inputs;
            this.outputCount = outputCount;
            this.outputs = outputs;
            this.byteTotals = byteTotals;
            this.splatBytes = splatBytes;
            this.splatCount = splatCount;
            this.workspaceCount = workspaceCount;
            this.scanFailures = scanFailures;
        }

        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId = traceUnit == null || !traceUnit.enabled()
                    ? null : traceUnit.beginInvocation();
            if (invocationId != null && traceUnit.enabled()) {
                traceUnit.invocationPlanned(
                        invocationId,
                        preparationPlan.feedValueIds().size(),
                        preparationPlan.targetValueIds().size(),
                        preparationPlan.internalValueIds().size(),
                        byteTotals.input(),
                        byteTotals.output(),
                        byteTotals.internal(),
                        splatBytes,
                        byteTotals.workspace(),
                        byteTotals.required(),
                        splatCount,
                        workspaceCount);
            }
            try {
                resource.run(inputCount, inputs, outputCount, outputs);
            } catch (MetalNativeApi.NativeFailure failure) {
                RuntimeException translated;
                try {
                    translated = failure.status() == MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS
                            || failure.status() == MetalNativeApi.Status.INVALID_ARGUMENT
                            ? scanFailure(traceUnit, invocationId) : null;
                } catch (RuntimeException | Error diagnosticFailure) {
                    if (diagnosticFailure != failure) failure.addSuppressed(diagnosticFailure);
                    if (traceUnit != null) traceUnit.invocationFailed(invocationId, failure);
                    throw failure;
                }
                if (translated != null) {
                    translated.initCause(failure);
                    if (traceUnit != null) traceUnit.invocationFailed(invocationId, translated);
                    throw translated;
                }
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
                    reproduced.initCause(failure);
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

        private RuntimeException scanFailure(MetalTraceProducer.PreparedUnit traceUnit,
                TraceInvocationId invocationId) {
            for (ScanFailureBinding binding : scanFailures) {
                binding.workspace().readFailure(binding.offset(), binding.record());
                MemorySegment record = binding.record();
                long category = record.getAtIndex(NATIVE_LONG, 0);
                if (category == 0) continue;
                long time = record.getAtIndex(NATIVE_LONG, 1);
                long unit = record.getAtIndex(NATIVE_LONG, 2);
                long operand = record.getAtIndex(NATIVE_LONG, 3);
                long ordinal = record.getAtIndex(NATIVE_LONG, 4);
                String context = "Metal Scan time=" + time + ", unit=" + unit
                        + ", operand=" + operand + ", ordinal=" + ordinal;
                if (traceUnit != null && traceUnit.enabled()) {
                    traceUnit.scanFailed(invocationId, category, time, unit, operand, ordinal);
                }
                return switch ((int) category) {
                    case 1 -> new IndexOutOfBoundsException(context + ": index out of bounds");
                    case 2 -> new IllegalArgumentException(context + ": duplicate scatter target");
                    case 3 -> new IllegalArgumentException(context + ": non-canonical BOOL byte");
                    default -> throw new IllegalStateException("unknown Metal Scan validation category " + category);
                };
            }
            return null;
        }

        private RuntimeException reproduceIndexFailure() {
            var descriptors = preparationPlan.descriptors();
            boolean[] feeds = new boolean[descriptors.size()];
            for (int feed : preparationPlan.feedValueIndices()) feeds[feed] = true;
            // Native checks boundary controls before any producer runs. Only after those pass
            // may a produced control be inspected; later producer buffers can be uninitialized.
            for (int pass = 0; pass < 2; pass++) {
                for (var node : preparationPlan.graphProgram().nodes()) {
                    int indexValue;
                    boolean length = false, nd = false, classIndex = false;
                    long time = 0L;
                    long[] words = node.attributeWords();
                    switch (node.kind()) {
                        case GATHER, GATHER_ELEMENTS, SCATTER_ELEMENTS, SCATTER_ADD -> indexValue = node.secondInputIndex();
                        case GATHER_ND, SCATTER_ND -> { indexValue = node.secondInputIndex(); nd = true; }
                        case ONE_HOT -> indexValue = node.firstInputIndex();
                        case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> {
                            indexValue = node.secondInputIndex(); classIndex = true;
                        }
                        case RNN_TANH, GRU_RESET_AFTER, LSTM -> {
                            indexValue = node.inputs()[1]; length = true;
                            time = descriptors.get(node.firstInputIndex()).shape().toLongArray()[0];
                        }
                        case GENERAL_SCAN -> {
                            if ((words[4] & 2L) == 0L) continue;
                            indexValue = node.inputs()[node.inputCount() - 1]; length = true; time = words[3];
                        }
                        default -> { continue; }
                    }
                    if (feeds[indexValue] != (pass == 0)) continue;
                    var indexDescriptor = descriptors.get(indexValue);
                    long elements = indexDescriptor.shape().knownElementCount().orElseThrow();
                    if (elements == 0L) continue;
                    DataType indexType = indexDescriptor.dataType();
                    int width = indexType.byteWidth();
                    long[] indexShape = indexDescriptor.shape().toLongArray();
                    var layout = indexDescriptor.layout().orElseGet(
                            () -> io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(indexDescriptor.shape()));
                    long[] strides = layout.strides();
                    long offset = layout.storageOffset();
                    long bytes = Math.multiplyExact(layout.referencedElementSpan(), width);
                    var indexBuffer = inputBuffers[indexValue];
                    if (indexBuffer == null) throw new IllegalStateException("Metal control has no stable executable buffer");
                    long[] dataShape = descriptors.get(node.firstInputIndex()).shape().toLongArray();
                    int batch = nd ? Math.toIntExact(words[0]) : 0;
                    long tuple = nd ? indexShape[indexShape.length - 1] : 0L;
                    String family = node.kind().name();
                    try (Arena arena = Arena.ofConfined()) {
                        MemorySegment indices = arena.allocate(bytes, width);
                        indexBuffer.download(0L, indices, 0L, bytes);
                        for (long ordinal = 0L; ordinal < elements; ordinal++) {
                            long value = readLogicalIndex(indices, ordinal, indexType, indexShape, strides, offset);
                            if (classIndex && words[2] != 0L
                                    && value == (indexType == DataType.INT32 ? (int)words[4] : words[4])) continue;
                            int axis = nd ? Math.addExact(batch, Math.toIntExact(ordinal % tuple))
                                    : length ? 0 : classIndex ? Math.toIntExact(words[0]) : node.axis();
                            long bound = length ? time : node.kind() == MetalMpsGraphProgram.NodeKind.ONE_HOT
                                    ? node.attributeValues()[0] : dataShape[axis];
                            if (value < 0L || (length ? value > bound : value >= bound)) {
                                if (length) return new IllegalArgumentException(family + " length at batch "
                                        + ordinal + " must be in [0," + bound + "]: " + value);
                                return new IndexOutOfBoundsException(family + " index at logical position "
                                        + ordinal + " for axis " + axis + " is out of bounds: value="
                                        + value + ", extent=" + bound);
                            }
                        }
                        boolean replacement = node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS
                                && (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.AXIS || words[1] == 1L)
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ND && words[1] == 1L;
                        if (replacement) {
                            long updates = nd ? elements / tuple : elements;
                            for (long later = 1L; later < updates; later++) {
                                for (long first = 0L; first < later; first++) {
                                    boolean same = nd ? sameNdTarget(indices, later, first, indexShape, tuple,
                                            batch, indexType, strides, offset)
                                            : sameScatterTarget(indices, later, first, indexShape, node.axis(),
                                                    indexType, strides, offset);
                                    if (same) return new IllegalArgumentException(family + " duplicate target at logical update "
                                            + later + "; first addressed at " + first);
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

        private static long readLogicalIndex(MemorySegment indices, long ordinal, DataType type,
                long[] shape, long[] strides, long offset) {
            long physical = offset;
            for (int axis = shape.length; axis-- > 0;) {
                physical = Math.addExact(physical, Math.multiplyExact(ordinal % shape[axis], strides[axis]));
                ordinal /= shape[axis];
            }
            return readIndex(indices, physical, type);
        }

        private static boolean sameNdTarget(MemorySegment indices, long left, long right,
                long[] shape, long tuple, int batch, DataType type, long[] strides, long offset) {
            for (int axis = 0; axis < batch; axis++)
                if (coordinate(left * tuple, shape, axis) != coordinate(right * tuple, shape, axis)) return false;
            for (long component = 0; component < tuple; component++)
                if (readLogicalIndex(indices, left * tuple + component, type, shape, strides, offset)
                        != readLogicalIndex(indices, right * tuple + component, type, shape, strides, offset)) return false;
            return true;
        }


        private static boolean sameScatterTarget(
                MemorySegment indices,
                long left,
                long right,
                long[] shape,
                int selectedAxis, DataType type, long[] strides, long offset) {
            for (int axis = 0; axis < shape.length; axis++) {
                if (axis == selectedAxis) {
                    if (readLogicalIndex(indices, left, type, shape, strides, offset)
                            != readLogicalIndex(indices, right, type, shape, strides, offset)) {
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
        private final long splatBytes;
        private final int splatCount;
        private final int workspaceCount;
        private CustomBoundInvocation(
                RunState runState,
                MetalNegPreparationPlan preparationPlan,
                MetalNegKernelPipelineResource resource,
                MetalBufferRepresentation input,
                MetalBufferRepresentation output,
                InvocationByteTotals byteTotals,
                long splatBytes,
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
            this.splatBytes = splatBytes;
            this.splatCount = splatCount;
            this.workspaceCount = workspaceCount;
        }
        @Override
        protected void executeBound() {
            MetalTraceProducer.PreparedUnit traceUnit = preparationPlan.traceUnit();
            TraceInvocationId invocationId = traceUnit == null || !traceUnit.enabled()
                    ? null : traceUnit.beginInvocation();
            if (invocationId != null && traceUnit.enabled()) {
                traceUnit.invocationPlanned(
                        invocationId,
                        1,
                        1,
                        0,
                        byteTotals.input(),
                        byteTotals.output(),
                        byteTotals.internal(),
                        splatBytes,
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

    private record ScanFailureBinding(MetalWorkspaceRepresentation workspace,
            long offset, MemorySegment record) {}

    /** Run-owned address table and explicitly declared, unpooled device scratch. */
    static final class AddressWorkspace implements WorkspaceRepresentation {
        private final MetalDeviceContext context;
        private final int pointerCount;
        private final Arena arena = Arena.ofShared();
        private final MemorySegment segment;
        private final MetalWorkspaceRepresentation[] scratch;
        private final long[] scratchBytes;
        private boolean closed;

        AddressWorkspace(MetalDeviceContext context, int pointerCount) {
            this(context, pointerCount, new long[0]);
        }

        AddressWorkspace(MetalDeviceContext context, int pointerCount, long[] scratchBytes) {
            this.context = Objects.requireNonNull(context, "context");
            if (pointerCount <= 0) throw new IllegalArgumentException("pointerCount must be positive");
            this.scratchBytes = scratchBytes.clone();
            this.pointerCount = Math.addExact(pointerCount, scratchBytes.length);
            this.scratch = new MetalWorkspaceRepresentation[scratchBytes.length];
            try {
                this.segment = arena.allocate(ADDRESS, this.pointerCount);
                for (int index = 0; index < scratch.length; index++) {
                    if (scratchBytes[index] <= 0L)
                        throw new IllegalArgumentException("numerical scratch extent must be positive");
                    scratch[index] = context.createWorkspace(scratchBytes[index]);
                }
            } catch (RuntimeException | Error failure) {
                for (var resource : scratch) {
                    if (resource == null) continue;
                    try { resource.close(); } catch (RuntimeException | Error closeFailure) {
                        if (closeFailure != failure) failure.addSuppressed(closeFailure);
                    }
                }
                try { arena.close(); } catch (RuntimeException | Error closeFailure) {
                    if (closeFailure != failure) failure.addSuppressed(closeFailure);
                }
                throw failure;
            }
        }

        synchronized boolean belongsTo(MetalDeviceContext expected) {
            return !closed && context == expected;
        }

        int pointerCount() { return pointerCount; }
        MemorySegment segment() { return segment; }
        int scratchCount() { return scratch.length; }
        boolean matchesScratch(long[] expected) { return java.util.Arrays.equals(scratchBytes, expected); }
        MetalNativeApi.Handle scratchHandle(int index) { return scratch[index].executionHandle(); }
        /** Binds only fixed metadata and bounded staging; never captures tensor payloads. */
        ScanFailureBinding[] scanFailures(MetalNegPreparationPlan plan) {
            var failures = new java.util.ArrayList<ScanFailureBinding>();
            int index = 0;
            for (var node : plan.graphProgram().nodes()) {
                if (MetalProgramWorkspace.scratchBytes(node, plan.programValueDescriptors()) == 0) continue;
                if (node.kind() == MetalMpsGraphProgram.NodeKind.GENERAL_SCAN) {
                    failures.add(new ScanFailureBinding(scratch[index],
                            Math.addExact(node.attributeWords()[30], 80),
                            arena.allocate(40, 8)));
                }
                index++;
            }
            return failures.toArray(ScanFailureBinding[]::new);
        }

        synchronized void set(int index, MetalNativeApi.Handle handle) {
            if (closed) throw new IllegalStateException("Metal address workspace is closed");
            segment.setAtIndex(ADDRESS, index, handle.carrier());
        }

        @Override
        public synchronized void close() {
            if (closed) return;
            closed = true;
            Throwable failure = null;
            for (int index = scratch.length; index-- > 0;) {
                try { scratch[index].close(); } catch (RuntimeException | Error closeFailure) {
                    if (failure == null) failure = closeFailure;
                    else if (failure != closeFailure) failure.addSuppressed(closeFailure);
                }
            }
            try { arena.close(); } catch (RuntimeException | Error closeFailure) {
                if (failure == null) failure = closeFailure;
                else if (failure != closeFailure) failure.addSuppressed(closeFailure);
            }
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
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
