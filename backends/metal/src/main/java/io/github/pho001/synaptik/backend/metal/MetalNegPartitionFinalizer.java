package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Validates assigned Metal supported-operation declarations and compiles the selected resource.
 *
 * <p>The finalizer changes no route or declaration. It compiles and owns the selected custom
 * singleton-NEG pipeline or typed whole-partition MPSGraph executable until the complete result
 * returns, and reverses that acquisition on every intervening failure while preserving the
 * original failure and distinct cleanup suppression.</p>
 */
final class MetalNegPartitionFinalizer
        implements BackendPartitionFinalizer<MetalNegPreparationPlan> {
    private final MetalDeviceContext context;
    private final FinalizedExecutableFactory executableFactory;

    /**
     * Creates a finalizer borrowing the same context supplied to analysis.
     *
     * @param context non-null open device context
     * @throws NullPointerException if {@code context} is {@code null}
     */
    MetalNegPartitionFinalizer(MetalDeviceContext context) {
        this(context, new FinalizedExecutableFactory() {
            @Override
            public MetalNegPreparedExecutable createMpsGraph(
                    MetalNegPreparationPlan plan,
                    io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
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
                return new MetalNegPreparedExecutable(
                        plan,
                        memoryPlan,
                        feedPlanIndices,
                        feedRepresentationIndices,
                        targetPlanIndices,
                        targetRepresentationIndices,
                        internalPlanIndices,
                        internalRepresentationIndices,
                        resource,
                        splatResources,
                        feedRequiredBytes,
                        targetRequiredBytes,
                        internalRequiredBytes,
                        workspacePlanIndex);
            }
        });
    }

    /**
     * Creates a finalizer with an injected deterministic executable-construction seam.
     *
     * @param context non-null open device context
     * @param executableFactory non-null cold recipe factory invoked after resource acquisition
     * @throws NullPointerException if an argument is {@code null}
     */
    MetalNegPartitionFinalizer(
            MetalDeviceContext context, FinalizedExecutableFactory executableFactory) {
        this.context = Objects.requireNonNull(context, "context");
        this.executableFactory = Objects.requireNonNull(executableFactory, "executableFactory");
    }

    /** @return the stable Metal backend ownership identity */
    @Override
    public BackendId backendId() {
        return MetalCapabilityProvider.METAL_BACKEND_ID;
    }

    /**
     * Compiles and returns one assigned executable plus its persistent resources.
     *
     * @param finalization non-null exact analyzed plan and complete shared assignments
     * @return non-null atomic result whose resources are in physical acquisition order
     * @throws NullPointerException if {@code finalization} is {@code null}
     * @throws IllegalArgumentException if ownership, declaration identity/order, assignment kind,
     *     or exact geometry disagrees with analysis
     * @throws RuntimeException if native compilation, construction, or rollback fails
     * @throws Error if compilation, construction, or rollback reports an error
     */
    @Override
    public BackendPartitionFinalizationResult finalizePartition(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization) {
        Objects.requireNonNull(finalization, "finalization");
        MetalNegPreparationPlan plan = finalization.analysis().plan();
        BackendPartitionFinalizationResult result;
        try {
            if (!finalization.analysis().partition().owner()
                    .equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
                throw new IllegalArgumentException("partition owner must be Metal");
            }
            if (plan.partition() != finalization.analysis().partition()) {
                throw new IllegalArgumentException(
                        "Metal NEG analyzed plan/finalized partition identity mismatch");
            }
            if (plan.context() != context) {
                throw new IllegalArgumentException(
                        "Metal NEG analysis/finalization context mismatch");
            }
            result = finalizeSelectedPartition(finalization, plan);
        } catch (RuntimeException | Error failure) {
            if (plan.traceUnit() != null) {
                plan.traceUnit().preparationFailed(failure);
            }
            throw failure;
        }
        try {
            if (plan.traceUnit() != null) {
                plan.traceUnit().preparationSucceeded();
            }
        } catch (Error fatal) {
            closeResultAfterTraceError(result, fatal);
            throw fatal;
        }
        return result;
    }

    private BackendPartitionFinalizationResult finalizeSelectedPartition(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            MetalNegPreparationPlan plan) {
        int expectedAssignments = plan.declarations().size()
                + (plan.addressWorkspace().isPresent() ? 1 : 0);
        if (finalization.assignments().size() != expectedAssignments) {
            throw new IllegalArgumentException("Metal NEG finalization requires exact declarations");
        }
        int feedCount = plan.feedValueIds().size();
        int[] feedPlanIndices = new int[feedCount];
        int[] feedRepresentationIndices = new int[feedCount];
        int[] targetPlanIndices = new int[plan.targetValueIds().size()];
        int[] targetRepresentationIndices = new int[plan.targetValueIds().size()];
        int[] internalPlanIndices = new int[plan.internalValueIds().size()];
        int[] internalRepresentationIndices = new int[plan.internalValueIds().size()];
        boolean[] assignedPlanIndices = new boolean[finalization.memoryPlan().buffers().size()];
        var assignedSlots = Collections.newSetFromMap(
                new IdentityHashMap<io.github.pho001.synaptik.runtime.memory.BufferSlot, Boolean>());
        for (int index = 0; index < plan.declarations().size(); index++) {
            if (!(finalization.assignments().get(index)
                    instanceof PreparationResourceAssignment.Buffer assignment)
                    || assignment.requirement() != plan.declarations().get(index)) {
                throw new IllegalArgumentException(
                        "Metal NEG declaration assignment identity or order disagrees");
            }
            if (assignedPlanIndices[assignment.planIndex()]
                    || !assignedSlots.add(assignment.slot())) {
                throw new IllegalArgumentException(
                        "Metal NEG buffer assignments repeat a plan index or slot");
            }
            assignedPlanIndices[assignment.planIndex()] = true;
            var entry = finalization.memoryPlan().buffers().get(assignment.planIndex());
            var declaration = plan.declarations().get(index);
            if (entry.slot() != assignment.slot()
                    || entry.byteSize() != declaration.byteSize()
                    || entry.byteAlignment() != declaration.byteAlignment()) {
                throw new IllegalArgumentException(
                        "Metal NEG declaration assignment geometry disagrees");
            }
            if (index < feedCount) {
                feedPlanIndices[index] = assignment.planIndex();
                feedRepresentationIndices[index] = assignment.representationIndex();
            } else if (index < feedCount + targetPlanIndices.length) {
                int targetIndex = index - feedCount;
                targetPlanIndices[targetIndex] = assignment.planIndex();
                targetRepresentationIndices[targetIndex] = assignment.representationIndex();
            } else {
                int internalIndex = index - feedCount - targetPlanIndices.length;
                internalPlanIndices[internalIndex] = assignment.planIndex();
                internalRepresentationIndices[internalIndex] =
                        assignment.representationIndex();
            }
        }
        boolean[] splatSources = plan.feedSplatSources();
        for (int index = 0; index < splatSources.length; index++) {
            if (splatSources[index] && feedRepresentationIndices[index] != 0) {
                throw new IllegalArgumentException(
                        "Metal source splat must own representation position zero");
            }
        }
        return switch (plan.route()) {
            case CUSTOM_SINGLE_NEG -> finalizeCustom(
                    finalization,
                    plan,
                    feedPlanIndices,
                    feedRepresentationIndices,
                    targetPlanIndices,
                    targetRepresentationIndices);
            case MPSGRAPH, CUSTOM_PROGRAM -> finalizeMpsGraph(
                    finalization,
                    plan,
                    feedPlanIndices,
                    feedRepresentationIndices,
                    targetPlanIndices,
                    targetRepresentationIndices,
                    internalPlanIndices,
                    internalRepresentationIndices);
        };
    }

    private BackendPartitionFinalizationResult finalizeMpsGraph(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            MetalNegPreparationPlan plan,
            int[] feedPlanIndices,
            int[] feedRepresentationIndices,
            int[] targetPlanIndices,
            int[] targetRepresentationIndices,
            int[] internalPlanIndices,
            int[] internalRepresentationIndices) {
        var workspaceRequirement = plan.addressWorkspace().orElseThrow();
        var last = finalization.assignments().getLast();
        if (!(last instanceof PreparationResourceAssignment.Workspace workspace)
                || workspace.requirement() != workspaceRequirement) {
            throw new IllegalArgumentException("Metal NEG address workspace assignment disagrees");
        }
        var workspaceEntry = finalization.memoryPlan().workspaces().get(workspace.planIndex());
        if (workspaceEntry.slot() != workspace.slot()
                || workspaceEntry.byteSize() != workspaceRequirement.byteSize()
                || workspaceEntry.byteAlignment() != workspaceRequirement.byteAlignment()) {
            throw new IllegalArgumentException("Metal NEG address workspace geometry disagrees");
        }
        var resources = new ArrayList<PreparedResource>();
        MetalMpsGraphExecutableResource resource = context.createMpsGraphExecutable(plan);
        addAcquired(resources, resource);
        try {
            List<Optional<MetalPreparedSplatResource>> splats =
                    acquireSplats(plan, resources);
            var executable = executableFactory.createMpsGraph(
                    plan,
                    finalization.memoryPlan(),
                    feedPlanIndices,
                    feedRepresentationIndices,
                    targetPlanIndices,
                    targetRepresentationIndices,
                    internalPlanIndices,
                    internalRepresentationIndices,
                    resource,
                    splats,
                    plan.feedRequiredBytes(),
                    plan.targetRequiredBytes(),
                    plan.internalRequiredBytes(),
                    workspace.planIndex());
            return new BackendPartitionFinalizationResult(executable, resources);
        } catch (RuntimeException | Error failure) {
            closeResourcesAfterFailure(resources, failure);
            throw failure;
        }
    }

    private BackendPartitionFinalizationResult finalizeCustom(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            MetalNegPreparationPlan plan,
            int[] feedPlanIndices,
            int[] feedRepresentationIndices,
            int[] targetPlanIndices,
            int[] targetRepresentationIndices) {
        if (feedPlanIndices.length != 1 || targetPlanIndices.length != 1
                || plan.addressWorkspace().isPresent()) {
            throw new IllegalArgumentException("Metal NEG custom declarations disagree");
        }
        var resources = new ArrayList<PreparedResource>();
        MetalNegKernelPipelineResource resource = context.createNegKernelPipeline(plan);
        addAcquired(resources, resource);
        try {
            List<Optional<MetalPreparedSplatResource>> splats =
                    acquireSplats(plan, resources);
            var executable = executableFactory.createCustom(
                    plan,
                    finalization.memoryPlan(),
                    feedPlanIndices[0],
                    feedRepresentationIndices[0],
                    targetPlanIndices[0],
                    targetRepresentationIndices[0],
                    resource,
                    splats);
            return new BackendPartitionFinalizationResult(executable, resources);
        } catch (RuntimeException | Error failure) {
            closeResourcesAfterFailure(resources, failure);
            throw failure;
        }
    }

    private List<Optional<MetalPreparedSplatResource>> acquireSplats(
            MetalNegPreparationPlan plan, List<PreparedResource> resources) {
        boolean[] sources = plan.feedSplatSources();
        long[] requiredBytes = plan.feedRequiredBytes();
        var result = new ArrayList<Optional<MetalPreparedSplatResource>>(sources.length);
        for (int index = 0; index < sources.length; index++) {
            if (!sources[index]) {
                result.add(Optional.empty());
                continue;
            }
            MetalPreparedSplatResource splat = MetalPreparedSplatResource.create(
                    context, requiredBytes[index], plan.feedSplats().get(index).orElseThrow());
            addAcquired(resources, splat);
            result.add(Optional.of(splat));
        }
        return List.copyOf(result);
    }

    private static void closeResultAfterTraceError(
            BackendPartitionFinalizationResult result, Error fatal) {
        closeResourcesAfterFailure(result.resources(), fatal);
    }

    private static void addAcquired(
            List<PreparedResource> resources, PreparedResource resource) {
        try {
            resources.add(resource);
        } catch (RuntimeException | Error failure) {
            try {
                resource.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) {
                    failure.addSuppressed(cleanup);
                }
            }
            throw failure;
        }
    }

    private static void closeResourcesAfterFailure(
            List<? extends PreparedResource> resources, Throwable failure) {
        for (int index = resources.size() - 1; index >= 0; index--) {
            try {
                resources.get(index).close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) {
                    failure.addSuppressed(cleanup);
                }
            }
        }
    }

    /** Cold construction seam whose failure is covered by finalizer-local rollback. */
    interface FinalizedExecutableFactory {
        /**
         * Constructs the immutable executable recipe after persistent resource acquisition.
         *
         * @param plan exact immutable analyzed backend plan
         * @param memoryPlan exact finalized memory plan
         * @param feedPlanIndices stable feed positions in {@code memoryPlan}
         * @param feedRepresentationIndices owner representation positions aligned with feeds
         * @param targetPlanIndices stable target positions in {@code memoryPlan}
         * @param targetRepresentationIndices owner representation positions aligned with targets
         * @param internalPlanIndices stable internal logical-value positions in {@code memoryPlan}
         * @param internalRepresentationIndices owner representation positions aligned with internals
         * @param resource acquired executable resource borrowed by the recipe
         * @param splatResources source-owned immutable splat resources aligned with feeds
         * @param feedRequiredBytes feed byte extents aligned with {@code feedPlanIndices}
         * @param targetRequiredBytes target byte extents aligned with {@code targetPlanIndices}
         * @param internalRequiredBytes internal byte extents aligned with {@code internalPlanIndices}
         * @param workspacePlanIndex assigned native-address workspace position
         * @return non-null immutable executable recipe
         * @throws RuntimeException if recipe construction fails
         * @throws Error if construction reports an error
         */
        MetalNegPreparedExecutable createMpsGraph(
                MetalNegPreparationPlan plan,
                io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
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
                int workspacePlanIndex);

        /**
         * Constructs the workspace-free custom executable recipe after pipeline acquisition.
         *
         * @param plan exact immutable analyzed backend plan
         * @param memoryPlan exact finalized memory plan
         * @param feedPlanIndex assigned singleton feed buffer position
         * @param feedRepresentationIndex assigned Metal feed representation position
         * @param targetPlanIndex assigned singleton target buffer position
         * @param targetRepresentationIndex assigned Metal target representation position
         * @param resource acquired custom pipeline borrowed by the recipe
         * @param splatResources source-owned immutable splat resources aligned with the feed
         * @return non-null immutable custom executable recipe
         * @throws RuntimeException if recipe construction fails
         * @throws Error if construction reports an error
         */
        default MetalNegPreparedExecutable createCustom(
                MetalNegPreparationPlan plan,
                io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
                int feedPlanIndex,
                int feedRepresentationIndex,
                int targetPlanIndex,
                int targetRepresentationIndex,
                MetalNegKernelPipelineResource resource,
                List<Optional<MetalPreparedSplatResource>> splatResources) {
            return new MetalNegPreparedExecutable(
                    plan,
                    memoryPlan,
                    feedPlanIndex,
                    feedRepresentationIndex,
                    targetPlanIndex,
                    targetRepresentationIndex,
                    resource,
                    splatResources);
        }
    }
}
