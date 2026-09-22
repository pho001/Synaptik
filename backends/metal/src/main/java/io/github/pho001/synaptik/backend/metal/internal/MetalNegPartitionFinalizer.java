package io.github.pho001.synaptik.backend.metal.internal;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import java.util.List;
import java.util.Objects;
import java.util.Collections;
import java.util.IdentityHashMap;

/**
 * Validates assigned Metal NEG declarations and compiles the selected typed persistent resource.
 *
 * <p>The finalizer changes no route or declaration. It compiles and owns the selected custom
 * pipeline or MPSGraph executable resource until the complete result returns, and reverses that
 * acquisition on every intervening failure while preserving the original failure and distinct
 * cleanup suppression.</p>
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
                    io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
                    int[] feedPlanIndices,
                    int[] targetPlanIndices,
                    MetalMpsGraphExecutableResource resource,
                    long[] feedRequiredBytes,
                    long[] targetRequiredBytes,
                    int workspacePlanIndex) {
                return new MetalNegPreparedExecutable(memoryPlan, feedPlanIndices,
                        targetPlanIndices, resource, feedRequiredBytes,
                        targetRequiredBytes, workspacePlanIndex);
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
     * Compiles and returns one assigned executable plus its sole persistent resource.
     *
     * @param finalization non-null exact analyzed plan and complete shared assignments
     * @return non-null atomic result whose resource list contains the executable owner once
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
        if (!finalization.analysis().partition().owner()
                .equals(MetalCapabilityProvider.METAL_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be Metal");
        }
        MetalNegPreparationPlan plan = finalization.analysis().plan();
        if (plan.partition() != finalization.analysis().partition()) {
            throw new IllegalArgumentException(
                    "Metal NEG analyzed plan/finalized partition identity mismatch");
        }
        if (plan.context() != context) {
            throw new IllegalArgumentException("Metal NEG analysis/finalization context mismatch");
        }
        int expectedAssignments = plan.declarations().size()
                + (plan.addressWorkspace().isPresent() ? 1 : 0);
        if (finalization.assignments().size() != expectedAssignments) {
            throw new IllegalArgumentException("Metal NEG finalization requires exact declarations");
        }
        int feedCount = plan.feedValueIds().size();
        int[] feedPlanIndices = new int[feedCount];
        int[] targetPlanIndices = new int[plan.targetValueIds().size()];
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
            if (index < feedCount) feedPlanIndices[index] = assignment.planIndex();
            else targetPlanIndices[index - feedCount] = assignment.planIndex();
        }
        return switch (plan.route()) {
            case CUSTOM_SINGLE_NEG -> finalizeCustom(
                    finalization, plan, feedPlanIndices, targetPlanIndices);
            case MPSGRAPH -> finalizeMpsGraph(
                    finalization, plan, feedPlanIndices, targetPlanIndices);
        };
    }

    private BackendPartitionFinalizationResult finalizeMpsGraph(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            MetalNegPreparationPlan plan,
            int[] feedPlanIndices,
            int[] targetPlanIndices) {
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
        MetalMpsGraphExecutableResource resource = context.createNegExecutable(plan);
        try {
            var executable = executableFactory.createMpsGraph(
                    finalization.memoryPlan(), feedPlanIndices, targetPlanIndices, resource,
                    plan.feedRequiredBytes(), plan.targetRequiredBytes(), workspace.planIndex());
            return new BackendPartitionFinalizationResult(executable, List.of(resource));
        } catch (RuntimeException | Error failure) {
            closeAfterFailure(resource, failure);
            throw failure;
        }
    }

    private BackendPartitionFinalizationResult finalizeCustom(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            MetalNegPreparationPlan plan,
            int[] feedPlanIndices,
            int[] targetPlanIndices) {
        if (feedPlanIndices.length != 1 || targetPlanIndices.length != 1
                || plan.addressWorkspace().isPresent()) {
            throw new IllegalArgumentException("Metal NEG custom declarations disagree");
        }
        MetalNegKernelPipelineResource resource = context.createNegKernelPipeline(plan);
        try {
            var executable = executableFactory.createCustom(
                    finalization.memoryPlan(), feedPlanIndices[0], targetPlanIndices[0], resource);
            return new BackendPartitionFinalizationResult(executable, List.of(resource));
        } catch (RuntimeException | Error failure) {
            closeAfterFailure(resource, failure);
            throw failure;
        }
    }

    private static void closeAfterFailure(
            io.github.pho001.synaptik.runtime.resource.PreparedResource resource,
            Throwable failure) {
        try {
            resource.close();
        } catch (RuntimeException | Error cleanup) {
            if (cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
        }
    }

    /** Cold construction seam whose failure is covered by finalizer-local rollback. */
    interface FinalizedExecutableFactory {
        /**
         * Constructs the immutable executable recipe after persistent resource acquisition.
         *
         * @param memoryPlan exact finalized memory plan
         * @param feedPlanIndices stable feed positions in {@code memoryPlan}
         * @param targetPlanIndices stable target positions in {@code memoryPlan}
         * @param resource acquired executable resource borrowed by the recipe
         * @param feedRequiredBytes feed byte extents aligned with {@code feedPlanIndices}
         * @param targetRequiredBytes target byte extents aligned with {@code targetPlanIndices}
         * @param workspacePlanIndex assigned address-workspace position
         * @return non-null immutable executable recipe
         * @throws RuntimeException if recipe construction fails
         * @throws Error if construction reports an error
         */
        MetalNegPreparedExecutable createMpsGraph(
                io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
                int[] feedPlanIndices,
                int[] targetPlanIndices,
                MetalMpsGraphExecutableResource resource,
                long[] feedRequiredBytes,
                long[] targetRequiredBytes,
                int workspacePlanIndex);

        /**
         * Constructs the workspace-free custom executable recipe after pipeline acquisition.
         *
         * @param memoryPlan exact finalized memory plan
         * @param feedPlanIndex assigned singleton feed position
         * @param targetPlanIndex assigned singleton target position
         * @param resource acquired custom pipeline borrowed by the recipe
         * @return non-null immutable custom executable recipe
         * @throws RuntimeException if recipe construction fails
         * @throws Error if construction reports an error
         */
        default MetalNegPreparedExecutable createCustom(
                io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
                int feedPlanIndex,
                int targetPlanIndex,
                MetalNegKernelPipelineResource resource) {
            return new MetalNegPreparedExecutable(
                    memoryPlan, feedPlanIndex, targetPlanIndex, resource);
        }
    }
}
