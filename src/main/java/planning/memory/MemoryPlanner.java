package planning.memory;

import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Computes tensor and region memory reuse plans for optimized graphs.
 *
 * <p>The planner is the public entry point for memory planning. The concrete planning responsibilities live in
 * package-local planners for tensor lifetimes, reusable slots, region value flow, region bindings, handoffs, runtime
 * binding policy, and summary reporting.
 *
 * <p>This class is stateless and thread-safe as long as the input graph and planning input are not mutated
 * concurrently while planning.
 */
public final class MemoryPlanner {
    private MemoryPlanner() {
    }

    /**
     * Plans memory with the default policy.
     *
     * @param sortedGraph tensors in topological execution order
     * @return memory plan for tensor lifetimes and reusable slots
     */
    public static MemoryPlan plan(List<Tensor> sortedGraph) {
        return plan(sortedGraph, MemoryPlannerPolicy.defaults());
    }

    /**
     * Plans tensor memory without region planning artifacts.
     *
     * @param sortedGraph tensors in topological execution order
     * @param policy memory reuse policy
     * @return memory plan for tensor lifetimes and reusable slots
     */
    public static MemoryPlan plan(List<Tensor> sortedGraph, MemoryPlannerPolicy policy) {
        Objects.requireNonNull(policy, "policy cannot be null");
        if (sortedGraph == null || sortedGraph.isEmpty()) {
            return emptyPlan(policy);
        }

        TensorLifetimePlan lifetimePlan = TensorLifetimePlanner.plan(sortedGraph);
        Map<Tensor, ReusableInterval> reusableIntervals = ReusableIntervalBuilder.build(
                sortedGraph,
                lifetimePlan.lifetimes(),
                policy
        );
        ReusableSlotAssignment slotAssignment = ReusableSlotAllocator.allocate(
                reusableIntervals.values().stream().toList(),
                lifetimePlan.forwardBoundaryIndex(),
                policy
        );
        MemoryPlanSummary summary = MemoryPlanSummaryBuilder.build(
                sortedGraph,
                lifetimePlan.lifetimes(),
                reusableIntervals,
                slotAssignment.slotByOwner(),
                slotAssignment.slotSizes(),
                lifetimePlan.forwardBoundaryIndex()
        );

        return new MemoryPlan(
                new TensorMemoryPlan(
                        lifetimePlan.lifetimes(),
                        reusableIntervals,
                        slotAssignment.slotByOwner(),
                        slotAssignment.slotSizes()
                ),
                RegionMemoryPlan.empty(),
                new RuntimeBindingPlan(
                        RuntimeMemoryBindingPolicyPlanner.forTensors(sortedGraph),
                        Map.of()
                ),
                policy,
                summary
        );
    }

    /**
     * Plans memory using full compile-planning input.
     *
     * <p>When planned regions are present, the returned plan includes structural memory view, region value lifetimes,
     * materialization decisions, region slot assignment, and handoff requirements. Tensor-level reuse is intentionally
     * empty on this compiled-node path; runtime binding consumes node-id and region value metadata.
     *
     * @param input compile-planning memory input
     * @param policy memory reuse policy
     * @return memory plan for runtime binding
     */
    public static MemoryPlan plan(MemoryPlanningInput input, MemoryPlannerPolicy policy) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(policy, "policy cannot be null");

        RegionValueFlowPlan flowPlan = RegionValueFlowPlanner.plan(input);
        RegionBindingAssignment bindingAssignment = RegionBindingAllocator.allocate(
                flowPlan.regionValueLifetimes().values().stream().toList()
        );
        List<RegionHandoffRequirement> handoffRequirements = RegionHandoffPlanner.plan(
                flowPlan.regionValueLifetimes().values().stream().toList()
        );

        return new MemoryPlan(
                TensorMemoryPlan.empty(),
                new RegionMemoryPlan(
                        flowPlan.structuralView(),
                        flowPlan.regionValueLifetimes(),
                        flowPlan.materializationPlan(),
                        bindingAssignment.bindingsByValueRef(),
                        bindingAssignment.slotByValueRef(),
                        bindingAssignment.slotSizes(),
                        flowPlan.tensorToGraphValueRef(),
                        flowPlan.nodeIdToGraphValueRef(),
                        handoffRequirements
                ),
                new RuntimeBindingPlan(
                        Map.of(),
                        RuntimeMemoryBindingPolicyPlanner.forNodeIds(input.compiledNodes())
                ),
                policy,
                emptySummary()
        );
    }

    private static MemoryPlan emptyPlan(MemoryPlannerPolicy policy) {
        return new MemoryPlan(
                TensorMemoryPlan.empty(),
                RegionMemoryPlan.empty(),
                RuntimeBindingPlan.empty(),
                policy,
                emptySummary()
        );
    }

    private static MemoryPlanSummary emptySummary() {
        return new MemoryPlanSummary(0, 0, 0, 0, 0.0d, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0, 0, 0.0);
    }
}
