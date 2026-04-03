package graph.optimizer.rules;

import graph.optimizer.OptimizationRule;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryRole;
import graph.optimizer.memory.NodeLifetime;
import graph.optimizer.memory.MemoryPlannerPolicy;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MemoryOptimizerRule implements OptimizationRule {
    private static final boolean ENABLE_MEMORY_REUSE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.enableMemoryReuse", "true"));
    private static volatile MemoryPlan lastPlan;
    private final MemoryPlannerPolicy policy;

    public MemoryOptimizerRule() {
        this(resolvePolicyFromProperties());
    }

    public MemoryOptimizerRule(MemoryPlannerPolicy policy) {
        this.policy = policy == null ? MemoryPlannerPolicy.defaults() : policy;
    }

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (!ENABLE_MEMORY_REUSE || sortedGraph == null || sortedGraph.isEmpty()) {
            return sortedGraph;
        }

        DataType graphType = validateUniformGraphType(sortedGraph);
        if (graphType == null || graphType == DataType.FLOAT16 || graphType == DataType.BOOL) {
            return sortedGraph;
        }

        MemoryPlan plan = MemoryPlanner.plan(sortedGraph, policy);
        lastPlan = plan;
        return switch (graphType) {
            case FLOAT64 -> applyFloat64(sortedGraph, plan);
            case FLOAT32 -> applyFloat32(sortedGraph, plan);
            case FLOAT16, BOOL -> sortedGraph;
        };
    }

    private DataType validateUniformGraphType(List<Tensor> sortedGraph) {
        DataType graphType = sortedGraph.get(0).getDataType();
        for (Tensor tensor : sortedGraph) {
            if (tensor.getDataType() != graphType) {
                return null;
            }
        }
        return graphType;
    }

    public static MemoryPlan lastPlan() {
        return lastPlan;
    }

    public static String lastExplain() {
        return lastPlan == null ? "MemoryPlan{unavailable}" : lastPlan.explain();
    }

    public static graph.optimizer.memory.MemoryPlanSummary lastSummary() {
        return lastPlan == null ? null : lastPlan.summary();
    }

    public MemoryPlannerPolicy policy() {
        return policy;
    }

    private List<Tensor> applyFloat64(List<Tensor> sortedGraph, MemoryPlan plan) {
        Map<Integer, double[]> slotBuffers = new HashMap<>();

        for (Tensor tensor : sortedGraph) {
            if (tensor.getOperation() == null) {
                continue;
            }

            NodeLifetime lifetime = plan.lifetimeOf(tensor);
            if (lifetime.role() == MemoryRole.VIEW_ALIAS) {
                tensor.aliasRuntimeFrom(lifetime.storageOwner());
            } else {
                Integer slotId = plan.slotIdOf(tensor);
                if (slotId != null) {
                    int slotSize = plan.slotSize(slotId);
                    double[] buffer = slotBuffers.computeIfAbsent(slotId, ignored -> new double[slotSize]);
                    Arrays.fill(buffer, 0.0d);
                    tensor.setData(buffer);
                } else {
                    tensor.setData(new double[tensor.getFlatDataSize()]);
                }
            }
        }

        return sortedGraph;
    }

    private List<Tensor> applyFloat32(List<Tensor> sortedGraph, MemoryPlan plan) {
        Map<Integer, float[]> slotBuffers = new HashMap<>();

        for (Tensor tensor : sortedGraph) {
            if (tensor.getOperation() == null) {
                continue;
            }

            NodeLifetime lifetime = plan.lifetimeOf(tensor);
            if (lifetime.role() == MemoryRole.VIEW_ALIAS) {
                tensor.aliasRuntimeFrom(lifetime.storageOwner());
            } else {
                Integer slotId = plan.slotIdOf(tensor);
                if (slotId != null) {
                    int slotSize = plan.slotSize(slotId);
                    float[] buffer = slotBuffers.computeIfAbsent(slotId, ignored -> new float[slotSize]);
                    Arrays.fill(buffer, 0.0f);
                    tensor.setFloat32Data(buffer);
                } else {
                    tensor.setFloat32Data(new float[tensor.getFlatDataSize()]);
                }
            }
        }

        return sortedGraph;
    }

    private static MemoryPlannerPolicy resolvePolicyFromProperties() {
        boolean separatePools = Boolean.parseBoolean(
                System.getProperty("cg.optimizer.memory.separateForwardBackwardPools", "true")
        );
        boolean allowCrossPhaseReuse = Boolean.parseBoolean(
                System.getProperty("cg.optimizer.memory.allowCrossPhaseReuse", "false")
        );
        boolean allowLargerBufferReuse = Boolean.parseBoolean(
                System.getProperty("cg.optimizer.memory.allowLargerBufferReuse", "false")
        );
        int minReusableBufferSize = Integer.parseInt(
                System.getProperty("cg.optimizer.memory.minReusableBufferSize", "1")
        );
        return new MemoryPlannerPolicy(
                separatePools,
                allowCrossPhaseReuse,
                allowLargerBufferReuse,
                minReusableBufferSize
        );
    }
}
