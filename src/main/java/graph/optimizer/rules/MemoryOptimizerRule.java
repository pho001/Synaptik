package graph.optimizer.rules;

import graph.optimizer.OptimizationRule;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryRole;
import graph.optimizer.memory.NodeLifetime;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MemoryOptimizerRule implements OptimizationRule {
    private static final boolean ENABLE_MEMORY_REUSE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.enableMemoryReuse", "true"));

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (!ENABLE_MEMORY_REUSE || sortedGraph == null || sortedGraph.isEmpty()) {
            return sortedGraph;
        }

        DataType graphType = validateUniformGraphType(sortedGraph);
        if (graphType == null || graphType == DataType.FLOAT16) {
            return sortedGraph;
        }

        MemoryPlan plan = MemoryPlanner.plan(sortedGraph);
        return switch (graphType) {
            case FLOAT64 -> applyFloat64(sortedGraph, plan);
            case FLOAT32 -> applyFloat32(sortedGraph, plan);
            case FLOAT16 -> sortedGraph;
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

    private List<Tensor> applyFloat64(List<Tensor> sortedGraph, MemoryPlan plan) {
        Map<Integer, ArrayDeque<double[]>> pool = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor tensor = sortedGraph.get(i);
            if (tensor.getOperation() == null) {
                continue;
            }

            NodeLifetime lifetime = plan.lifetimeOf(tensor);
            if (lifetime.role() == MemoryRole.VIEW_ALIAS) {
                tensor.aliasRuntimeFrom(lifetime.storageOwner());
            } else if (plan.isReusableOwner(tensor)) {
                int size = tensor.getFlatDataSize();
                double[] buffer = acquireF64(size, pool);
                if (buffer != null) {
                    Arrays.fill(buffer, 0.0d);
                    tensor.setData(buffer);
                } else {
                    tensor.setData(new double[size]);
                }
            } else {
                tensor.setData(new double[tensor.getFlatDataSize()]);
            }

            releaseInputsF64(sortedGraph, plan, i, tensor, pool);
        }

        return sortedGraph;
    }

    private List<Tensor> applyFloat32(List<Tensor> sortedGraph, MemoryPlan plan) {
        Map<Integer, ArrayDeque<float[]>> pool = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor tensor = sortedGraph.get(i);
            if (tensor.getOperation() == null) {
                continue;
            }

            NodeLifetime lifetime = plan.lifetimeOf(tensor);
            if (lifetime.role() == MemoryRole.VIEW_ALIAS) {
                tensor.aliasRuntimeFrom(lifetime.storageOwner());
            } else if (plan.isReusableOwner(tensor)) {
                int size = tensor.getFlatDataSize();
                float[] buffer = acquireF32(size, pool);
                if (buffer != null) {
                    Arrays.fill(buffer, 0.0f);
                    tensor.setFloat32Data(buffer);
                } else {
                    tensor.setFloat32Data(new float[size]);
                }
            } else {
                tensor.setFloat32Data(new float[tensor.getFlatDataSize()]);
            }

            releaseInputsF32(sortedGraph, plan, i, tensor, pool);
        }

        return sortedGraph;
    }

    private void releaseInputsF64(
            List<Tensor> sortedGraph,
            MemoryPlan plan,
            int index,
            Tensor consumer,
            Map<Integer, ArrayDeque<double[]>> pool
    ) {
        List<Tensor> inputs = consumer.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return;
        }

        Set<Tensor> releasedOwners = new HashSet<>();
        for (Tensor input : inputs) {
            Tensor owner = plan.storageOwnerOf(input);
            if (!releasedOwners.add(owner)) {
                continue;
            }
            if (!plan.isReusableOwner(owner)) {
                continue;
            }
            if (plan.lastReadIndexOf(owner) != index) {
                continue;
            }
            double[] data = owner.getFloat64Data();
            if (data != null) {
                releaseF64(data, pool);
            }
        }
    }

    private void releaseInputsF32(
            List<Tensor> sortedGraph,
            MemoryPlan plan,
            int index,
            Tensor consumer,
            Map<Integer, ArrayDeque<float[]>> pool
    ) {
        List<Tensor> inputs = consumer.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return;
        }

        Set<Tensor> releasedOwners = new HashSet<>();
        for (Tensor input : inputs) {
            Tensor owner = plan.storageOwnerOf(input);
            if (!releasedOwners.add(owner)) {
                continue;
            }
            if (!plan.isReusableOwner(owner)) {
                continue;
            }
            if (plan.lastReadIndexOf(owner) != index) {
                continue;
            }
            float[] data = owner.getFloat32Data();
            if (data != null) {
                releaseF32(data, pool);
            }
        }
    }

    private double[] acquireF64(int size, Map<Integer, ArrayDeque<double[]>> pool) {
        ArrayDeque<double[]> buffers = pool.get(size);
        return buffers == null || buffers.isEmpty() ? null : buffers.pop();
    }

    private void releaseF64(double[] buffer, Map<Integer, ArrayDeque<double[]>> pool) {
        pool.computeIfAbsent(buffer.length, ignored -> new ArrayDeque<>()).push(buffer);
    }

    private float[] acquireF32(int size, Map<Integer, ArrayDeque<float[]>> pool) {
        ArrayDeque<float[]> buffers = pool.get(size);
        return buffers == null || buffers.isEmpty() ? null : buffers.pop();
    }

    private void releaseF32(float[] buffer, Map<Integer, ArrayDeque<float[]>> pool) {
        pool.computeIfAbsent(buffer.length, ignored -> new ArrayDeque<>()).push(buffer);
    }
}
