package graph.optimizer.rules;

import graph.optimizer.OptimizationRule;
import tensor.DataType;
import tensor.Tensor;
import java.util.*;

public class MemoryOptimizerRule implements OptimizationRule {
    private static final boolean ENABLE_MEMORY_REUSE =
            Boolean.parseBoolean(System.getProperty("cg.optimizer.enableMemoryReuse", "true"));

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (!ENABLE_MEMORY_REUSE) {
            return sortedGraph;
        }
        if (sortedGraph.isEmpty()) return sortedGraph;
        DataType graphType = sortedGraph.get(0).getDataType();
        for (Tensor t : sortedGraph) {
            if (t.getDataType() != graphType) {
                return sortedGraph;
            }
        }
        if (graphType == DataType.FLOAT64) {
            return applyFloat64(sortedGraph);
        }
        if (graphType == DataType.FLOAT32) {
            return applyFloat32(sortedGraph);
        }
        return sortedGraph;
    }

    private List<Tensor> applyFloat64(List<Tensor> sortedGraph) {
        StoragePlan plan = buildStoragePlan(sortedGraph);
        Map<Integer, Stack<double[]>> freeBuffers = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getOperation() == null) {
                continue;
            }

            if (aliasesInput0AtRuntime(t)) {
                t.aliasRuntimeFrom(t.getPrevTensors().get(0));
            } else {
                int size = t.getFlatDataSize();
                double[] buffer = getFromPool(size, freeBuffers);
                if (buffer != null) {
                    Arrays.fill(buffer, 0.0d);
                    t.setData(buffer);
                } else {
                    t.setData(new double[size]);
                }
            }

            List<Tensor> prev = t.getPrevTensors();
            if (prev == null || prev.isEmpty()) {
                continue;
            }
            Set<Integer> releasedClasses = new HashSet<>();
            for (Tensor input : prev) {
                if (input.getOperation() == null) {
                    continue;
                }
                int cls = plan.storageClassOf(input);
                if (!releasedClasses.add(cls)) {
                    continue;
                }
                if (plan.lastUseOfClass(cls) != i) {
                    continue;
                }

                double[] inputData = input.getFloat64Data();
                double[] outputData = t.getFloat64Data();
                if (inputData == null) {
                    continue;
                }
                if (outputData != null && inputData == outputData) {
                    continue;
                }
                releaseToPool(inputData, freeBuffers);
            }
        }
        return sortedGraph;
    }

    private List<Tensor> applyFloat32(List<Tensor> sortedGraph) {
        StoragePlan plan = buildStoragePlan(sortedGraph);
        Map<Integer, Stack<float[]>> freeBuffers = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getOperation() == null) {
                continue;
            }

            if (aliasesInput0AtRuntime(t)) {
                t.aliasRuntimeFrom(t.getPrevTensors().get(0));
            } else {
                int size = t.getFlatDataSize();
                float[] buffer = getFromPoolF32(size, freeBuffers);
                if (buffer != null) {
                    Arrays.fill(buffer, 0.0f);
                    t.setFloat32Data(buffer);
                } else {
                    t.setFloat32Data(new float[size]);
                }
            }

            List<Tensor> prev = t.getPrevTensors();
            if (prev == null || prev.isEmpty()) {
                continue;
            }
            Set<Integer> releasedClasses = new HashSet<>();
            for (Tensor input : prev) {
                if (input.getOperation() == null) {
                    continue;
                }
                int cls = plan.storageClassOf(input);
                if (!releasedClasses.add(cls)) {
                    continue;
                }
                if (plan.lastUseOfClass(cls) != i) {
                    continue;
                }

                float[] inputData = input.getFloat32Data();
                float[] outputData = t.getFloat32Data();
                if (inputData == null) {
                    continue;
                }
                if (outputData != null && inputData == outputData) {
                    continue;
                }
                releaseToPoolF32(inputData, freeBuffers);
            }
        }
        return sortedGraph;
    }

    private StoragePlan buildStoragePlan(List<Tensor> sortedGraph) {
        int n = sortedGraph.size();
        if (n == 0) {
            return new StoragePlan(Collections.emptyMap(), new int[0]);
        }
        Map<Tensor, Integer> indexByTensor = new HashMap<>(n * 2);
        for (int i = 0; i < n; i++) {
            indexByTensor.put(sortedGraph.get(i), i);
        }

        UnionFind uf = new UnionFind(n);
        for (Tensor t : sortedGraph) {
            if (!aliasesInput0AtRuntime(t)) {
                continue;
            }
            List<Tensor> prev = t.getPrevTensors();
            if (prev == null || prev.isEmpty()) {
                continue;
            }
            Integer tid = indexByTensor.get(t);
            Integer pid = indexByTensor.get(prev.get(0));
            if (tid != null && pid != null) {
                uf.union(tid, pid);
            }
        }

        int[] classLastUse = new int[n];
        Arrays.fill(classLastUse, -1);
        int[] consumerCounts = new int[n];

        for (int i = 0; i < n; i++) {
            Tensor t = sortedGraph.get(i);
            List<Tensor> prev = t.getPrevTensors();
            if (prev == null || prev.isEmpty()) {
                continue;
            }
            for (Tensor input : prev) {
                Integer pid = indexByTensor.get(input);
                if (pid == null) {
                    continue;
                }
                consumerCounts[pid]++;
                int cls = uf.find(pid);
                classLastUse[cls] = Math.max(classLastUse[cls], i);
            }
        }

        for (int i = 0; i < n; i++) {
            if (consumerCounts[i] == 0) {
                int cls = uf.find(i);
                classLastUse[cls] = Integer.MAX_VALUE;
            }
        }

        for (int i = 0; i < n; i++) {
            classLastUse[i] = classLastUse[uf.find(i)];
        }
        return new StoragePlan(indexByTensor, classLastUse);
    }

    private boolean aliasesInput0AtRuntime(Tensor t) {
        if (t == null || t.getOperation() == null) {
            return false;
        }
        List<Tensor> prev = t.getPrevTensors();
        if (prev == null || prev.isEmpty()) {
            return false;
        }
        return switch (t.getOperation().opType()) {
            case NOOP, PERMUTE -> true;
            case RESHAPE, EXPAND_DIMS, SQUEEZE -> prev.get(0).isContiguous();
            default -> false;
        };
    }

    private static final class StoragePlan {
        private final Map<Tensor, Integer> tensorIndex;
        private final int[] classLastUse;

        private StoragePlan(Map<Tensor, Integer> tensorIndex, int[] classLastUse) {
            this.tensorIndex = tensorIndex;
            this.classLastUse = classLastUse;
        }

        int storageClassOf(Tensor t) {
            Integer idx = tensorIndex.get(t);
            return idx == null ? -1 : idx;
        }

        int lastUseOfClass(int cls) {
            if (cls < 0 || cls >= classLastUse.length) {
                return Integer.MAX_VALUE;
            }
            return classLastUse[cls];
        }
    }

    private static final class UnionFind {
        private final int[] parent;
        private final int[] rank;

        private UnionFind(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        int find(int x) {
            int p = parent[x];
            if (p != x) {
                parent[x] = find(p);
            }
            return parent[x];
        }

        void union(int a, int b) {
            int ra = find(a);
            int rb = find(b);
            if (ra == rb) {
                return;
            }
            if (rank[ra] < rank[rb]) {
                parent[ra] = rb;
            } else if (rank[ra] > rank[rb]) {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra]++;
            }
        }
    }

    private double[] getFromPool(int size, Map<Integer, Stack<double[]>> pool) {
        Stack<double[]> buffers = pool.get(size);
        return (buffers != null && !buffers.isEmpty()) ? buffers.pop() : null;
    }

    private void releaseToPool(double[] buffer, Map<Integer, Stack<double[]>> pool) {
        if (buffer == null) return;
        pool.computeIfAbsent(buffer.length, k -> new Stack<>()).push(buffer);
    }

    private float[] getFromPoolF32(int size, Map<Integer, Stack<float[]>> pool) {
        Stack<float[]> buffers = pool.get(size);
        return (buffers != null && !buffers.isEmpty()) ? buffers.pop() : null;
    }

    private void releaseToPoolF32(float[] buffer, Map<Integer, Stack<float[]>> pool) {
        if (buffer == null) return;
        pool.computeIfAbsent(buffer.length, k -> new Stack<>()).push(buffer);
    }
}
