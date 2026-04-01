package graph.codegen;

import operations.Operation;
import operations.mulScalar;
import operations.pow;
import operations.sum;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

public final class FusedPlanBuilder {
    private FusedPlanBuilder() {}

    public static FusedExpressionPlan build(
            List<Tensor> cluster,
            List<Tensor> externalInputs,
            Tensor root
    ) {
        Objects.requireNonNull(cluster, "cluster cannot be null");
        Objects.requireNonNull(externalInputs, "externalInputs cannot be null");
        Objects.requireNonNull(root, "root cannot be null");

        List<Tensor> safeCluster = List.copyOf(FusedAsmSupport.buildTopologicalOrder(root, cluster));
        List<Tensor> safeExternalInputs = List.copyOf(externalInputs);

        Snapshot snapshot = snapshotCluster(safeCluster, root);
        List<Tensor> compiledCluster = snapshot.cluster();
        Tensor compiledRoot = snapshot.root();

        IdentityHashMap<Tensor, Integer> refs = new IdentityHashMap<>();
        for (int i = 0; i < safeExternalInputs.size(); i++) {
            refs.put(safeExternalInputs.get(i), i);
        }

        List<FusedExternalInputPlan> inputPlans = buildInputPlans(safeExternalInputs, compiledRoot);

        List<FusedNodePlan> nodes = new ArrayList<>(compiledCluster.size());
        for (int i = 0; i < compiledCluster.size(); i++) {
            Tensor tensor = compiledCluster.get(i);
            Operation operation = tensor.getOperation();
            if (operation == null) {
                throw new IllegalArgumentException("Cluster node at index " + i + " does not have an operation");
            }

            int outputRef = safeExternalInputs.size() + i;
            List<Integer> inputRefs = new ArrayList<>();

            List<Tensor> prev = tensor.getPrevTensors();
            if (prev != null) {
                for (Tensor parent : prev) {
                    Integer ref = refs.get(parent);
                    if (ref == null) {
                        throw new IllegalStateException(
                                "Missing fused input reference for node " + tensor.getLabel()
                        );
                    }
                    inputRefs.add(ref);
                }
            }

            refs.put(tensor, outputRef);

            nodes.add(new FusedNodePlan(
                    i,
                    operation.opType(),
                    inputRefs,
                    outputRef,
                    extractParameter(operation)
            ));
        }

        Integer outputRef = refs.get(compiledRoot);
        if (outputRef == null) {
            throw new IllegalStateException("Missing output ref for fused root");
        }

        return new FusedExpressionPlan(
                nodes,
                inputPlans,
                outputRef
        );
    }

    private static Object extractParameter(Operation operation) {
        if (operation instanceof pow p) {
            return p.getExponent();
        }
        if (operation instanceof mulScalar m) {
            return m.getScalar();
        }
        if (operation instanceof sum s) {
            return s.getDimension();
        }
        return null;
    }

    private static Snapshot snapshotCluster(List<Tensor> cluster, Tensor root) {
        IdentityHashMap<Tensor, Tensor> clones = new IdentityHashMap<>();
        for (Tensor original : cluster) {
            Tensor clone = new Tensor(
                    original.getShape().clone(),
                    new ArrayList<>(),
                    original.getOperation(),
                    original.getLabel(),
                    original.getDataType()
            );
            clone.setBackward(original.isBackward());
            clones.put(original, clone);
        }

        for (Tensor original : cluster) {
            Tensor clone = clones.get(original);
            List<Tensor> prev = original.getPrevTensors();
            if (prev == null) {
                clone.setPrevTensors(null);
                continue;
            }
            List<Tensor> mappedPrev = new ArrayList<>(prev.size());
            for (Tensor parent : prev) {
                mappedPrev.add(clones.getOrDefault(parent, parent));
            }
            clone.setPrevTensors(mappedPrev);
        }

        List<Tensor> clusterCopy = new ArrayList<>(cluster.size());
        for (Tensor original : cluster) {
            clusterCopy.add(clones.get(original));
        }
        Tensor rootCopy = clones.get(root);
        if (rootCopy == null) {
            throw new IllegalStateException("Missing fused root clone");
        }
        return new Snapshot(List.copyOf(clusterCopy), rootCopy);
    }

    private static List<FusedExternalInputPlan> buildInputPlans(List<Tensor> externalInputs, Tensor outputTensor) {
        int[] outShape = outputTensor.getShape();
        int[] outStrides = tensor.TensorMetadata.computeStrides(outShape);
        List<FusedExternalInputPlan> plans = new ArrayList<>(externalInputs.size());
        for (int i = 0; i < externalInputs.size(); i++) {
            Tensor input = externalInputs.get(i);
            tensor.BroadcastPlan plan = tensor.BroadcastPlanner.plan(input.getShape(), input.getStrides(), outShape, outStrides);
            if (!java.util.Arrays.equals(plan.outShape(), outShape)) {
                throw new IllegalArgumentException("Fused broadcast shape mismatch for external input");
            }
            int[] eff = plan.aEffStrides();
            boolean direct = java.util.Arrays.equals(eff, outStrides);
            plans.add(new FusedExternalInputPlan(i, direct, outShape, outStrides, eff));
        }
        return java.util.List.copyOf(plans);
    }

    private record Snapshot(List<Tensor> cluster, Tensor root) {}
}
