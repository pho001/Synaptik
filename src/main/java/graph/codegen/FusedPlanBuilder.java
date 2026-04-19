package graph.codegen;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.Tensor;
import tensor.DataType;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.Arrays;
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
            CanonicalNode canonical = canonicalize(operation, inputRefs);

            nodes.add(new FusedNodePlan(
                    i,
                    canonical.opType(),
                    canonical.inputRefs(),
                    outputRef,
                    tensor.getDataType(),
                    canonical.attributes()
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

    private static CanonicalNode canonicalize(Operation operation, List<Integer> inputRefs) {
        if (operation instanceof pow p && inputRefs.size() == 1) {
            double exponent = p.getExponent();
            int inputRef = inputRefs.get(0);
            if (exponent == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        new ScalarDoubleAttribute(1.0d)
                );
            }
            if (exponent == 1.0d) {
                return new CanonicalNode(
                        Operation.OpType.NOOP,
                        List.of(inputRef),
                        NoAttributes.INSTANCE
                );
            }
            if (exponent == -1.0d) {
                return new CanonicalNode(
                        Operation.OpType.INV,
                        List.of(inputRef),
                        NoAttributes.INSTANCE
                );
            }
            if (exponent == 2.0d) {
                return new CanonicalNode(
                        Operation.OpType.MUL,
                        List.of(inputRef, inputRef),
                        NoAttributes.INSTANCE
                );
            }
        }
        if (operation instanceof mulScalar m && inputRefs.size() == 1) {
            double scalar = m.getScalar();
            int inputRef = inputRefs.get(0);
            if (scalar == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        new ScalarDoubleAttribute(0.0d)
                );
            }
            if (scalar == 1.0d) {
                return new CanonicalNode(
                        Operation.OpType.NOOP,
                        List.of(inputRef),
                        NoAttributes.INSTANCE
                );
            }
            if (scalar == -1.0d) {
                return new CanonicalNode(
                        Operation.OpType.NEG,
                        List.of(inputRef),
                        NoAttributes.INSTANCE
                );
            }
        }
        return new CanonicalNode(
                operation.opType(),
                List.copyOf(inputRefs),
                extractAttributes(operation)
        );
    }

    private static FusedNodeAttributes extractAttributes(Operation operation) {
        if (operation instanceof pow p) {
            return new ScalarDoubleAttribute(p.getExponent());
        }
        if (operation instanceof mulScalar m) {
            return new ScalarDoubleAttribute(m.getScalar());
        }
        if (operation instanceof clampMin c) {
            return new ScalarDoubleAttribute(c.getMinValue());
        }
        if (operation instanceof clampMax c) {
            return new ScalarDoubleAttribute(c.getMaxValue());
        }
        if (operation.opType() == Operation.OpType.WHERE) {
            return new WhereAttributes();
        }
        return NoAttributes.INSTANCE;
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
            TensorInternalAccess.setBackward(clone, original.isBackward());
            clones.put(original, clone);
        }

        for (Tensor original : cluster) {
            Tensor clone = clones.get(original);
            List<Tensor> prev = original.getPrevTensors();
            if (prev == null) {
                TensorInternalAccess.setPrevTensors(clone, null);
                continue;
            }
            List<Tensor> mappedPrev = new ArrayList<>(prev.size());
            for (Tensor parent : prev) {
                mappedPrev.add(clones.getOrDefault(parent, parent));
            }
            TensorInternalAccess.setPrevTensors(clone, mappedPrev);
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
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        List<FusedExternalInputPlan> plans = new ArrayList<>(externalInputs.size());
        for (int i = 0; i < externalInputs.size(); i++) {
            Tensor input = externalInputs.get(i);
            tensor.BroadcastPlan plan = tensor.BroadcastPlanner.plan(input.getShape(), input.getStrides(), outShape, outDenseStrides);
            if (!java.util.Arrays.equals(plan.outShape(), outShape)) {
                throw new IllegalArgumentException("Fused broadcast shape mismatch for external input");
            }
            int[] eff = plan.aEffStrides();
            plans.add(new FusedExternalInputPlan(
                    i,
                    input.getDataType(),
                    outShape,
                    outDenseStrides,
                    input.getStorageOffsetUnsafe(),
                    eff,
                    classifyAccessKind(eff, outDenseStrides, input.getStorageOffsetUnsafe())
            ));
        }
        return java.util.List.copyOf(plans);
    }

    private static FusedAccessKind classifyAccessKind(int[] effectiveStrides, int[] denseStrides, int storageOffset) {
        boolean broadcast = false;
        for (int stride : effectiveStrides) {
            if (stride == 0) {
                broadcast = true;
                break;
            }
        }
        if (broadcast) {
            return FusedAccessKind.BROADCAST_STRIDED;
        }
        if (storageOffset != 0) {
            if (Arrays.equals(effectiveStrides, denseStrides)) {
                return FusedAccessKind.OFFSET_CONTIGUOUS;
            }
            return FusedAccessKind.OFFSET_STRIDED;
        }
        if (Arrays.equals(effectiveStrides, denseStrides)) {
            return FusedAccessKind.DIRECT_CONTIGUOUS;
        }
        return FusedAccessKind.DIRECT_STRIDED;
    }

    private record Snapshot(List<Tensor> cluster, Tensor root) {}
    private record CanonicalNode(
            Operation.OpType opType,
            List<Integer> inputRefs,
            FusedNodeAttributes attributes
    ) {}
}
