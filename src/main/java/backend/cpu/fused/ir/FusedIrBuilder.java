package backend.cpu.fused.ir;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import tensor.DataType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Internal builder that lowers a tensor cluster into a fused expression plan.
 */
public final class FusedIrBuilder {
    private FusedIrBuilder() {}

    public static FusedExpressionPlan build(
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            java.util.function.IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        Objects.requireNonNull(orderedNodeIds, "orderedNodeIds cannot be null");
        Objects.requireNonNull(externalInputNodeIds, "externalInputNodeIds cannot be null");
        Objects.requireNonNull(compiledNodeResolver, "compiledNodeResolver cannot be null");
        Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }

        java.util.HashMap<Integer, Integer> refs = new java.util.HashMap<>();
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            refs.put(externalInputNodeIds.get(i), i);
        }

        CompiledTensorDescriptor outputDescriptor = descriptorIndex.byNodeId(orderedNodeIds.getLast());
        List<FusedExternalInputPlan> inputPlans = buildInputPlans(externalInputNodeIds, outputDescriptor, descriptorIndex);

        List<FusedNodePlan> nodes = new ArrayList<>(orderedNodeIds.size());
        for (int i = 0; i < orderedNodeIds.size(); i++) {
            int nodeId = orderedNodeIds.get(i);
            CompiledNode node = compiledNodeResolver.apply(nodeId);
            if (node == null || node.operation() == null) {
                throw new IllegalArgumentException("Fused node " + nodeId + " does not have an operation");
            }
            int outputRef = externalInputNodeIds.size() + i;
            List<Integer> inputRefs = new ArrayList<>();
            for (int inputNodeId : node.inputIds()) {
                Integer ref = refs.get(inputNodeId);
                if (ref == null) {
                    throw new IllegalStateException("Missing fused input reference for nodeId=" + nodeId);
                }
                inputRefs.add(ref);
            }
            refs.put(nodeId, outputRef);
            CanonicalNode canonical = canonicalize(node.operation(), inputRefs);
            nodes.add(new FusedNodePlan(
                    i,
                    canonical.opType(),
                    canonical.inputRefs(),
                    outputRef,
                    descriptorIndex.byNodeId(nodeId).dataType(),
                    canonical.attributes()
            ));
        }

        Integer outputRef = refs.get(orderedNodeIds.getLast());
        if (outputRef == null) {
            throw new IllegalStateException("Missing output ref for fused root nodeId=" + orderedNodeIds.getLast());
        }
        return new FusedExpressionPlan(nodes, inputPlans, outputRef);
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

    private static List<FusedExternalInputPlan> buildInputPlans(
            List<Integer> externalInputNodeIds,
            CompiledTensorDescriptor output,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        int[] outShape = output.shape();
        int[] outDenseStrides = tensor.TensorMetadata.computeStrides(outShape);
        List<FusedExternalInputPlan> plans = new ArrayList<>(externalInputNodeIds.size());
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            CompiledTensorDescriptor input = descriptorIndex.byNodeId(externalInputNodeIds.get(i));
            tensor.layout.BroadcastPlan plan = tensor.layout.BroadcastPlanner.plan(
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides
            );
            if (!java.util.Arrays.equals(plan.outShape(), outShape)) {
                throw new IllegalArgumentException("Fused broadcast shape mismatch for external input nodeId=" + input.nodeId());
            }
            int[] eff = plan.aEffStrides();
            plans.add(new FusedExternalInputPlan(
                    i,
                    input.dataType(),
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides,
                    input.storageOffset(),
                    eff,
                    classifyAccessKind(eff, outDenseStrides, input.storageOffset())
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

    private record CanonicalNode(
            Operation.OpType opType,
            List<Integer> inputRefs,
            FusedNodeAttributes attributes
    ) {}
}
