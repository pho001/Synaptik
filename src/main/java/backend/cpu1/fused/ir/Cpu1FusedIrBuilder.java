package backend.cpu1.fused.ir;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.TensorMetadata;
import tensor.layout.BroadcastPlan;
import tensor.layout.BroadcastPlanner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.function.IntFunction;

public final class Cpu1FusedIrBuilder {
    private Cpu1FusedIrBuilder() {
    }

    public static Cpu1FusedExpressionPlan build(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (compiledNodeResolver == null) {
            throw new IllegalArgumentException("compiledNodeResolver cannot be null");
        }
        if (descriptorIndex == null) {
            throw new IllegalArgumentException("descriptorIndex cannot be null");
        }
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }
        return build(
                orderedNodeIds,
                externalInputNodeIds(orderedNodeIds, compiledNodeResolver),
                compiledNodeResolver,
                descriptorIndex
        );
    }

    public static Cpu1FusedExpressionPlan build(
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (externalInputNodeIds == null) {
            throw new IllegalArgumentException("externalInputNodeIds cannot be null");
        }
        if (compiledNodeResolver == null) {
            throw new IllegalArgumentException("compiledNodeResolver cannot be null");
        }
        if (descriptorIndex == null) {
            throw new IllegalArgumentException("descriptorIndex cannot be null");
        }
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }

        java.util.HashMap<Integer, Integer> refs = new java.util.HashMap<>();
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            refs.put(externalInputNodeIds.get(i), i);
        }

        CompiledTensorDescriptor outputDescriptor = descriptorIndex.byNodeId(orderedNodeIds.getLast());
        List<Cpu1FusedInputPlan> inputPlans = buildInputPlans(externalInputNodeIds, outputDescriptor, descriptorIndex);

        List<Cpu1FusedNodePlan> nodes = new ArrayList<>(orderedNodeIds.size());
        for (int i = 0; i < orderedNodeIds.size(); i++) {
            int nodeId = orderedNodeIds.get(i);
            CompiledNode node = compiledNodeResolver.apply(nodeId);
            if (node == null || node.operation() == null) {
                throw new IllegalArgumentException("Fused node " + nodeId + " does not have an operation");
            }
            int outputRef = externalInputNodeIds.size() + i;
            List<Integer> inputRefs = new ArrayList<>(node.inputIds().size());
            for (int inputNodeId : node.inputIds()) {
                Integer ref = refs.get(inputNodeId);
                if (ref == null) {
                    throw new IllegalStateException("Missing fused input reference for nodeId=" + nodeId);
                }
                inputRefs.add(ref);
            }
            refs.put(nodeId, outputRef);
            CanonicalNode canonical = canonicalize(node.operation(), inputRefs);
            nodes.add(new Cpu1FusedNodePlan(
                    i,
                    nodeId,
                    canonical.opType(),
                    canonical.inputRefs(),
                    outputRef,
                    descriptorIndex.byNodeId(nodeId).dataType(),
                    canonical.scalarParameter()
            ));
        }

        Integer outputRef = refs.get(orderedNodeIds.getLast());
        if (outputRef == null) {
            throw new IllegalStateException("Missing output ref for fused root nodeId=" + orderedNodeIds.getLast());
        }
        return new Cpu1FusedExpressionPlan(nodes, inputPlans, outputRef);
    }

    private static List<Integer> externalInputNodeIds(
            List<Integer> orderedNodeIds,
            IntFunction<CompiledNode> compiledNodeResolver
    ) {
        LinkedHashSet<Integer> chainNodeIds = new LinkedHashSet<>(orderedNodeIds);
        LinkedHashSet<Integer> externalInputs = new LinkedHashSet<>();
        for (int nodeId : orderedNodeIds) {
            CompiledNode node = requireNode(compiledNodeResolver, nodeId, "fused unit");
            for (int inputNodeId : node.inputIds()) {
                if (chainNodeIds.contains(inputNodeId)) {
                    continue;
                }
                requireNode(compiledNodeResolver, inputNodeId, "fused unit input");
                externalInputs.add(inputNodeId);
            }
        }
        return List.copyOf(externalInputs);
    }

    private static CompiledNode requireNode(
            IntFunction<CompiledNode> compiledNodeResolver,
            int nodeId,
            String context
    ) {
        CompiledNode node = compiledNodeResolver.apply(nodeId);
        if (node == null) {
            throw new IllegalStateException("Missing compiled node for " + context + " nodeId=" + nodeId);
        }
        return node;
    }

    private static CanonicalNode canonicalize(Operation operation, List<Integer> inputRefs) {
        if (operation instanceof pow p && inputRefs.size() == 1) {
            double exponent = p.getExponent();
            int inputRef = inputRefs.getFirst();
            if (exponent == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        Cpu1FusedScalarParameter.of(1.0f, 1.0d)
                );
            }
            if (exponent == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (exponent == -1.0d) {
                return new CanonicalNode(Operation.OpType.INV, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (exponent == 2.0d) {
                return new CanonicalNode(Operation.OpType.MUL, List.of(inputRef, inputRef), Cpu1FusedScalarParameter.NONE);
            }
        }
        if (operation instanceof mulScalar m && inputRefs.size() == 1) {
            double scalar = m.getScalar();
            int inputRef = inputRefs.getFirst();
            if (scalar == 0.0d) {
                return new CanonicalNode(
                        Operation.OpType.CONST_SCALAR,
                        List.of(),
                        Cpu1FusedScalarParameter.of(0.0f, 0.0d)
                );
            }
            if (scalar == 1.0d) {
                return new CanonicalNode(Operation.OpType.NOOP, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
            if (scalar == -1.0d) {
                return new CanonicalNode(Operation.OpType.NEG, List.of(inputRef), Cpu1FusedScalarParameter.NONE);
            }
        }
        return new CanonicalNode(operation.opType(), List.copyOf(inputRefs), scalarParameter(operation));
    }

    private static Cpu1FusedScalarParameter scalarParameter(Operation operation) {
        if (operation instanceof pow p) {
            return Cpu1FusedScalarParameter.of(p.getExponentF32(), p.getExponent());
        }
        if (operation instanceof mulScalar m) {
            return Cpu1FusedScalarParameter.of(m.getScalarF32(), m.getScalar());
        }
        if (operation instanceof clampMin c) {
            return Cpu1FusedScalarParameter.of(c.getMinValueF32(), c.getMinValue());
        }
        if (operation instanceof clampMax c) {
            return Cpu1FusedScalarParameter.of(c.getMaxValueF32(), c.getMaxValue());
        }
        return Cpu1FusedScalarParameter.NONE;
    }

    private static List<Cpu1FusedInputPlan> buildInputPlans(
            List<Integer> externalInputNodeIds,
            CompiledTensorDescriptor output,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        int[] outShape = output.shape();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        List<Cpu1FusedInputPlan> plans = new ArrayList<>(externalInputNodeIds.size());
        for (int i = 0; i < externalInputNodeIds.size(); i++) {
            int inputNodeId = externalInputNodeIds.get(i);
            CompiledTensorDescriptor input = descriptorIndex.byNodeId(inputNodeId);
            BroadcastPlan plan = BroadcastPlanner.plan(
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides
            );
            if (!Arrays.equals(plan.outShape(), outShape)) {
                throw new IllegalArgumentException("Fused broadcast shape mismatch for external input nodeId=" + inputNodeId);
            }
            int[] effectiveStrides = plan.aEffStrides();
            plans.add(new Cpu1FusedInputPlan(
                    i,
                    inputNodeId,
                    input.dataType(),
                    input.shape(),
                    input.strides(),
                    outShape,
                    outDenseStrides,
                    input.storageOffset(),
                    effectiveStrides,
                    classifyAccessKind(effectiveStrides, outDenseStrides, input.storageOffset())
            ));
        }
        return List.copyOf(plans);
    }

    private static Cpu1FusedAccessKind classifyAccessKind(
            int[] effectiveStrides,
            int[] denseStrides,
            int storageOffset
    ) {
        boolean broadcast = false;
        for (int stride : effectiveStrides) {
            if (stride == 0) {
                broadcast = true;
                break;
            }
        }
        if (broadcast) {
            return Cpu1FusedAccessKind.BROADCAST_STRIDED;
        }
        if (storageOffset != 0) {
            return Arrays.equals(effectiveStrides, denseStrides)
                    ? Cpu1FusedAccessKind.OFFSET_CONTIGUOUS
                    : Cpu1FusedAccessKind.OFFSET_STRIDED;
        }
        return Arrays.equals(effectiveStrides, denseStrides)
                ? Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                : Cpu1FusedAccessKind.DIRECT_STRIDED;
    }

    private record CanonicalNode(
            Operation.OpType opType,
            List<Integer> inputRefs,
            Cpu1FusedScalarParameter scalarParameter
    ) {
    }
}
