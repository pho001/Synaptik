package graph.optimizer.partition.apple;

import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;
import graph.optimizer.partition.model.AcceleratorDagInput;
import graph.optimizer.partition.model.AcceleratorDagNode;
import graph.optimizer.partition.model.AcceleratorDagNodeType;
import graph.optimizer.partition.model.AcceleratorDagSpec;
import graph.optimizer.partition.model.AcceleratorSubgraphSpec;
import graph.optimizer.partition.model.AcceleratorDagValueRef;
import graph.optimizer.partition.model.AcceleratorDagValueRefKind;
import graph.optimizer.partition.model.AcceleratorPostOp;
import graph.optimizer.partition.model.AcceleratorPostOpType;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.layout.expandDims;
import operations.layout.permute;
import operations.layout.squeeze;
import operations.reduction.softmax;
import operations.elementwise.where.where;
import tensor.DataType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class AppleGpuSubgraphLowerer {
    public AppleGpuSubgraphLoweringResult tryLower(AcceleratorSubgraphSpec subgraph, BackendPrepareContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }

        CompiledNode compute = context.compiledNode(subgraph.computeNodeId());
        if (compute == null || compute.operation() == null) {
            return null;
        }
        boolean linear = compute.operation().opType() == Operation.OpType.LINEAR;
        if (!linear && compute.operation().opType() != Operation.OpType.MATMUL) {
            return null;
        }
        if ((!linear && compute.inputIds().size() != 2) || (linear && (compute.inputIds().size() != 2 && compute.inputIds().size() != 3))) {
            return null;
        }

        CompiledNode left = context.compiledNode(compute.inputIds().getFirst());
        CompiledNode right = context.compiledNode(compute.inputIds().get(1));
        if (left == null || right == null) {
            return null;
        }
        int[] leftShape = left.shape();
        int[] rightShape = right.shape();
        int[] outShape = compute.shape();
        if (linear) {
            if (leftShape.length != 2 || rightShape.length != 2 || outShape.length != 2) {
                return null;
            }
        } else if (leftShape.length < 2 || leftShape.length > 4
                || rightShape.length < 2 || rightShape.length > 4
                || outShape.length < 2 || outShape.length > 4) {
            return null;
        }
        AcceleratorDagSpec dagSpec = buildDagSpec(subgraph, context);
        if (dagSpec == null) {
            return null;
        }
        AppleGpuMatMulSpec matMulSpec = tryBuildLegacyMatMulSpec(subgraph, context, compute, leftShape, rightShape, outShape);
        long estimatedWork = (long) leftShape[0] * rightShape[1] * leftShape[1]
                + (long) leftShape[0] * rightShape[1] * Math.max(0, dagSpec.nodes().size() - 1);

        return new AppleGpuSubgraphLoweringResult(
                compute.id(),
                matMulSpec,
                dagSpec,
                estimatedWork
        );
    }

    private AppleGpuMatMulSpec tryBuildLegacyMatMulSpec(
            AcceleratorSubgraphSpec subgraph,
            BackendPrepareContext context,
            CompiledNode compute,
            int[] leftShape,
            int[] rightShape,
            int[] outShape
    ) {
        boolean linear = compute.operation().opType() == Operation.OpType.LINEAR;
        int biasInputNodeId = -1;
        boolean biasVector = false;
        ArrayList<AcceleratorPostOp> postOps = new ArrayList<>();
        int outputNodeId = compute.id();

        if (linear && compute.inputIds().size() == 3) {
            CompiledNode bias = context.compiledNode(compute.inputIds().get(2));
            if (bias == null || !bias.contiguous() || bias.hasStorageOffset()) {
                return null;
            }
            int[] biasShape = bias.shape();
            if (biasShape.length != 1 || biasShape[0] != rightShape[1]) {
                return null;
            }
            biasInputNodeId = bias.id();
            biasVector = true;
        }

        List<Integer> nodeIds = subgraph.orderedNodeIds();
        if (nodeIds.size() >= 2) {
            CompiledNode second = context.compiledNode(nodeIds.get(1));
            if (second == null || second.operation() == null) {
                return null;
            }
            if (!linear && second.operation().opType() == Operation.OpType.ADD) {
                if (second.inputIds().size() != 2 || !second.inputIds().contains(compute.id())) {
                    return null;
                }
                int otherInputId = second.inputIds().getFirst() == compute.id()
                        ? second.inputIds().get(1)
                        : second.inputIds().getFirst();
                CompiledNode bias = context.compiledNode(otherInputId);
                if (bias == null || bias.dataType() != compute.dataType() || !bias.contiguous() || bias.hasStorageOffset()) {
                    return null;
                }
                int[] biasShape = bias.shape();
                if (biasShape.length == 1 && biasShape[0] == rightShape[1]) {
                    biasVector = true;
                } else if (!(biasShape.length == 2 && biasShape[0] == outShape[0] && biasShape[1] == rightShape[1])) {
                    return null;
                }
                biasInputNodeId = otherInputId;
                outputNodeId = second.id();
            }
        }
        int postOpStart = (!linear && nodeIds.size() >= 2 && context.compiledNode(nodeIds.get(1)) != null
                && context.compiledNode(nodeIds.get(1)).operation() != null
                && context.compiledNode(nodeIds.get(1)).operation().opType() == Operation.OpType.ADD
                && outputNodeId == nodeIds.get(1))
                ? 2
                : 1;
        for (int i = postOpStart; i < nodeIds.size(); i++) {
            CompiledNode node = context.compiledNode(nodeIds.get(i));
            if (node == null || node.operation() == null) {
                return null;
            }
            AcceleratorPostOp resolved = resolvePostOp(node, outputNodeId, context, compute);
            if (resolved == null) {
                return null;
            }
            postOps.add(resolved);
            outputNodeId = node.id();
        }
        return new AppleGpuMatMulSpec(
                compute.inputIds().getFirst(),
                compute.inputIds().get(1),
                biasInputNodeId,
                outputNodeId,
                leftShape[0],
                rightShape[1],
                leftShape[1],
                biasVector,
                postOps
        );
    }

    private AcceleratorDagSpec buildDagSpec(AcceleratorSubgraphSpec subgraph, BackendPrepareContext context) {
        AcceleratorDagSpec specializedSdpa = tryBuildSpecializedSdpaDagSpec(subgraph, context);
        if (specializedSdpa != null) {
            return specializedSdpa;
        }
        Map<Integer, Integer> externalInputIndex = new HashMap<>();
        List<AcceleratorDagInput> externalInputs = new ArrayList<>(subgraph.externalInputNodeIds().size());
        for (int i = 0; i < subgraph.externalInputNodeIds().size(); i++) {
            int nodeId = subgraph.externalInputNodeIds().get(i);
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null) {
                return null;
            }
            int[] shape = node.shape();
            if (shape.length < 1 || shape.length > 4) {
                return null;
            }
            externalInputIndex.put(nodeId, i);
            externalInputs.add(new AcceleratorDagInput(nodeId, java.util.Arrays.stream(shape).boxed().toList(), node.dataType()));
        }

        Map<Integer, Integer> loweredNodeIndex = new HashMap<>();
        List<AcceleratorDagNode> nodes = new ArrayList<>(subgraph.orderedNodeIds().size());
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node == null || node.operation() == null) {
                return null;
            }
            AcceleratorDagNodeType type = resolveDagNodeType(node.operation().opType());
            if (type == null) {
                return null;
            }
            AcceleratorDagValueRef input0 = resolveDagValueRef(node.inputIds(), 0, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input1 = resolveDagValueRef(node.inputIds(), 1, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input2 = resolveDagValueRef(node.inputIds(), 2, externalInputIndex, loweredNodeIndex);
            AcceleratorDagValueRef input3 = resolveDagValueRef(node.inputIds(), 3, externalInputIndex, loweredNodeIndex);
            if ((node.inputIds().size() >= 1 && input0.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 2 && input1.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 3 && input2.kind() == AcceleratorDagValueRefKind.NONE)
                    || (node.inputIds().size() >= 4 && input3.kind() == AcceleratorDagValueRefKind.NONE)) {
                return null;
            }
            int[] shape = node.shape();
            if (shape.length < 1 || shape.length > 4) {
                return null;
            }
            int scalarValueBits = resolveScalarValueBits(node);
            if (type == AcceleratorDagNodeType.PERMUTE && scalarValueBits == Integer.MIN_VALUE) {
                return null;
            }
            nodes.add(new AcceleratorDagNode(
                    nodeId,
                    type,
                    input0,
                    input1,
                    input2,
                    input3,
                    scalarValueBits,
                    shape.length,
                    shape[0],
                    shape.length >= 2 ? shape[1] : 1,
                    shape.length >= 3 ? shape[2] : 1,
                    shape.length >= 4 ? shape[3] : 1
            ));
            loweredNodeIndex.put(nodeId, nodes.size() - 1);
        }
        int outputNodeId = subgraph.outputNodeIds().isEmpty() ? subgraph.orderedNodeIds().getLast() : subgraph.outputNodeIds().getFirst();
        Integer outputNodeIndex = loweredNodeIndex.get(outputNodeId);
        if (outputNodeIndex == null) {
            return null;
        }
        return new AcceleratorDagSpec(externalInputs, nodes, outputNodeIndex, outputNodeId);
    }

    private AcceleratorDagSpec tryBuildSpecializedSdpaDagSpec(AcceleratorSubgraphSpec subgraph, BackendPrepareContext context) {
        if (subgraph == null || context == null || subgraph.orderedNodeIds().isEmpty()) {
            return null;
        }
        List<Integer> nodeIds = subgraph.orderedNodeIds();
        int outputNodeId = subgraph.outputNodeIds().isEmpty() ? nodeIds.getLast() : subgraph.outputNodeIds().getFirst();
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        if (outputNode == null || outputNode.operation() == null || outputNode.operation().opType() != Operation.OpType.MATMUL) {
            return null;
        }
        if (outputNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode softmaxNode = context.compiledNode(outputNode.inputIds().getFirst());
        CompiledNode valueNode = context.compiledNode(outputNode.inputIds().get(1));
        if (softmaxNode == null
                || softmaxNode.operation() == null
                || softmaxNode.operation().opType() != Operation.OpType.SOFTMAX
                || !(softmaxNode.operation() instanceof softmax softmaxOp)
                || valueNode == null) {
            return null;
        }
        int[] outputShape = outputNode.shape();
        if (outputShape.length < 3 || outputShape.length > 4 || softmaxOp.getDimension() != outputShape.length - 1) {
            return null;
        }

        CompiledNode maskedOrScaled = context.compiledNode(softmaxNode.inputIds().getFirst());
        CompiledNode maskNode = null;
        CompiledNode scoreNode = maskedOrScaled;
        if (maskedOrScaled != null && maskedOrScaled.operation() instanceof where) {
            if (maskedOrScaled.inputIds().size() != 3) {
                return null;
            }
            maskNode = context.compiledNode(maskedOrScaled.inputIds().getFirst());
            scoreNode = context.compiledNode(maskedOrScaled.inputIds().get(1));
            CompiledNode fillNode = context.compiledNode(maskedOrScaled.inputIds().get(2));
            if (maskNode == null
                    || maskNode.dataType() != DataType.BOOL
                    || fillNode == null
                    || fillNode.operation() != null
                    || fillNode.flatDataSize() != 1) {
                return null;
            }
        }
        if (scoreNode == null || scoreNode.operation() == null || scoreNode.operation().opType() != Operation.OpType.MUL_SCALAR) {
            return null;
        }
        if (!(scoreNode.operation() instanceof mulScalar mulScalarOp) || scoreNode.inputIds().size() != 1) {
            return null;
        }
        CompiledNode qkNode = context.compiledNode(scoreNode.inputIds().getFirst());
        if (qkNode == null || qkNode.operation() == null || qkNode.operation().opType() != Operation.OpType.MATMUL || qkNode.inputIds().size() != 2) {
            return null;
        }
        CompiledNode queryNode = context.compiledNode(qkNode.inputIds().getFirst());
        CompiledNode permutedKeyNode = context.compiledNode(qkNode.inputIds().get(1));
        if (queryNode == null || permutedKeyNode == null || permutedKeyNode.operation() == null || permutedKeyNode.operation().opType() != Operation.OpType.PERMUTE) {
            return null;
        }
        int permuteBits = resolveScalarValueBits(permutedKeyNode);
        if (permuteBits == Integer.MIN_VALUE) {
            return null;
        }
        CompiledNode keyNode = context.compiledNode(permutedKeyNode.inputIds().getFirst());
        if (keyNode == null
                || queryNode.dataType() != DataType.FLOAT32
                || keyNode.dataType() != DataType.FLOAT32
                || valueNode.dataType() != DataType.FLOAT32
                || valueNode.shape().length < 3
                || valueNode.shape().length > 4) {
            return null;
        }

        List<AcceleratorDagInput> externalInputs = new ArrayList<>();
        externalInputs.add(new AcceleratorDagInput(queryNode.id(), java.util.Arrays.stream(queryNode.shape()).boxed().toList(), queryNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(keyNode.id(), java.util.Arrays.stream(keyNode.shape()).boxed().toList(), keyNode.dataType()));
        externalInputs.add(new AcceleratorDagInput(valueNode.id(), java.util.Arrays.stream(valueNode.shape()).boxed().toList(), valueNode.dataType()));
        AcceleratorDagValueRef maskRef = AcceleratorDagValueRef.none();
        if (maskNode != null) {
            externalInputs.add(new AcceleratorDagInput(maskNode.id(), java.util.Arrays.stream(maskNode.shape()).boxed().toList(), maskNode.dataType()));
            maskRef = AcceleratorDagValueRef.externalInput(3);
        }

        AcceleratorDagNode sdpaNode = new AcceleratorDagNode(
                outputNode.id(),
                AcceleratorDagNodeType.SDPA,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.externalInput(2),
                maskRef,
                Float.floatToIntBits(mulScalarOp.getScalarF32()),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1
        );
        return new AcceleratorDagSpec(externalInputs, List.of(sdpaNode), 0, outputNode.id());
    }

    private AcceleratorDagValueRef resolveDagValueRef(
            List<Integer> inputIds,
            int inputIndex,
            Map<Integer, Integer> externalInputIndex,
            Map<Integer, Integer> loweredNodeIndex
    ) {
        if (inputIds == null || inputIndex >= inputIds.size()) {
            return AcceleratorDagValueRef.none();
        }
        int inputNodeId = inputIds.get(inputIndex);
        Integer external = externalInputIndex.get(inputNodeId);
        if (external != null) {
            return AcceleratorDagValueRef.externalInput(external);
        }
        Integer nodeOutput = loweredNodeIndex.get(inputNodeId);
        return nodeOutput == null ? AcceleratorDagValueRef.none() : AcceleratorDagValueRef.nodeOutput(nodeOutput);
    }

    private AcceleratorDagNodeType resolveDagNodeType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case MATMUL -> AcceleratorDagNodeType.MATMUL;
            case LINEAR -> AcceleratorDagNodeType.LINEAR;
            case ADD -> AcceleratorDagNodeType.ADD;
            case SUB -> AcceleratorDagNodeType.SUB;
            case MUL -> AcceleratorDagNodeType.MUL;
            case DIV -> AcceleratorDagNodeType.DIV;
            case RELU -> AcceleratorDagNodeType.RELU;
            case TANH, FAST_TANH -> AcceleratorDagNodeType.TANH;
            case SIGMOID -> AcceleratorDagNodeType.SIGMOID;
            case ABS -> AcceleratorDagNodeType.ABS;
            case EXP, FAST_EXP -> AcceleratorDagNodeType.EXP;
            case LOG -> AcceleratorDagNodeType.LOG;
            case NEG -> AcceleratorDagNodeType.NEG;
            case SQRT -> AcceleratorDagNodeType.SQRT;
            case INV -> AcceleratorDagNodeType.INV;
            case CLAMP_MIN -> AcceleratorDagNodeType.CLAMP_MIN;
            case CLAMP_MAX -> AcceleratorDagNodeType.CLAMP_MAX;
            case RESHAPE -> AcceleratorDagNodeType.RESHAPE;
            case CONTIGUOUS, NOOP -> AcceleratorDagNodeType.CONTIGUOUS;
            case PERMUTE -> AcceleratorDagNodeType.PERMUTE;
            case EXPAND_DIMS -> AcceleratorDagNodeType.EXPAND_DIMS;
            case SQUEEZE -> AcceleratorDagNodeType.SQUEEZE;
            case MUL_SCALAR -> AcceleratorDagNodeType.MUL_SCALAR;
            case WHERE -> AcceleratorDagNodeType.WHERE;
            case SOFTMAX -> AcceleratorDagNodeType.SOFTMAX;
            case SCALED_DOT_PRODUCT_ATTENTION -> AcceleratorDagNodeType.SDPA;
            default -> null;
        };
    }

    private int resolveScalarValueBits(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return 0;
        }
        return switch (node.operation().opType()) {
            case CLAMP_MIN -> node.operation() instanceof clampMin clamp ? Float.floatToIntBits(clamp.getMinValueF32()) : 0;
            case CLAMP_MAX -> node.operation() instanceof clampMax clamp ? Float.floatToIntBits(clamp.getMaxValueF32()) : 0;
            case MUL_SCALAR -> node.operation() instanceof mulScalar op ? Float.floatToIntBits(op.getScalarF32()) : Integer.MIN_VALUE;
            case SOFTMAX -> node.operation() instanceof softmax op ? op.getDimension() : Integer.MIN_VALUE;
            case PERMUTE -> encodePermuteMode(node);
            case EXPAND_DIMS -> node.operation() instanceof expandDims op ? op.getAxis() : Integer.MIN_VALUE;
            case SQUEEZE -> node.operation() instanceof squeeze op ? op.getAxis() : Integer.MIN_VALUE;
            default -> 0;
        };
    }

    private int encodePermuteMode(CompiledNode node) {
        if (!(node.operation() instanceof permute permuteOp)) {
            return 0;
        }
        int[] axes = permuteOp.getAxes();
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4 || axes.length != shape.length) {
            return Integer.MIN_VALUE;
        }
        int encoded = shape.length & 0xFF;
        boolean[] seen = new boolean[shape.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= shape.length || seen[axis]) {
                return Integer.MIN_VALUE;
            }
            seen[axis] = true;
            encoded |= (axis & 0xF) << (8 + i * 4);
        }
        return encoded;
    }

    private AcceleratorPostOp resolvePostOp(
            CompiledNode node,
            int currentOutputNodeId,
            BackendPrepareContext context,
            CompiledNode computeNode
    ) {
        if (node.inputIds().size() == 1 && node.inputIds().getFirst() == currentOutputNodeId) {
            AcceleratorPostOpType unaryType = resolveUnaryPostOpType(node.operation().opType());
            if (unaryType == null) {
                return null;
            }
            if (unaryType == AcceleratorPostOpType.CLAMP_MIN) {
                if (!(node.operation() instanceof clampMin clamp)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, clamp.getMinValueF32());
            }
            if (unaryType == AcceleratorPostOpType.CLAMP_MAX) {
                if (!(node.operation() instanceof clampMax clamp)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, clamp.getMaxValueF32());
            }
            if (unaryType == AcceleratorPostOpType.MUL_SCALAR) {
                if (!(node.operation() instanceof mulScalar mul)) {
                    return null;
                }
                return AcceleratorPostOp.scalarUnary(unaryType, mul.getScalarF32());
            }
            return AcceleratorPostOp.unary(unaryType);
        }
        if (node.inputIds().size() != 2 || !node.inputIds().contains(currentOutputNodeId)) {
            return null;
        }
        AcceleratorPostOpType binaryType = resolveBinaryPostOpType(node.operation().opType());
        if (binaryType == null) {
            return null;
        }
        int otherInputNodeId = node.inputIds().getFirst() == currentOutputNodeId
                ? node.inputIds().get(1)
                : node.inputIds().getFirst();
        CompiledNode other = context.compiledNode(otherInputNodeId);
        if (other == null
                || other.dataType() != computeNode.dataType()
                || !other.contiguous()
                || other.hasStorageOffset()) {
            return null;
        }
        int[] otherShape = other.shape();
        int[] outputShape = computeNode.shape();
        boolean inputVector;
        if (otherShape.length == 1 && otherShape[0] == outputShape[1]) {
            inputVector = true;
        } else if (otherShape.length == 2 && otherShape[0] == outputShape[0] && otherShape[1] == outputShape[1]) {
            inputVector = false;
        } else {
            return null;
        }
        return AcceleratorPostOp.binary(binaryType, otherInputNodeId, inputVector);
    }

    private AcceleratorPostOpType resolveUnaryPostOpType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case RELU -> AcceleratorPostOpType.RELU;
            case TANH, FAST_TANH -> AcceleratorPostOpType.TANH;
            case SIGMOID -> AcceleratorPostOpType.SIGMOID;
            case ABS -> AcceleratorPostOpType.ABS;
            case EXP, FAST_EXP -> AcceleratorPostOpType.EXP;
            case LOG -> AcceleratorPostOpType.LOG;
            case NEG -> AcceleratorPostOpType.NEG;
            case SQRT -> AcceleratorPostOpType.SQRT;
            case INV -> AcceleratorPostOpType.INV;
            case CLAMP_MIN -> AcceleratorPostOpType.CLAMP_MIN;
            case CLAMP_MAX -> AcceleratorPostOpType.CLAMP_MAX;
            case MUL_SCALAR -> AcceleratorPostOpType.MUL_SCALAR;
            default -> null;
        };
    }

    private AcceleratorPostOpType resolveBinaryPostOpType(Operation.OpType opType) {
        if (opType == null) {
            return null;
        }
        return switch (opType) {
            case ADD -> AcceleratorPostOpType.ADD;
            case MUL -> AcceleratorPostOpType.MUL;
            case DIV -> AcceleratorPostOpType.DIV;
            case SUB -> AcceleratorPostOpType.SUB;
            default -> null;
        };
    }
}
