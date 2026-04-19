package graph.optimizer.rewrite;

import graph.optimizer.OptimizationRule;
import graph.optimizer.OptimizerGraphSupport;
import operations.Operation;
import operations.elementwise.unary.mulScalar;
import operations.layout.permute;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionBackward;
import operations.linalg.scaledDotProductAttentionWeights;
import operations.reduction.softmax;
import operations.reduction.softmaxGrad;
import operations.elementwise.where.where;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class AttentionBackwardLoweringRewrite implements OptimizationRule {
    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> originalRoots = OptimizerGraphSupport.observableRoots(sortedGraph);
        Map<AttentionKey, List<Tensor>> attentionIndex = buildAttentionIndex(sortedGraph);
        Set<Tensor> forwardReachable = collectForwardReachable(sortedGraph);
        Map<Tensor, Tensor> replacements = new HashMap<>();
        List<Tensor> optimized = new ArrayList<>(sortedGraph.size());

        for (Tensor tensor : sortedGraph) {
            OptimizerGraphSupport.rewriteInputs(tensor, replacements);
            Tensor rewritten = rewriteTensor(tensor, attentionIndex, forwardReachable);
            if (rewritten != tensor) {
                TensorInternalAccess.setBackward(rewritten, tensor.isBackward());
                replacements.put(tensor, rewritten);
                optimized.add(rewritten);
            } else {
                optimized.add(tensor);
            }
        }

        if (replacements.isEmpty()) {
            return optimized;
        }

        for (Tensor tensor : sortedGraph) {
            Tensor resolvedGradient = OptimizerGraphSupport.resolveReplacement(tensor.getGradient(), replacements);
            if (resolvedGradient != null) {
                TensorInternalAccess.setGradient(tensor, resolvedGradient);
            }
        }

        return OptimizerGraphSupport.rebuildTopologicalClosureFromRoots(
                OptimizerGraphSupport.resolveRoots(originalRoots, replacements)
        );
    }

    private static Tensor rewriteTensor(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        if (!tensor.isBackward() || !supportsLoweredBackward(tensor.getDataType())) {
            return tensor;
        }

        AttentionBackwardMatch match = matchValueGradRaw(tensor, attentionIndex, forwardReachable);
        if (match == null) {
            match = matchQueryGradRaw(tensor, attentionIndex, forwardReachable);
        }
        if (match == null) {
            match = matchKeyGradRaw(tensor, attentionIndex, forwardReachable);
        }
        if (match == null) {
            return tensor;
        }

        Tensor lowered = new Tensor(
                tensor.getShapeUnsafe().clone(),
                List.of(match.attentionOut(), match.outGrad()),
                new scaledDotProductAttentionBackward(match.outputKind()),
                "scaledDotProductAttentionBackward",
                tensor.getDataType()
        );
        lowered.setRequiresGrad(false);
        return lowered;
    }

    private static AttentionBackwardMatch matchValueGradRaw(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        if (!isOp(tensor, Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        Tensor weights = matchSwappedLastTwoAxes(inputs.get(0));
        if (weights == null) {
            return null;
        }
        Tensor attentionOut = resolveAttentionOut(weights, attentionIndex, forwardReachable);
        if (attentionOut == null) {
            return null;
        }
        return new AttentionBackwardMatch(attentionOut, inputs.get(1), scaledDotProductAttentionBackward.OutputKind.VALUE);
    }

    private static AttentionBackwardMatch matchQueryGradRaw(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        if (!isOp(tensor, Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        DScoreMatch dScores = matchDScores(inputs.get(0), attentionIndex, forwardReachable);
        if (dScores == null || !matchesAttentionKeyOperand(inputs.get(1), dScores.key())) {
            return null;
        }
        return new AttentionBackwardMatch(dScores.attentionOut(), dScores.outGrad(), scaledDotProductAttentionBackward.OutputKind.QUERY);
    }

    private static AttentionBackwardMatch matchKeyGradRaw(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        if (!(tensor != null && tensor.getOperation() instanceof permute permuteOp) || !isLastTwoAxesSwap(permuteOp.getAxes())) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1 || !isOp(inputs.getFirst(), Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> matMulInputs = inputs.getFirst().getPrevTensors();
        if (matMulInputs == null || matMulInputs.size() != 2) {
            return null;
        }
        Tensor queryT = matchSwappedLastTwoAxes(matMulInputs.get(0));
        if (queryT == null) {
            return null;
        }
        DScoreMatch dScores = matchDScores(matMulInputs.get(1), attentionIndex, forwardReachable);
        if (dScores == null || queryT != dScores.query()) {
            return null;
        }
        return new AttentionBackwardMatch(dScores.attentionOut(), dScores.outGrad(), scaledDotProductAttentionBackward.OutputKind.KEY);
    }

    private static DScoreMatch matchDScores(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        Tensor candidate = tensor;
        double scale = 1.0d;
        Tensor mask = null;

        if (candidate != null && candidate.getOperation() instanceof mulScalar mulScalarOp) {
            List<Tensor> inputs = candidate.getPrevTensors();
            if (inputs == null || inputs.size() != 1 || !(mulScalarOp.getScalar() > 0.0d)) {
                return null;
            }
            scale = mulScalarOp.getScalar();
            candidate = inputs.getFirst();
        }

        if (candidate != null && candidate.getOperation() instanceof where) {
            List<Tensor> inputs = candidate.getPrevTensors();
            if (inputs == null || inputs.size() != 3) {
                return null;
            }
            if (inputs.get(0) == null || inputs.get(0).getDataType() != DataType.BOOL || !isZeroTensor(inputs.get(2))) {
                return null;
            }
            mask = canonicalizeMask(inputs.get(0));
            candidate = inputs.get(1);
        }

        if (!(candidate != null && candidate.getOperation() instanceof softmaxGrad softmaxGradOp)) {
            return null;
        }
        List<Tensor> inputs = candidate.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        Tensor attentionOut = resolveAttentionOut(inputs.get(0), attentionIndex, forwardReachable);
        if (attentionOut == null || !(attentionOut.getOperation() instanceof scaledDotProductAttention attention)) {
            return null;
        }
        List<Tensor> attentionInputs = attentionOut.getPrevTensors();
        Tensor expectedMask = attentionInputs.size() == 4 ? canonicalizeMask(attentionInputs.get(3)) : null;
        if (mask != expectedMask) {
            return null;
        }
        if (softmaxGradOp.getDimension() != inputs.get(0).getShapeUnsafe().length - 1) {
            return null;
        }
        if (Math.abs(scale - attention.getScale()) > 1e-12d) {
            return null;
        }
        Tensor query = attentionInputs.get(0);
        Tensor key = attentionInputs.get(1);
        Tensor value = attentionInputs.get(2);
        Tensor outGrad = matchDWeights(inputs.get(1), value);
        if (outGrad == null) {
            return null;
        }
        return new DScoreMatch(attentionOut, outGrad, query, key);
    }

    private static Tensor matchDWeights(Tensor tensor, Tensor value) {
        if (!isOp(tensor, Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        Tensor valueT = matchSwappedLastTwoAxes(inputs.get(1));
        if (valueT != value) {
            return null;
        }
        return inputs.get(0);
    }

    private static Tensor resolveAttentionOut(
            Tensor tensor,
            Map<AttentionKey, List<Tensor>> attentionIndex,
            Set<Tensor> forwardReachable
    ) {
        if (tensor == null) {
            return null;
        }
        if (tensor.getOperation() instanceof scaledDotProductAttentionWeights) {
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs != null && inputs.size() == 1 && inputs.getFirst().getOperation() instanceof scaledDotProductAttention) {
                return inputs.getFirst();
            }
            return null;
        }
        if (!(tensor.getOperation() instanceof softmax) || forwardReachable.contains(tensor)) {
            return null;
        }
        AttentionScoreMatch match = matchAttentionScores(tensor);
        if (match == null) {
            return null;
        }
        List<Tensor> matches = attentionIndex.getOrDefault(
                new AttentionKey(match.query(), match.key(), match.mask(), Double.doubleToLongBits(match.scale())),
                List.of()
        );
        return matches.size() == 1 ? matches.getFirst() : null;
    }

    private static Set<Tensor> collectForwardReachable(List<Tensor> sortedGraph) {
        Set<Tensor> reachable = new HashSet<>();
        Tensor forwardOutput = null;
        for (Tensor tensor : sortedGraph) {
            if (Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(tensor.getLabel())) {
                forwardOutput = tensor;
                break;
            }
        }
        collectInputs(forwardOutput, reachable);
        return reachable;
    }

    private static Map<AttentionKey, List<Tensor>> buildAttentionIndex(List<Tensor> sortedGraph) {
        Map<AttentionKey, List<Tensor>> index = new HashMap<>();
        for (Tensor tensor : sortedGraph) {
            if (!(tensor.getOperation() instanceof scaledDotProductAttention attention)) {
                continue;
            }
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs == null || (inputs.size() != 3 && inputs.size() != 4)) {
                continue;
            }
            Tensor query = inputs.get(0);
            Tensor key = inputs.get(1);
            Tensor mask = inputs.size() == 4 ? canonicalizeMask(inputs.get(3)) : null;
            AttentionKey keySig = new AttentionKey(query, key, mask, Double.doubleToLongBits(attention.getScale()));
            index.computeIfAbsent(keySig, ignored -> new ArrayList<>()).add(tensor);
        }
        return index;
    }

    private static AttentionScoreMatch matchAttentionScores(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        return matchScores(inputs.getFirst());
    }

    private static AttentionScoreMatch matchScores(Tensor tensor) {
        if (tensor != null && tensor.getOperation() instanceof where) {
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs == null || inputs.size() != 3) {
                return null;
            }
            Tensor mask = inputs.get(0);
            Tensor kept = inputs.get(1);
            Tensor fill = inputs.get(2);
            if (mask == null || mask.getDataType() != DataType.BOOL || !isMaskFillScalar(fill)) {
                return null;
            }
            AttentionScoreMatch keptMatch = matchScaledQkMatMul(kept);
            if (keptMatch == null) {
                return null;
            }
            return new AttentionScoreMatch(keptMatch.query(), keptMatch.key(), canonicalizeMask(mask), keptMatch.scale());
        }
        return matchScaledQkMatMul(tensor);
    }

    private static AttentionScoreMatch matchScaledQkMatMul(Tensor tensor) {
        if (tensor == null) {
            return null;
        }
        double scale = 1.0d;
        Tensor candidate = tensor;
        if (tensor.getOperation() instanceof mulScalar mulScalarOp) {
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs == null || inputs.size() != 1 || !(mulScalarOp.getScalar() > 0.0d)) {
                return null;
            }
            scale = mulScalarOp.getScalar();
            candidate = inputs.getFirst();
        }
        if (!isOp(candidate, Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> matMulInputs = candidate.getPrevTensors();
        if (matMulInputs == null || matMulInputs.size() != 2) {
            return null;
        }
        Tensor query = matMulInputs.get(0);
        Tensor key = matchSwappedLastTwoAxes(matMulInputs.get(1));
        if (key == null) {
            return null;
        }
        return new AttentionScoreMatch(query, key, null, scale);
    }

    private static Tensor matchSwappedLastTwoAxes(Tensor tensor) {
        if (!(tensor != null && tensor.getOperation() instanceof permute permuteOp)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        int[] axes = permuteOp.getAxes();
        if (!isLastTwoAxesSwap(axes)) {
            return null;
        }
        return inputs.getFirst();
    }

    private static boolean matchesAttentionKeyOperand(Tensor tensor, Tensor key) {
        if (tensor == key) {
            return true;
        }
        if (!(tensor != null && tensor.getOperation() instanceof permute permuteOp) || !isLastTwoAxesSwap(permuteOp.getAxes())) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        return inputs != null && inputs.size() == 1 && matchSwappedLastTwoAxes(inputs.getFirst()) == key;
    }

    private static boolean isLastTwoAxesSwap(int[] axes) {
        if (axes == null || axes.length < 2) {
            return false;
        }
        for (int i = 0; i < axes.length - 2; i++) {
            if (axes[i] != i) {
                return false;
            }
        }
        return axes[axes.length - 2] == axes.length - 1
                && axes[axes.length - 1] == axes.length - 2;
    }

    private static boolean isMaskFillScalar(Tensor tensor) {
        if (tensor == null || tensor.getOperation() != null || tensor.getFlatDataSize() != 1) {
            return false;
        }
        double expected = maskFillValue(tensor.getDataType());
        double actual = tensor.scalarAsDouble();
        double tolerance = Math.max(1e-6d, Math.abs(expected) * 1e-6d);
        return Math.abs(actual - expected) <= tolerance;
    }

    private static boolean isZeroTensor(Tensor tensor) {
        if (tensor == null || tensor.getOperation() != null || tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            return false;
        }
        return tensor.getFlatDataSize() == 1 ? tensor.scalarAsDouble() == 0.0d : isAllZero(tensor);
    }

    private static boolean isAllZero(Tensor tensor) {
        if (tensor.getFlatDataSize() == 1) {
            return tensor.scalarAsDouble() == 0.0d;
        }
        double[] values = tensor.toDoubleArrayCopy();
        for (double value : values) {
            if (value != 0.0d) {
                return false;
            }
        }
        return true;
    }

    private static Tensor canonicalizeMask(Tensor mask) {
        Tensor current = mask;
        while (current != null
                && current.getOperation() != null
                && current.getOperation().opType() == Operation.OpType.EXPAND
                && current.getPrevTensors() != null
                && current.getPrevTensors().size() == 1) {
            current = current.getPrevTensors().getFirst();
        }
        return current;
    }

    private static void collectInputs(Tensor tensor, Set<Tensor> visited) {
        if (tensor == null || !visited.add(tensor) || tensor.getPrevTensors() == null) {
            return;
        }
        for (Tensor input : tensor.getPrevTensors()) {
            collectInputs(input, visited);
        }
    }

    private static boolean supportsLoweredBackward(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, BOOL -> Double.NaN;
        };
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    private record AttentionKey(Tensor query, Tensor key, Tensor mask, long scaleBits) {}

    private record AttentionScoreMatch(Tensor query, Tensor key, Tensor mask, double scale) {}

    private record AttentionBackwardMatch(
            Tensor attentionOut,
            Tensor outGrad,
            scaledDotProductAttentionBackward.OutputKind outputKind
    ) {}

    private record DScoreMatch(Tensor attentionOut, Tensor outGrad, Tensor query, Tensor key) {}
}
