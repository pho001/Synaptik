package graph;

import config.optimizer.RewriteConfig;
import operations.Operation;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.where.where;
import operations.index.gather;
import operations.layout.noop;
import operations.layout.permute;
import operations.loss.crossEntropyLossIndices;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;
import tensor.options.AttentionOptions;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Forward-only compile phase that rebuilds semantic-safe canonical forms without mutating the original user graph.
 *
 * <p>The canonicalizer runs before backward graph construction. It creates replacement tensors for forward expressions
 * that are safe to lower semantically, preserving a source-tensor map so later compile artifacts can publish data and
 * gradients back to user-visible tensors. The result is still a tensor-level graph with valid backward lambdas.
 */
public final class SemanticForwardCanonicalizer {
    private final RewriteConfig config;

    /**
     * Creates a canonicalizer.
     *
     * @param config rewrite configuration, or {@code null} for defaults
     */
    public SemanticForwardCanonicalizer(RewriteConfig config) {
        this.config = config == null ? RewriteConfig.defaults() : config;
    }

    /**
     * Canonicalizes the forward graph while preserving the original forward root mapping.
     *
     * @param forwardGraph original forward graph in topological order
     * @param forwardOutput semantic forward output
     * @param originalForwardRoot user-visible root tensor that should receive published data
     * @return canonical graph, canonical forward output, and source tensor mappings
     */
    public Result canonicalize(List<Tensor> forwardGraph, Tensor forwardOutput, Tensor originalForwardRoot) {
        Objects.requireNonNull(forwardGraph, "forwardGraph cannot be null");
        Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        Objects.requireNonNull(originalForwardRoot, "originalForwardRoot cannot be null");

        IdentityHashMap<Tensor, Tensor> cache = new IdentityHashMap<>();
        IdentityHashMap<Tensor, Tensor> sourceTensors = new IdentityHashMap<>();
        Tensor canonicalForwardOutput;
        try {
            canonicalForwardOutput = rewriteTensor(forwardOutput, cache, sourceTensors);
        } catch (UnsupportedRebuildException ex) {
            return new Result(forwardGraph, forwardOutput, Map.of());
        }

        Tensor actualForwardRoot = requireForwardRoot(canonicalForwardOutput);
        if (actualForwardRoot != originalForwardRoot) {
            sourceTensors.put(actualForwardRoot, originalForwardRoot);
        }
        return new Result(canonicalForwardOutput.topologicalSort(), canonicalForwardOutput, Map.copyOf(sourceTensors));
    }

    private Tensor rewriteTensor(
            Tensor tensor,
            IdentityHashMap<Tensor, Tensor> cache,
            IdentityHashMap<Tensor, Tensor> sourceTensors
    ) {
        Tensor existing = cache.get(tensor);
        if (existing != null) {
            return existing;
        }
        if (tensor == null || tensor.getOperation() == null) {
            cache.put(tensor, tensor);
            return tensor;
        }

        List<Tensor> originalInputs = tensor.getPrevTensors();
        Tensor[] rewrittenInputs = new Tensor[originalInputs == null ? 0 : originalInputs.size()];
        boolean inputChanged = false;
        for (int i = 0; i < rewrittenInputs.length; i++) {
            rewrittenInputs[i] = rewriteTensor(originalInputs.get(i), cache, sourceTensors);
            inputChanged |= rewrittenInputs[i] != originalInputs.get(i);
        }

        Tensor lowered = trySemanticLowering(tensor, rewrittenInputs);
        Tensor result;
        if (lowered != null) {
            result = lowered;
            sourceTensors.put(result, tensor);
        } else if (inputChanged) {
            result = rebuildEquivalent(tensor, rewrittenInputs);
            sourceTensors.put(result, tensor);
        } else {
            result = tensor;
        }
        cache.put(tensor, result);
        return result;
    }

    private Tensor trySemanticLowering(Tensor tensor, Tensor[] inputs) {
        if (config.piecewiseLowering().anyEnabled()) {
            Tensor piecewise = tryPiecewiseLowering(tensor, inputs);
            if (piecewise != null) {
                return piecewise;
            }
        }
        Tensor linear = tryLinearLowering(tensor, inputs);
        if (linear != null) {
            return linear;
        }
        Tensor loss = tryLossForwardLowering(tensor, inputs);
        if (loss != null) {
            return loss;
        }
        return tryAttentionLowering(tensor, inputs);
    }

    private Tensor tryPiecewiseLowering(Tensor tensor, Tensor[] inputs) {
        return switch (tensor.getOperation().opType()) {
            case INV -> config.piecewiseLowering().canonicalSigmoid() ? tryCanonicalSigmoid(inputs) : null;
            case WHERE -> tryWherePatterns(inputs);
            default -> null;
        };
    }

    private Tensor tryCanonicalSigmoid(Tensor[] inputs) {
        if (inputs.length != 1) {
            return null;
        }
        Tensor add = inputs[0];
        if (!isOp(add, Operation.OpType.ADD)) {
            return null;
        }
        List<Tensor> addInputs = add.getPrevTensors();
        Tensor left = addInputs.get(0);
        Tensor right = addInputs.get(1);
        Tensor expNode = isConstant(left, 1.0) ? right : isConstant(right, 1.0) ? left : null;
        if (expNode == null || !isOp(expNode, Operation.OpType.EXP)) {
            return null;
        }
        Tensor source = extractNegatedSource(expNode.getPrevTensors().get(0));
        return source == null ? null : source.sigmoid();
    }

    private Tensor tryWherePatterns(Tensor[] inputs) {
        if (inputs.length != 3) {
            return null;
        }
        Tensor condition = inputs[0];
        Tensor ifTrue = inputs[1];
        Tensor ifFalse = inputs[2];

        if (config.piecewiseLowering().reluLikeWhere()) {
            Tensor relu = tryLowerRelu(condition, ifTrue, ifFalse);
            if (relu != null) {
                return relu;
            }
        }
        if (config.piecewiseLowering().clampLikeWhere()) {
            Tensor clamp = tryLowerClamp(condition, ifTrue, ifFalse);
            if (clamp != null) {
                return clamp;
            }
        }
        return null;
    }

    private Tensor tryLowerRelu(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (!isOp(condition, Operation.OpType.GT)) {
            return null;
        }
        Tensor source = condition.getPrevTensors().get(0);
        Tensor threshold = condition.getPrevTensors().get(1);
        if (source == ifTrue && isConstant(threshold, 0.0) && isZeroTensorLike(ifFalse, source)) {
            return source.relu();
        }
        return null;
    }

    private Tensor tryLowerClamp(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        if (isOp(condition, Operation.OpType.LT)) {
            Tensor source = condition.getPrevTensors().get(0);
            Tensor threshold = condition.getPrevTensors().get(1);
            if (source == ifFalse && isScalarConstantLike(ifTrue, threshold)) {
                return source.clampMin(threshold.scalarAsDouble());
            }
        }
        if (isOp(condition, Operation.OpType.GT)) {
            Tensor source = condition.getPrevTensors().get(0);
            Tensor threshold = condition.getPrevTensors().get(1);
            if (source == ifFalse && isScalarConstantLike(ifTrue, threshold)) {
                return source.clampMax(threshold.scalarAsDouble());
            }
        }
        return null;
    }

    private Tensor tryLinearLowering(Tensor tensor, Tensor[] inputs) {
        if (tensor.getOperation().opType() != Operation.OpType.ADD || inputs.length != 2) {
            return null;
        }
        Tensor candidate = tryLowerLinear(inputs[0], inputs[1], tensor);
        return candidate != tensor ? candidate : tryLowerLinear(inputs[1], inputs[0], tensor);
    }

    private Tensor tryLowerLinear(Tensor first, Tensor second, Tensor originalAdd) {
        if (first == null || second == null || !(first.getOperation() instanceof operations.linalg.matmul)) {
            return originalAdd;
        }
        if (!isBias(second)) {
            return originalAdd;
        }
        Tensor input = first.getPrevTensors().get(0);
        Tensor weight = first.getPrevTensors().get(1);
        if (!matchesLinearShape(input, weight, second, first)) {
            return originalAdd;
        }
        return input.linear(weight, second);
    }

    private Tensor tryLossForwardLowering(Tensor tensor, Tensor[] inputs) {
        return switch (tensor.getOperation().opType()) {
            case NEG -> lowerCrossEntropyFromIndices(tensor, inputs, LossReduction.NONE);
            case SUM -> lowerReducedCrossEntropyFromIndices(tensor, inputs, LossReduction.SUM);
            case MEAN -> lowerReducedCrossEntropyFromIndices(tensor, inputs, LossReduction.MEAN);
            default -> null;
        };
    }

    private Tensor lowerReducedCrossEntropyFromIndices(Tensor tensor, Tensor[] inputs, LossReduction reduction) {
        if (inputs.length != 1) {
            return null;
        }
        if (reduction == LossReduction.SUM && (!(tensor.getOperation() instanceof sum sumOp) || sumOp.getDimension() != -1)) {
            return null;
        }
        if (reduction == LossReduction.MEAN && (!(tensor.getOperation() instanceof mean meanOp) || meanOp.getDimension() != -1)) {
            return null;
        }
        Match match = matchPerSampleCrossEntropy(inputs[0]);
        if (match == null) {
            return null;
        }
        return match.logits().crossEntropyLossFromIndices(match.targetIndices(), match.classDimension(), reduction);
    }

    private Tensor lowerCrossEntropyFromIndices(Tensor tensor, Tensor[] inputs, LossReduction reduction) {
        Match match = matchPerSampleCrossEntropy(tensor.getOperation().opType() == Operation.OpType.NEG && inputs.length == 1
                ? rebuildNeg(inputs[0])
                : tensor);
        if (match == null) {
            return null;
        }
        return match.logits().crossEntropyLossFromIndices(match.targetIndices(), match.classDimension(), reduction);
    }

    private Tensor tryAttentionLowering(Tensor tensor, Tensor[] inputs) {
        if (tensor.getOperation().opType() != Operation.OpType.MATMUL || inputs.length != 2) {
            return null;
        }
        Tensor weights = inputs[0];
        Tensor value = inputs[1];
        if (!(weights.getOperation() instanceof softmax softmaxOp)) {
            return null;
        }
        if (softmaxOp.getDimension() != weights.getShapeUnsafe().length - 1) {
            return null;
        }
        List<Tensor> weightInputs = weights.getPrevTensors();
        if (weightInputs == null || weightInputs.size() != 1) {
            return null;
        }
        ScoreMatch scores = matchAttentionScores(weightInputs.get(0));
        if (scores == null || !Arrays.equals(tensor.getShapeUnsafe(), expectedAttentionOutputShape(scores.query(), value))) {
            return null;
        }
        AttentionOptions options = AttentionOptions.defaults().withScale(scores.scale());
        return scores.mask() == null
                ? scores.query().scaledDotProductAttention(scores.key(), value, options)
                : scores.query().scaledDotProductAttention(scores.key(), value, scores.mask(), options);
    }

    private ScoreMatch matchAttentionScores(Tensor tensor) {
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
            ScoreMatch keptMatch = matchScaledQkMatMul(kept);
            if (keptMatch == null) {
                return null;
            }
            return new ScoreMatch(keptMatch.query(), keptMatch.key(), mask, keptMatch.scale());
        }
        return matchScaledQkMatMul(tensor);
    }

    private ScoreMatch matchScaledQkMatMul(Tensor tensor) {
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
            candidate = inputs.get(0);
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
        return key == null ? null : new ScoreMatch(query, key, null, scale);
    }

    private Tensor matchSwappedLastTwoAxes(Tensor tensor) {
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
        return inputs.get(0);
    }

    private Tensor rebuildEquivalent(Tensor original, Tensor[] inputs) {
        return switch (original.getOperation().opType()) {
            case NOOP -> inputs[0].forwardOutput();
            case SUM -> rebuildSum(original, inputs);
            case MEAN -> rebuildMean(original, inputs);
            default -> throw new UnsupportedRebuildException(original);
        };
    }

    private Tensor rebuildSum(Tensor original, Tensor[] inputs) {
        if (!(original.getOperation() instanceof sum sumOp) || inputs.length != 1) {
            throw new UnsupportedRebuildException(original);
        }
        return sumOp.getDimension() == -1
                ? inputs[0].sum()
                : inputs[0].sum(sumOp.getDimension(), sumOp.keepDims());
    }

    private Tensor rebuildMean(Tensor original, Tensor[] inputs) {
        if (!(original.getOperation() instanceof mean meanOp) || inputs.length != 1) {
            throw new UnsupportedRebuildException(original);
        }
        return meanOp.getDimension() == -1
                ? inputs[0].mean()
                : inputs[0].mean(meanOp.getDimension(), meanOp.keepDims());
    }

    private Match matchPerSampleCrossEntropy(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null || tensor.getOperation().opType() != Operation.OpType.NEG) {
            return null;
        }
        List<Tensor> negInputs = tensor.getPrevTensors();
        if (negInputs == null || negInputs.size() != 1) {
            return null;
        }
        Tensor gathered = negInputs.get(0);
        if (!(gathered.getOperation() instanceof gather gatherOp)) {
            return null;
        }
        List<Tensor> gatherInputs = gathered.getPrevTensors();
        if (gatherInputs == null || gatherInputs.size() != 2) {
            return null;
        }
        Tensor logSoftmaxTensor = gatherInputs.get(0);
        Tensor targetIndices = gatherInputs.get(1);
        if (!(logSoftmaxTensor.getOperation() instanceof logSoftmax logSoftmaxOp)) {
            return null;
        }
        List<Tensor> softmaxInputs = logSoftmaxTensor.getPrevTensors();
        if (softmaxInputs == null || softmaxInputs.size() != 1) {
            return null;
        }
        if (gatherOp.getDimension() != logSoftmaxOp.getDimension()) {
            return null;
        }
        return new Match(softmaxInputs.get(0), targetIndices, logSoftmaxOp.getDimension());
    }

    private Tensor rebuildNeg(Tensor input) {
        return input.neg();
    }

    private boolean isBias(Tensor tensor) {
        if (tensor.getDataType() == DataType.BOOL || tensor.getDataType() == DataType.INT32) {
            return false;
        }
        int[] shape = tensor.getShapeUnsafe();
        return shape.length == 1 && shape[0] > 0;
    }

    private boolean matchesLinearShape(Tensor input, Tensor weight, Tensor bias, Tensor matmulOut) {
        int[] inputShape = input.getShapeUnsafe();
        int[] weightShape = weight.getShapeUnsafe();
        int[] biasShape = bias.getShapeUnsafe();
        int[] outShape = matmulOut.getShapeUnsafe();

        if (inputShape.length < 2 || weightShape.length != 2 || biasShape.length != 1 || outShape.length != inputShape.length) {
            return false;
        }
        int inFeatures = inputShape[inputShape.length - 1];
        int outFeatures = weightShape[1];
        if (weightShape[0] != inFeatures || biasShape[0] != outFeatures) {
            return false;
        }
        if (outShape[outShape.length - 1] != outFeatures) {
            return false;
        }
        for (int i = 0; i < outShape.length - 1; i++) {
            if (outShape[i] != inputShape[i]) {
                return false;
            }
        }
        return true;
    }

    private static Tensor requireForwardRoot(Tensor forwardOutput) {
        List<Tensor> inputs = forwardOutput.getPrevTensors();
        if (inputs == null || inputs.size() != 1 || inputs.get(0) == null) {
            throw new IllegalStateException("System forward output must have exactly one input.");
        }
        return inputs.get(0);
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    private static boolean isConstant(Tensor tensor, double expected) {
        return tensor != null
                && tensor.getOperation() == null
                && tensor.getFlatDataSize() == 1
                && Math.abs(tensor.scalarAsDouble() - expected) < 1e-12;
    }

    private static Tensor extractNegatedSource(Tensor tensor) {
        if (isOp(tensor, Operation.OpType.NEG)) {
            return tensor.getPrevTensors().get(0);
        }
        if (isOp(tensor, Operation.OpType.MUL_SCALAR)
                && tensor.getOperation() instanceof mulScalar mulScalar
                && Math.abs(mulScalar.getScalar() + 1.0) < 1e-12) {
            return tensor.getPrevTensors().get(0);
        }
        return null;
    }

    private static boolean isScalarConstantLike(Tensor candidate, Tensor reference) {
        return candidate != null
                && reference != null
                && candidate.getOperation() == null
                && reference.getOperation() == null
                && candidate.getFlatDataSize() == 1
                && reference.getFlatDataSize() == 1
                && candidate.getDataType() == reference.getDataType()
                && Math.abs(candidate.scalarAsDouble() - reference.scalarAsDouble()) < 1e-12;
    }

    private static boolean isZeroTensorLike(Tensor candidate, Tensor reference) {
        if (candidate == null || reference == null) {
            return false;
        }
        if (candidate.getOperation() != null || candidate.getDataType() != reference.getDataType()) {
            return false;
        }
        if (!Arrays.equals(candidate.getShapeUnsafe(), reference.getShapeUnsafe())) {
            return false;
        }
        double[] values = candidate.toDoubleArrayCopy();
        for (double value : values) {
            if (Math.abs(value) > 1e-12) {
                return false;
            }
        }
        return true;
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
        double expected = switch (tensor.getDataType()) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, BOOL -> Double.NaN;
        };
        double actual = tensor.scalarAsDouble();
        double tolerance = Math.max(1e-6d, Math.abs(expected) * 1e-6d);
        return Math.abs(actual - expected) <= tolerance;
    }

    private static int[] expectedAttentionOutputShape(Tensor query, Tensor value) {
        int[] qShape = query.getShapeUnsafe();
        int[] vShape = value.getShapeUnsafe();
        int[] out = qShape.clone();
        out[out.length - 1] = vShape[vShape.length - 1];
        return out;
    }

    public record Result(
            List<Tensor> graph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> sourceTensors
    ) {
        public Result {
            graph = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
            Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            sourceTensors = Map.copyOf(Objects.requireNonNull(sourceTensors, "sourceTensors cannot be null"));
        }
    }

    private record Match(Tensor logits, Tensor targetIndices, int classDimension) {}

    private record ScoreMatch(Tensor query, Tensor key, Tensor mask, double scale) {}

    private static final class UnsupportedRebuildException extends RuntimeException {
        private UnsupportedRebuildException(Tensor tensor) {
            super(tensor == null ? "" : tensor.getLabel());
        }
    }
}
