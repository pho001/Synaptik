package graph;

import config.optimizer.RewriteConfig;
import operations.Operation;
import operations.elementwise.where.where;
import operations.layout.noop;
import operations.reduction.mean;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Forward-only compile phase that rebuilds light semantic canonical forms without mutating the original user graph.
 *
 * <p>The canonicalizer runs before backward graph construction. It creates replacement tensors for forward expressions
 * that are safe to canonicalize before autograd, preserving a source-tensor map so later compile artifacts can publish
 * data and gradients back to user-visible tensors. Heavy executable lowering belongs to backend-owned region lowering,
 * not this pre-partition graph cleanup phase.
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
        return null;
    }

    private Tensor tryPiecewiseLowering(Tensor tensor, Tensor[] inputs) {
        return switch (tensor.getOperation().opType()) {
            case WHERE -> tryWherePatterns(inputs);
            default -> null;
        };
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
        if (!java.util.Arrays.equals(candidate.getShapeUnsafe(), reference.getShapeUnsafe())) {
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

    private static final class UnsupportedRebuildException extends RuntimeException {
        private UnsupportedRebuildException(Tensor tensor) {
            super(tensor == null ? "" : tensor.getLabel());
        }
    }
}
