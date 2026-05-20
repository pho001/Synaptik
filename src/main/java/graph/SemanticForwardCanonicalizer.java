package graph;

import config.optimizer.RewriteConfig;
import graph.optimizer.rewrite.canonical.PiecewisePatternLowerer;
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
 * not this pre-partition graph simplification phase.
 */
public final class SemanticForwardCanonicalizer {
    private static final int ZERO_TENSOR_SCAN_LIMIT = 4096;

    private final RewriteConfig config;
    private final PiecewisePatternLowerer piecewiseLowerer;

    /**
     * Creates a canonicalizer.
     *
     * @param config rewrite configuration, or {@code null} for defaults
     */
    public SemanticForwardCanonicalizer(RewriteConfig config) {
        this.config = config == null ? RewriteConfig.defaults() : config;
        this.piecewiseLowerer = new PiecewisePatternLowerer(
                this.config.piecewiseLowering(),
                ZERO_TENSOR_SCAN_LIMIT
        );
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
            case WHERE -> inputs.length == 3 ? piecewiseLowerer.lowerWhere(inputs[0], inputs[1], inputs[2]) : null;
            default -> null;
        };
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
