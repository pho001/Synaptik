package graph.optimizer.state;

import graph.optimizer.OptimizerGraph;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Cumulative identity mapping from optimizer input tensors to their rewritten tensors.
 */
public final class GraphRewriteMap {
    private static final GraphRewriteMap EMPTY = new GraphRewriteMap(new IdentityHashMap<>());

    private final IdentityHashMap<Tensor, Tensor> replacements;

    private GraphRewriteMap(IdentityHashMap<Tensor, Tensor> replacements) {
        this.replacements = identityCopy(replacements);
    }

    public static GraphRewriteMap empty() {
        return EMPTY;
    }

    public Tensor resolve(Tensor tensor) {
        Tensor resolved = OptimizerGraph.resolveReplacement(tensor, replacements);
        return resolved == null ? tensor : resolved;
    }

    public GraphRewriteMap withReplacements(Map<Tensor, Tensor> newReplacements) {
        if (newReplacements == null || newReplacements.isEmpty()) {
            return this;
        }
        IdentityHashMap<Tensor, Tensor> merged = identityCopy(replacements);
        for (Map.Entry<Tensor, Tensor> entry : merged.entrySet()) {
            Tensor resolved = OptimizerGraph.resolveReplacement(entry.getValue(), newReplacements);
            if (resolved != null) {
                entry.setValue(resolved);
            }
        }
        for (Map.Entry<Tensor, Tensor> entry : newReplacements.entrySet()) {
            Tensor source = Objects.requireNonNull(entry.getKey(), "rewrite source tensor cannot be null");
            Tensor target = Objects.requireNonNull(entry.getValue(), "rewrite target tensor cannot be null");
            Tensor resolved = OptimizerGraph.resolveReplacement(target, newReplacements);
            merged.put(source, resolved == null ? target : resolved);
        }
        return new GraphRewriteMap(merged);
    }

    private static IdentityHashMap<Tensor, Tensor> identityCopy(Map<Tensor, Tensor> input) {
        IdentityHashMap<Tensor, Tensor> copy = new IdentityHashMap<>();
        if (input == null) {
            return copy;
        }
        for (Map.Entry<Tensor, Tensor> entry : input.entrySet()) {
            copy.put(
                    Objects.requireNonNull(entry.getKey(), "rewrite source tensor cannot be null"),
                    Objects.requireNonNull(entry.getValue(), "rewrite target tensor cannot be null")
            );
        }
        return copy;
    }
}
