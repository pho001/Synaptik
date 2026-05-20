package graph.compile.session;

import graph.SemanticForwardCanonicalizer;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Captures the semantic forward graph and optional canonicalization mapping for one compile.
 */
final class ForwardGraphCapture {
    private ForwardGraphCapture() {
    }

    record Result(
            List<Tensor> forwardGraph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors
    ) {
        public Result {
            forwardGraph = List.copyOf(forwardGraph == null ? List.of() : forwardGraph);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            publicationTensors = identityCopy(publicationTensors);
        }
    }

    static Result capture(Tensor rootTensor, SemanticForwardCanonicalizer forwardCanonicalizer) {
        Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        Tensor semanticForwardOutput = rootTensor.forwardOutput();
        if (forwardCanonicalizer == null) {
            return new Result(
                    semanticForwardOutput.topologicalSort(),
                    semanticForwardOutput,
                    new IdentityHashMap<>()
            );
        }
        SemanticForwardCanonicalizer.Result canonicalized = forwardCanonicalizer.canonicalize(
                semanticForwardOutput.topologicalSort(),
                semanticForwardOutput,
                rootTensor
        );
        return new Result(
                canonicalized.graph(),
                canonicalized.forwardOutput(),
                canonicalized.publicationTensors()
        );
    }

    static Map<Tensor, Tensor> identityCopy(Map<Tensor, Tensor> source) {
        IdentityHashMap<Tensor, Tensor> copy = new IdentityHashMap<>();
        if (source != null) {
            copy.putAll(source);
        }
        return java.util.Collections.unmodifiableMap(copy);
    }
}
