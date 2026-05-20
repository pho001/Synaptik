package graph.compile.session;

import backend.runtime.ExecutionMode;
import graph.compile.intent.BackendIntentPropagator;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Captures the optimizer graph snapshot, runs graph optimizer rules, and composes publication mappings.
 */
final class OptimizerSnapshotStage {
    private OptimizerSnapshotStage() {
    }

    record Result(
            List<Tensor> optimizedGraph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors,
            OptimizerState optimizerState
    ) {
        public Result {
            optimizedGraph = List.copyOf(optimizedGraph == null ? List.of() : optimizedGraph);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            publicationTensors = ForwardGraphCapture.identityCopy(publicationTensors);
            optimizerState = Objects.requireNonNull(optimizerState, "optimizerState cannot be null");
        }
    }

    static Result optimize(
            GraphOptimizer optimizer,
            List<Tensor> workingGraph,
            Tensor forwardOutput,
            Map<Tensor, Tensor> publicationTensors,
            boolean supportsBackward
    ) {
        Objects.requireNonNull(optimizer, "optimizer cannot be null");
        List<Tensor> graph = List.copyOf(workingGraph == null ? List.of() : workingGraph);
        BackendIntentPropagator.propagateBackwardClosure(graph);
        OptimizerGraphSnapshot snapshot = OptimizerGraphSnapshot.capture(graph, forwardOutput);
        OptimizerState optimizedState = optimizer.optimize(
                OptimizerState.ofGraph(
                        snapshot.graph(),
                        snapshot.forwardOutput()
                ).withExecutionMetadata(
                        supportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD,
                        supportsBackward,
                        snapshot.graph().indexOf(snapshot.forwardOutput())
                )
        );
        return new Result(
                optimizedState.graph(),
                optimizedState.forwardOutput(),
                composePublicationTensors(snapshot, publicationTensors),
                optimizedState
        );
    }

    private static Map<Tensor, Tensor> composePublicationTensors(
            OptimizerGraphSnapshot snapshot,
            Map<Tensor, Tensor> publicationTensors
    ) {
        Map<Tensor, Tensor> sources = publicationTensors == null ? Map.of() : publicationTensors;
        IdentityHashMap<Tensor, Tensor> composed = new IdentityHashMap<>();
        for (Map.Entry<Tensor, Tensor> entry : snapshot.originalBySnapshot().entrySet()) {
            Tensor original = entry.getValue();
            composed.put(entry.getKey(), sources.getOrDefault(original, original));
        }
        return composed;
    }
}
