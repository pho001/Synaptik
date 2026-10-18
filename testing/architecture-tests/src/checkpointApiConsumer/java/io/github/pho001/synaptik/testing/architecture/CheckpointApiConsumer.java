package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** Compile-only witness with only Checkpoint, no direct Model, Compiler, NN or Engine dependency. */
final class CheckpointApiConsumer {
    private CheckpointApiConsumer() { }

    /**
     * Exercises all named generic declarations through Checkpoint's exported Model boundary.
     * @param path trusted target/source path
     * @param input prototype borrowed without storage
     * @param state named constant with readable host values
     * @param limits explicit encoded limits
     * @return ordinary rebound outputs
     * @throws IOException if publication or loading fails
     */
    static List<Tensor> roundtrip(Path path, Tensor input, Tensor state, ArtifactLimits limits) throws IOException {
        var leaf = new GraphCheckpoint.Leaf("value", GraphCheckpoint.Role.CONSTANT, state);
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.add(state)), List.of(leaf));
        GraphCheckpoint.save(path, GraphCheckpoint.copy(source), limits);
        return GraphCheckpoint.rebind(GraphCheckpoint.load(path, limits), List.of(input));
    }
}
