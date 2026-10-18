package io.github.pho001.synaptik.nn.checkpoint.api;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public-only file-free consumer evidence; model internals and Training are not imported. */
final class ModelCheckpointEnvelopeApiTest {
    @Test
    void comparesCompleteParameterDescriptorsAndRawBytesByExactPath() throws Exception {
        var limits = new ArtifactLimits(4096, 2048, 8, 32, 1024, 2048);
        byte[] config = new byte[32];
        ArtifactEnvelope envelope;
        try (Engine engine = Engine.standard()) {
            envelope = ModelCheckpoint.capture(new Tree(-0.0f), engine, config, null, limits);
        }
        var inspection = ModelCheckpoint.inspect(envelope, limits);
        var bytes = ModelCheckpoint.canonicalStateBytes(envelope, limits);
        assertEquals(List.of("z重み", "a branch.重み/🦉 space", "a branch.flag"),
                new ArrayList<>(bytes.keySet()));
        var parameters = inspection.states().stream()
                .filter(state -> state.kind() == StateKind.PARAMETER).toList();
        assertEquals(List.of("z重み", "a branch.重み/🦉 space"),
                parameters.stream().map(state -> state.path()).toList());
        for (var parameter : parameters) {
            assertEquals(DataType.FLOAT32, parameter.dataType());
            assertEquals(Shape.scalar(), parameter.shape());
            assertEquals(4, parameter.byteLength());
            assertEquals(0x80000000, bytes.get(parameter.path()).getInt(0));
        }
        Tree target = new Tree(1.0f);
        ModelCheckpoint.load(envelope, target, config, null, limits);
        try (Engine engine = Engine.standard()) {
            var restored = ModelCheckpoint.capture(target, engine, config, null, limits);
            assertEquals(bytes, ModelCheckpoint.canonicalStateBytes(restored, limits));
            assertEquals(inspection.states(), ModelCheckpoint.inspect(restored, limits).states());
        }
    }

    private static final class Tree extends Module {
        /**
         * Builds a complete tree with parameter paths spanning direct and child ownership.
         *
         * @param value scalar FLOAT32 logical value for both independent parameters
         */
        Tree(float value) {
            parameter("z重み", TensorFactory.scalar(value, Optional.empty(), true));
            child("a branch", new Branch(value));
        }
    }

    private static final class Branch extends Module {
        /**
         * Declares a Unicode parameter path and an integral buffer to prove role filtering.
         *
         * @param value scalar FLOAT32 logical parameter value
         */
        Branch(float value) {
            parameter("重み/🦉 space", TensorFactory.scalar(value, Optional.empty(), true));
            buffer("flag", TensorFactory.scalar(1, Optional.empty(), false));
        }
    }
}
