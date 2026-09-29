package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.nio.ByteBuffer;

/** Compile-only witness whose sole declared dependency is extensions:training. */
final class TrainingApiConsumer {
    private TrainingApiConsumer() { }

    static TrainingSession open(Engine engine, Module module, Tensor objective) {
        return TrainingSession.open(engine, module, objective, new Sgd(0.1d));
    }

    static DataType dataType(TrainingState.ParameterState state) {
        return state.dataType();
    }

    static Shape shape(TrainingState.ParameterState state) {
        return state.shape();
    }

    static ByteBuffer masterParameterBytes(TrainingState.ParameterState state) {
        return state.masterParameterBytes();
    }
}
