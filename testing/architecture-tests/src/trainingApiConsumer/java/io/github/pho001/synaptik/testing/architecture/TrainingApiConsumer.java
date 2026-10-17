package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.training.TrainingParameters;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.nio.ByteBuffer;

/** Compile-only witness whose sole declared dependency is extensions:training. */
final class TrainingApiConsumer {
    private TrainingApiConsumer() { }

    /**
     * Opens from an NN-independent live owner through the sole Training dependency.
     * @param engine borrowed open Engine
     * @param parameters live ordered source, not an NN wrapper
     * @param objective scalar objective using exact source Tensors
     * @return newly owned session
     */
    static TrainingSession open(Engine engine, TrainingParameters parameters, Tensor objective) {
        return TrainingSession.open(engine, parameters, objective, new Sgd(0.1d));
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
