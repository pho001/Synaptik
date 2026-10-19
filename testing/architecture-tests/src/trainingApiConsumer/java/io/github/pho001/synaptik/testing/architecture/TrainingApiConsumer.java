package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.training.TrainingParameters;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.TrainingStep;
import java.nio.ByteBuffer;
import java.util.List;

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

    /**
     * Proves positional Training execution is usable without an NN or inward project edge.
     * @param session borrowed open Training owner
     * @param values stored values in session.inputs order, excluding parameters and seed
     * @return detached step after a successful atomic update
     * @throws RuntimeException if lifecycle, input admission or execution rejects the call
     */
    static TrainingStep runValues(TrainingSession session, List<Tensor> values) {
        return session.runValues(values, GradientMode.RESET_AND_STEP);
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
