package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.training.TrainingParameters;

/**
 * Compile-only witness that NN exports the Model types needed by ordinary public API calls.
 *
 * <p>Its isolated source set declares only {@code extensions:nn}, with no direct Model dependency
 * or inherited Training/test classpath to hide an export regression.</p>
 */
final class NnApiConsumer {
    /** Prevents instantiation of this compile-only witness. */
    private NnApiConsumer() { }

    /**
     * Compiles old public NN persistence signatures using only NN's exported dependencies.
     * @param module fully bound caller-owned NN tree
     * @param engine borrowed public materialization facade
     * @param limits explicit artifact bounds
     * @param fingerprint application-verified input identity
     * @return detached typed inspection
     * @throws java.io.IOException if capture validation fails
     */
    static io.github.pho001.synaptik.nn.checkpoint.ModelCheckpointInspection inspect(
            io.github.pho001.synaptik.nn.module.Module module,
            io.github.pho001.synaptik.engine.Engine engine,
            io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits limits,
            io.github.pho001.synaptik.nn.checkpoint.InputArtifactFingerprint fingerprint)
            throws java.io.IOException {
        var envelope = io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint.capture(
                module, engine, fingerprint.bytes(), null, limits);
        return io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint.inspect(envelope, limits);
    }

    /**
     * Obtains live admission through NN's exported Training edge without a direct dependency.
     * @param module caller-owned initialized Module
     * @return immutable membership with live owner validation
     */
    static TrainingParameters parameters(io.github.pho001.synaptik.nn.module.Module module) {
        TrainingParameters parameters = module.trainingParameters();
        parameters.validateUnchanged();
        return parameters;
    }

    /**
     * Reads the exact Tensor currently bound to a caller-supplied parameter without copying it.
     *
     * @param parameter non-null parameter whose binding is read
     * @return the non-null current Tensor reference
     * @throws NullPointerException if {@code parameter} is null
     */
    static Tensor value(Parameter parameter) {
        return parameter.value();
    }

    /**
     * Reads the Model data type through a parameter's public Tensor descriptor.
     *
     * @param parameter non-null parameter whose binding is inspected
     * @return the non-null data type of the current binding
     * @throws NullPointerException if {@code parameter} is null
     */
    static DataType dataType(Parameter parameter) {
        return parameter.value().descriptor().dataType();
    }

    /**
     * Reads the immutable Model Shape through a parameter's public Tensor descriptor.
     *
     * @param parameter non-null parameter whose binding is inspected
     * @return the non-null Shape of the current binding, without copying it
     * @throws NullPointerException if {@code parameter} is null
     */
    static Shape shape(Parameter parameter) {
        return parameter.value().descriptor().shape();
    }

    /**
     * Defines and applies an identity Model using Tensor input/output types supplied by NN.
     *
     * @param input non-null caller-owned Tensor; no storage is read or modified
     * @return the exact input reference, without compilation or execution
     * @throws NullPointerException if {@code input} is null
     */
    static Tensor forward(Tensor input) {
        Model<Tensor, Tensor> model = Model.define(topology -> value -> value);
        return model.forward(input);
    }
}
