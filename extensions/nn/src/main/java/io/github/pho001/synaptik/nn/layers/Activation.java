package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationStep;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.util.Objects;

/**
 * A closed, stateless floating activation in a declarative Sequential tree.
 *
 * <p>Delegates to ordinary Model operations without evaluation, storage, callbacks or mode effects.</p>
 */
public final class Activation extends UnaryTensorModule {
    private final ActivationKind kind;

    /**
     * Selects the exact Model activation to apply.
     *
     * @param kind non-null closed activation kind retained unchanged
     * @throws NullPointerException if kind is null
     */
    public Activation(ActivationKind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
    }

    /**
     * Returns the immutable activation configuration.
     *
     * @return non-null snapshot with the selected kind and no Tensor references
     */
    public ActivationStep specification() {
        return new ActivationStep(kind);
    }

    /**
     * Constructs one ordinary Model activation expression, independent of module mode.
     *
     * @param input non-null floating Tensor retained in result provenance and never mutated
     * @return non-null storage-free expression with Model-defined type, Shape and gradient metadata
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if Model rejects the input type
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input");
        return switch (kind) {
            case RELU -> input.relu();
            case TANH -> input.tanh();
            case SIGMOID -> input.sigmoid();
        };
    }
}
