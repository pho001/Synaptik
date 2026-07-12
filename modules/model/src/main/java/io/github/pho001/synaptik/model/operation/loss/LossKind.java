package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent loss semantics.
 *
 * <p>A loss occurrence combines ordered prediction and target inputs. This family currently owns
 * only mean-squared error; categorical losses and executable gradient, compiler, backend, runtime,
 * and training behavior are outside this model semantic identity.</p>
 */
public enum LossKind implements OperationKind {
    /**
     * Computes squared differences between exact-shape prediction and target values and applies
     * the explicit reduction carried by {@link MeanSquaredErrorAttrs}.
     */
    MEAN_SQUARED_ERROR;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1));

    /**
     * Returns the fixed two-input, one-output mean-squared-error signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
