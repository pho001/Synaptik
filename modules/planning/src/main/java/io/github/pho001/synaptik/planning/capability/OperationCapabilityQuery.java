package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Objects;

/**
 * Describes one structurally valid operation occurrence for a backend capability question.
 *
 * <p>The numerical profile supplies the immutable graph-wide result-set identity, while the
 * operation supplies backend-independent semantics and the ordered descriptor lists describe the
 * occurrence's logical input and output positions. Construction snapshots list membership,
 * retains the exact profile, operation, and descriptor references, and validates only the input
 * and output counts declared by the operation's signature. It does not interpret profile
 * semantics or validate descriptor compatibility, graph closure, backend availability, hard
 * requirements, scoring, preparation, routing, or execution.</p>
+
 * <p>Record equality, hashing, and diagnostic text use the profile, operation, and both ordered
 * snapshots. The diagnostic text is not a serialization or dispatch contract.</p>
 *
 * @param numericalProfile the non-null immutable graph-wide numerical-profile identity; retained
 *     by exact reference and not interpreted
 * @param operation the non-null immutable backend-independent operation semantics; retained by
 *     exact reference
 * @param inputs the non-null ordered input descriptors; every element must be non-null, and list
 *     membership is copied while descriptor references are retained
 * @param outputs the non-null ordered output descriptors; every element must be non-null, and
 *     list membership is copied while descriptor references are retained
 */
public record OperationCapabilityQuery(
        NumericalProfile numericalProfile,
        Operation operation,
        List<TensorDescriptor> inputs,
        List<TensorDescriptor> outputs) {
    /**
     * Creates an immutable capability question for one structurally valid operation occurrence.
     *
     * <p>Validation checks the profile, operation, and list references in declaration order,
     * scans input elements in encounter order, snapshots the inputs, then scans and snapshots the
     * outputs. Only after both snapshots exist does the operation signature validate their
     * occurrence counts. No profile, descriptor, or cross-layer semantic interpretation is
     * performed.</p>
     *
     * @param numericalProfile the non-null graph-wide numerical-profile identity retained exactly
     * @param operation the non-null immutable operation semantics; retained by exact reference
     * @param inputs the non-null ordered input descriptors; every element must be non-null, list
     *     membership is copied, and descriptor references are retained
     * @param outputs the non-null ordered output descriptors; every element must be non-null, list
     *     membership is copied, and descriptor references are retained
     * @throws NullPointerException if {@code numericalProfile}, {@code operation}, {@code inputs},
     *     {@code outputs}, or a descriptor element is {@code null}; element failures identify the
     *     first null encounter as {@code inputs[index]} or {@code outputs[index]}
     * @throws IllegalArgumentException if the input or output count is outside the operation
     *     signature's inclusive accepted range
     * @throws IllegalStateException if the operation kind's signature declaration has become
     *     missing or malformed
     */
    public OperationCapabilityQuery {
        numericalProfile = Objects.requireNonNull(numericalProfile, "numericalProfile");
        operation = Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");

        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }
        inputs = List.copyOf(inputs);

        for (int index = 0; index < outputs.size(); index++) {
            Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
        }
        outputs = List.copyOf(outputs);

        operation.signature().validateOccurrence(inputs.size(), outputs.size());
    }

    /**
     * Returns the backend-independent semantics of this operation occurrence.
     *
     * @return the exact non-null {@link Operation} reference supplied at construction
     */
    @Override
    public Operation operation() {
        return operation;
    }

    /**
     * Returns the ordered input descriptors of this operation occurrence.
     *
     * @return the non-null immutable membership snapshot in encounter order; elements are the
     *     exact non-null descriptor references supplied at construction
     */
    @Override
    public List<TensorDescriptor> inputs() {
        return inputs;
    }

    /**
     * Returns the ordered output descriptors of this operation occurrence.
     *
     * @return the non-null immutable membership snapshot in encounter order; elements are the
     *     exact non-null descriptor references supplied at construction
     */
    @Override
    public List<TensorDescriptor> outputs() {
        return outputs;
    }
}
