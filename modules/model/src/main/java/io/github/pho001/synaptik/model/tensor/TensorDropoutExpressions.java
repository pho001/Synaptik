package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free, explicit-state dropout Tensor expressions.
 *
 * <p>This field-free helper creates one producer with ordered inputs {@code [input, state]} and
 * ordered outputs {@code [output, keep mask, next state]}. It exposes only output and next state,
 * while the shared producer retains the BOOL mask descriptor at slot one for later compiler-owned
 * capture and backward construction. The factory necessarily creates a wrapper for that slot, but
 * this helper discards it from the public result. It neither samples nor executes, and it selects
 * no random algorithm, backend route, inference mode, or gradient rule.</p>
 */
final class TensorDropoutExpressions {
    /** Prevents instantiation because dropout construction is stateless and package-local. */
    private TensorDropoutExpressions() {
    }

    /**
     * Creates one fresh explicit-state training-dropout occurrence.
     *
     * <p>Validation checks input nullity, floating eligibility, probability, and state nullity in
     * that order before factory allocation. A successful call creates exactly three indexed
     * storage-free Tensor wrappers and identifiers in output order under one producer. The output
     * preserves the input's exact floating type, Shape reference, and gradient eligibility; the
     * mask is BOOL and the next state is INT64 Shape[2]. Every output has unresolved layout, no
     * label, and no storage.</p>
     *
     * @param input non-null floating Tensor retained as producer input zero and not mutated
     * @param probability finite drop probability in {@code [0.0, 1.0)}
     * @param state non-null opaque graph RNG state whose exact Tensor is producer input one
     * @return fresh public output and freshly wrapped next state that select slots zero and two of
     *     the same producer; never null
     * @throws NullPointerException if {@code input} or {@code state} is null, checked in the
     *     documented order with the parameter name as the message
     * @throws IllegalArgumentException if the input type is not floating or probability is outside
     *     its accepted domain
     * @throws IllegalStateException if Tensor identifier space is exhausted; earlier output IDs
     *     allocated by this occurrence are not rolled back
     */
    static DropoutResult apply(Tensor input, double probability, GraphRngState state) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input.descriptor();
        DataType inputDataType = inputDescriptor.dataType();
        if (!inputDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "dropout input data type must be floating: " + inputDataType);
        }
        DropoutAttrs attrs = new DropoutAttrs(probability);
        Objects.requireNonNull(state, "state");
        Tensor stateTensor = state.tensor();

        TensorDescriptor outputDescriptor = new TensorDescriptor(
                inputDataType,
                inputDescriptor.shape(),
                Optional.empty(),
                inputDescriptor.requiresGrad());
        TensorDescriptor maskDescriptor = new TensorDescriptor(
                DataType.BOOL,
                inputDescriptor.shape(),
                Optional.empty(),
                false);
        TensorDescriptor stateDescriptor = new TensorDescriptor(
                DataType.INT64,
                Shape.of(2),
                Optional.empty(),
                false);
        Operation operation = new Operation(DropoutKind.DROPOUT, attrs);
        List<Tensor> outputs = TensorFactory.createDerivedOutputs(
                operation,
                List.of(input, stateTensor),
                List.of(outputDescriptor, maskDescriptor, stateDescriptor));

        Tensor output = outputs.get(0);
        outputs.get(1);
        GraphRngState nextState = new GraphRngState(outputs.get(2));
        return new DropoutResult(output, nextState);
    }
}
