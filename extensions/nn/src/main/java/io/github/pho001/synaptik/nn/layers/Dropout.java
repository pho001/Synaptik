package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.Module;
import java.util.Objects;

/**
 * Mode-sensitive inverted dropout with explicit caller-threaded graph RNG state.
 *
 * <p>The layer stores one finite drop probability numerically in {@code [0.0, 1.0)}. It declares
 * no parameter, buffer, child, seed, generator, counter, or other random state. The immutable
 * {@link ForwardContext} supplied to {@link #forward(Tensor, GraphRngState, ForwardContext)} is
 * authoritative for that call, even when it originated from another module or differs from this
 * layer's current inherited mode.</p>
 *
 * <p>Training delegates exactly once to {@link Tensor#dropout(double, GraphRngState)} and returns
 * a fresh {@link DropoutForwardResult} containing that Model occurrence's exact public output and
 * next-state references. Evaluation creates no Model occurrence and returns a fresh result
 * containing the exact input and incoming-state references. Consequently evaluation accepts any
 * non-null Tensor data type, while training retains Model's floating-input requirement.</p>
 *
 * <p>Forward construction records symbolic Model semantics or performs the evaluation bypass. It
 * does not draw random values, evaluate a Tensor, manage a random stream between calls, mutate the
 * supplied state, compile a graph, select a backend, or execute work. Callers express sequential
 * random consumption by passing a returned next state to a later call; reusing an incoming state
 * deliberately expresses a branch.</p>
 */
public final class Dropout extends Module {
    private final double probability;

    /**
     * Creates a parameterless and bufferless dropout layer.
     *
     * <p>Construction validates through the Model-owned {@link DropoutAttrs} contract, retains the
     * exact validated primitive value, and retains no attributes object. Both signed zero values
     * are accepted and preserved without normalization. Construction creates no Tensor, producer,
     * graph RNG state, random source, or Tensor identifier.</p>
     *
     * @param probability finite drop probability numerically in {@code [0.0, 1.0)}; either signed
     *     zero is retained exactly
     * @throws IllegalArgumentException if {@code probability} is non-finite, negative, or at
     *     least one
     */
    public Dropout(double probability) {
        DropoutAttrs attrs = new DropoutAttrs(probability);
        this.probability = attrs.probability();
    }

    /**
     * Applies the branch selected by one explicit immutable forward context.
     *
     * <p>Arguments are null-checked in input, state, then context order before the context mode is
     * read. In evaluation, the method creates only a fresh NN result retaining the exact input and
     * incoming state; it performs no Model validation or Tensor allocation. In training, it calls
     * {@link Tensor#dropout(double, GraphRngState)} exactly once with the stored probability and
     * supplied state, then wraps the exact Model output and next state in a fresh NN result.</p>
     *
     * <p>The method never reads this module's current {@link #mode()} implicitly and changes
     * neither module mode nor either supplied value. A training Model allocation failure may
     * consume a prefix of its three output identifiers under the existing no-rollback contract;
     * no NN result or hidden layer state is retained.</p>
     *
     * @param input non-null Tensor; every data type is accepted by evaluation, while training
     *     requires a floating type through the delegated Model contract
     * @param state non-null caller-owned explicit graph RNG state; retained unchanged in
     *     evaluation or consumed as the exact Model state input in training
     * @param context non-null immutable mode snapshot whose mode alone selects this call's branch
     * @return a non-null fresh result containing exact branch-selected output and next-state
     *     references
     * @throws NullPointerException if {@code input}, {@code state}, or {@code context} is null,
     *     checked in that order with the parameter name as the message
     * @throws IllegalArgumentException if training is selected and Model rejects a non-floating
     *     input data type
     * @throws IllegalStateException if training is selected and Tensor identifier space is
     *     exhausted; identifiers already allocated for the Model occurrence are not rolled back
     */
    public DropoutForwardResult forward(
            Tensor input, GraphRngState state, ForwardContext context) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        GraphRngState suppliedState = Objects.requireNonNull(state, "state");
        ForwardContext suppliedContext = Objects.requireNonNull(context, "context");

        return switch (suppliedContext.mode()) {
            case EVALUATION -> new DropoutForwardResult(suppliedInput, suppliedState);
            case TRAINING -> {
                DropoutResult result = suppliedInput.dropout(probability, suppliedState);
                yield new DropoutForwardResult(result.output(), result.nextState());
            }
        };
    }
}
