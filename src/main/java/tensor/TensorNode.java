package tensor;

import operations.Operation;
import tensor.autograd.GradientRule;

import java.util.ArrayList;
import java.util.List;

/**
 * Internal graph-node facts for a public logical {@link Tensor}.
 */
final class TensorNode {
    private Operation operation;
    private List<Tensor> inputs;
    private GradientRule gradientRule;
    private boolean backward;

    TensorNode(Operation operation, List<Tensor> inputs) {
        this.operation = operation;
        setInputs(inputs);
    }

    Operation operation() {
        return operation;
    }

    void setOperation(Operation operation) {
        this.operation = operation;
    }

    List<Tensor> inputs() {
        return inputs;
    }

    void setInputs(List<Tensor> inputs) {
        this.inputs = inputs == null ? null : new ArrayList<>(inputs);
    }

    GradientRule gradientRule() {
        return gradientRule;
    }

    void setGradientRule(GradientRule gradientRule) {
        this.gradientRule = gradientRule;
    }

    boolean backward() {
        return backward;
    }

    void setBackward(boolean backward) {
        this.backward = backward;
    }

}
