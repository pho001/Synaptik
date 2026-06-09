package backend.cpu1.exec;

import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.offset.Cpu1GenericOffsetPlan;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import tensor.DataType;

import java.util.List;

/**
 * Runtime arguments for a prepared generated fused elementwise kernel.
 */
public final class Cpu1FusedKernelArgs {
    private final Cpu1PreparedFusedElementwiseUnit preparedUnit;
    private final List<Cpu1TensorView> inputs;
    private final Cpu1TensorView output;
    private final Cpu1GenericOffsetPlan[] inputGenericOffsetPlans;
    private Cpu1GenericOffsetPlan outputGenericOffsetPlan;

    public Cpu1FusedKernelArgs(
            Cpu1PreparedFusedElementwiseUnit preparedUnit,
            List<Cpu1TensorView> inputs,
            Cpu1TensorView output
    ) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        if (inputs == null) {
            throw new IllegalArgumentException("inputs cannot be null");
        }
        if (output == null) {
            throw new IllegalArgumentException("output cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.inputs = List.copyOf(inputs);
        this.output = output;
        validate();
        this.inputGenericOffsetPlans = new Cpu1GenericOffsetPlan[this.inputs.size()];
    }

    public Cpu1PreparedFusedElementwiseUnit preparedUnit() {
        return preparedUnit;
    }

    public Cpu1TensorView input(int index) {
        return inputs.get(index);
    }

    public List<Cpu1TensorView> inputs() {
        return inputs;
    }

    public Cpu1TensorView output() {
        return output;
    }

    public int elementCount() {
        return preparedUnit.elementCount();
    }

    public Cpu1GenericOffsetPlan inputGenericOffsetPlan(int inputIndex) {
        Cpu1GenericOffsetPlan plan = inputGenericOffsetPlans[inputIndex];
        if (plan == null) {
            plan = Cpu1GenericOffsetPlan.forView(inputs.get(inputIndex));
            inputGenericOffsetPlans[inputIndex] = plan;
        }
        return plan;
    }

    public Cpu1GenericOffsetPlan outputGenericOffsetPlan() {
        if (outputGenericOffsetPlan == null) {
            outputGenericOffsetPlan = Cpu1GenericOffsetPlan.forView(output);
        }
        return outputGenericOffsetPlan;
    }

    private void validate() {
        if (inputs.size() != preparedUnit.inputNodeIds().size()) {
            throw new IllegalArgumentException("Expected " + preparedUnit.inputNodeIds().size()
                    + " fused inputs, got " + inputs.size());
        }
        if (output.dataType() != preparedUnit.outputDataType()) {
            throw new IllegalArgumentException("Fused output dtype " + output.dataType()
                    + " does not match prepared dtype " + preparedUnit.outputDataType());
        }
        if (output.elementCount() != preparedUnit.elementCount()) {
            throw new IllegalArgumentException("Fused output element count " + output.elementCount()
                    + " does not match prepared element count " + preparedUnit.elementCount());
        }
        requireSupportedView(output, "output");
        for (int i = 0; i < inputs.size(); i++) {
            Cpu1TensorView input = inputs.get(i);
            Cpu1FusedInputPlan inputPlan = preparedUnit.plan().inputs().get(i);
            if (input.dataType() != inputPlan.dataType()) {
                throw new IllegalArgumentException("Fused input " + i + " dtype " + input.dataType()
                        + " does not match prepared dtype " + inputPlan.dataType());
            }
            if (input.elementCount() != preparedUnit.elementCount()) {
                throw new IllegalArgumentException("Fused input " + i + " element count " + input.elementCount()
                        + " does not match prepared element count " + preparedUnit.elementCount());
            }
            requireSupportedView(input, "input " + i);
        }
    }

    private void requireSupportedView(Cpu1TensorView view, String role) {
        if (view.dataType() != DataType.FLOAT32
                && view.dataType() != DataType.FLOAT64
                && view.dataType() != DataType.BFLOAT16
                && view.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("cpu1 fused only supports FLOAT32/FLOAT64/BFLOAT16/BOOL "
                    + role + " views.");
        }
        if (view.storageKind() != preparedUnit.storageKind()) {
            throw new IllegalArgumentException("cpu1 fused " + role + " storage kind " + view.storageKind()
                    + " does not match prepared storage kind " + preparedUnit.storageKind());
        }
    }
}
