package backend.cpu1.exec;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.offset.Cpu1GenericOffsetPlan;
import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Runtime kernel arguments bound from the current execution run.
 */
public final class Cpu1KernelArgs {
    private final Cpu1PreparedElementwiseUnit preparedUnit;
    private final List<Cpu1TensorView> inputs;
    private final Cpu1GenericOffsetPlan[] inputGenericOffsetPlans;
    private final Cpu1TensorView output;
    private final Cpu1Workspace workspace;
    private Cpu1GenericOffsetPlan outputGenericOffsetPlan;

    public Cpu1KernelArgs(Cpu1PreparedElementwiseUnit preparedUnit, List<Cpu1TensorView> inputs, Cpu1TensorView output) {
        this(preparedUnit, inputs, output, null);
    }

    public Cpu1KernelArgs(
            Cpu1PreparedElementwiseUnit preparedUnit,
            List<Cpu1TensorView> inputs,
            Cpu1TensorView output,
            Cpu1Workspace workspace
    ) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
        this.inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs cannot be null"));
        this.output = Objects.requireNonNull(output, "output cannot be null");
        this.workspace = workspace;
        validate();
        this.inputGenericOffsetPlans = preparedUnit.layoutKind() == Cpu1LayoutKind.STRIDED_GENERIC
                ? new Cpu1GenericOffsetPlan[this.inputs.size()]
                : null;
    }

    public Cpu1PreparedElementwiseUnit preparedUnit() {
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

    public boolean hasWorkspace() {
        return workspace != null;
    }

    public Cpu1Workspace workspace() {
        if (workspace == null) {
            throw new IllegalStateException("This cpu1 kernel invocation does not have workspace.");
        }
        return workspace;
    }

    public Cpu1GenericOffsetPlan inputGenericOffsetPlan(int inputIndex) {
        requireGenericStridedLayout();
        Cpu1GenericOffsetPlan plan = inputGenericOffsetPlans[inputIndex];
        if (plan == null) {
            plan = Cpu1GenericOffsetPlan.forView(inputs.get(inputIndex));
            inputGenericOffsetPlans[inputIndex] = plan;
        }
        return plan;
    }

    public Cpu1GenericOffsetPlan outputGenericOffsetPlan() {
        requireGenericStridedLayout();
        if (outputGenericOffsetPlan == null) {
            outputGenericOffsetPlan = Cpu1GenericOffsetPlan.forView(output);
        }
        return outputGenericOffsetPlan;
    }

    public int elementCount() {
        return preparedUnit.elementCount();
    }

    public float scalarF32() {
        return preparedUnit.scalarParameterF32();
    }

    public double scalarF64() {
        return preparedUnit.scalarParameterF64();
    }

    private void validate() {
        if (inputs.size() != preparedUnit.inputNodeIds().size()) {
            throw new IllegalArgumentException("Expected " + preparedUnit.inputNodeIds().size()
                    + " inputs, got " + inputs.size());
        }
        if (output.dataType() != preparedUnit.dataType()) {
            throw new IllegalArgumentException("Output dtype " + output.dataType()
                    + " does not match prepared dtype " + preparedUnit.dataType());
        }
        requireSupportedView(output, "output");
        if (output.elementCount() != preparedUnit.elementCount()) {
            throw new IllegalArgumentException("Output element count " + output.elementCount()
                    + " does not match prepared element count " + preparedUnit.elementCount());
        }
        if (preparedUnit.layoutKind() == Cpu1LayoutKind.BROADCAST_INNER && !output.contiguous()) {
            throw new IllegalArgumentException("cpu1 BROADCAST_INNER requires contiguous output view.");
        }
        for (int i = 0; i < inputs.size(); i++) {
            Cpu1TensorView input = inputs.get(i);
            DataType expectedInputDataType = preparedUnit.inputDataType(i);
            if (input.dataType() != expectedInputDataType) {
                throw new IllegalArgumentException("Input " + i + " dtype " + input.dataType()
                        + " does not match prepared input dtype " + expectedInputDataType);
            }
            requireSupportedView(input, "input " + i);
            if (input.elementCount() != preparedUnit.elementCount()) {
                throw new IllegalArgumentException("Input " + i + " element count " + input.elementCount()
                        + " does not match prepared element count " + preparedUnit.elementCount());
            }
            if (preparedUnit.layoutKind() == Cpu1LayoutKind.BROADCAST_INNER && !isBroadcastInnerVectorInput(input)) {
                throw new IllegalArgumentException("Input " + i
                        + " is not compatible with cpu1 BROADCAST_INNER vector layout.");
            }
        }
    }

    private void requireSupportedView(Cpu1TensorView view, String role) {
        if (view.dataType() != DataType.FLOAT32
                && view.dataType() != DataType.FLOAT64
                && view.dataType() != DataType.BFLOAT16
                && view.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("cpu1 only supports FLOAT32/FLOAT64/BFLOAT16/BOOL " + role + " views.");
        }
        if (view.storageKind() != preparedUnit.storageKind()) {
            throw new IllegalArgumentException("cpu1 " + role + " storage kind " + view.storageKind()
                    + " does not match prepared storage kind " + preparedUnit.storageKind());
        }
    }

    private void requireGenericStridedLayout() {
        if (preparedUnit.layoutKind() != Cpu1LayoutKind.STRIDED_GENERIC) {
            throw new IllegalStateException("Generic offset plans are available only for STRIDED_GENERIC units.");
        }
    }

    private static boolean isBroadcastInnerVectorInput(Cpu1TensorView view) {
        if (view.contiguous() || isBroadcastScalar(view)) {
            return true;
        }
        int rank = view.rank();
        if (rank == 0) {
            return false;
        }
        for (int dim = 0; dim < rank - 1; dim++) {
            if (view.stride(dim) != 0) {
                return false;
            }
        }
        return view.stride(rank - 1) == 1;
    }

    private static boolean isBroadcastScalar(Cpu1TensorView view) {
        for (int dim = 0; dim < view.rank(); dim++) {
            if (view.stride(dim) != 0) {
                return false;
            }
        }
        return true;
    }

}
