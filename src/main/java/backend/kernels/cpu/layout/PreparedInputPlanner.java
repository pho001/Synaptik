package backend.kernels.cpu.layout;

import backend.CpuPreparedInput;
import backend.kernels.cpu.plan.PreparedTypeContract;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorRemap;

import java.util.ArrayList;
import java.util.List;

public final class PreparedInputPlanner {
    private PreparedInputPlanner() {
    }

    public static PreparedInputsResult plan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            PreparedTypeContract typeContract,
            CpuExecutionPlanner planner,
            boolean stridedPath
    ) {
        if (inputs.isEmpty()) {
            return new PreparedInputsResult(List.of(), List.of());
        }

        if (PreparedInputPolicy.bypassPreparation(op)
                && !PreparedInputPolicy.requiresPreparedInputs(op, inputs, node, typeContract, planner)) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        if (stridedPath) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        List<CpuPreparedInput> preparedInputs = new ArrayList<>();
        List<Tensor> runtimeInputs = new ArrayList<>(inputs.size());

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("Input tensor at index " + i + " is null");
            }

            DataType expectedInputType = typeContract.expectedInputTypes().get(i);

            if (!PreparedInputPolicy.requiresPreparedInput(op, input, node, expectedInputType, planner)) {
                runtimeInputs.add(input);
                continue;
            }

            if (!PreparedInputPolicy.canConvertPreparedInput(input.getDataType(), expectedInputType)) {
                throw new IllegalArgumentException("Unsupported prepared input conversion for op="
                        + (op == null ? "null" : op.opType())
                        + ", inputIndex=" + i
                        + ", sourceType=" + input.getDataType()
                        + ", expectedType=" + expectedInputType);
            }

            Tensor preparedTensor = createPreparedTensor(input, expectedInputType, node, i);
            TensorRemap.RemapPlan remapPlan = TensorRemap.buildPlan(input, preparedTensor);
            preparedInputs.add(new CpuPreparedInput(i, preparedTensor, remapPlan));
            runtimeInputs.add(preparedTensor);
        }

        return new PreparedInputsResult(preparedInputs, runtimeInputs);
    }

    private static Tensor createPreparedTensor(Tensor source, DataType targetType, Tensor node, int inputIndex) {
        String baseLabel = node != null && node.getLabel() != null ? node.getLabel() : "node";
        String label = baseLabel + "_prepared_input_" + inputIndex;
        return new Tensor(source.getShape().clone(), new ArrayList<>(), label, targetType);
    }
}
