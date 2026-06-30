package backend.cpu.prepare.layout;

import backend.cpu.plan.CpuPreparedInput;
import backend.cpu.plan.layout.PreparedInputsResult;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.plan.PreparedTypeContract;
import backend.cpu.prepare.CpuExecutionPlanner;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.LayoutClass;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.layout.TensorRemap;

import java.util.ArrayList;
import java.util.List;

public final class PreparedInputPlanner {
    private PreparedInputPlanner() {
    }

    public static PreparedInputsResult plan(
            Operation op,
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node,
            PreparedTypeContract typeContract,
            CpuExecutionPlanner planner,
            StridedLayoutDecision layoutDecision
    ) {
        if (inputs.isEmpty()) {
            return new PreparedInputsResult(List.of(), List.of());
        }

        if (PreparedInputPolicy.bypassPreparation(op)
                && !PreparedInputPolicy.requiresPreparedInputs(op, inputs, node, typeContract, planner)) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        if (layoutDecision != null && layoutDecision.useStridedPath()) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        List<CpuPreparedInput> preparedInputs = new ArrayList<>();
        List<CompiledTensorDescriptor> runtimeInputs = new ArrayList<>(inputs.size());

        for (int i = 0; i < inputs.size(); i++) {
            CompiledTensorDescriptor input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("Input tensor at index " + i + " is null");
            }

            DataType expectedInputType = typeContract.expectedInputTypes().get(i);

            boolean forcePrepared = layoutDecision != null && layoutDecision.shouldForcePrepareInput(i);
            if (!forcePrepared && !PreparedInputPolicy.requiresPreparedInput(op, input, node, expectedInputType, planner)) {
                runtimeInputs.add(input);
                continue;
            }

            if (!PreparedInputPolicy.canConvertPreparedInput(input.dataType(), expectedInputType)) {
                throw new IllegalArgumentException("Unsupported prepared input conversion for op="
                        + (op == null ? "null" : op.opType())
                        + ", inputIndex=" + i
                        + ", sourceType=" + input.dataType()
                        + ", expectedType=" + expectedInputType);
            }

            Tensor preparedTensor = createPreparedTensor(input, expectedInputType, node, i);
            TensorRemap.RemapPlan remapPlan = TensorRemap.buildPlan(
                    input.shape(),
                    input.strides(),
                    input.storageOffset(),
                    preparedTensor.getShapeUnsafe(),
                    preparedTensor.getStridesUnsafe(),
                    preparedTensor.getStorageOffsetUnsafe()
            );
            preparedInputs.add(new CpuPreparedInput(i, preparedTensor, remapPlan));
            runtimeInputs.add(preparedDescriptor(input, expectedInputType));
        }

        return new PreparedInputsResult(preparedInputs, runtimeInputs);
    }

    private static Tensor createPreparedTensor(
            CompiledTensorDescriptor source,
            DataType targetType,
            CompiledTensorDescriptor node,
            int inputIndex
    ) {
        String baseLabel = node == null ? "node" : "node_" + node.nodeId();
        String label = baseLabel + "_prepared_input_" + inputIndex;
        return new Tensor(source.shape(), new ArrayList<>(), label, targetType);
    }

    private static CompiledTensorDescriptor preparedDescriptor(CompiledTensorDescriptor source, DataType dataType) {
        int[] shape = source.shape();
        int[] strides = TensorMetadata.computeStrides(shape);
        long logicalElementCount = source.logicalElementCount();
        long byteLength = Math.multiplyExact(logicalElementCount, bytesPerElement(dataType));
        return new CompiledTensorDescriptor(
                source.nodeId(),
                source.opType(),
                dataType,
                shape,
                shape.length,
                strides,
                0,
                logicalElementCount,
                logicalElementCount,
                byteLength,
                byteLength,
                LayoutClass.DENSE_CONTIGUOUS,
                true,
                false,
                false,
                false,
                source.leaf(),
                source.backwardNode(),
                source.requiresGrad(),
                source.trainableParameter(),
                source.inputIds()
        );
    }

    private static int bytesPerElement(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }
}
