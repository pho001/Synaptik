package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;

/**
 * Runtime wrapper for a prepared cpu1 CONV2D node.
 */
public final class Cpu1Conv2dExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedConv2dUnit preparedUnit;

    public Cpu1Conv2dExecutableUnit(Cpu1PreparedConv2dUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedConv2dUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor weightTensor = context.runtimeTensorForNodeId(preparedUnit.weightNodeId());
        Tensor biasTensor = preparedUnit.hasBias()
                ? context.runtimeTensorForNodeId(preparedUnit.biasNodeId())
                : null;
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView input;
        Cpu1TensorView weight;
        Cpu1TensorView bias;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            context.requireCpuReadable(preparedUnit.weightNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            if (preparedUnit.hasBias()) {
                context.requireCpuReadable(preparedUnit.biasNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            }
            input = Cpu1TensorView.fromTensor(inputTensor);
            weight = Cpu1TensorView.fromTensor(weightTensor);
            bias = biasTensor == null ? null : Cpu1TensorView.fromTensor(biasTensor);
            output = Cpu1TensorView.fromTensor(outputTensor);
            requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("weight", weight, Cpu1StorageKind.JAVA_ARRAY);
            if (bias != null) {
                requireDenseNoOffset("bias", bias, Cpu1StorageKind.JAVA_ARRAY);
            }
            requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        } else if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    preparedUnit.inputNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage nativeWeight = context.requireNativeReadable(
                    preparedUnit.weightNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage nativeBias = preparedUnit.hasBias()
                    ? context.requireNativeReadable(preparedUnit.biasNodeId(), CpuMaterializationReason.CPU_CONSUMER)
                    : null;
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.nodeId(),
                    preparedUnit.dataType(),
                    preparedUnit.outputElementCount(),
                    "cpu1-conv2d-node-" + preparedUnit.nodeId()
            );
            input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
            weight = Cpu1TensorView.fromNativeStorage(weightTensor, nativeWeight);
            bias = nativeBias == null ? null : Cpu1TensorView.fromNativeStorage(biasTensor, nativeBias);
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("weight", weight, Cpu1StorageKind.MEMORY_SEGMENT);
            if (bias != null) {
                requireDenseNoOffset("bias", bias, Cpu1StorageKind.MEMORY_SEGMENT);
            }
            requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        } else {
            throw new UnsupportedOperationException("cpu1 CONV2D executable does not support storage "
                    + preparedUnit.storageKind());
        }
        requireShape("input", input.shape(), preparedUnit.inputShape());
        requireShape("weight", weight.shape(), preparedUnit.weightShape());
        if (bias != null) {
            requireShape("bias", bias.shape(), preparedUnit.biasShape());
        }
        requireShape("output", output.shape(), preparedUnit.outputShape());
        if (output.elementCount() != preparedUnit.outputElementCount()) {
            throw new UnsupportedOperationException("cpu1 CONV2D output element count mismatch. expected="
                    + preparedUnit.outputElementCount() + ", actual=" + output.elementCount());
        }

        preparedUnit.kernel().run(preparedUnit, input, weight, bias, output);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 CONV2D wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(
                    preparedUnit.nodeId(),
                    nativeOutput,
                    "cpu1 CONV2D wrote native CPU segment"
            );
        }
    }

    private static void requireDenseNoOffset(
            String role,
            Cpu1TensorView view,
            Cpu1StorageKind expectedStorageKind
    ) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 CONV2D dense direct route supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 CONV2D dense direct route supports only dense "
                    + "contiguous no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 CONV2D " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }
}
