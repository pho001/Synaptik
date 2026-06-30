package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;

/**
 * Runtime wrapper for a prepared cpu1 RMSNorm node.
 */
public final class Cpu1RmsNormExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedRmsNormUnit preparedUnit;

    public Cpu1RmsNormExecutableUnit(Cpu1PreparedRmsNormUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedRmsNormUnit preparedUnit() {
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
        Tensor gammaTensor = context.runtimeTensorForNodeId(preparedUnit.gammaNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView input;
        Cpu1TensorView gamma;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            context.requireCpuReadable(preparedUnit.gammaNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            input = Cpu1TensorView.fromTensor(inputTensor);
            gamma = Cpu1TensorView.fromTensor(gammaTensor);
            output = Cpu1TensorView.fromTensor(outputTensor);
            requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        } else if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    preparedUnit.inputNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage nativeGamma = context.requireNativeReadable(
                    preparedUnit.gammaNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.nodeId(),
                    preparedUnit.dataType(),
                    preparedUnit.outputElementCount(),
                    "cpu1-rmsnorm-node-" + preparedUnit.nodeId()
            );
            input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
            gamma = Cpu1TensorView.fromNativeStorage(gammaTensor, nativeGamma);
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        } else {
            throw new UnsupportedOperationException("cpu1 RMS_NORM executable does not support storage "
                    + preparedUnit.storageKind());
        }
        requireShape("input", input.shape(), preparedUnit.inputShape());
        requireShape("output", output.shape(), preparedUnit.inputShape());
        if (output.elementCount() != preparedUnit.outputElementCount()) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM output element count mismatch. expected="
                    + preparedUnit.outputElementCount() + ", actual=" + output.elementCount());
        }

        preparedUnit.kernel().run(preparedUnit, input, gamma, output);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 RMS_NORM wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(
                    preparedUnit.nodeId(),
                    nativeOutput,
                    "cpu1 RMS_NORM wrote native CPU segment"
            );
        }
    }

    private static void requireDenseNoOffset(
            String role,
            Cpu1TensorView view,
            Cpu1StorageKind expectedStorageKind
    ) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM dense slice supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM dense slice supports only dense "
                    + "contiguous no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 RMS_NORM " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }
}
