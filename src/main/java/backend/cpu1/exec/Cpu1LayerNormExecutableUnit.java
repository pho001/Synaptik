package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.TensorResidencyState;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;

/**
 * Runtime wrapper for a prepared cpu1 LayerNorm node.
 */
public final class Cpu1LayerNormExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedLayerNormUnit preparedUnit;

    public Cpu1LayerNormExecutableUnit(Cpu1PreparedLayerNormUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedLayerNormUnit preparedUnit() {
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
        Tensor betaTensor = context.runtimeTensorForNodeId(preparedUnit.betaNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView input;
        Cpu1TensorView gamma;
        Cpu1TensorView beta;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            requireCpuArrayCurrent("input", preparedUnit.inputNodeId(), context);
            requireCpuArrayCurrent("gamma", preparedUnit.gammaNodeId(), context);
            requireCpuArrayCurrent("beta", preparedUnit.betaNodeId(), context);
            input = Cpu1TensorView.fromTensor(inputTensor);
            gamma = Cpu1TensorView.fromTensor(gammaTensor);
            beta = Cpu1TensorView.fromTensor(betaTensor);
            output = Cpu1TensorView.fromTensor(outputTensor);
            requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("beta", beta, Cpu1StorageKind.JAVA_ARRAY);
            requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        } else if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = requireNativeCurrent("input", preparedUnit.inputNodeId(), context);
            NativeTensorStorage nativeGamma = requireNativeCurrent("gamma", preparedUnit.gammaNodeId(), context);
            NativeTensorStorage nativeBeta = requireNativeCurrent("beta", preparedUnit.betaNodeId(), context);
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.nodeId(),
                    preparedUnit.dataType(),
                    preparedUnit.outputElementCount(),
                    "cpu1-layernorm-node-" + preparedUnit.nodeId()
            );
            input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
            gamma = Cpu1TensorView.fromNativeStorage(gammaTensor, nativeGamma);
            beta = Cpu1TensorView.fromNativeStorage(betaTensor, nativeBeta);
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("beta", beta, Cpu1StorageKind.MEMORY_SEGMENT);
            requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        } else {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM executable does not support storage "
                    + preparedUnit.storageKind());
        }
        requireShape("input", input.shape(), preparedUnit.inputShape());
        requireShape("output", output.shape(), preparedUnit.inputShape());
        if (output.elementCount() != preparedUnit.outputElementCount()) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM output element count mismatch. expected="
                    + preparedUnit.outputElementCount() + ", actual=" + output.elementCount());
        }

        preparedUnit.kernel().run(preparedUnit, input, gamma, beta, output);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 LAYER_NORM wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(
                    preparedUnit.nodeId(),
                    nativeOutput,
                    "cpu1 LAYER_NORM wrote native CPU segment"
            );
        }
    }

    private static void requireCpuArrayCurrent(String role, int nodeId, ExecutionContext context) {
        TensorResidencyState residency = context.residencyForNodeId(nodeId);
        if (residency == null || residency.cpuCurrent()) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 LAYER_NORM JAVA_ARRAY requires current CPU array "
                + role + " storage for nodeId=" + nodeId + "; residency=" + residency.residency()
                + ", nativeCurrent=" + residency.nativeCurrent()
                + ", deviceCurrent=" + residency.deviceCurrent()
                + ", reason=" + residency.lastTransitionReason());
    }

    private static NativeTensorStorage requireNativeCurrent(String role, int nodeId, ExecutionContext context) {
        TensorResidencyState residency = context.residencyForNodeId(nodeId);
        NativeTensorStorage storage = context.nativeStorageForNodeId(nodeId);
        if (residency != null && residency.nativeCurrent() && storage != null) {
            storage.ensureOpen();
            return storage;
        }
        throw new UnsupportedOperationException("cpu1 LAYER_NORM MEMORY_SEGMENT requires current native CPU segment "
                + role + " storage for nodeId=" + nodeId + "; residency="
                + (residency == null ? "unknown" : residency.residency())
                + ", cpuCurrent=" + (residency != null && residency.cpuCurrent())
                + ", nativeCurrent=" + (residency != null && residency.nativeCurrent())
                + ", deviceCurrent=" + (residency != null && residency.deviceCurrent())
                + ", nativeStorageAttached=" + (storage != null)
                + ", reason=" + (residency == null ? "" : residency.lastTransitionReason()));
    }

    private static void requireDenseNoOffset(
            String role,
            Cpu1TensorView view,
            Cpu1StorageKind expectedStorageKind
    ) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM dense slice supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM dense slice supports only dense "
                    + "contiguous no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 LAYER_NORM " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }
}
