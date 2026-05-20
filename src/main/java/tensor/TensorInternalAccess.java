package tensor;
import tensor.storage.TensorStorage;

import backend.ComputeBackend;
import operations.Operation;
import tensor.autograd.GradientRule;

import java.util.List;

/**
 * Internal mutation/read access for graph construction, rewrites, planning, and runtime plumbing.
 * This type is intentionally separate from the public {@link Tensor} surface so that model-building
 * code does not treat structural graph mutation as normal user-facing API.
 */
public final class TensorInternalAccess {
    private TensorInternalAccess() {
    }

    public static void setGradient(Tensor tensor, Tensor gradient) {
        tensor.setGradientInternal(gradient);
    }

    public static void clearGradient(Tensor tensor) {
        tensor.setGradientInternal(null);
    }

    public static void setBackward(Tensor tensor, boolean backward) {
        tensor.setBackwardInternal(backward);
    }

    public static void setGradientRule(Tensor tensor, GradientRule gradientRule) {
        tensor.setGradientRuleInternal(gradientRule);
    }

    public static GradientRule gradientRule(Tensor tensor) {
        return tensor.gradientRuleInternal();
    }

    public static void setOperation(Tensor tensor, Operation operation) {
        tensor.setOperationInternal(operation);
    }

    public static void setPrevTensors(Tensor tensor, List<Tensor> prevTensors) {
        tensor.setPrevTensorsInternal(prevTensors);
    }

    public static void setBackendIntent(Tensor tensor, ComputeBackend backend) {
        tensor.setBackendIntentInternal(backend);
    }

    public static ComputeBackend backendIntent(Tensor tensor) {
        return tensor.backendIntentInternal();
    }

    public static Operation operation(Tensor tensor) {
        return tensor.getOperation();
    }

    public static List<Tensor> inputs(Tensor tensor) {
        return tensor.prevTensorsRef();
    }

    public static void buildBackwardGraph(Tensor tensor) {
        tensor.buildBackwardGraphInternal();
    }

    public static void aliasRuntimeFrom(Tensor target, Tensor source) {
        target.aliasRuntimeFromInternal(source);
    }

    public static void replaceStorage(Tensor target, TensorStorage storage) {
        target.replaceStorageInternal(storage);
    }

    public static TensorStorage storage(Tensor tensor) {
        return tensor.storageInternal();
    }

    public static double getByStorageOffset(Tensor tensor, int offset) {
        return tensor.getByStorageOffset(offset);
    }

    public static void setByStorageOffset(Tensor tensor, int offset, double value) {
        tensor.setByStorageOffset(offset, value);
    }

    public static float[] float32Data(Tensor tensor) {
        return tensor.float32DataInternal();
    }

    public static double[] float64Data(Tensor tensor) {
        return tensor.float64DataInternal();
    }

    public static short[] bfloat16Data(Tensor tensor) {
        return tensor.bfloat16DataInternal();
    }

    public static int[] int32Data(Tensor tensor) {
        return tensor.int32DataInternal();
    }

    public static long[] int64Data(Tensor tensor) {
        return tensor.int64DataInternal();
    }

    public static byte[] boolData(Tensor tensor) {
        return tensor.boolDataInternal();
    }

    public static void markStorageModified(Tensor tensor) {
        tensor.markStorageModifiedInternal();
    }

    public static List<Tensor> prevTensors(Tensor tensor) {
        return tensor.prevTensorsRef();
    }
}
