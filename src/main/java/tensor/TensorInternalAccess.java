package tensor;

import backend.ComputeBackend;
import operations.Operation;

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

    public static void setBackwardFunction(Tensor tensor, Runnable backwardFunction) {
        tensor.setBackwardFunctionInternal(backwardFunction);
    }

    public static void setOperation(Tensor tensor, Operation operation) {
        tensor.setOperationInternal(operation);
    }

    public static void setPrevTensors(Tensor tensor, List<Tensor> prevTensors) {
        tensor.setPrevTensorsInternal(prevTensors);
    }

    public static void setBackend(Tensor tensor, ComputeBackend backend) {
        tensor.setBackendInternal(backend);
    }

    public static void buildBackwardGraph(Tensor tensor) {
        tensor.buildBackwardGraphInternal();
    }

    public static void aliasRuntimeFrom(Tensor target, Tensor source) {
        target.aliasRuntimeFromInternal(source);
    }

    public static List<Tensor> prevTensors(Tensor tensor) {
        return tensor.prevTensorsRef();
    }
}
