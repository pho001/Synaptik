package graph.execution.plan;

import tensor.Tensor;

import java.util.function.Supplier;

/**
 * Backend-neutral sink for run-scoped prepared execution state.
 */
public interface PreparedRuntimeStateAllocator {
    /**
     * Returns the run-local fork for a prepared workspace template.
     *
     * @param template immutable/prepared workspace template
     * @param forkFactory factory used once per distinct template in one run
     * @return run-local workspace fork
     */
    Object forkWorkspace(Object template, Supplier<?> forkFactory);

    /**
     * Registers a run-local workspace for a compiled node.
     *
     * @param nodeId compiled node id
     * @param workspace run-local workspace
     */
    void putWorkspace(int nodeId, Object workspace);

    /**
     * Registers a run-local prepared input tensor.
     *
     * @param nodeId compiled node id
     * @param inputIndex input index
     * @param tensor run-local prepared input tensor
     */
    void putPreparedInputTensor(int nodeId, int inputIndex, Tensor tensor);
}
