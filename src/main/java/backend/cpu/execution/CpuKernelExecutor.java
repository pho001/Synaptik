package backend.cpu.execution;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.layout.CpuLayoutOutputStorageDeferredKernel;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageResolver;
import runtime.execution.ExecutionContext;
import runtime.execution.PreparedStepMetadata;
import operations.Operation;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public final class CpuKernelExecutor {
    private final CpuStorageResolver storageResolver;

    public CpuKernelExecutor() {
        this(new CpuStorageResolver());
    }

    public CpuKernelExecutor(CpuStorageResolver storageResolver) {
        this.storageResolver = Objects.requireNonNull(storageResolver, "storageResolver cannot be null");
    }

    public CpuKernelResult execute(
            CpuKernel kernel,
            Operation operation,
            List<Tensor> inputs,
            Tensor output,
            int nodeId,
            List<Integer> inputNodeIds,
            CpuNodeExecutionPlan plan,
            ExecutionContext executionContext,
            PreparedStepMetadata metadata,
            List<PreparedStepMetadata> inputMetadatas
    ) {
        Objects.requireNonNull(kernel, "kernel cannot be null");
        Objects.requireNonNull(operation, "operation cannot be null");
        Objects.requireNonNull(inputs, "inputs cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        Objects.requireNonNull(inputNodeIds, "inputNodeIds cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        Objects.requireNonNull(executionContext, "executionContext cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");

        CpuNodeWorkspace workspace = executionContext.workspaceForNodeId(nodeId) == null
                ? null
                : executionContext.requireWorkspace(nodeId, CpuNodeWorkspace.class);
        if (workspace != null) {
            workspace.clearFloatContinuation();
        }

        CpuKernelContext context = new CpuKernelContext(
                nodeId,
                inputNodeIds,
                plan,
                executionContext,
                metadata,
                inputMetadatas,
                operation
        );

        if (kernel instanceof CpuLayoutOutputStorageDeferredKernel) {
            return kernel.execute(new CpuKernelCall(
                    operation,
                    inputs,
                    output,
                    List.of(),
                    null,
                    plan,
                    context,
                    workspace
            ));
        }

        CpuStorageBindings storage = storageResolver.bindRuntime(
                inputNodeIds,
                inputs,
                nodeId,
                output,
                executionContext
        );
        return kernel.execute(new CpuKernelCall(
                operation,
                inputs,
                output,
                storage,
                plan,
                context,
                workspace
        ));
    }
}
