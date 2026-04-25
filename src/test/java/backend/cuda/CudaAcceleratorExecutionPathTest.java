package backend.cuda;

import backend.ComputeBackend;
import backend.ComputeEngine;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import org.junit.jupiter.api.Test;
import tensor.Tensor;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaAcceleratorExecutionPathTest {
    @Test
    void computeEngineUsesAcceleratorExecutableForCudaMetadataWhenPresent() {
        Tensor a = Tensor.scalar(1.0);
        Tensor b = Tensor.scalar(2.0);
        Tensor out = a.add(b);
        CompiledNode node = CompiledNode.snapshot(out.topologicalSort()).getLast();

        AtomicBoolean executed = new AtomicBoolean(false);
        PreparedAcceleratorExecutable executable = new PreparedAcceleratorExecutable() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_CUDA;
            }

            @Override
            public void execute(ExecutionContext context) {
                executed.set(true);
            }
        };

        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_CUDA,
                null,
                null,
                null,
                null,
                executable,
                PartitionExecutionRole.ANCHOR
        );

        ComputeEngine.compute(
                node,
                metadata,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD)
        );

        assertTrue(executed.get());
    }
}
