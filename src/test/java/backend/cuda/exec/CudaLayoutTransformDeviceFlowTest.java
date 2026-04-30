package backend.cuda.exec;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.execution.trace.ExecutionStepTrace;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaLayoutTransformDeviceFlowTest {
    @Test
    void reshapePermuteContiguousFlowAvoidsIntermediateCpuMaterialization() {
        Tensor source = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "cudaLayoutSource", DataType.FLOAT32);
        Tensor reshape = source.reshape(3, 2);
        Tensor permute = reshape.permute(1, 0);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(source, reshape, permute));
        CompiledNode sourceNode = nodeFor(nodes, source);
        CompiledNode reshapeNode = nodeFor(nodes, reshape);
        CompiledNode permuteNode = nodeFor(nodes, permute);

        PreparedExecution prepared = prepared(
                nodes,
                source,
                sourceNode,
                List.of(
                        cudaSourceStep(sourceNode),
                        cudaLayoutStep(reshapeNode),
                        cudaLayoutStep(permuteNode)
                )
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.CPU_CONSUMER));
        ExecutionStepTrace permuteTrace = trace.steps().stream()
                .filter(step -> step.opType().equals(Operation.OpType.PERMUTE.name()))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = permuteTrace.metadata().attributes();
        assertEquals("GPU_CUDA", attrs.get("acceleratorBufferBackend"));
        assertEquals("GPU_LAYOUT_VIEW_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("METADATA_ONLY_VIEW", attrs.get("gpuLayoutTransformKind"));
        assertEquals("HOST_SHARED_DEVICE_BUFFER", attrs.get("storageResidency"));
        assertEquals("GPU_CUDA", attrs.get("deviceBufferBackend"));
    }

    @Test
    void unsupportedDirectNonDenseCudaConsumerFallsBackVisibly() {
        Tensor base = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "cudaBase", DataType.FLOAT32);
        Tensor nonDenseSource = base.permute(1, 0);
        Tensor contiguous = nonDenseSource.contiguous();
        CompiledGraph compiled = CompiledGraph.compile(contiguous, OptimizerConfig.noOptimization());
        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        CompiledNode baseNode = nodeFor(nodes, base);
        CompiledNode sourceNode = nodeFor(nodes, nonDenseSource);
        CompiledNode contiguousNode = nodeFor(nodes, contiguous);
        CompiledNodeExecutionMetadata contiguousMetadata = compiled.prepare(RuntimeConfig.inferenceDefaults())
                .executionSteps()
                .stream()
                .filter(step -> step.compiledNode().id() == contiguousNode.id())
                .map(PreparedNodeExecution::metadata)
                .findFirst()
                .orElseThrow();

        PreparedExecution prepared = prepared(
                nodes,
                base,
                baseNode,
                List.of(
                        cudaSourceStep(sourceNode),
                        cpuLayoutStep(contiguousNode, contiguousMetadata)
                )
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);

        ExecutionStepTrace contiguousTrace = trace.steps().stream()
                .filter(step -> step.opType().equals(Operation.OpType.CONTIGUOUS.name()))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = contiguousTrace.metadata().attributes();
        assertEquals("GPU_CUDA", attrs.get("acceleratorBufferBackend"));
        assertEquals("GPU_LAYOUT_TRANSFORM_UNSUPPORTED", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("UNAVAILABLE", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("CPU_ARRAY", attrs.get("storageResidency"));
        assertTrue(String.valueOf(attrs.get("acceleratorBufferReason"))
                .contains("no dense layout materializer registered"));
    }

    private static PreparedExecution prepared(
            List<CompiledNode> nodes,
            Tensor rootTensor,
            CompiledNode forwardOutputNode,
            List<PreparedNodeExecution> steps
    ) {
        return new PreparedExecution(
                RuntimeConfig.inferenceDefaults(),
                false,
                steps,
                steps,
                List.of(),
                nodes,
                Map.of(),
                rootTensor,
                forwardOutputNode,
                null,
                null,
                graph.execution.trace.PrepareTrace.skipped()
        );
    }

    private static PreparedNodeExecution cudaSourceStep(CompiledNode node) {
        return new PreparedNodeExecution(
                node,
                new CompiledNodeExecutionMetadata(
                        ComputeBackend.GPU_CUDA,
                        null,
                        null,
                        null,
                        null,
                        new SyntheticCudaSourceExecutable(node.id()),
                        null,
                        List.of(),
                        PartitionExecutionRole.NONE
                )
        );
    }

    private static PreparedNodeExecution cudaLayoutStep(CompiledNode node) {
        return new PreparedNodeExecution(node, metadata(ComputeBackend.GPU_CUDA));
    }

    private static PreparedNodeExecution cpuLayoutStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        return new PreparedNodeExecution(node, metadata);
    }

    private static CompiledNodeExecutionMetadata metadata(ComputeBackend backend) {
        return new CompiledNodeExecutionMetadata(
                backend,
                null,
                null,
                null,
                null,
                null,
                null,
                List.of(),
                PartitionExecutionRole.NONE
        );
    }

    private static CompiledNode nodeFor(List<CompiledNode> nodes, Tensor tensor) {
        return nodes.stream()
                .filter(node -> node.semanticTensor() == tensor || node.sourceTensor() == tensor)
                .findFirst()
                .orElseThrow();
    }

    private record SyntheticCudaSourceExecutable(int nodeId) implements PreparedAcceleratorExecutable {
        @Override
        public ComputeBackend backend() {
            return ComputeBackend.GPU_CUDA;
        }

        @Override
        public void execute(ExecutionContext context) {
            Tensor tensor = context.runtimeTensorForNodeId(nodeId);
            AcceleratorBufferLayout layout = AcceleratorBufferLayout.fromTensor(tensor);
            CudaBufferBinding binding = new CudaBufferBinding(
                    nodeId,
                    layout,
                    new CudaBufferHandle(MemorySegment.ofAddress(30_000L + nodeId), layout.logicalByteLength(), false),
                    CudaBufferAccess.READ_WRITE
            );
            context.attachDeviceBufferBinding(
                    nodeId,
                    binding,
                    StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                    "synthetic CUDA source binding"
            );
        }

        @Override
        public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    AcceleratorBufferBindingMode.AUTO,
                    AcceleratorBufferExecutionPath.BUFFER_BINDING,
                    true,
                    false,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "synthetic CUDA source buffer binding",
                    List.of(),
                    List.of()
            );
        }
    }
}
