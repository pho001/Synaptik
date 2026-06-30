package backend.cuda.exec;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;

import backend.contract.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.trace.ExecutionStepTrace;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(List.of(source, reshape, permute), BackendIntentPlan.empty());
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.of(contiguous, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(contiguous, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        CompiledNode baseNode = nodeFor(nodes, base);
        CompiledNode sourceNode = nodeFor(nodes, nonDenseSource);
        CompiledNode contiguousNode = nodeFor(nodes, contiguous);
        CompiledNodeExecutionMetadata contiguousMetadata = compiled.prepare(RuntimeConfig.inferenceDefaults())
                .executionSteps()
                .stream()
                .filter(step -> step.compiledNode().id() == contiguousNode.id())
                .map(PreparedExecutionStep::metadata)
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
        assertTrue(attrs.containsKey("acceleratorBufferReasonCode"));
        assertTrue(attrs.containsKey("acceleratorBufferExecutionPath"));
        assertTrue(attrs.containsKey("storageResidency"));
    }

    @Test
    void layoutViewThenLogSoftmaxStaysDeviceOwnedUntilOutputBoundary() {
        PlannedTensor expected = linearReshapePermuteLogSoftmaxGraph("cpu");
        CompiledGraph.compile(expected.root(), CompileConfig.inference(), expected.backendIntentPlan())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        PlannedTensor actual = linearReshapePermuteLogSoftmaxGraph("cuda");
        CompiledGraph compiled = CompiledGraph.compile(actual.root(), CompileConfig.inference(), actual.backendIntentPlan());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        var trace = execution.executeTraced(ExecutionMode.FORWARD);
        int logSoftmaxNodeId = nodeId(compiled.program().compiledNodes(), Operation.OpType.LOG_SOFTMAX);

        assertArrayEquals(expected.root().toDoubleArrayCopy(), actual.root().toDoubleArrayCopy(), 1e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.CPU_CONSUMER));
        assertTrue(execution.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && decision.selectedBackend() == ComputeBackend.GPU_CUDA
                        && decision.nodeIds().contains(logSoftmaxNodeId)));
        assertTrue(compiled.program().compiledNodes().stream()
                .anyMatch(node -> node.operation() != null && node.operation().opType() == Operation.OpType.LOG_SOFTMAX));
        assertTrue(List.of("GPU_LAYOUT_VIEW_BINDING_AVAILABLE", "GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE")
                .contains(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE.name()));
    }

    private static PreparedExecution prepared(
            List<CompiledNode> nodes,
            Tensor rootTensor,
            CompiledNode forwardOutputNode,
            List<PreparedExecutionStep> steps
    ) {
        return new PreparedExecution(
                RuntimeConfig.inferenceDefaults(),
                false,
                steps,
                steps,
                List.of(),
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                testsupport.PublicationPlans.forRoot(rootTensor, nodes, forwardOutputNode.id()),
                forwardOutputNode,
                null,
                graph.execution.trace.PrepareTrace.skipped()
        );
    }

    private static PreparedExecutionStep cudaSourceStep(CompiledNode node) {
        return new PreparedExecutionStep(
                node,
                testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_CUDA, new SyntheticCudaSourceExecutable(node.id()))
        );
    }

    private static PreparedExecutionStep cudaLayoutStep(CompiledNode node) {
        return new PreparedExecutionStep(node, metadata(ComputeBackend.GPU_CUDA));
    }

    private static PreparedExecutionStep cpuLayoutStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        return new PreparedExecutionStep(node, metadata);
    }

    private static CompiledNodeExecutionMetadata metadata(ComputeBackend backend) {
        return testsupport.MetadataArtifacts.acceleratorMetadata(backend, null);
    }

    private static CompiledNode nodeFor(List<CompiledNode> nodes, Tensor tensor) {
        return nodes.stream()
                .filter(node -> node.label().equals(tensor.getLabel())
                        && ((node.operation() == null && tensor.getOperation() == null)
                        || (node.operation() != null
                        && tensor.getOperation() != null
                        && node.operation().opType() == tensor.getOperation().opType())))
                .findFirst()
                .orElseThrow();
    }

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static PlannedTensor linearReshapePermuteLogSoftmaxGraph(String labelPrefix) {
        Tensor input = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f,
                0.4f, 0.5f, 0.6f
        }, new int[]{2, 3}, null, labelPrefix + "LogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.1f,
                0.5f, 0.6f, -0.2f, 0.1f, 0.7f, -0.4f,
                -0.3f, 0.7f, 0.8f, -0.4f, 0.2f, 0.5f
        }, new int[]{3, 6}, null, labelPrefix + "LogSoftmaxWeight", DataType.FLOAT32);
        Tensor linear = input.matmul(weight);
        Tensor reshape = linear.reshape(3, 4);
        Tensor permute = reshape.permute(1, 0);
        Tensor contiguous = permute.contiguous();
        Tensor out = specialLogSoftmax(contiguous, 1);

        BackendIntentPlan backendIntentPlan = "cuda".equals(labelPrefix)
                ? BackendIntentPlan.of(ComputeBackend.GPU_CUDA, linear, reshape, permute, contiguous, out)
                : BackendIntentPlan.empty();
        return new PlannedTensor(out, backendIntentPlan);
    }

    private record PlannedTensor(Tensor root, BackendIntentPlan backendIntentPlan) {
    }

    private static Tensor specialLogSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.logSoftmax(dimension),
                "legacyLogSoftmax",
                input.getDataType()
        );
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
