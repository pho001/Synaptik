package backend.cuda.exec;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferAllocator;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.lowering.LoweringFamily;
import runtime.contract.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import runtime.contract.StorageResidency;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import trace.execution.RunTrace;
import trace.execution.CpuMaterializationTrace;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreparedCudaExecutableBufferPolicyTest {
    @Test
    void contiguousViewMaterializesDenseDeviceOutputWithoutCpuRoundTrip() {
        AcceleratorBufferLayout sourceLayout = new AcceleratorBufferLayout(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6,
                24,
                backend.accelerator.buffer.AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        CudaBufferBinding source = new CudaBufferBinding(
                1,
                sourceLayout,
                new CudaBufferHandle(MemorySegment.ofAddress(123), 24, false),
                CudaBufferAccess.READ
        );

        var decision = backend.accelerator.buffer.AcceleratorLayoutTransformPlanner.decide(
                new backend.accelerator.buffer.AcceleratorLayoutTransformRequest(
                        backend.contract.ComputeBackend.GPU_CUDA.name(),
                        1,
                        2,
                        Operation.OpType.CONTIGUOUS,
                        sourceLayout,
                        targetLayout,
                        source,
                        false
                ));

        assertEquals(backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE, decision.reasonCode());
    }

    @Test
    void contiguousViewFallsBackVisiblyWhenLayoutTransformUnavailable() {
        CudaGraphBridge bridge = new FakeCudaBridge(true);

        assertFalse(bridge.supportsLayoutMaterialization());
        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> bridge.materializeLayout(CudaBridgeContext.unavailable("test"), null, null));
        assertTrue(failure.getMessage().contains("GPU layout materialization")
                || failure.getMessage().contains("layout materialization"));
    }

    @Test
    void cudaBufferPathExecutesWithoutTensorArrayBridge() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastExecutionStats().inputBytes() > 0L);
        assertTrue(executable.lastExecutionStats().outputBytes() > 0L);
        assertEquals(fixture.inputNode().id(), bridge.lastBufferInputs.getFirst().nodeId());
        assertEquals(fixture.outputNode().id(), bridge.lastBufferOutputs.getFirst().nodeId());
        assertNotNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED,
                fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void elementwiseChainBufferBindingKeepsIntermediatesDeviceOwnedWithoutCpuConsumerMaterialization() {
        ElementwiseChainFixture fixture = elementwiseChainFixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = elementwiseChainExecutable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.expNode().id()).residency());
        assertTrue(fixture.state().cpuMaterializationTraces().stream().noneMatch(trace ->
                (trace.nodeId() == fixture.addNode().id() || trace.nodeId() == fixture.reluNode().id())
                        && trace.reason() == CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void phaseNineteenCudaMultiOpBufferPathKeepsInteriorDeviceOwned() {
        ElementwiseChainFixture fixture = elementwiseChainFixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = elementwiseChainExecutable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.expNode().id()).residency());
        assertTrue(fixture.state().cpuMaterializationTraces().stream()
                .noneMatch(trace -> trace.reason() == CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT));
        assertTrue(fixture.state().cpuMaterializationTraces().stream().noneMatch(trace ->
                (trace.nodeId() == fixture.addNode().id() || trace.nodeId() == fixture.reluNode().id())
                        && trace.reason() == CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void cudaBufferPathMaterializesGraphOutputThroughExecutionState() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());
        fixture.state().requireCpuReadable(fixture.outputNode().id(), CpuMaterializationReason.GRAPH_OUTPUT);

        assertArrayEquals(new float[]{0f, 3f}, fixture.context()
                .runtimeTensorForNodeId(fixture.outputNode().id())
                .toFloat32ArrayCopy(), 0f);
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        List<CpuMaterializationTrace> traces = fixture.state().cpuMaterializationTraces();
        assertEquals(1, traces.size());
        CpuMaterializationTrace trace = traces.getFirst();
        assertEquals(fixture.outputNode().id(), trace.nodeId());
        assertEquals(CpuMaterializationReason.GRAPH_OUTPUT, trace.reason());
        assertEquals("GPU_CUDA", trace.materializedFrom());
        assertEquals(StorageResidency.DEVICE_OWNED, trace.sourceResidency());
        assertTrue(trace.bytes() > 0);
        assertTrue(trace.completed());
    }

    @Test
    void cudaBufferFailureFallsBackInAutoMode() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true, true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.CPU_FALLBACK, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(executable.lastAcceleratorBufferDecision().reason()
                .startsWith("CUDA buffer binding execution failed:"));
        assertArrayEquals(new float[]{0f, 3f}, fixture.context()
                .runtimeTensorForNodeId(fixture.outputNode().id())
                .toFloat32ArrayCopy(), 0f);
        assertFalse(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void cudaBufferFailureThrowsInRequireMode() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true, true);
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("NATIVE_BUFFER_EXECUTION_FAILED"));
        assertTrue(failure.getMessage().contains("CUDA buffer binding execution failed:"));
    }

    @Test
    void requiredModeBridgeUnavailableThrowsBridgeUnavailableBeforeTensorArray() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = FakeCudaBridge.unavailableBridge("synthetic CUDA unavailable");
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("Accelerator buffer path is required for GPU_CUDA but unavailable:"));
        assertTrue(failure.getMessage().contains("BRIDGE_UNAVAILABLE"));
    }

    @Test
    void requiredModeNativeBufferAbiUnavailableThrowsBeforeTensorArray() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(false);
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("Accelerator buffer path is required for GPU_CUDA but unavailable:"));
        assertTrue(failure.getMessage().contains("NATIVE_BUFFER_ABI_UNAVAILABLE"));
    }

    @Test
    void requiredModeNativeBufferFailureThrowsNativeBufferExecutionFailed() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true, true);
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("Accelerator buffer path is required for GPU_CUDA but unavailable:"));
        assertTrue(failure.getMessage().contains("NATIVE_BUFFER_EXECUTION_FAILED"));
    }

    @Test
    void requiredModeMissingBindingAndStaleCpuThrowsInputNotCpuCurrent() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        fixture.state().markDeviceCurrent(
                fixture.inputNode().id(),
                StorageResidency.DEVICE_OWNED,
                "GPU_CUDA",
                "synthetic stale host input"
        );
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferReasonCode.INPUT_NOT_CPU_CURRENT,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("INPUT_NOT_CPU_CURRENT"));
    }

    @Test
    void tracedCudaBufferExecutionPublishesReasonAndStatsAttributes() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());
        var acceleratorMetadata = testsupport.MetadataArtifacts.acceleratorMetadata(backend.contract.ComputeBackend.GPU_CUDA, executable);
        PreparedExecution prepared = new PreparedExecution(
                RuntimeConfig.inferenceDefaults(),
                false,
                List.of(new PreparedExecutionStep(fixture.outputNode(), acceleratorMetadata)),
                List.of(new PreparedExecutionStep(fixture.outputNode(), acceleratorMetadata)),
                List.of(),
                fixture.nodes(),
                CompiledTensorDescriptorBuilder.build(fixture.nodes()),
                testsupport.PublicationPlans.forRoot(fixture.rootTensor(), fixture.nodes(), fixture.outputNode().id()),
                fixture.outputNode(),
                null,
                trace.prepare.PrepareTrace.skipped()
        );

        RunTrace trace = prepared.executeTraced(ExecutionMode.FORWARD);

        Map<String, Object> attrs = trace.steps().getFirst().metadata().attributes();
        assertEquals("GPU_CUDA", attrs.get("acceleratorBufferBackend"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("CUDA dense FLOAT32 buffer metadata accepted", attrs.get("acceleratorBufferReason"));
        assertEquals("BUFFER_BINDING", attrs.get("cudaExecutionPath"));
        assertEquals("", attrs.get("cudaFallbackReason"));
        assertTrue(((Number) attrs.get("acceleratorInputBytes")).longValue() > 0L);
    }

    @Test
    void adjacentCudaRegionsReuseDeviceBufferBinding() {
        TwoStageFixture fixture = twoStageFixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable first = executable(
                fixture.inputNode(),
                fixture.middleNode(),
                fixture.metadata(),
                bridge,
                AcceleratorBackendConfig.defaults()
        );
        PreparedCudaExecutable second = executable(
                fixture.middleNode(),
                fixture.outputNode(),
                fixture.metadata(),
                bridge,
                AcceleratorBackendConfig.defaults()
        );

        first.execute(fixture.context());
        CudaBufferBinding middleBinding = (CudaBufferBinding) fixture.state()
                .deviceBufferBindingForNodeId(fixture.middleNode().id());
        second.execute(fixture.context());

        assertEquals(2, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(middleBinding.nativeHandleIdentity(), bridge.lastBufferInputs.getFirst().nativeHandleIdentity());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, second.lastAcceleratorBufferDecision().path());
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
    }

    @Test
    void adjacentCudaRegionRejectsDifferentBackendBinding() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new NonCudaBinding(fixture.inputNode().id(), layout(fixture.inputNode())),
                StorageResidency.DEVICE_OWNED,
                "foreign device binding"
        );
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("INPUT_BINDING_UNAVAILABLE"));
    }

    @Test
    void adjacentCudaRegionRejectsMismatchedLayoutBinding() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new CudaBufferBinding(
                        fixture.inputNode().id(),
                        AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{1}, new int[]{1}, 0, 1),
                        new CudaBufferHandle(MemorySegment.ofAddress(50_000), Float.BYTES, false),
                        CudaBufferAccess.READ_WRITE
                ),
                StorageResidency.DEVICE_OWNED,
                "mismatched CUDA binding"
        );
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("INPUT_BINDING_UNAVAILABLE"));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("does not match expected shape"));
    }

    private static PreparedCudaExecutable executable(
            Fixture fixture,
            CudaGraphBridge bridge,
            AcceleratorBackendConfig backendConfig
    ) {
        return executable(fixture.inputNode(), fixture.outputNode(), fixture.metadata(), bridge, backendConfig);
    }

    private static PreparedCudaExecutable elementwiseChainExecutable(
            ElementwiseChainFixture fixture,
            CudaGraphBridge bridge,
            AcceleratorBackendConfig backendConfig
    ) {
        AcceleratorDagSpec dag = elementwiseChainDag(fixture);
        return new PreparedCudaExecutable(
                dag,
                LoweringFamily.CUDA_GRAPH_REGION,
                bridge,
                List.of(),
                backendConfig,
                GpuCompoundRegionSummary.supported(
                        backend.contract.ComputeBackend.GPU_CUDA,
                        GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(fixture.addNode().id(), fixture.reluNode().id(), fixture.expNode().id()),
                        List.of(fixture.inputA().id(), fixture.inputB().id()),
                        List.of(fixture.expNode().id()),
                        List.of("ADD", "RELU", "EXP"),
                        List.of(),
                        "test elementwise chain"
                )
        );
    }

    private static PreparedCudaExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            CudaGraphBridge bridge,
            AcceleratorBackendConfig backendConfig
    ) {
        CompiledNodeExecutionMetadata fallbackMetadata = metadata.getOrDefault(
                outputNode.id(),
                testsupport.MetadataArtifacts.metadata(backend.contract.ComputeBackend.CPU)
        );
        return new PreparedCudaExecutable(
                dag(inputNode, outputNode),
                LoweringFamily.CUDA_GRAPH_REGION,
                bridge,
                List.of(new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                        outputNode,
                        fallbackMetadata
                )),
                backendConfig
        );
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{-2f, 3f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        CompiledGraph compiled = CompiledGraph.compile(output, CompileConfig.noGraphOptimizationBaseline());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        CompiledNode outputNode = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .findFirst()
                .orElseThrow();
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == outputNode.inputIds().getFirst())
                .findFirst()
                .orElseThrow();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : compiled.prepare(RuntimeConfig.inferenceDefaults()).executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new Fixture(input, output, nodes, inputNode, outputNode, metadata, state, context);
    }

    private static TwoStageFixture twoStageFixture() {
        Tensor input = new Tensor(new float[]{-2f, 3f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor middle = input.relu();
        Tensor output = middle.relu();
        CompiledGraph compiled = CompiledGraph.compile(output, CompileConfig.noGraphOptimizationBaseline());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        List<CompiledNode> reluNodes = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .sorted(java.util.Comparator.comparingInt(CompiledNode::id))
                .toList();
        CompiledNode middleNode = reluNodes.get(0);
        CompiledNode outputNode = reluNodes.get(1);
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == middleNode.inputIds().getFirst())
                .findFirst()
                .orElseThrow();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : compiled.prepare(RuntimeConfig.inferenceDefaults()).executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new TwoStageFixture(inputNode, middleNode, outputNode, metadata, state, context);
    }

    private static ElementwiseChainFixture elementwiseChainFixture() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "chain_a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "chain_b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor exp = relu.exp();
        CompiledGraph compiled = CompiledGraph.compile(exp, CompileConfig.noGraphOptimizationBaseline());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        CompiledNode addNode = operationNode(nodes, Operation.OpType.ADD);
        CompiledNode reluNode = operationNode(nodes, Operation.OpType.RELU);
        CompiledNode expNode = operationNode(nodes, Operation.OpType.EXP);
        CompiledNode inputA = nodes.stream().filter(node -> node.id() == addNode.inputIds().get(0)).findFirst().orElseThrow();
        CompiledNode inputB = nodes.stream().filter(node -> node.id() == addNode.inputIds().get(1)).findFirst().orElseThrow();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : compiled.prepare(RuntimeConfig.inferenceDefaults()).executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new ElementwiseChainFixture(inputA, inputB, addNode, reluNode, expNode, state, context);
    }

    private static AcceleratorDagSpec dag(CompiledNode inputNode, CompiledNode outputNode) {
        AcceleratorDagInput input = new AcceleratorDagInput(
                inputNode.id(),
                Arrays.stream(inputNode.shape()).boxed().toList(),
                inputNode.dataType()
        );
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNode.id(),
                AcceleratorDagNodeType.RELU,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0,
                1,
                2,
                1,
                1,
                1
        );
        return new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNode.id()));
    }

    private static AcceleratorDagSpec elementwiseChainDag(ElementwiseChainFixture fixture) {
        return new AcceleratorDagSpec(
                List.of(
                        new AcceleratorDagInput(fixture.inputA().id(), Arrays.stream(fixture.inputA().shape()).boxed().toList(), fixture.inputA().dataType()),
                        new AcceleratorDagInput(fixture.inputB().id(), Arrays.stream(fixture.inputB().shape()).boxed().toList(), fixture.inputB().dataType())
                ),
                List.of(
                        new AcceleratorDagNode(fixture.addNode().id(), AcceleratorDagNodeType.ADD, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1),
                        new AcceleratorDagNode(fixture.reluNode().id(), AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1),
                        new AcceleratorDagNode(fixture.expNode().id(), AcceleratorDagNodeType.EXP, AcceleratorDagValueRef.nodeOutput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1)
                ),
                List.of(2),
                List.of(fixture.expNode().id())
        );
    }

    private static CompiledNode operationNode(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow();
    }

    private record Fixture(
            Tensor inputTensor,
            Tensor rootTensor,
            List<CompiledNode> nodes,
            CompiledNode inputNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record TwoStageFixture(
            CompiledNode inputNode,
            CompiledNode middleNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record ElementwiseChainFixture(
            CompiledNode inputA,
            CompiledNode inputB,
            CompiledNode addNode,
            CompiledNode reluNode,
            CompiledNode expNode,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record NonCudaBinding(int nodeId, AcceleratorBufferLayout layout) implements DeviceBufferBinding {
        @Override
        public String backendId() {
            return "GPU_TEST";
        }

        @Override
        public AcceleratorBufferAccessMode accessMode() {
            return AcceleratorBufferAccessMode.READ_WRITE;
        }

        @Override
        public String nativeHandleIdentity() {
            return "test-non-cuda-" + nodeId;
        }

        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String describe() {
            return "NonCudaBinding{nodeId=" + nodeId + '}';
        }
    }

    private static AcceleratorBufferLayout layout(CompiledNode node) {
        return AcceleratorBufferLayout.of(
                node.dataType(),
                node.shape(),
                node.strides(),
                node.storageOffset(),
                node.flatDataSize()
        );
    }

    private static final class FakeCudaBridge implements CudaGraphBridge {
        private final boolean supportsBufferBindings;
        private final boolean failBufferExecution;
        private final boolean bridgeAvailable;
        private final boolean contextAvailable;
        private final boolean executableAvailable;
        private final String unavailableReason;
        private final Map<Long, float[]> buffers = new HashMap<>();
        private long nextHandle = 10_000L;
        private int tensorExecutions;
        private int bufferExecutions;
        private List<CudaBufferBinding> lastBufferInputs = List.of();
        private List<CudaBufferBinding> lastBufferOutputs = List.of();

        private FakeCudaBridge(boolean supportsBufferBindings) {
            this(supportsBufferBindings, false, true, true, true, "");
        }

        private FakeCudaBridge(boolean supportsBufferBindings, boolean failBufferExecution) {
            this(supportsBufferBindings, failBufferExecution, true, true, true, "");
        }

        private FakeCudaBridge(
                boolean supportsBufferBindings,
                boolean failBufferExecution,
                boolean bridgeAvailable,
                boolean contextAvailable,
                boolean executableAvailable,
                String unavailableReason
        ) {
            this.supportsBufferBindings = supportsBufferBindings;
            this.failBufferExecution = failBufferExecution;
            this.bridgeAvailable = bridgeAvailable;
            this.contextAvailable = contextAvailable;
            this.executableAvailable = executableAvailable;
            this.unavailableReason = unavailableReason == null ? "" : unavailableReason;
        }

        private static FakeCudaBridge unavailableBridge(String reason) {
            return new FakeCudaBridge(true, false, false, false, false, reason);
        }

        @Override
        public boolean isAvailable() {
            return bridgeAvailable;
        }

        @Override
        public String unavailableReason() {
            return unavailableReason;
        }

        @Override
        public CudaBridgeContext createContext() {
            return contextAvailable
                    ? new CudaBridgeContext(true, MemorySegment.ofAddress(1), "")
                    : CudaBridgeContext.unavailable("synthetic CUDA context unavailable");
        }

        @Override
        public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
            if (!executableAvailable) {
                return CudaBridgeExecutable.unavailable("synthetic CUDA executable unavailable");
            }
            return new CudaBridgeExecutable(
                    true,
                    MemorySegment.ofAddress(2),
                    "",
                    false,
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::nodeId).toList(),
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    dagSpec.outputNodeIds(),
                    dagSpec.outputNodeIds().stream().map(ignored -> DataType.FLOAT32).toList()
            );
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public CudaBufferAllocator createBufferAllocator(CudaBridgeContext bridgeContext) {
            if (!supportsBufferBindings) {
                return CudaBufferAllocator.unavailable("fake CUDA buffer bindings disabled");
            }
            return CudaBufferAllocator.available(new CudaBufferAllocator.NativeAccess() {
                @Override
                public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
                    long address = nextHandle++;
                    float[] data = new float[Math.toIntExact(byteLength / Float.BYTES)];
                    if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                        MemorySegment source = initialData.reinterpret(initialDataBytes);
                        for (int i = 0; i < data.length; i++) {
                            data[i] = source.get(JAVA_FLOAT, (long) i * Float.BYTES);
                        }
                    }
                    buffers.put(address, data);
                    return new CudaBufferHandle(MemorySegment.ofAddress(address), byteLength, true);
                }

                @Override
                public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
                    float[] data = buffers.get(handle.handle().address());
                    if (data == null) {
                        throw new IllegalStateException("unknown fake CUDA handle");
                    }
                    for (int i = 0; i < data.length; i++) {
                        destination.set(JAVA_FLOAT, (long) i * Float.BYTES, data[i]);
                    }
                }

                @Override
                public void destroyBuffer(CudaBufferHandle handle) {
                    buffers.remove(handle.handle().address());
                }
            });
        }

        @Override
        public void executeBuffers(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<CudaBufferBinding> inputs,
                List<CudaBufferBinding> outputs
        ) {
            bufferExecutions++;
            lastBufferInputs = List.copyOf(inputs);
            lastBufferOutputs = List.copyOf(outputs);
            if (failBufferExecution) {
                throw new UnsupportedOperationException("synthetic CUDA buffer failure");
            }
            float[] input = buffers.get(inputs.getFirst().handle().handle().address());
            float[] output = buffers.get(outputs.getFirst().handle().handle().address());
            for (int i = 0; i < output.length; i++) {
                output[i] = Math.max(0f, input[i]);
            }
        }

        @Override
        public void execute(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            tensorExecutions++;
            throw new AssertionError("tensor-array bridge should not execute in CUDA buffer tests");
        }
    }
}
