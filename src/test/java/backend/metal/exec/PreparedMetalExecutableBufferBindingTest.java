package backend.metal.exec;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferLayoutClass;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.exec.AcceleratorPreparedInputSite;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsBridgeExecutionPath;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.buffer.MetalAcceleratorBufferBinder;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.metal.kernel.MetalCustomKernelBridge;
import backend.metal.kernel.MetalCustomKernelCandidate;
import backend.metal.kernel.MetalCustomKernelCapabilities;
import backend.metal.kernel.MetalCustomKernelExecutable;
import backend.metal.lowering.MetalPartitionPlan;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.DeviceTransferPolicy;
import config.runtime.RuntimeConfig;
import graph.execution.trace.HostDeviceTransferKind;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import operations.Operation;
import operations.elementwise.unary.relu;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeFloat32Storage;
import tensor.Tensor;

import java.lang.reflect.Method;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreparedMetalExecutableBufferBindingTest {
    @Test
    void contiguousViewMaterializesDenseDeviceOutputWithoutCpuRoundTrip() {
        AcceleratorBufferLayout sourceLayout = layout(
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW,
                new int[]{3, 2},
                new int[]{1, 3},
                0
        );
        AcceleratorBufferLayout targetLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );

        var decision = backend.accelerator.buffer.AcceleratorLayoutTransformPlanner.decide(
                new backend.accelerator.buffer.AcceleratorLayoutTransformRequest(
                        ComputeBackend.GPU_METAL.name(),
                        1,
                        2,
                        Operation.OpType.CONTIGUOUS,
                        sourceLayout,
                        targetLayout,
                        binding(1, MetalBufferAccess.READ, 24, sourceLayout),
                        false
                ));

        assertEquals(backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION, decision.kind());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE, decision.reasonCode());
    }

    @Test
    void contiguousViewFallsBackVisiblyWhenLayoutTransformUnavailable() {
        MetalMpsGraphBridge bridge = new FakeBridge(true);

        assertFalse(bridge.supportsLayoutMaterialization());
        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> bridge.materializeLayout(MetalMpsBridgeContext.unavailable("test"), null, null));
        assertTrue(failure.getMessage().contains("GPU layout materialization")
                || failure.getMessage().contains("layout materialization"));
    }

    @Test
    void layoutPolicyClassifiesDenseOutputAsDirectDenseBuffer() throws Exception {
        Object decision = metalLayoutPolicyDecision("output", layout(
                AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS,
                new int[]{2},
                new int[]{1},
                0
        ));

        assertTrue(policyAccepted(decision));
        assertFalse(policyRequiresDensePhysicalLogicalView(decision));
        assertEquals("DIRECT_DENSE_BUFFER", policyAction(decision));
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, policyReasonCode(decision));
        assertEquals("", policyReason(decision));
    }

    @Test
    void layoutPolicyClassifiesZeroOffsetOutputAsDensePhysicalLogicalView() throws Exception {
        Object decision = metalLayoutPolicyDecision("output", layout(
                AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW,
                new int[]{2, 2},
                new int[]{2, 1},
                0
        ));

        assertTrue(policyAccepted(decision));
        assertTrue(policyRequiresDensePhysicalLogicalView(decision));
        assertEquals("DENSE_PHYSICAL_LOGICAL_VIEW", policyAction(decision));
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, policyReasonCode(decision));
        assertTrue(policyReason(decision).contains("policyAction=DENSE_PHYSICAL_LOGICAL_VIEW"));
        assertTrue(policyReason(decision).contains("layoutClass=ZERO_OFFSET_VIEW"));
    }

    @Test
    void layoutPolicyClassifiesPermutedOutputAsDensePhysicalLogicalView() throws Exception {
        Object decision = metalLayoutPolicyDecision("output", layout(
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW,
                new int[]{2, 2},
                new int[]{1, 2},
                0
        ));

        assertTrue(policyAccepted(decision));
        assertTrue(policyRequiresDensePhysicalLogicalView(decision));
        assertEquals("DENSE_PHYSICAL_LOGICAL_VIEW", policyAction(decision));
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, policyReasonCode(decision));
        assertTrue(policyReason(decision).contains("policyAction=DENSE_PHYSICAL_LOGICAL_VIEW"));
        assertTrue(policyReason(decision).contains("layoutClass=PERMUTED_OR_STRIDED_VIEW"));
    }

    @Test
    void layoutPolicyRejectsBroadcastOutputBeforeNativeExecution() throws Exception {
        Object decision = metalLayoutPolicyDecision("output", layout(
                AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW,
                new int[]{3, 2},
                new int[]{0, 1},
                0
        ));

        assertFalse(policyAccepted(decision));
        assertFalse(policyRequiresDensePhysicalLogicalView(decision));
        assertEquals("REJECT", policyAction(decision));
        assertEquals(AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED, policyReasonCode(decision));
        assertTrue(policyReason(decision).contains("broadcast zero-stride layout is not supported by Metal buffer execution"));
        assertTrue(policyReason(decision).contains("policyAction=REJECT"));
        assertTrue(policyReason(decision).contains("layoutClass=BROADCAST_ZERO_STRIDE_VIEW"));
        assertTrue(policyReason(decision).contains("storageOffset=0"));
        assertTrue(policyReason(decision).contains("strides=[0, 1]"));
    }

    @Test
    void layoutPolicyRejectsCpuUploadForNonDenseInput() throws Exception {
        Object decision = metalLayoutPolicyDecision("cpuUploadInput", layout(
                AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW,
                new int[]{2, 2},
                new int[]{2, 1},
                0
        ));

        assertFalse(policyAccepted(decision));
        assertFalse(policyRequiresDensePhysicalLogicalView(decision));
        assertEquals("REJECT", policyAction(decision));
        assertEquals(AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED, policyReasonCode(decision));
        assertTrue(policyReason(decision).contains("cpuUploadRequires=DENSE_CONTIGUOUS"));
        assertTrue(policyReason(decision).contains("policyAction=REJECT"));
        assertTrue(policyReason(decision).contains("layoutClass=ZERO_OFFSET_VIEW"));
        assertTrue(policyReason(decision).contains("storageOffset=0"));
        assertTrue(policyReason(decision).contains("strides=[2, 1]"));
    }

    @Test
    void usesBufferBindingPathWhenBridgeAndAllBindingsAreAvailable() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        assertTrue(executable.preparedTransportPlan().contains("preferredPath=BUFFER_BINDING"));
        assertEquals(MetalExecutionRoute.MPS_GRAPH, executable.routeDecision().selectedRoute());
        assertEquals(MetalRouteReasonCode.MPS_GRAPH_SELECTED, executable.routeDecision().reasonCode());
        assertTrue(executable.routeDecision().rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE));
        assertEquals(-1L, executable.routeDecision().estimatedCopyCost());
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(fixture.outputNode().id(), MetalBufferAccess.READ_WRITE, 8);
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void mpsGraphFirstRouteExecutesThroughBufferBridgeWhenCustomKernelIsEligible() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        FakeCustomKernelBridge customBridge = new FakeCustomKernelBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge, customBridge);
        assertTrue(executable.preparedTransportPlan().contains("preferredPath=BUFFER_BINDING"));
        assertEquals(MetalExecutionRoute.MPS_GRAPH, executable.routeDecision().selectedRoute());
        assertEquals(MetalRouteReasonCode.MPS_GRAPH_SELECTED, executable.routeDecision().reasonCode());
        assertTrue(executable.routeDecision().customKernelAvailable());
        assertTrue(executable.routeDecision().detail().contains("metalRegionLowering=MPSGRAPH_DAG"));
        assertTrue(executable.routeDecision().detail().contains("metalExecutionRoute=MPS_GRAPH"));
        assertTrue(executable.routeDecision().rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_NOT_PROFITABLE));
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(fixture.outputNode().id(), MetalBufferAccess.READ_WRITE, 8);
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(0, customBridge.bufferExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(MetalCustomKernelCandidate.RELU_F32_KERNEL_ID, executable.customKernelExecutable().kernelId());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void customKernelBridgeDoesNotOverrideMpsGraphForUnsupportedMultiNodeCandidate() {
        ElementwiseChainFixture fixture = elementwiseChainFixture();
        FakeBridge bridge = new FakeBridge(true);
        FakeCustomKernelBridge customBridge = new FakeCustomKernelBridge(true);
        PreparedMetalExecutable executable = elementwiseChainExecutable(fixture, bridge, customBridge);
        assertEquals(MetalExecutionRoute.MPS_GRAPH, executable.routeDecision().selectedRoute());
        assertFalse(executable.routeDecision().customKernelAvailable());
        assertTrue(executable.routeDecision().rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY));

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, customBridge.bufferExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
    }

    @Test
    void customKernelRouteReportsMpsGraphWhenRuntimeBindingsAreNotDense() {
        Fixture fixture = nonContiguousOutputFixture();
        FakeBridge bridge = new FakeBridge(true);
        FakeCustomKernelBridge customBridge = new FakeCustomKernelBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge, customBridge);
        assertEquals(MetalExecutionRoute.MPS_GRAPH, executable.routeDecision().selectedRoute());
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_NOT_PROFITABLE));

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, customBridge.bufferExecutions);
        assertEquals(MetalExecutionRoute.MPS_GRAPH, executable.routeDecision().selectedRoute());
        assertEquals(MetalRouteReasonCode.MPS_GRAPH_SELECTED, executable.routeDecision().reasonCode());
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_NOT_PROFITABLE));
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
    }

    @Test
    void tracedBufferBindingStepPublishesMetalRouteAttributes() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_METAL, executable, backend.accelerator.exec.PartitionExecutionRole.NONE);
        PreparedExecutionStep step = new PreparedExecutionStep(fixture.outputNode(), metadata);
        PreparedExecution prepared = new PreparedExecution(
                RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                List.of(fixture.inputNode(), fixture.outputNode()),
                CompiledTensorDescriptorBuilder.build(List.of(fixture.inputNode(), fixture.outputNode())),
                testsupport.PublicationPlans.forRoot(
                        tensorForNode(fixture.outputNode()),
                        List.of(fixture.inputNode(), fixture.outputNode()),
                        fixture.outputNode().id()
                ),
                fixture.outputNode(),
                null,
                graph.execution.trace.PrepareTrace.skipped()
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        Map<String, Object> attrs = trace.steps().getFirst().metadata().attributes();

        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("MPS_GRAPH", attrs.get("metalExecutionRoute"));
        assertEquals("MPS_GRAPH_SELECTED", attrs.get("metalRouteReasonCode"));
        assertEquals(List.of("CUSTOM_KERNEL"), attrs.get("metalRouteRejectedRoutes"));
        assertEquals(List.of("CUSTOM_KERNEL_UNAVAILABLE"), attrs.get("metalRouteRejectedReasonCodes"));
        assertEquals(-1L, attrs.get("metalRouteEstimatedCopyCost"));
        assertEquals(false, attrs.get("metalRouteNativeCopyCostKnown"));
        assertEquals(false, attrs.get("metalRouteCustomKernelAvailable"));
        assertEquals("MetalBackendRouteCostModel", attrs.get("metalRouteCostModel"));
        assertEquals("metal-prepared-execution-route", attrs.get("metalRouteCostInputKind"));
        assertEquals("MPS_GRAPH_SELECTED", attrs.get("metalRouteCostReason"));
        assertTrue(((List<?>) attrs.get("metalRouteCostTopContributors")).stream()
                .anyMatch(value -> String.valueOf(value).contains("estimatedRouteCost")));
        assertEquals(false, attrs.get("metalOutputBufferWriteProbeSupported"));
        assertEquals("MPSGRAPH_RESULT_COPY", attrs.get("metalNativeCopyStrategy"));
        assertEquals(false, attrs.get("metalOutputBufferWriteProven"));
        assertEquals("COPY_REQUIRED", attrs.get("metalOutputBufferWriteStatus"));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
    }

    @Test
    void tracedCustomKernelStepPublishesRouteAndExecutionPathAttributes() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge, new FakeCustomKernelBridge(true));
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_METAL, executable, backend.accelerator.exec.PartitionExecutionRole.NONE);
        PreparedExecutionStep step = new PreparedExecutionStep(fixture.outputNode(), metadata);
        PreparedExecution prepared = new PreparedExecution(
                RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                List.of(fixture.inputNode(), fixture.outputNode()),
                CompiledTensorDescriptorBuilder.build(List.of(fixture.inputNode(), fixture.outputNode())),
                testsupport.PublicationPlans.forRoot(
                        tensorForNode(fixture.outputNode()),
                        List.of(fixture.inputNode(), fixture.outputNode()),
                        fixture.outputNode().id()
                ),
                fixture.outputNode(),
                null,
                graph.execution.trace.PrepareTrace.skipped()
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        Map<String, Object> attrs = trace.steps().getFirst().metadata().attributes();

        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("MPS_GRAPH", attrs.get("metalExecutionRoute"));
        assertEquals("MPS_GRAPH_SELECTED", attrs.get("metalRouteReasonCode"));
        assertEquals(List.of("CUSTOM_KERNEL"), attrs.get("metalRouteRejectedRoutes"));
        assertEquals(List.of("CUSTOM_KERNEL_NOT_PROFITABLE"), attrs.get("metalRouteRejectedReasonCodes"));
        assertEquals(true, attrs.get("metalRouteCustomKernelAvailable"));
        assertEquals("MetalBackendRouteCostModel", attrs.get("metalRouteCostModel"));
        assertEquals("MPS_GRAPH_SELECTED", attrs.get("metalRouteCostReason"));
        assertTrue(((List<?>) attrs.get("metalRouteCostComponents")).stream()
                .anyMatch(value -> String.valueOf(value).contains("customKernelAvailable")));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
    }


    @Test
    void adjacentDeviceOwnedInputUsesBufferBindingWithoutCpuMaterialization() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.DEVICE_OWNED,
                "adjacent device-owned input"
        );
        fixture.state().reserveDeviceBufferBinding(
                fixture.outputNode().id(),
                binding(fixture.outputNode().id(), MetalBufferAccess.READ_WRITE, 8)
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
    }

    @Test
    void elementwiseChainBufferBindingKeepsIntermediatesDeviceOwnedWithoutCpuConsumerMaterialization() {
        ElementwiseChainFixture fixture = elementwiseChainFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = elementwiseChainExecutable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.expNode().id()).residency());
        assertTrue(fixture.state().cpuMaterializationTraces().stream().noneMatch(trace ->
                (trace.nodeId() == fixture.addNode().id() || trace.nodeId() == fixture.reluNode().id())
                        && trace.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void phaseNineteenMetalMultiOpBufferPathKeepsInteriorDeviceOwned() {
        ElementwiseChainFixture fixture = elementwiseChainFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = elementwiseChainExecutable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.expNode().id()).residency());
        assertTrue(fixture.state().cpuMaterializationTraces().stream()
                .noneMatch(trace -> trace.reason() == backend.memory.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT));
        assertTrue(fixture.state().cpuMaterializationTraces().stream().noneMatch(trace ->
                (trace.nodeId() == fixture.addNode().id() || trace.nodeId() == fixture.reluNode().id())
                        && trace.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void bufferOffUsesTensorArrayWithoutAllocatorPreflight() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, true, 0)
                )
        );
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(0, bridge.bufferAllocations);
        assertEquals(AcceleratorBufferExecutionPath.TENSOR_ARRAY, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferBindingMode.OFF, executable.lastAcceleratorBufferDecision().mode());
        assertTrue(executable.lastBufferBindingDecision().contains("buffer bindings disabled"));
    }

    @Test
    void bufferRequireFailsWhenBridgeDoesNotSupportBufferBindings() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );
        assertTrue(executable.preparedTransportPlan().contains("preferredPath=UNAVAILABLE_REQUIRED"));
        assertTrue(executable.preparedTransportPlan().contains("reasonCode=NATIVE_BUFFER_ABI_UNAVAILABLE"));
        assertEquals(MetalExecutionRoute.UNAVAILABLE_REQUIRED, executable.routeDecision().selectedRoute());
        assertEquals(MetalRouteReasonCode.UNAVAILABLE_REQUIRED, executable.routeDecision().reasonCode());
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE));

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertTrue(failure.getMessage().contains("NATIVE_BUFFER_ABI_UNAVAILABLE"));
        assertTrue(failure.getMessage().contains("native Metal buffer ABI unavailable: bridge does not support buffer bindings"));
        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertTrue(executable.lastAcceleratorBufferDecision().required());
        assertEquals(
                AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode()
        );
        assertEquals(
                "native Metal buffer ABI unavailable: bridge does not support buffer bindings",
                executable.lastAcceleratorBufferDecision().reason()
        );
    }

    @Test
    void preparedInputUploadIsExecutionLocalAndDoesNotPromoteSemanticInputBinding() {
        Fixture fixture = nonContiguousInputFixture();
        FakeBridge bridge = new FakeBridge(true);
        MetalAcceleratorBufferBinder binder = new MetalAcceleratorBufferBinder(bridge, bridge.createContext());
        Tensor semanticInput = fixture.context().runtimeTensorForNodeId(fixture.inputNode().id());
        Tensor preparedInput = new Tensor(new float[]{1f, 3f, 2f, 4f}, fixture.inputNode().shape(), null, "prepared", DataType.FLOAT32);
        ResolvedAcceleratorInputs resolved = new ResolvedAcceleratorInputs(
                List.of(fixture.inputNode().id()),
                List.of(semanticInput),
                List.of(preparedInput),
                List.of(true),
                List.of(new AcceleratorPreparedInputSite(
                        fixture.inputNode().id(),
                        fixture.outputNode().id(),
                        0,
                        true
                ))
        );
        AcceleratorBufferRequest request = new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                fixture.outputNode().flatDataSize(),
                List.of(fixture.inputNode().id()),
                List.of(DataType.FLOAT32),
                List.of(AcceleratorBufferLayout.fromTensor(semanticInput)),
                List.of(fixture.outputNode().id()),
                List.of(DataType.FLOAT32),
                List.of(AcceleratorBufferLayout.fromTensor(fixture.context().runtimeTensorForNodeId(fixture.outputNode().id()))),
                false
        );

        var decision = binder.decide(request, resolved, AcceleratorBufferConfig.defaults(), fixture.context());
        var bindings = binder.resolve(request, resolved, decision, fixture.context());

        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, decision.path());
        assertTrue(decision.preparedInputUsed());
        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, decision.inputs().getFirst().layout().layoutClass());
        assertEquals(1, bindings.inputs().size());
        assertEquals(fixture.inputNode().id(), bindings.inputs().getFirst().nodeId());
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.inputNode().id()));
        assertEquals(2, bridge.bufferAllocations);
    }

    @Test
    void nativeInputUploadsDirectlyAndRecordsTransferTrace() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        MetalAcceleratorBufferBinder binder = new MetalAcceleratorBufferBinder(bridge, bridge.createContext());
        NativeFloat32Storage storage = (NativeFloat32Storage) new NativeCpuStorageFactory()
                .allocate(DataType.FLOAT32, fixture.inputNode().flatDataSize(), "native-metal-input");
        for (int i = 0; i < fixture.inputNode().flatDataSize(); i++) {
            storage.setFloat32At(i, i + 1f);
        }
        fixture.state().attachNativeStorage(fixture.inputNode().id(), storage, "native input");
        Tensor input = fixture.context().runtimeTensorForNodeId(fixture.inputNode().id());
        AcceleratorBufferRequest request = singleInputRequest(fixture, input);
        ResolvedAcceleratorInputs resolved = singleInputResolved(fixture, input);

        var decision = binder.decide(request, resolved, AcceleratorBufferConfig.defaults(), fixture.context());
        var bindings = binder.resolve(request, resolved, decision, fixture.context());

        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, decision.path());
        assertEquals(1, bindings.inputs().size());
        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                fixture.state().residencyForNodeId(fixture.inputNode().id()).residency());
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
        var transfer = fixture.state().hostDeviceTransferTraces().stream()
                .filter(entry -> entry.transferKind() == HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY)
                .findFirst()
                .orElseThrow();
        assertEquals(fixture.inputNode().id(), transfer.nodeId());
        assertTrue(transfer.fallbackReason().isBlank());
        long expectedBytes = (long) fixture.inputNode().flatDataSize() * Float.BYTES;
        assertEquals(0L, transfer.javaArrayBytes());
        assertEquals(expectedBytes, transfer.nativeBytes());
        assertTrue(transfer.directTransferSupported());
    }

    @Test
    void requireDirectAllowsDenseNativeInputDirectUpload() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        MetalAcceleratorBufferBinder binder = new MetalAcceleratorBufferBinder(bridge, bridge.createContext());
        NativeFloat32Storage storage = (NativeFloat32Storage) new NativeCpuStorageFactory()
                .allocate(DataType.FLOAT32, fixture.inputNode().flatDataSize(), "native-metal-input");
        fixture.state().attachNativeStorage(fixture.inputNode().id(), storage, "native input");
        ExecutionContext requireDirectContext = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults().withDeviceTransferPolicy(DeviceTransferPolicy.REQUIRE_DIRECT),
                ExecutionMode.FORWARD,
                Map.of(),
                fixture.state()
        );
        Tensor input = requireDirectContext.runtimeTensorForNodeId(fixture.inputNode().id());
        AcceleratorBufferRequest request = singleInputRequest(fixture, input);
        ResolvedAcceleratorInputs resolved = singleInputResolved(fixture, input);

        var decision = binder.decide(request, resolved, AcceleratorBufferConfig.defaults(), requireDirectContext);
        var bindings = binder.resolve(request, resolved, decision, requireDirectContext);

        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, decision.path());
        assertEquals(1, bindings.inputs().size());
        assertTrue(fixture.state().hostDeviceTransferTraces().stream()
                .anyMatch(entry -> entry.transferKind() == HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY
                        && entry.javaArrayBytes() == 0L));
    }

    @Test
    void metalOutputMaterializesDirectlyToNativeFloat32Storage() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);

        executable.execute(fixture.context());
        var storage = fixture.context().requireNativeReadable(
                fixture.outputNode().id(),
                backend.memory.CpuMaterializationReason.CPU_CONSUMER
        );

        assertTrue(storage instanceof NativeFloat32Storage);
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
        assertTrue(fixture.state().hostDeviceTransferTraces().stream()
                .anyMatch(entry -> entry.nodeId() == fixture.outputNode().id()
                        && entry.transferKind() == HostDeviceTransferKind.DEVICE_TO_NATIVE_SEGMENT_COPY
                        && entry.javaArrayBytes() == 0L
                        && entry.nativeBytes() == fixture.outputNode().flatDataSize() * Float.BYTES));
    }

    @Test
    void existingDeviceInputWithPermutedLayoutMaterializesDenseBeforeBufferCompute() {
        Fixture fixture = nonContiguousInputFixture();
        FakeBridge bridge = new FakeBridge(true, false, false, true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        Tensor input = fixture.context().runtimeTensorForNodeId(fixture.inputNode().id());
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(
                        fixture.inputNode().id(),
                        MetalBufferAccess.READ,
                        16,
                        AcceleratorBufferLayout.fromTensor(input)
                ),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(
                fixture.outputNode().id(),
                MetalBufferAccess.READ_WRITE,
                16,
                fixture.outputNode().shape(),
                4
        );
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(1, bridge.layoutMaterializations);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertTrue(executable.lastAcceleratorBufferDecision().inputs().getFirst().accepted());
        assertTrue(executable.lastAcceleratorBufferDecision().inputs().getFirst().reason()
                .contains("policyAction=DENSE_PHYSICAL_LOGICAL_VIEW"));
        assertTrue(executable.lastAcceleratorBufferDecision().inputs().getFirst().reason()
                .contains("layoutClass=PERMUTED_OR_STRIDED_VIEW"));
        assertEquals(
                AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS,
                bridge.lastBufferInputs.getFirst().layout().layoutClass()
        );
        assertEquals(
                AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS,
                fixture.state().deviceBufferBindingForNodeId(fixture.inputNode().id()).layout().layoutClass()
        );
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void existingDeviceInputWithPermutedLayoutFallsBackWhenGpuLayoutRepairUnavailable() {
        Fixture fixture = nonContiguousInputFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        Tensor input = fixture.context().runtimeTensorForNodeId(fixture.inputNode().id());
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(
                        fixture.inputNode().id(),
                        MetalBufferAccess.READ,
                        16,
                        AcceleratorBufferLayout.fromTensor(input)
                ),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(0, bridge.layoutMaterializations);
        assertEquals(AcceleratorBufferExecutionPath.TENSOR_ARRAY, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED, executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("must be materialized to DENSE_CONTIGUOUS"));
    }

    @Test
    void bufferBindingOutputWithPrivateStorageBecomesDeviceOwned() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(
                fixture.outputNode().id(),
                MetalBufferAccess.READ_WRITE,
                8,
                fixture.outputNode().shape(),
                2,
                "private"
        );
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void allocatesMissingOutputBindingWhenBufferBridgeIsAvailable() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void zeroOffsetViewOutputUsesBufferBindingWhenLogicalViewMaterializationIsSupported() {
        assertLogicalViewOutputUsesBufferBinding(
                zeroOffsetViewOutputFixture(),
                AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW
        );
    }

    @Test
    void permutedOutputUsesBufferBindingWhenDensePhysicalLogicalViewIsSupported() {
        assertLogicalViewOutputUsesBufferBinding(
                nonContiguousOutputFixture(),
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW
        );
    }

    @Test
    void nonZeroOffsetViewOutputUsesBufferBindingWhenDensePhysicalLogicalViewIsSupported() {
        assertLogicalViewOutputUsesBufferBinding(
                nonZeroOffsetOutputFixture(),
                AcceleratorBufferLayoutClass.NON_ZERO_OFFSET_VIEW
        );
    }

    @Test
    void existingLogicalViewDeviceBindingIsDenselyRepairedForAdjacentMetalExecutableWithoutCpuMaterialization() {
        Fixture firstFixture = nonContiguousOutputFixture();
        FakeBridge bridge = new FakeBridge(true, false, false, true);
        PreparedMetalExecutable first = executable(firstFixture, bridge);

        first.execute(firstFixture.context());

        MetalBufferBinding intermediate = (MetalBufferBinding) firstFixture.state()
                .deviceBufferBindingForNodeId(firstFixture.outputNode().id());
        assertEquals(AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW, intermediate.layout().layoutClass());
        assertEquals(StorageResidency.DEVICE_OWNED, firstFixture.state()
                .residencyForNodeId(firstFixture.outputNode().id()).residency());

        Tensor output = firstFixture.context().runtimeTensorForNodeId(firstFixture.outputNode().id()).relu();
        Fixture secondFixture = fixture(
                firstFixture.context().runtimeTensorForNodeId(firstFixture.outputNode().id()),
                output
        );
        MetalBufferBinding secondInputBinding = new MetalBufferBinding(
                secondFixture.inputNode().id(),
                intermediate.layout(),
                intermediate.handle(),
                MetalBufferAccess.READ_WRITE
        );
        secondFixture.state().attachDeviceBufferBinding(
                secondFixture.inputNode().id(),
                secondInputBinding,
                StorageResidency.DEVICE_OWNED,
                "logical-view intermediate"
        );
        PreparedMetalExecutable second = executable(secondFixture, bridge);

        second.execute(secondFixture.context());

        assertEquals(2, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(1, bridge.layoutMaterializations);
        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, bridge.lastBufferInputs.getFirst().layout().layoutClass());
        assertEquals(AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS, secondFixture.state()
                .deviceBufferBindingForNodeId(secondFixture.inputNode().id()).layout().layoutClass());
        assertTrue(firstFixture.state().cpuMaterializationTraces().isEmpty());
        assertTrue(secondFixture.state().cpuMaterializationTraces().isEmpty());
        assertEquals(StorageResidency.DEVICE_OWNED, secondFixture.state()
                .residencyForNodeId(secondFixture.outputNode().id()).residency());
    }

    @Test
    void permutedOutputReservesDensePhysicalLogicalViewBuffer() {
        Fixture fixture = nonContiguousOutputFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        assertFalse(fixture.context().runtimeTensorForNodeId(fixture.outputNode().id()).isContiguous());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW,
                executable.lastAcceleratorBufferDecision().outputs().getFirst().layout().layoutClass()
        );
        assertTrue(executable.lastAcceleratorBufferDecision().outputs().getFirst().reason()
                .contains("policyAction=DENSE_PHYSICAL_LOGICAL_VIEW"));
        assertEquals(
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW,
                bridge.lastBufferOutputs.getFirst().layout().layoutClass()
        );
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void requiredBufferModeUsesPermutedOutputBufferBinding() {
        Fixture fixture = nonContiguousOutputFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertTrue(executable.lastAcceleratorBufferDecision().required());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(
                AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW,
                executable.lastAcceleratorBufferDecision().outputs().getFirst().layout().layoutClass()
        );
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void reportsZeroOffsetViewOutputLayoutClass() {
        assertLogicalViewOutputUsesBufferBinding(
                zeroOffsetViewOutputFixture(),
                AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW
        );
    }

    @Test
    void reportsNonZeroOffsetViewOutputLayoutClass() {
        assertLogicalViewOutputUsesBufferBinding(
                nonZeroOffsetOutputFixture(),
                AcceleratorBufferLayoutClass.NON_ZERO_OFFSET_VIEW
        );
    }

    @Test
    void broadcastZeroStrideOutputFallsBackWithOutputLayoutUnsupported() {
        assertOutputLayoutFallback(
                broadcastOutputFixture(),
                AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW,
                "policyAction=REJECT",
                "layoutClass=BROADCAST_ZERO_STRIDE_VIEW"
        );
    }

    @Test
    void unsupportedOutputLayoutFallsBackWithOutputLayoutUnsupported() {
        assertOutputLayoutFallback(
                unsupportedOutputFixture(),
                AcceleratorBufferLayoutClass.UNSUPPORTED,
                "policyAction=REJECT",
                "layoutClass=UNSUPPORTED"
        );
    }

    @Test
    void requiredBufferModeRejectsBroadcastOutputBeforeTensorArrayExecution() {
        assertRequiredOutputLayoutUnavailable(
                broadcastOutputFixture(),
                AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW,
                "layoutClass=BROADCAST_ZERO_STRIDE_VIEW"
        );
    }

    @Test
    void requiredBufferModeRejectsUnsupportedOutputBeforeTensorArrayExecution() {
        assertRequiredOutputLayoutUnavailable(
                unsupportedOutputFixture(),
                AcceleratorBufferLayoutClass.UNSUPPORTED,
                "layoutClass=UNSUPPORTED"
        );
    }

    @Test
    void doesNotAllocateOutputBufferWhenInputBufferPreflightFails() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new NonMetalBinding(fixture.inputNode().id(), 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "non-metal shared input"
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(0, bridge.bufferAllocations);
        assertTrue(executable.lastBufferBindingDecision().contains("external input"));
        assertTrue(executable.lastBufferBindingDecision().contains("binding is not Metal-compatible"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void adjacentMetalExecutionsReuseIntermediateBufferWithoutCpuMaterialization() {
        TwoStageFixture fixture = twoStageFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable first = executable(fixture.inputNode(), fixture.middleNode(), bridge);
        PreparedMetalExecutable second = executable(fixture.middleNode(), fixture.outputNode(), bridge);

        first.execute(fixture.context());
        MetalBufferBinding middleBinding = (MetalBufferBinding) fixture.state()
                .deviceBufferBindingForNodeId(fixture.middleNode().id());
        assertEquals(MetalBufferAccess.READ_WRITE, middleBinding.access());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.middleNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.middleNode().id()));

        second.execute(fixture.context());

        assertEquals(2, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(middleBinding, bridge.lastBufferInputs.getFirst());
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void usesTensorArrayPathWhenBridgeDoesNotSupportBufferBindings() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        assertEquals(MetalExecutionRoute.TENSOR_ARRAY, executable.routeDecision().selectedRoute());
        assertEquals(MetalRouteReasonCode.BUFFER_ABI_UNAVAILABLE, executable.routeDecision().reasonCode());
        assertTrue(executable.routeDecision().rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE));
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        fixture.state().reserveDeviceBufferBinding(
                fixture.outputNode().id(),
                binding(fixture.outputNode().id(), MetalBufferAccess.WRITE, 8)
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY, executable.lastExecutionStats().executionPath());
        assertEquals(
                AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode()
        );
        assertEquals(
                "native Metal buffer ABI unavailable: bridge does not support buffer bindings",
                executable.lastAcceleratorBufferDecision().reason()
        );
        assertTrue(executable.lastBufferBindingDecision().contains("native Metal buffer ABI unavailable"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void bufferBindingExecutionFailureFallsBackWithoutPromotingOutput() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true, false, true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        fixture.state().reserveDeviceBufferBinding(
                fixture.outputNode().id(),
                binding(fixture.outputNode().id(), MetalBufferAccess.WRITE, 8)
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastExecutionStats().fallbackReason().contains("buffer binding execution failed"));
        assertTrue(executable.lastBufferBindingDecision().contains("buffer binding execution failed"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void tensorArrayExecutionFailureFallsBackWithTraceReason() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false, true, false);
        PreparedMetalExecutable executable = executable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastExecutionStats().fallbackReason().contains("tensor-array bridge execution failed"));
        assertTrue(executable.lastBufferBindingDecision().contains("native Metal buffer ABI unavailable"));
    }

    @Test
    void requestRejectsLayoutListsThatDoNotMatchNodeIdLists() {
        AcceleratorBufferLayout dense = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2},
                new int[]{1},
                0,
                2
        );

        assertThrows(IllegalArgumentException.class, () -> new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                2,
                List.of(1),
                List.of(DataType.FLOAT32),
                List.of(),
                List.of(2),
                List.of(DataType.FLOAT32),
                List.of(dense),
                false
        ));
    }

    @Test
    void requestRejectsDTypeListsThatDoNotMatchNodeIdLists() {
        AcceleratorBufferLayout dense = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2},
                new int[]{1},
                0,
                2
        );

        assertThrows(IllegalArgumentException.class, () -> new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                2,
                List.of(1),
                List.of(),
                List.of(dense),
                List.of(2),
                List.of(DataType.FLOAT32),
                List.of(dense),
                false
        ));
        assertThrows(IllegalArgumentException.class, () -> new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                2,
                List.of(1),
                List.of(DataType.FLOAT32),
                List.of(dense),
                List.of(2),
                List.of(),
                List.of(dense),
                false
        ));
    }

    @Test
    void cpuFallbackPublishesEveryInternalStepAsCpuCurrent() {
        TwoStageFixture fixture = twoStageFixture();
        FakeBridge bridge = new FakeBridge(false, true, false);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(
                        new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                                fixture.middleNode(),
                                fixture.metadata().get(fixture.middleNode().id())
                        ),
                        new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                                fixture.outputNode(),
                                fixture.metadata().get(fixture.outputNode().id())
                        )
                )
        );

        executable.execute(fixture.context());

        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertEquals(StorageResidency.CPU_ARRAY, fixture.state().residencyForNodeId(fixture.middleNode().id()).residency());
        assertTrue(fixture.state().residencyForNodeId(fixture.middleNode().id()).cpuCurrent());
        assertEquals(StorageResidency.CPU_ARRAY, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().residencyForNodeId(fixture.outputNode().id()).cpuCurrent());
        fixture.context().requireCpuReadable(fixture.middleNode().id(), backend.memory.CpuMaterializationReason.CPU_CONSUMER);
        fixture.context().requireCpuReadable(fixture.outputNode().id(), backend.memory.CpuMaterializationReason.CPU_CONSUMER);
    }

    private static PreparedMetalExecutable executable(Fixture fixture, MetalMpsGraphBridge bridge) {
        return executable(fixture.inputNode(), fixture.outputNode(), bridge);
    }

    private static PreparedMetalExecutable executable(
            Fixture fixture,
            MetalMpsGraphBridge bridge,
            MetalCustomKernelBridge customKernelBridge
    ) {
        return executable(fixture.inputNode(), fixture.outputNode(), bridge, List.of(), AcceleratorBackendConfig.defaults(), customKernelBridge);
    }

    private static PreparedMetalExecutable elementwiseChainExecutable(
            ElementwiseChainFixture fixture,
            MetalMpsGraphBridge bridge
    ) {
        return elementwiseChainExecutable(fixture, bridge, MetalCustomKernelBridge.unavailable());
    }

    private static PreparedMetalExecutable elementwiseChainExecutable(
            ElementwiseChainFixture fixture,
            MetalMpsGraphBridge bridge,
            MetalCustomKernelBridge customKernelBridge
    ) {
        return new PreparedMetalExecutable(
                elementwiseChainPlan(fixture),
                backend.lowering.LoweringFamily.METAL_GRAPH_REGION,
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults(),
                customKernelBridge
        );
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge
    ) {
        return executable(inputNode, outputNode, bridge, List.of());
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        return executable(inputNode, outputNode, bridge, cpuFallbackSteps, AcceleratorBackendConfig.defaults());
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig
    ) {
        return executable(inputNode, outputNode, bridge, cpuFallbackSteps, backendConfig, MetalCustomKernelBridge.unavailable());
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig,
            MetalCustomKernelBridge customKernelBridge
    ) {
        return new PreparedMetalExecutable(
                plan(inputNode, outputNode),
                backend.lowering.LoweringFamily.METAL_GRAPH_REGION,
                bridge,
                cpuFallbackSteps,
                backendConfig,
                customKernelBridge
        );
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        return fixture(input, output);
    }

    private static TwoStageFixture twoStageFixture() {
        Tensor input = new Tensor(new float[]{1f, -2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor middle = input.relu();
        Tensor output = middle.relu();
        CompiledGraph compiled = CompiledGraph.compile(output, CompileConfig.cpuOnlyBaseline());
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
        return new TwoStageFixture(inputNode, middleNode, outputNode, state, context, metadata);
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

    private static Fixture nonContiguousInputFixture() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor input = base.permute(1, 0);
        Tensor output = input.relu();
        return fixture(input, output);
    }

    private static Fixture nonContiguousOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor output = input.permute(1, 0);
        return directFixture(input, output);
    }

    private static Fixture zeroOffsetViewOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new int[]{2, 2, 2}, null, "input", DataType.FLOAT32);
        Tensor output = input.select(1, 0);
        return directFixture(input, output);
    }

    private static Fixture nonZeroOffsetOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor output = input.select(0, 1);
        return directFixture(input, output);
    }

    private static Fixture broadcastOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f}, new int[]{1, 2}, null, "input", DataType.FLOAT32);
        Tensor output = input.expand(new int[]{3, 2});
        return directFixture(input, output);
    }

    private static Fixture unsupportedOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(
                new int[]{2, 3},
                new int[]{3, -1},
                0,
                List.of(input),
                new relu(),
                "unsupported_output",
                DataType.FLOAT32
        );
        return directFixture(input, output);
    }

    private static Fixture directFixture(Tensor input, Tensor output) {
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(input, output));
        CompiledNode inputNode = nodes.get(0);
        CompiledNode outputNode = nodes.get(1);
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                metadata,
                outputNode.id(),
                testsupport.PublicationPlans.forRoot(output, nodes, outputNode.id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new Fixture(inputNode, outputNode, state, context);
    }

    private static void assertOutputLayoutFallback(
            Fixture fixture,
            AcceleratorBufferLayoutClass layoutClass,
            String expectedPolicyAction,
            String expectedLayoutClassText
    ) {
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertEquals(AcceleratorBufferExecutionPath.TENSOR_ARRAY, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(layoutClass, executable.lastAcceleratorBufferDecision().outputs().getFirst().layout().layoutClass());
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains(expectedPolicyAction));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains(expectedLayoutClassText));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("storageOffset="));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("strides="));
        assertTrue(executable.lastBufferBindingDecision().contains(expectedPolicyAction));
        assertTrue(executable.lastBufferBindingDecision().contains(expectedLayoutClassText));
        assertTrue(executable.lastBufferBindingDecision().contains("storageOffset="));
        assertTrue(executable.lastBufferBindingDecision().contains("strides=["));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    private static void assertLogicalViewOutputUsesBufferBinding(
            Fixture fixture,
            AcceleratorBufferLayoutClass expectedLayoutClass
    ) {
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(expectedLayoutClass, executable.lastAcceleratorBufferDecision().outputs().getFirst().layout().layoutClass());
        assertTrue(executable.lastAcceleratorBufferDecision().outputs().getFirst().accepted());
        assertTrue(executable.lastAcceleratorBufferDecision().outputs().getFirst().reason()
                .contains("policyAction=DENSE_PHYSICAL_LOGICAL_VIEW"));
        MetalBufferBinding outputBinding = (MetalBufferBinding) fixture.state()
                .deviceBufferBindingForNodeId(fixture.outputNode().id());
        assertEquals(expectedLayoutClass, outputBinding.layout().layoutClass());
        assertEquals(outputBinding.layout().logicalByteLength(), outputBinding.handle().byteLength());
        assertEquals(outputBinding, bridge.lastBufferOutputs.getFirst());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
    }

    private static void assertRequiredOutputLayoutUnavailable(
            Fixture fixture,
            AcceleratorBufferLayoutClass layoutClass,
            String expectedLayoutClassText
    ) {
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertTrue(failure.getMessage().contains("OUTPUT_LAYOUT_UNSUPPORTED"));
        assertTrue(failure.getMessage().contains("policyAction=REJECT"));
        assertTrue(failure.getMessage().contains(expectedLayoutClassText));
        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(0, bridge.bufferAllocations);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertTrue(executable.lastAcceleratorBufferDecision().required());
        assertEquals(AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED, executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(layoutClass, executable.lastAcceleratorBufferDecision().outputs().getFirst().layout().layoutClass());
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("policyAction=REJECT"));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains(expectedLayoutClassText));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("storageOffset="));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("strides="));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    private static Fixture fixture(Tensor input, Tensor output) {
        CompiledGraph compiled = CompiledGraph.compile(output, CompileConfig.noGraphOptimizationBaseline());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        CompiledNode outputNode = nodes.stream()
                .filter(node -> node.id() == compiled.publication().nodeIdsByPublicationTarget().get(output)
                        || (node.operation() != null && node.operation().opType() == Operation.OpType.RELU))
                .findFirst()
                .orElseThrow();
        int inputNodeId = outputNode.inputIds().getFirst();
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == inputNodeId)
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
        return new Fixture(inputNode, outputNode, state, context);
    }

    private static CpuNodeExecutionPlan cpuPlan() {
        CpuLayoutPlan layoutPlan = new CpuLayoutPlan(
                StridedLayoutDecision.NONE,
                DataType.FLOAT32,
                0,
                null,
                null,
                List.of()
        );
        return new CpuNodeExecutionPlan(layoutPlan, null, false, 1, 0, null, null, null, null, null, null);
    }

    private static AcceleratorBufferRequest singleInputRequest(Fixture fixture, Tensor input) {
        return new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                fixture.outputNode().flatDataSize(),
                List.of(fixture.inputNode().id()),
                List.of(DataType.FLOAT32),
                List.of(AcceleratorBufferLayout.fromTensor(input)),
                List.of(fixture.outputNode().id()),
                List.of(DataType.FLOAT32),
                List.of(AcceleratorBufferLayout.fromTensor(fixture.context().runtimeTensorForNodeId(fixture.outputNode().id()))),
                false
        );
    }

    private static ResolvedAcceleratorInputs singleInputResolved(Fixture fixture, Tensor input) {
        return new ResolvedAcceleratorInputs(
                List.of(fixture.inputNode().id()),
                List.of(input),
                List.of(),
                List.of(false),
                List.of()
        );
    }

    private static MetalPartitionPlan plan(CompiledNode inputNode, CompiledNode outputNode) {
        AcceleratorDagInput input = new AcceleratorDagInput(
                inputNode.id(),
                shapeList(inputNode.shape()),
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
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNode.id()));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNode.id(),
                List.of(outputNode.id()),
                List.of(new AcceleratorSubgraphOp(outputNode.id(), Operation.OpType.RELU)),
                List.of(inputNode.id()),
                List.of(outputNode.id())
        );
        return new MetalPartitionPlan(
                outputNode.id(),
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNode.id(), null, dag, outputNode.flatDataSize())
        );
    }

    private static MetalPartitionPlan elementwiseChainPlan(ElementwiseChainFixture fixture) {
        List<Integer> orderedNodeIds = List.of(fixture.addNode().id(), fixture.reluNode().id(), fixture.expNode().id());
        List<Integer> externalInputIds = List.of(fixture.inputA().id(), fixture.inputB().id());
        List<Integer> outputNodeIds = List.of(fixture.expNode().id());
        AcceleratorDagSpec dag = new AcceleratorDagSpec(
                List.of(
                        new AcceleratorDagInput(fixture.inputA().id(), shapeList(fixture.inputA().shape()), fixture.inputA().dataType()),
                        new AcceleratorDagInput(fixture.inputB().id(), shapeList(fixture.inputB().shape()), fixture.inputB().dataType())
                ),
                List.of(
                        new AcceleratorDagNode(fixture.addNode().id(), AcceleratorDagNodeType.ADD, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1),
                        new AcceleratorDagNode(fixture.reluNode().id(), AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1),
                        new AcceleratorDagNode(fixture.expNode().id(), AcceleratorDagNodeType.EXP, AcceleratorDagValueRef.nodeOutput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, fixture.expNode().flatDataSize(), 1, 1, 1)
                ),
                List.of(2),
                outputNodeIds
        );
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                fixture.addNode().id(),
                orderedNodeIds,
                List.of(
                        new AcceleratorSubgraphOp(fixture.addNode().id(), Operation.OpType.ADD),
                        new AcceleratorSubgraphOp(fixture.reluNode().id(), Operation.OpType.RELU),
                        new AcceleratorSubgraphOp(fixture.expNode().id(), Operation.OpType.EXP)
                ),
                externalInputIds,
                outputNodeIds
        );
        GpuCompoundRegionSummary summary = GpuCompoundRegionSummary.supported(
                ComputeBackend.GPU_METAL,
                GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                orderedNodeIds,
                externalInputIds,
                outputNodeIds,
                List.of("ADD", "RELU", "EXP"),
                List.of(),
                "test elementwise chain"
        );
        return new MetalPartitionPlan(
                fixture.expNode().id(),
                subgraph,
                new AcceleratorSubgraphLoweringResult(fixture.addNode().id(), null, dag, fixture.expNode().flatDataSize(), summary)
        );
    }

    private static CompiledNode operationNode(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow();
    }

    private static MetalBufferBinding binding(int nodeId, MetalBufferAccess access, long bytes) {
        return binding(nodeId, access, bytes, new int[]{2}, 2);
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            int[] shape,
            long elementCount
    ) {
        return binding(nodeId, access, bytes, shape, elementCount, "shared");
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            int[] shape,
            long elementCount,
            String storageMode
    ) {
        return binding(
                nodeId,
                access,
                bytes,
                AcceleratorBufferLayout.of(DataType.FLOAT32, shape, denseStrides(shape), 0, elementCount),
                storageMode
        );
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            AcceleratorBufferLayout layout
    ) {
        return binding(nodeId, access, bytes, layout, "shared");
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            AcceleratorBufferLayout layout,
            String storageMode
    ) {
        return new MetalBufferBinding(
                nodeId,
                layout,
                new MetalBufferHandle(MemorySegment.ofAddress(nodeId + 1L), bytes, storageMode, "test", false),
                access
        );
    }

    private static int[] denseStrides(int[] shape) {
        return tensor.TensorMetadata.computeStrides(shape);
    }

    private static AcceleratorBufferLayout layout(
            AcceleratorBufferLayoutClass layoutClass,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        long elementCount = 1;
        for (int extent : shape) {
            elementCount = Math.multiplyExact(elementCount, extent);
        }
        return new AcceleratorBufferLayout(
                DataType.FLOAT32,
                shape,
                strides,
                storageOffset,
                elementCount,
                AcceleratorBufferLayout.byteLength(DataType.FLOAT32, elementCount),
                layoutClass
        );
    }

    private static Object metalLayoutPolicyDecision(String methodName, AcceleratorBufferLayout layout) throws Exception {
        Class<?> policyClass = Class.forName("backend.metal.buffer.MetalLayoutPolicy");
        Method method = policyClass.getDeclaredMethod(methodName, AcceleratorBufferLayout.class);
        method.setAccessible(true);
        return method.invoke(null, layout);
    }

    private static boolean policyAccepted(Object decision) throws Exception {
        Method method = decision.getClass().getDeclaredMethod("accepted");
        method.setAccessible(true);
        return (boolean) method.invoke(decision);
    }

    private static boolean policyRequiresDensePhysicalLogicalView(Object decision) throws Exception {
        Method method = decision.getClass().getDeclaredMethod("requiresDensePhysicalLogicalView");
        method.setAccessible(true);
        return (boolean) method.invoke(decision);
    }

    private static String policyAction(Object decision) throws Exception {
        Method method = decision.getClass().getDeclaredMethod("action");
        method.setAccessible(true);
        return method.invoke(decision).toString();
    }

    private static AcceleratorBufferReasonCode policyReasonCode(Object decision) throws Exception {
        Method method = decision.getClass().getDeclaredMethod("reasonCode");
        method.setAccessible(true);
        return (AcceleratorBufferReasonCode) method.invoke(decision);
    }

    private static String policyReason(Object decision) throws Exception {
        Method method = decision.getClass().getDeclaredMethod("reason");
        method.setAccessible(true);
        return (String) method.invoke(decision);
    }

    private static List<Integer> shapeList(int[] shape) {
        return java.util.Arrays.stream(shape).boxed().toList();
    }

    private static Tensor tensorForNode(CompiledNode node) {
        return new Tensor(node.shape(), null, node.label(), node.dataType());
    }

    private record Fixture(
            CompiledNode inputNode,
            CompiledNode outputNode,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record TwoStageFixture(
            CompiledNode inputNode,
            CompiledNode middleNode,
            CompiledNode outputNode,
            ExecutionState state,
            ExecutionContext context,
            Map<Integer, CompiledNodeExecutionMetadata> metadata
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

    private record NonMetalBinding(int nodeId, long logicalByteLength) implements DeviceBufferBinding {
        @Override
        public String backendId() {
            return "GPU_TEST";
        }

        @Override
        public AcceleratorBufferLayout layout() {
            return AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{(int) (logicalByteLength / Float.BYTES)},
                    new int[]{1},
                    0,
                    logicalByteLength / Float.BYTES
            );
        }

        @Override
        public AcceleratorBufferAccessMode accessMode() {
            return AcceleratorBufferAccessMode.READ_WRITE;
        }

        @Override
        public String nativeHandleIdentity() {
            return "non-metal-" + nodeId;
        }

        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String describe() {
            return "non-metal nodeId=" + nodeId;
        }
    }

    private static final class FakeCustomKernelBridge implements MetalCustomKernelBridge {
        private final boolean available;
        private int bufferExecutions;
        private List<MetalBufferBinding> lastBufferInputs = List.of();
        private List<MetalBufferBinding> lastBufferOutputs = List.of();

        private FakeCustomKernelBridge(boolean available) {
            this.available = available;
        }

        @Override
        public MetalCustomKernelCapabilities capabilities() {
            return available
                    ? new MetalCustomKernelCapabilities(true, MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED, "")
                    : MetalCustomKernelCapabilities.unavailable("fake custom kernel bridge unavailable");
        }

        @Override
        public MetalCustomKernelExecutable compile(MetalPartitionPlan plan) {
            if (!available) {
                return MetalCustomKernelExecutable.unavailable("fake custom kernel bridge unavailable");
            }
            MetalCustomKernelCandidate candidate = MetalCustomKernelCandidate.evaluate(plan);
            if (!candidate.supported()) {
                return MetalCustomKernelExecutable.unavailable(candidate.reason());
            }
            return new MetalCustomKernelExecutable(
                    true,
                    candidate.kernelId(),
                    candidate.primitiveIds(),
                    MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                    ""
            );
        }

        @Override
        public MetalMpsBridgeExecutionStats executeBuffers(
                MetalMpsBridgeContext context,
                MetalCustomKernelExecutable executable,
                List<MetalBufferBinding> externalInputs,
                List<MetalBufferBinding> outputs
        ) {
            bufferExecutions++;
            lastBufferInputs = List.copyOf(externalInputs);
            lastBufferOutputs = List.copyOf(outputs);
            return new MetalMpsBridgeExecutionStats(
                    false,
                    "",
                    MetalMpsBridgeExecutionPath.CUSTOM_KERNEL,
                    externalInputs.size(),
                    outputs.size(),
                    8,
                    8,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1
            );
        }
    }

    private static final class FakeBridge implements MetalMpsGraphBridge {
        private final boolean supportsBufferBindings;
        private final boolean failTensorExecution;
        private final boolean failBufferExecution;
        private final boolean supportsLayoutMaterialization;
        private int tensorExecutions;
        private int bufferExecutions;
        private int bufferAllocations;
        private int layoutMaterializations;
        private List<MetalBufferBinding> lastBufferInputs = List.of();
        private List<MetalBufferBinding> lastBufferOutputs = List.of();

        private FakeBridge(boolean supportsBufferBindings) {
            this(supportsBufferBindings, false, false, false);
        }

        private FakeBridge(boolean supportsBufferBindings, boolean failTensorExecution, boolean failBufferExecution) {
            this(supportsBufferBindings, failTensorExecution, failBufferExecution, false);
        }

        private FakeBridge(
                boolean supportsBufferBindings,
                boolean failTensorExecution,
                boolean failBufferExecution,
                boolean supportsLayoutMaterialization
        ) {
            this.supportsBufferBindings = supportsBufferBindings;
            this.failTensorExecution = failTensorExecution;
            this.failBufferExecution = failBufferExecution;
            this.supportsLayoutMaterialization = supportsLayoutMaterialization;
        }

        @Override
        public boolean isAvailable() {
            return true;
        }

        @Override
        public String unavailableReason() {
            return "";
        }

        @Override
        public MetalMpsBridgeContext createContext() {
            return new MetalMpsBridgeContext(true, MemorySegment.ofAddress(1), "");
        }

        @Override
        public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
            return new MetalMpsBridgeExecutable(
                    true,
                    MemorySegment.ofAddress(2),
                    "",
                    false,
                    plan.externalInputNodeIds(),
                    plan.lowering().dagSpec().externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    plan.producedOutputNodeIds(),
                    plan.lowering().dagSpec().outputNodeIndices().stream()
                            .map(index -> plan.lowering().dagSpec().nodes().get(index).outputDataType())
                            .toList(),
                    plan.lowering().dagSpec().outputNodeIndices()
            );
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public boolean supportsLayoutMaterialization() {
            return supportsLayoutMaterialization;
        }

        @Override
        public void materializeLayout(
                MetalMpsBridgeContext context,
                MetalBufferBinding source,
                MetalBufferBinding destination
        ) {
            if (!supportsLayoutMaterialization) {
                throw new UnsupportedOperationException("fake layout materialization disabled");
            }
            layoutMaterializations++;
        }

        @Override
        public MetalBufferAllocator createBufferAllocator(MetalMpsBridgeContext bridgeContext) {
            if (!supportsBufferBindings) {
                return MetalBufferAllocator.unavailable("fake bridge buffer bindings disabled");
            }
            return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
                @Override
                public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                    bufferAllocations++;
                    return new MetalBufferHandle(MemorySegment.ofAddress(1000L + byteLength), byteLength, "shared", "test", true);
                }

                @Override
                public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                }

                @Override
                public void destroyBuffer(MetalBufferHandle handle) {
                }
            });
        }

        @Override
        public MetalMpsBridgeExecutionStats execute(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            tensorExecutions++;
            if (failTensorExecution) {
                throw new UnsupportedOperationException("synthetic tensor bridge failure");
            }
            return new MetalMpsBridgeExecutionStats(
                    false,
                    "",
                    MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY,
                    externalInputs.size(),
                    outputs.size(),
                    8,
                    8,
                    1,
                    1,
                    1,
                    0,
                    1,
                    4
            );
        }

        @Override
        public MetalMpsBridgeExecutionStats executeBuffers(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<MetalBufferBinding> externalInputs,
                List<MetalBufferBinding> outputs
        ) {
            bufferExecutions++;
            lastBufferInputs = List.copyOf(externalInputs);
            lastBufferOutputs = List.copyOf(outputs);
            if (failBufferExecution) {
                throw new UnsupportedOperationException("synthetic buffer bridge failure");
            }
            return new MetalMpsBridgeExecutionStats(
                    false,
                    "",
                    MetalMpsBridgeExecutionPath.BUFFER_BINDING,
                    externalInputs.size(),
                    outputs.size(),
                    8,
                    8,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1
            );
        }
    }
}
