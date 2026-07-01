package backend.metal.exec;

import runtime.device.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.metal.bridge.MetalMpsBridgeCapabilities;
import backend.metal.bridge.MetalMpsCapabilityCode;
import backend.metal.kernel.MetalCustomKernelCandidate;
import backend.metal.kernel.MetalCustomKernelCapabilities;
import backend.metal.kernel.MetalCustomKernelExecutable;
import backend.metal.lowering.MetalPartitionPlan;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import graph.optimizer.cost.CostDirection;
import graph.optimizer.cost.CostExplanation;
import graph.optimizer.cost.CostScore;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalExecutionRouterTest {
    @Test
    void selectsScopedReluCustomKernelWhenEligible() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.FLOAT32),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                bufferTransport(),
                customCapabilities(),
                customExecutable()
        );

        assertEquals(MetalExecutionRoute.CUSTOM_KERNEL, decision.selectedRoute());
        assertEquals(MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED, decision.reasonCode());
        assertTrue(decision.customKernelAvailable());
        assertTrue(decision.detail().contains("kernelId=" + MetalCustomKernelCandidate.RELU_F32_KERNEL_ID));
        assertTrue(decision.detail().contains("metalPartitionLowering=CUSTOM_KERNEL_DAG"));
        assertTrue(decision.detail().contains("metalExecutionRoute=CUSTOM_KERNEL"));
        assertFalse(decision.rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertFalse(decision.rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_NOT_PROFITABLE));
    }

    @Test
    void rejectsCustomKernelWhenCandidateIsMultiNodeEvenIfExecutableClaimsAvailable() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                multiNodePlan(),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                bufferTransport(),
                customCapabilities(),
                customExecutable()
        );

        assertEquals(MetalExecutionRoute.MPS_GRAPH, decision.selectedRoute());
        assertEquals(MetalRouteReasonCode.MPS_GRAPH_SELECTED, decision.reasonCode());
        assertFalse(decision.customKernelAvailable());
        assertTrue(decision.rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertTrue(decision.rejectedReasonCodes().contains(MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY));
        assertTrue(decision.detail().contains("custom kernel rejected: UNSUPPORTED_OPERATION_FAMILY"));
    }

    @Test
    void rejectsCustomKernelWhenCandidateDTypeIsNotFloat32() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.BFLOAT16),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                bufferTransport(),
                customCapabilities(),
                customExecutable()
        );

        assertEquals(MetalExecutionRoute.MPS_GRAPH, decision.selectedRoute());
        assertFalse(decision.customKernelAvailable());
        assertTrue(decision.rejectedReasonCodes().contains(MetalRouteReasonCode.UNSUPPORTED_DTYPE));
    }

    @Test
    void keepsMpsGraphWhenCustomExecutableIsUnavailable() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.FLOAT32),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                bufferTransport(),
                customCapabilities(),
                MetalCustomKernelExecutable.unavailable("native custom kernel symbol unavailable")
        );

        assertEquals(MetalExecutionRoute.MPS_GRAPH, decision.selectedRoute());
        assertFalse(decision.customKernelAvailable());
        assertTrue(decision.rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
        assertTrue(decision.rejectedReasonCodes().contains(MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE));

        CostScore score = decision.toCostScore();
        CostExplanation explanation = score.explain(decision.reasonCode().name());
        assertEquals("MetalBackendRouteCostModel", score.modelName());
        assertEquals("metal-prepared-execution-route", score.inputKind());
        assertEquals("MPS_GRAPH_SELECTED", explanation.reasonCode());
        assertEquals(8.0d, component(score, "estimatedRouteCost").value());
        assertEquals(0.0d, component(score, "tensorArrayFallback").value());
        assertEquals(0.0d, component(score, "cpuFallback").value());
        assertEquals("MPS_GRAPH", component(score, "selectedRoute").reason());
    }

    @Test
    void routeCostScoreMarksTensorArrayFallback() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.FLOAT32),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                tensorArrayTransport(),
                MetalCustomKernelCapabilities.unavailable("custom route unavailable"),
                MetalCustomKernelExecutable.unavailable("custom route unavailable")
        );

        CostScore score = decision.toCostScore();

        assertEquals(MetalExecutionRoute.TENSOR_ARRAY, decision.selectedRoute());
        assertEquals(MetalRouteReasonCode.BUFFER_ABI_UNAVAILABLE, decision.reasonCode());
        assertEquals(24.0d, component(score, "estimatedRouteCost").value());
        assertEquals(1.0d, component(score, "tensorArrayFallback").value());
        assertEquals(0.0d, component(score, "cpuFallback").value());
        assertEquals(CostDirection.LOWER_IS_BETTER, component(score, "tensorArrayFallback").direction());
        assertEquals("TENSOR_ARRAY", component(score, "selectedRoute").reason());
    }

    @Test
    void routeCostScoreMarksCpuFallback() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.FLOAT32),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                staticCpuFallbackTransport(),
                MetalCustomKernelCapabilities.unavailable("custom route unavailable"),
                MetalCustomKernelExecutable.unavailable("custom route unavailable")
        );

        CostScore score = decision.toCostScore();

        assertEquals(MetalExecutionRoute.CPU_FALLBACK, decision.selectedRoute());
        assertEquals(MetalRouteReasonCode.UNSUPPORTED_LAYOUT, decision.reasonCode());
        assertEquals(16.0d, component(score, "estimatedRouteCost").value());
        assertEquals(0.0d, component(score, "tensorArrayFallback").value());
        assertEquals(1.0d, component(score, "cpuFallback").value());
        assertEquals(0.0d, component(score, "unavailableRequired").value());
        assertEquals("CPU_FALLBACK", component(score, "selectedRoute").reason());
    }

    @Test
    void routeCostScoreMarksUnavailableRequired() {
        MetalRouteDecision decision = MetalExecutionRouter.decide(
                reluPlan(DataType.FLOAT32),
                capabilities(),
                AcceleratorBackendConfig.defaults(),
                unavailableRequiredTransport(),
                MetalCustomKernelCapabilities.unavailable("custom route unavailable"),
                MetalCustomKernelExecutable.unavailable("custom route unavailable")
        );

        CostScore score = decision.toCostScore();

        assertEquals(MetalExecutionRoute.UNAVAILABLE_REQUIRED, decision.selectedRoute());
        assertEquals(MetalRouteReasonCode.UNAVAILABLE_REQUIRED, decision.reasonCode());
        assertEquals("UNAVAILABLE_REQUIRED", score.explain(decision.reasonCode().name()).reasonCode());
        assertEquals(16.0d, component(score, "estimatedRouteCost").value());
        assertEquals(1.0d, component(score, "unavailableRequired").value());
        assertEquals(0.0d, component(score, "bridgeAvailable").value());
        assertEquals("UNAVAILABLE_REQUIRED", component(score, "selectedRoute").reason());
    }

    private static MetalExecutionRouter.TransportEvidence bufferTransport() {
        return new MetalExecutionRouter.TransportEvidence(
                MetalExecutionRouter.TransportPath.BUFFER_BINDING,
                AcceleratorBufferBindingMode.AUTO,
                AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                "",
                true,
                true,
                true,
                true,
                true,
                false,
                8,
                0
        );
    }

    private static MetalExecutionRouter.TransportEvidence tensorArrayTransport() {
        return new MetalExecutionRouter.TransportEvidence(
                MetalExecutionRouter.TransportPath.TENSOR_ARRAY,
                AcceleratorBufferBindingMode.AUTO,
                AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                "native buffer ABI unavailable",
                true,
                true,
                true,
                false,
                true,
                false,
                8,
                0
        );
    }

    private static MetalExecutionRouter.TransportEvidence staticCpuFallbackTransport() {
        return new MetalExecutionRouter.TransportEvidence(
                MetalExecutionRouter.TransportPath.STATIC_CPU_FALLBACK,
                AcceleratorBufferBindingMode.AUTO,
                AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                "input layout unsupported",
                true,
                true,
                true,
                true,
                true,
                false,
                8,
                0
        );
    }

    private static MetalExecutionRouter.TransportEvidence unavailableRequiredTransport() {
        return new MetalExecutionRouter.TransportEvidence(
                MetalExecutionRouter.TransportPath.UNAVAILABLE_REQUIRED,
                AcceleratorBufferBindingMode.REQUIRE,
                AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                "bridge unavailable",
                false,
                false,
                false,
                false,
                true,
                false,
                8,
                0
        );
    }

    private static graph.optimizer.cost.CostComponent component(CostScore score, String name) {
        return score.components().stream()
                .filter(component -> component.name().equals(name))
                .findFirst()
                .orElseThrow();
    }

    private static MetalMpsBridgeCapabilities capabilities() {
        return new MetalMpsBridgeCapabilities(
                true,
                true,
                true,
                true,
                true,
                true,
                2,
                true,
                3,
                MetalMpsCapabilityCode.AVAILABLE,
                ""
        );
    }

    private static MetalCustomKernelCapabilities customCapabilities() {
        return new MetalCustomKernelCapabilities(true, MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED, "");
    }

    private static MetalCustomKernelExecutable customExecutable() {
        return new MetalCustomKernelExecutable(
                true,
                MetalCustomKernelCandidate.RELU_F32_KERNEL_ID,
                List.of("p0"),
                MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                ""
        );
    }

    private static MetalPartitionPlan reluPlan(DataType dataType) {
        AcceleratorDagNode node = new AcceleratorDagNode(
                2,
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
                1,
                dataType
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(
                List.of(new AcceleratorDagInput(1, List.of(2), dataType)),
                List.of(node),
                List.of(0),
                List.of(2)
        );
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                2,
                List.of(2),
                List.of(new AcceleratorSubgraphOp(2, Operation.OpType.RELU)),
                List.of(1),
                List.of(2)
        );
        return new MetalPartitionPlan(2, subgraph, new AcceleratorSubgraphLoweringResult(2, null, dag, 8));
    }

    private static MetalPartitionPlan multiNodePlan() {
        AcceleratorDagSpec dag = new AcceleratorDagSpec(
                List.of(new AcceleratorDagInput(1, List.of(2), DataType.FLOAT32)),
                List.of(
                        new AcceleratorDagNode(
                                2,
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
                                1,
                                DataType.FLOAT32
                        ),
                        new AcceleratorDagNode(
                                3,
                                AcceleratorDagNodeType.EXP,
                                AcceleratorDagValueRef.nodeOutput(0),
                                AcceleratorDagValueRef.none(),
                                AcceleratorDagValueRef.none(),
                                AcceleratorDagValueRef.none(),
                                0,
                                1,
                                2,
                                1,
                                1,
                                1,
                                DataType.FLOAT32
                        )
                ),
                List.of(1),
                List.of(3)
        );
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                2,
                List.of(2, 3),
                List.of(
                        new AcceleratorSubgraphOp(2, Operation.OpType.RELU),
                        new AcceleratorSubgraphOp(3, Operation.OpType.EXP)
                ),
                List.of(1),
                List.of(3)
        );
        return new MetalPartitionPlan(2, subgraph, new AcceleratorSubgraphLoweringResult(2, null, dag, 16));
    }
}
