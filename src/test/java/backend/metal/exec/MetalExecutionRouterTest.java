package backend.metal.exec;

import backend.accelerator.buffer.AcceleratorBufferReasonCode;
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
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalExecutionRouterTest {
    @Test
    void selectsCustomKernelOnlyForScopedReluBufferCandidate() {
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
        assertTrue(decision.rejectedRoutes().contains(MetalExecutionRoute.MPS_GRAPH));
        assertFalse(decision.rejectedRoutes().contains(MetalExecutionRoute.CUSTOM_KERNEL));
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
