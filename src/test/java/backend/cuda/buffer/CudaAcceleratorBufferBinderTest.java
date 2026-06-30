package backend.cuda.buffer;

import backend.contract.ComputeBackend;
import runtime.device.buffer.AcceleratorBufferDecision;
import runtime.device.buffer.AcceleratorBufferExecutionPath;
import runtime.device.buffer.AcceleratorBufferLayout;
import runtime.device.buffer.AcceleratorBufferReasonCode;
import runtime.device.buffer.AcceleratorBufferRequest;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaAcceleratorBufferBinderTest {
    @Test
    void denseFloat32LayoutsAreAcceptedByPolicy() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(request(dense(DataType.FLOAT32), dense(DataType.FLOAT32)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, decision.path());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, decision.reasonCode());
        assertEquals(1, decision.inputs().size());
        assertEquals(1, decision.outputs().size());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, decision.inputs().getFirst().reasonCode());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE, decision.outputs().getFirst().reasonCode());
    }

    @Test
    void missingNativeBufferAbiFallsBackToTensorArrayInAutoMode() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(false))
                .decide(request(dense(DataType.FLOAT32), dense(DataType.FLOAT32)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferExecutionPath.TENSOR_ARRAY, decision.path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE, decision.reasonCode());
    }

    @Test
    void missingNativeBufferAbiIsUnavailableInRequireMode() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(false))
                .decide(
                        request(dense(DataType.FLOAT32), dense(DataType.FLOAT32)),
                        AcceleratorBufferConfig.defaults().withBindingMode(AcceleratorBufferBindingMode.REQUIRE)
                );

        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, decision.path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE, decision.reasonCode());
    }

    @Test
    void unsupportedInputDTypeIsRejected() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(request(dense(DataType.INT32), dense(DataType.FLOAT32)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED, decision.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED, decision.inputs().getFirst().reasonCode());
        assertTrue(decision.reason().contains("backend=GPU_CUDA"));
        assertTrue(decision.reason().contains("role=COMPUTE_INPUT"));
        assertTrue(decision.reason().contains("dtype=INT32"));
        assertTrue(decision.reason().contains("RESIDENCY_ONLY_NOT_COMPUTE"));
    }

    @Test
    void unsupportedOutputDTypeIsRejected() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(request(dense(DataType.FLOAT32), dense(DataType.FLOAT64)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED, decision.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.OUTPUT_DTYPE_UNSUPPORTED, decision.outputs().getFirst().reasonCode());
        assertTrue(decision.reason().contains("role=COMPUTE_OUTPUT"));
        assertTrue(decision.reason().contains("dtype=FLOAT64"));
        assertTrue(decision.reason().contains("UNSUPPORTED_DTYPE"));
    }

    @Test
    void unsupportedInputLayoutIsRejected() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(request(strided(DataType.FLOAT32), dense(DataType.FLOAT32)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE, decision.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE, decision.inputs().getFirst().reasonCode());
    }

    @Test
    void unsupportedOutputLayoutIsRejected() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(request(dense(DataType.FLOAT32), strided(DataType.FLOAT32)), AcceleratorBufferConfig.defaults());

        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE, decision.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE, decision.outputs().getFirst().reasonCode());
    }

    @Test
    void requiredModeRejectsNonDenseLayoutWhenLayoutAbiV2Unavailable() {
        AcceleratorBufferDecision decision = new CudaAcceleratorBufferBinder(new FakeBridge(true))
                .decide(
                        request(strided(DataType.FLOAT32), dense(DataType.FLOAT32)),
                        AcceleratorBufferConfig.defaults().withBindingMode(AcceleratorBufferBindingMode.REQUIRE)
                );

        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, decision.path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE, decision.reasonCode());
        assertEquals(true, decision.required());
        assertEquals(false, decision.allowed());
        org.junit.jupiter.api.Assertions.assertTrue(decision.reason().contains("layout ABI v2"));
    }

    private static AcceleratorBufferRequest request(AcceleratorBufferLayout input, AcceleratorBufferLayout output) {
        return new AcceleratorBufferRequest(
                ComputeBackend.GPU_CUDA,
                1L,
                List.of(1),
                List.of(input.dataType()),
                List.of(input),
                List.of(2),
                List.of(output.dataType()),
                List.of(output),
                false
        );
    }

    private static AcceleratorBufferLayout dense(DataType dataType) {
        return AcceleratorBufferLayout.of(dataType, new int[]{2, 2}, new int[]{2, 1}, 0, 4);
    }

    private static AcceleratorBufferLayout strided(DataType dataType) {
        return AcceleratorBufferLayout.of(dataType, new int[]{2, 2}, new int[]{1, 2}, 0, 4);
    }

    private static final class FakeBridge implements CudaGraphBridge {
        private final boolean supportsBufferBindings;

        private FakeBridge(boolean supportsBufferBindings) {
            this.supportsBufferBindings = supportsBufferBindings;
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
        public CudaBridgeContext createContext() {
            return CudaBridgeContext.unavailable("not used");
        }

        @Override
        public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
            return CudaBridgeExecutable.unavailable("not used");
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public void execute(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            throw new AssertionError("not used");
        }
    }
}
