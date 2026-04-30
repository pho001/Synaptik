package backend.cuda.exec;

import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.lowering.LoweringFamily;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertFalse;

class PreparedCudaExecutableBufferPolicyTest {
    @Test
    void requireBufferModeFailsBecauseCudaBufferBindingsAreNotImplementedYet() {
        PreparedCudaExecutable executable = new PreparedCudaExecutable(
                dag(),
                LoweringFamily.CUDA_GRAPH_REGION,
                new FakeCudaBridge(),
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(null));

        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferBindingMode.REQUIRE, executable.lastAcceleratorBufferDecision().mode());
        assertEquals(
                AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode()
        );
        assertEquals("GPU_CUDA", executable.lastAcceleratorBufferDecision().backend().name());
        assertEquals("Accelerator buffer path is required for GPU_CUDA but unavailable: "
                + "REQUIRED_BUFFER_EXECUTION_UNAVAILABLE: CUDA bridge does not support required buffer bindings",
                failure.getMessage());
    }

    @Test
    void requireBufferModeFailsEvenIfCudaBridgeAdvertisesBufferSupportWithoutImplementation() {
        PreparedCudaExecutable executable = new PreparedCudaExecutable(
                dag(),
                LoweringFamily.CUDA_GRAPH_REGION,
                new FakeCudaBridge(true),
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(null));

        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferBindingMode.REQUIRE, executable.lastAcceleratorBufferDecision().mode());
        assertEquals(
                AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode()
        );
        assertEquals("GPU_CUDA", executable.lastAcceleratorBufferDecision().backend().name());
        assertEquals("Accelerator buffer path is required for GPU_CUDA but unavailable: "
                        + "REQUIRED_BUFFER_EXECUTION_UNAVAILABLE: CUDA prepared executable does not implement buffer binding execution",
                failure.getMessage());
    }

    @Test
    void requireModeFailsBeforeTensorListExecutionEvenWhenDenseMetadataAccepted() {
        AtomicBoolean executed = new AtomicBoolean(false);
        PreparedCudaExecutable executable = new PreparedCudaExecutable(
                dag(),
                LoweringFamily.CUDA_GRAPH_REGION,
                new FakeCudaBridge(true, executed),
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(null));

        assertFalse(executed.get());
        assertEquals(AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals("Accelerator buffer path is required for GPU_CUDA but unavailable: "
                        + "REQUIRED_BUFFER_EXECUTION_UNAVAILABLE: CUDA prepared executable does not implement buffer binding execution",
                failure.getMessage());
    }

    private static AcceleratorDagSpec dag() {
        return new AcceleratorDagSpec(
                List.of(new AcceleratorDagInput(1, List.of(2), DataType.FLOAT32)),
                List.of(new AcceleratorDagNode(
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
                        1
                )),
                List.of(0),
                List.of(2)
        );
    }

    private static final class FakeCudaBridge implements CudaGraphBridge {
        private final boolean supportsBufferBindings;
        private final AtomicBoolean executed;

        private FakeCudaBridge() {
            this(false);
        }

        private FakeCudaBridge(boolean supportsBufferBindings) {
            this(supportsBufferBindings, new AtomicBoolean(false));
        }

        private FakeCudaBridge(boolean supportsBufferBindings, AtomicBoolean executed) {
            this.supportsBufferBindings = supportsBufferBindings;
            this.executed = executed;
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
            return new CudaBridgeContext(true, MemorySegment.ofAddress(1), "");
        }

        @Override
        public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
            return new CudaBridgeExecutable(
                    true,
                    MemorySegment.ofAddress(2),
                    "",
                    false,
                    List.of(1),
                    List.of(DataType.FLOAT32),
                    List.of(2),
                    List.of(DataType.FLOAT32)
            );
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
            executed.set(true);
            throw new AssertionError("REQUIRE mode must fail before tensor-list execution.");
        }
    }
}
