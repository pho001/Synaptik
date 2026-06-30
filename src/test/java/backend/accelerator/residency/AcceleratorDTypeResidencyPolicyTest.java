package backend.accelerator.residency;

import backend.contract.ComputeBackend;
import runtime.device.buffer.AcceleratorBufferLayout;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorDTypeResidencyPolicyTest {
    @Test
    void representsBFLOAT16INT32AndBOOLBufferMetadata() {
        assertEquals(8L, AcceleratorBufferLayout.of(DataType.BFLOAT16, new int[]{4}, new int[]{1}, 0, 4).logicalByteLength());
        assertEquals(16L, AcceleratorBufferLayout.of(DataType.INT32, new int[]{4}, new int[]{1}, 0, 4).logicalByteLength());
        assertEquals(4L, AcceleratorBufferLayout.of(DataType.BOOL, new int[]{4}, new int[]{1}, 0, 4).logicalByteLength());

        assertTrue(AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_METAL, DataType.BFLOAT16).residentRepresentable());
        assertTrue(AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_METAL, DataType.BOOL).residentRepresentable());
        assertTrue(AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_METAL, DataType.INT32).residentRepresentable());
        assertTrue(AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_CUDA, DataType.INT32).residentRepresentable());
        assertTrue(AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_CUDA, DataType.BOOL).residentRepresentable());
    }

    @Test
    void metalAllowsBoolPredicateInputComputeOutputAndInternalValue() {
        AcceleratorDTypeResidencyDecision input = AcceleratorDTypeResidencyPolicy.forExternalInput(ComputeBackend.GPU_METAL, DataType.BOOL);
        assertTrue(input.nativeInputLegal());
        assertFalse(input.rejected());

        AcceleratorDTypeResidencyDecision compute = AcceleratorDTypeResidencyPolicy.forCompute(ComputeBackend.GPU_METAL, DataType.BOOL);
        assertTrue(compute.nativeComputeLegal());
        assertFalse(compute.rejected());
        assertTrue(compute.detail().contains("backend=GPU_METAL"));
        assertTrue(compute.detail().contains("dtype=BOOL"));

        AcceleratorDTypeResidencyDecision output = AcceleratorDTypeResidencyPolicy.forOutput(ComputeBackend.GPU_METAL, DataType.BOOL);
        assertTrue(output.nativeOutputLegal());
        assertFalse(output.rejected());

        AcceleratorDTypeResidencyDecision internal = AcceleratorDTypeResidencyPolicy.forInternalValue(ComputeBackend.GPU_METAL, DataType.BOOL);
        assertTrue(internal.residentRepresentable());
        assertFalse(internal.rejected());
    }

    @Test
    void metalAllowsBFLOAT16ComputeAndOutputButStillRejectsINT32() {
        AcceleratorDTypeResidencyDecision bf16 = AcceleratorDTypeResidencyPolicy.forCompute(ComputeBackend.GPU_METAL, DataType.BFLOAT16);
        AcceleratorDTypeResidencyDecision i32Input = AcceleratorDTypeResidencyPolicy.forExternalInput(ComputeBackend.GPU_METAL, DataType.INT32);
        AcceleratorDTypeResidencyDecision i32 = AcceleratorDTypeResidencyPolicy.forOutput(ComputeBackend.GPU_METAL, DataType.INT32);

        assertTrue(bf16.nativeComputeLegal());
        assertFalse(bf16.rejected());
        assertTrue(i32Input.nativeInputLegal());
        assertFalse(i32Input.rejected());
        assertTrue(i32Input.detail().contains("dtype=INT32"));
        assertTrue(bf16.detail().contains("backend=GPU_METAL"));
        assertTrue(bf16.detail().contains("role=compute"));
        assertTrue(bf16.detail().contains("dtype=BFLOAT16"));
        assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE, i32.reason());
        assertTrue(i32.detail().contains("dtype=INT32"));
    }

    @Test
    void cudaRejectsNonFloat32NativeBufferResidencyWithStableUnsupportedDTypeReason() {
        for (DataType dataType : new DataType[]{DataType.BFLOAT16, DataType.INT32, DataType.BOOL, DataType.FLOAT64}) {
            AcceleratorDTypeResidencyDecision decision = AcceleratorDTypeResidencyPolicy.forCompute(ComputeBackend.GPU_CUDA, dataType);
            assertTrue(decision.residentRepresentable());
            assertFalse(decision.nativeComputeLegal());
            assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE, decision.reason());
            assertTrue(decision.detail().contains("backend=GPU_CUDA"));
            assertTrue(decision.detail().contains("role=compute"));
            assertTrue(decision.detail().contains("dtype=" + dataType));
        }
    }

    @Test
    void float32AndFloat64DecisionsRemainStable() {
        AcceleratorDTypeResidencyDecision metalF32 = AcceleratorDTypeResidencyPolicy.forCompute(ComputeBackend.GPU_METAL, DataType.FLOAT32);
        assertTrue(metalF32.nativeComputeLegal());
        assertTrue(metalF32.residentRepresentable());
        assertFalse(metalF32.rejected());

        AcceleratorDTypeResidencyDecision cudaF32 = AcceleratorDTypeResidencyPolicy.forOutput(ComputeBackend.GPU_CUDA, DataType.FLOAT32);
        assertTrue(cudaF32.nativeOutputLegal());
        assertFalse(cudaF32.rejected());

        AcceleratorDTypeResidencyDecision metalF64 = AcceleratorDTypeResidencyPolicy.forCompute(ComputeBackend.GPU_METAL, DataType.FLOAT64);
        assertTrue(metalF64.residentRepresentable());
        assertFalse(metalF64.nativeComputeLegal());
        assertEquals(GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE, metalF64.reason());
    }
}
