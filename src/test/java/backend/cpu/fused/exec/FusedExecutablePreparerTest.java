package backend.cpu.fused.exec;

import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.NoAttributes;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.numeric.FusedValueLane;
import backend.cpu.fused.plan.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedExecutionPlan;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedSignatureBuilder;
import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.ResolvedCpuComputeContract;
import config.runtime.FusedExecutionPolicy;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FusedExecutablePreparerTest {
    @Test
    void segmentContractHardFailsAsmPreparationEvenWhenBackendFallbackAllowed() {
        RuntimeException asmFailure = new IllegalStateException("forced asm failure");
        AtomicInteger asmCalls = new AtomicInteger();
        FusedExecutablePreparer preparer = new FusedExecutablePreparer(plan -> {
            asmCalls.incrementAndGet();
            throw asmFailure;
        });

        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> preparer.prepare(
                        plan(segmentContract()),
                        FusedExecutionPolicy.defaultsInference().withAllowBackendFallback(true)
                )
        );

        assertEquals(1, asmCalls.get());
        assertSame(asmFailure, error.getCause());
        assertTrue(error.getMessage().contains("CPU_MEMORY_SEGMENT fused execution is ASM-only"));
        assertTrue(error.getMessage().contains("refusing Java-array interpreter fallback"));
    }

    @Test
    void arrayContractUsesInterpreterFallbackWhenBackendFallbackAllowed() {
        FusedExecutablePreparer preparer = new FusedExecutablePreparer(plan -> {
            throw new IllegalStateException("forced asm failure");
        });

        PreparedFusedExecutable executable = preparer.prepare(
                plan(arrayContract()),
                FusedExecutionPolicy.defaultsInference().withAllowBackendFallback(true)
        );

        assertInstanceOf(InterpretedPreparedFusedExecutable.class, executable);

        Tensor a = new Tensor(new float[]{1.0f, -4.0f, 3.0f, 5.0f}, new int[]{4}, null, "array_a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2.0f, 2.0f, -8.0f, 1.0f}, new int[]{4}, null, "array_b", DataType.FLOAT32);
        Tensor out = new Tensor(new int[]{4}, null, "array_out", DataType.FLOAT32);

        executable.applyRangeScalar(List.of(a, b), out, null, 0, 4);

        assertArrayEquals(new double[]{3.0, -2.0, -5.0, 6.0}, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void interpreterRejectsMemorySegmentContracts() {
        FusedExecutionPlan plan = plan(segmentContract());

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> new InterpretedPreparedFusedExecutable(
                        plan.descriptor().getPlan(),
                        plan.descriptor().getNumericContract(),
                        plan.descriptor().getApproximationContract()
                )
        );

        assertTrue(error.getMessage().contains("CPU_JAVA_ARRAY-only"));
        assertTrue(error.getMessage().contains("CPU_MEMORY_SEGMENT fused execution is ASM-only"));
    }

    private static FusedExecutionPlan plan(FusedNumericContract numericContract) {
        FusedExpressionPlan expressionPlan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(
                        0,
                        Operation.OpType.ADD,
                        List.of(0, 1),
                        2,
                        DataType.FLOAT32,
                        NoAttributes.INSTANCE
                )),
                List.of(
                        input(0),
                        input(1)
                ),
                2
        );
        FusedOperation operation = new FusedOperation(
                "preparer-contract-test",
                numericContract,
                FusedApproximationContract.STRICT,
                false,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                FusedSignatureBuilder.buildFromPlan(
                        expressionPlan,
                        numericContract,
                        FusedApproximationContract.STRICT
                ),
                expressionPlan
        );
        return new FusedExecutionPlan(
                operation,
                new ResolvedCpuComputeContract(
                        DataType.FLOAT32,
                        CpuComputeDType.F32,
                        CpuExecutionBackend.CPU_FUSED,
                        CpuAccumulateDType.NONE
                ),
                4,
                1,
                1
        );
    }

    private static FusedExternalInputPlan input(int index) {
        return new FusedExternalInputPlan(
                index,
                DataType.FLOAT32,
                new int[]{4},
                new int[]{1},
                0,
                new int[]{1},
                FusedAccessKind.DIRECT_CONTIGUOUS
        );
    }

    private static FusedNumericContract arrayContract() {
        return new FusedNumericContract(
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedValueLane.F32,
                FusedComputeKind.F32,
                FusedValueLane.F32
        );
    }

    private static FusedNumericContract segmentContract() {
        return new FusedNumericContract(
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedValueLane.F32,
                FusedComputeKind.F32,
                FusedValueLane.F32
        );
    }
}
