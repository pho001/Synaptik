package backend.cpu1;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.runtime.RuntimeConfig;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1DispatchPolicyTest {
    @Test
    void elementwiseDecisionPreservesExplicitConfigForSupportedVectorPath() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();

        Cpu1DispatchDecision decision = policy.decideElementwise(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                128,
                Cpu1PrepareConfig.vectorParallel(4)
        );

        assertEquals(Operation.OpType.ADD, decision.kernelOpType());
        assertEquals(Cpu1CostClass.CHEAP_ELEMENTWISE, decision.costClass());
        assertEquals(Cpu1VectorizationKind.VECTOR, decision.requestedVectorizationKind());
        assertEquals(4, decision.plannedWorkers());
        assertEquals(4, decision.launchConfig().workerCount());
        assertEquals(0, decision.launchConfig().chunkSize());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, decision.storageKind());
        assertEquals(Cpu1VectorizationKind.VECTOR,
                policy.kernelVectorizationKind(decision, Cpu1LayoutKind.CONTIGUOUS));
    }

    @Test
    void approximationConfigSelectsFastKernelButFallsBackToScalarKernelFamily() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        Cpu1PrepareConfig config = Cpu1PrepareConfig.vectorSingleThread().withApproximation(true, true);

        Cpu1DispatchDecision decision = policy.decideElementwise(
                Operation.OpType.EXP,
                DataType.FLOAT32,
                128,
                config
        );

        assertEquals(Operation.OpType.FAST_EXP, decision.kernelOpType());
        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, decision.costClass());
        assertFalse(policy.canUseBroadcastInnerVectorLayout(decision));
        assertEquals(Cpu1VectorizationKind.SCALAR,
                policy.kernelVectorizationKind(decision, Cpu1LayoutKind.CONTIGUOUS));
    }

    @Test
    void vectorRequestFallsBackToScalarForStridedLayouts() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        Cpu1DispatchDecision decision = policy.decideElementwise(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                128,
                Cpu1PrepareConfig.vectorSingleThread()
        );

        assertEquals(Cpu1VectorizationKind.SCALAR,
                policy.kernelVectorizationKind(decision, Cpu1LayoutKind.STRIDED_RANK2));
    }

    @Test
    void rejectsNegativeElementCount() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();

        assertThrows(IllegalArgumentException.class, () -> policy.decideElementwise(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                -1,
                Cpu1PrepareConfig.scalarSingleThread()
        ));
    }

    @Test
    void automaticDispatchKeepsSmallCheapElementwiseScalarSingleThread() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);

        Cpu1DispatchDecision decision = policy.decideElementwise(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                tuned.cheapVectorMinSize() - 1,
                Cpu1PrepareConfig.automatic(tuned, 4)
        );

        assertEquals(Cpu1VectorizationKind.SCALAR, decision.requestedVectorizationKind());
        assertEquals(1, decision.plannedWorkers());
        assertEquals(1, decision.launchConfig().workerCount());
        assertEquals(Cpu1VectorizationKind.SCALAR,
                policy.kernelVectorizationKind(decision, Cpu1LayoutKind.CONTIGUOUS));
    }

    @Test
    void automaticDispatchUsesVectorParallelForLargeCheapElementwise() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);

        Cpu1DispatchDecision decision = policy.decideElementwise(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                tuned.cheapParallelMinSize(),
                Cpu1PrepareConfig.automatic(tuned, 4)
        );

        assertEquals(Cpu1VectorizationKind.VECTOR, decision.requestedVectorizationKind());
        assertEquals(4, decision.plannedWorkers());
        assertEquals(4, decision.launchConfig().workerCount());
        assertEquals(decision.vectorChunkSize(), decision.launchConfig().chunkSize());
        assertEquals(tuned.minScalarChunkSize(), decision.scalarChunkSize());
        assertEquals(tuned.minVectorChunkSize(), decision.vectorChunkSize());
        assertEquals(Cpu1VectorizationKind.VECTOR,
                policy.kernelVectorizationKind(decision, Cpu1LayoutKind.CONTIGUOUS));
    }

    @Test
    void automaticFactoryUsesRuntimeProfileCpuKernelConfig() {
        CpuKernelConfig expected = RuntimeConfig.inferenceDefaults(DataType.FLOAT32).cpuKernelConfig();

        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(DataType.FLOAT32, ExecutionMode.FORWARD, 4);

        assertEquals(expected.cheapVectorMinSize(), config.cpuKernelConfig().cheapVectorMinSize());
        assertEquals(expected.transcendentalVectorMinSize(), config.cpuKernelConfig().transcendentalVectorMinSize());
        assertEquals(expected.cheapParallelMinSize(), config.cpuKernelConfig().cheapParallelMinSize());
        assertEquals(expected.transcendentalParallelMinSize(), config.cpuKernelConfig().transcendentalParallelMinSize());
        assertEquals(expected.minScalarChunkSize(), config.cpuKernelConfig().minScalarChunkSize());
        assertEquals(expected.minVectorChunkSize(), config.cpuKernelConfig().minVectorChunkSize());
    }
}
