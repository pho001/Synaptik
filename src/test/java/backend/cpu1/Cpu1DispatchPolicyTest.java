package backend.cpu1;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.fused.ir.Cpu1FusedScalarParameter;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.ExecutionMode;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import config.runtime.RuntimeConfig;
import operations.elementwise.binary.add;
import operations.elementwise.unary.exp;
import operations.elementwise.unary.sqrt;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1DispatchPolicyTest {
    @Test
    void elementwiseDecisionPreservesExplicitConfigForSupportedVectorPath() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();

        Cpu1DispatchDecision decision = policy.decideElementwise(
                new add(),
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
                new exp(),
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
                new add(),
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
                new add(),
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
                new add(),
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
                new add(),
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
    void automaticMemorySegmentF32CheapElementwiseUsesNativeThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.MEMORY_SEGMENT);

        Cpu1DispatchDecision belowNative = policy.decideElementwise(
                new add(),
                DataType.FLOAT32,
                tuned.nativeF32CheapVectorMinSize() - 1,
                config
        );
        Cpu1DispatchDecision atNative = policy.decideElementwise(
                new add(),
                DataType.FLOAT32,
                tuned.nativeF32CheapVectorMinSize(),
                config
        );

        assertEquals(Cpu1VectorizationKind.SCALAR, belowNative.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atNative.requestedVectorizationKind());
    }

    @Test
    void automaticMemorySegmentF64CheapElementwiseUsesNativeThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.MEMORY_SEGMENT);

        Cpu1DispatchDecision belowNative = policy.decideElementwise(
                new add(),
                DataType.FLOAT64,
                tuned.nativeF64CheapVectorMinSize() - 1,
                config
        );
        Cpu1DispatchDecision atNative = policy.decideElementwise(
                new add(),
                DataType.FLOAT64,
                tuned.nativeF64CheapVectorMinSize(),
                config
        );

        assertEquals(Cpu1VectorizationKind.SCALAR, belowNative.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atNative.requestedVectorizationKind());
    }

    @Test
    void automaticArrayCheapElementwiseKeepsCheapVectorThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1DispatchDecision belowCheap = policy.decideElementwise(
                new add(),
                DataType.FLOAT32,
                tuned.cheapVectorMinSize() - 1,
                config
        );
        Cpu1DispatchDecision atCheap = policy.decideElementwise(
                new add(),
                DataType.FLOAT32,
                tuned.cheapVectorMinSize(),
                config
        );

        assertEquals(Cpu1VectorizationKind.SCALAR, belowCheap.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atCheap.requestedVectorizationKind());
    }

    @Test
    void automaticMemorySegmentExpensiveElementwiseKeepsTranscendentalThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.MEMORY_SEGMENT);

        Cpu1DispatchDecision belowTranscendental = policy.decideElementwise(
                new exp(),
                DataType.FLOAT32,
                tuned.transcendentalVectorMinSize() - 1,
                config
        );
        Cpu1DispatchDecision atTranscendental = policy.decideElementwise(
                new exp(),
                DataType.FLOAT32,
                tuned.transcendentalVectorMinSize(),
                config
        );

        assertEquals(Cpu1VectorizationKind.SCALAR, belowTranscendental.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atTranscendental.requestedVectorizationKind());
    }

    @Test
    void automaticMediumElementwiseUsesNonCheapThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1DispatchDecision belowNonCheap = policy.decideElementwise(
                new sqrt(),
                DataType.FLOAT32,
                tuned.transcendentalVectorMinSize() - 1,
                config
        );
        Cpu1DispatchDecision atNonCheap = policy.decideElementwise(
                new sqrt(),
                DataType.FLOAT32,
                tuned.transcendentalVectorMinSize(),
                config
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, belowNonCheap.costClass());
        assertEquals(Cpu1VectorizationKind.SCALAR, belowNonCheap.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atNonCheap.requestedVectorizationKind());
    }

    @Test
    void fusedDecisionUsesMostExpensiveSourceOperationCost() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = cpuKernelConfigWithNativeCheapThresholds(16, 64, 128);

        Cpu1FusedDispatchDecision decision = policy.decideFusedElementwise(
                twoNodeFusedPlan(),
                List.of(new add(), new sqrt()),
                DataType.FLOAT32,
                tuned.transcendentalVectorMinSize() - 1,
                Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.JAVA_ARRAY)
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, decision.costClass());
        assertEquals(Cpu1VectorizationKind.SCALAR, decision.requestedVectorizationKind());
    }

    @Test
    void automaticFusedCheapVectorUsesFusedCheapThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = fusedThresholdConfig(256, 256, 16, 32, 512, 512, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1FusedDispatchDecision belowFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.ADD),
                List.of(new add()),
                DataType.FLOAT32,
                tuned.fusedCheapVectorMinSize() - 1,
                config
        );
        Cpu1FusedDispatchDecision atFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.ADD),
                List.of(new add()),
                DataType.FLOAT32,
                tuned.fusedCheapVectorMinSize(),
                config
        );

        assertEquals(Cpu1CostClass.CHEAP_ELEMENTWISE, atFused.costClass());
        assertEquals(Cpu1VectorizationKind.SCALAR, belowFused.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atFused.requestedVectorizationKind());
    }

    @Test
    void automaticFusedExpensiveVectorUsesFusedTranscendentalThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = fusedThresholdConfig(256, 512, 16, 32, 512, 512, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 1, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1FusedDispatchDecision belowFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.SQRT),
                List.of(new sqrt()),
                DataType.FLOAT32,
                tuned.fusedTranscendentalVectorMinSize() - 1,
                config
        );
        Cpu1FusedDispatchDecision atFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.SQRT),
                List.of(new sqrt()),
                DataType.FLOAT32,
                tuned.fusedTranscendentalVectorMinSize(),
                config
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, atFused.costClass());
        assertEquals(Cpu1VectorizationKind.SCALAR, belowFused.requestedVectorizationKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, atFused.requestedVectorizationKind());
    }

    @Test
    void automaticFusedCheapParallelUsesFusedCheapThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = fusedThresholdConfig(256, 512, 256, 512, 512, 512, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 4, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1FusedDispatchDecision belowFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.ADD),
                List.of(new add()),
                DataType.FLOAT32,
                tuned.fusedCheapParallelMinSize() - 1,
                config
        );
        Cpu1FusedDispatchDecision atFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.ADD),
                List.of(new add()),
                DataType.FLOAT32,
                tuned.fusedCheapParallelMinSize(),
                config
        );

        assertEquals(Cpu1CostClass.CHEAP_ELEMENTWISE, atFused.costClass());
        assertEquals(1, belowFused.plannedWorkers());
        assertEquals(4, atFused.plannedWorkers());
        assertEquals(4, atFused.launchConfig().workerCount());
    }

    @Test
    void automaticFusedExpensiveParallelUsesFusedTranscendentalThreshold() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();
        CpuKernelConfig tuned = fusedThresholdConfig(256, 512, 256, 512, 512, 512, 64, 128);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(tuned, 4, Cpu1StorageKind.JAVA_ARRAY);

        Cpu1FusedDispatchDecision belowFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.SQRT),
                List.of(new sqrt()),
                DataType.FLOAT32,
                tuned.fusedTranscendentalParallelMinSize() - 1,
                config
        );
        Cpu1FusedDispatchDecision atFused = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.SQRT),
                List.of(new sqrt()),
                DataType.FLOAT32,
                tuned.fusedTranscendentalParallelMinSize(),
                config
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, atFused.costClass());
        assertEquals(1, belowFused.plannedWorkers());
        assertEquals(4, atFused.plannedWorkers());
        assertEquals(4, atFused.launchConfig().workerCount());
    }

    @Test
    void fusedDecisionDoesNotUseOpTypeAsCostMetadata() {
        Cpu1DispatchPolicy policy = new Cpu1DispatchPolicy();

        Cpu1FusedDispatchDecision decision = policy.decideFusedElementwise(
                oneNodeFusedPlan(Operation.OpType.ADD),
                List.of(new SyntheticCostOperation(Operation.OpType.ADD, Operation.OpComputationalCost.EXPENSIVE)),
                DataType.FLOAT32,
                128,
                Cpu1PrepareConfig.vectorSingleThread()
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, decision.costClass());
        assertEquals(Cpu1VectorizationKind.VECTOR, decision.requestedVectorizationKind());
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

    private static CpuKernelConfig fusedThresholdConfig(
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize
    ) {
        return new CpuKernelConfig(
                4, 32, 32, 32,
                cheapVectorMinSize,
                transcendentalVectorMinSize,
                fusedCheapVectorMinSize,
                fusedTranscendentalVectorMinSize,
                1_000_000,
                cheapParallelMinSize,
                transcendentalParallelMinSize,
                fusedCheapParallelMinSize,
                fusedTranscendentalParallelMinSize,
                1_000_000,
                1_000_000,
                4, 2, 1,
                4_096, 8_192, 16_384,
                16_384,
                0,
                SumAccuracyMode.FAST,
                2_000_000,
                AttentionMatMulPolicy.AUTO
        );
    }

    private static CpuKernelConfig cpuKernelConfigWithNativeCheapThresholds(
            int cheapVectorMinSize,
            int nativeF32CheapVectorMinSize,
            int nativeF64CheapVectorMinSize
    ) {
        return new CpuKernelConfig(
                4, 32, 32, 32,
                cheapVectorMinSize, 256, cheapVectorMinSize, 256, 256, 256,
                Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE,
                Integer.MAX_VALUE, Integer.MAX_VALUE,
                1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000,
                4, 2, 1,
                4_096, 8_192, 16_384, 16_384,
                0,
                SumAccuracyMode.FAST,
                2_000_000,
                AttentionMatMulPolicy.AUTO,
                CpuMatMulMicroKernel.AUTO,
                CpuMatMulMicroKernel.AUTO,
                32, 32, 32,
                nativeF32CheapVectorMinSize,
                nativeF64CheapVectorMinSize
        );
    }

    private static Cpu1FusedExpressionPlan oneNodeFusedPlan(Operation.OpType opType) {
        return new Cpu1FusedExpressionPlan(
                List.of(new Cpu1FusedNodePlan(
                        0,
                        10,
                        opType,
                        List.of(),
                        0,
                        DataType.FLOAT32,
                        Cpu1FusedScalarParameter.NONE
                )),
                List.of(),
                0
        );
    }

    private static Cpu1FusedExpressionPlan twoNodeFusedPlan() {
        return new Cpu1FusedExpressionPlan(
                List.of(
                        new Cpu1FusedNodePlan(
                                0,
                                10,
                                Operation.OpType.ADD,
                                List.of(),
                                0,
                                DataType.FLOAT32,
                                Cpu1FusedScalarParameter.NONE
                        ),
                        new Cpu1FusedNodePlan(
                                1,
                                11,
                                Operation.OpType.SQRT,
                                List.of(0),
                                1,
                                DataType.FLOAT32,
                                Cpu1FusedScalarParameter.NONE
                        )
                ),
                List.of(),
                1
        );
    }

    private record SyntheticCostOperation(
            Operation.OpType opType,
            Operation.OpComputationalCost computationalCost
    ) implements Operation {
        @Override
        public OpArityClass arityClass() {
            return OpArityClass.ELEMENT_WISE;
        }

        @Override
        public boolean isFusable() {
            return true;
        }

        @Override
        public OpSemanticFamily semanticFamily() {
            return OpSemanticFamily.ARITHMETIC;
        }

        @Override
        public OpResultKind resultKind() {
            return OpResultKind.NUMERIC;
        }

        @Override
        public String getExpression() {
            return "synthetic";
        }
    }
}
