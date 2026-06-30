package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.loss.crossentropy.Cpu1DenseCrossEntropyKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import trace.backend.StepTraceContribution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1DenseCrossEntropyLossExecutionContractTest {
    @Test
    void executesF64DenseTargetsMeanLoss() {
        Tensor logits = new Tensor(new double[]{
                1.0d, 2.0d, 3.0d,
                0.0d, 0.0d, 0.0d
        }, new int[]{2, 3}, null, "denseCeF64Logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0.0d, 0.0d, 1.0d,
                0.25d, 0.75d, 0.0d
        }, new int[]{2, 3}, null, "denseCeF64Targets", DataType.FLOAT64);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F64_ARRAY_DENSE_SCALAR,
                artifact.preparedDenseCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        double expected = (denseLossRow(
                new double[]{1.0d, 2.0d, 3.0d},
                new double[]{0.0d, 0.0d, 1.0d}
        ) + denseLossRow(
                new double[]{0.0d, 0.0d, 0.0d},
                new double[]{0.25d, 0.75d, 0.0d}
        )) / 2.0d;
        assertArrayEquals(new double[]{expected}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertLossTrace(fixture, artifact, 1, 3, 1, 2, DataType.FLOAT64);
    }

    @Test
    void executesF32DenseTargetsWithClassAxisZero() {
        Tensor logits = new Tensor(new float[]{
                1.0f, 0.0f,
                2.0f, 1.0f,
                3.0f, 0.0f
        }, new int[]{3, 2}, null, "denseCeF32Axis0Logits", DataType.FLOAT32);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.25f,
                1.0f, 0.75f,
                0.0f, 0.0f
        }, new int[]{3, 2}, null, "denseCeF32Axis0Targets", DataType.FLOAT32);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 0));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedDenseCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        double expected = (denseLossRow(
                new double[]{1.0d, 2.0d, 3.0d},
                new double[]{0.0d, 1.0d, 0.0d}
        ) + denseLossRow(
                new double[]{0.0d, 1.0d, 0.0d},
                new double[]{0.25d, 0.75d, 0.0d}
        )) / 2.0d;
        assertArrayEquals(new float[]{(float) expected}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertLossTrace(fixture, artifact, 0, 3, 2, 2, DataType.FLOAT32);
    }

    @Test
    void executesBf16DenseTargets() {
        Tensor logits = new Tensor(bf16Bits(
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "denseCeBf16Logits", DataType.BFLOAT16);
        Tensor targets = new Tensor(bf16Bits(
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "denseCeBf16Targets", DataType.BFLOAT16);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_BF16_ARRAY_DENSE_SCALAR,
                artifact.preparedDenseCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        float expectedValue = (float) ((denseLossRow(
                new double[]{1.0d, 2.0d, 3.0d},
                new double[]{0.0d, 0.0d, 1.0d}
        ) + denseLossRow(
                new double[]{0.0d, 0.0d, 0.0d},
                new double[]{1.0d, 0.0d, 0.0d}
        )) / 2.0d);
        float expected = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(expectedValue));
        assertEquals(expected, TensorDTypeOps.fromBFloat16Bits(actual.toBFloat16BitsArrayCopy()[0]), 0.0f);
    }

    @Test
    void executesParallelF64DenseTargetsMeanLoss() {
        Tensor logits = new Tensor(new double[]{
                1.0d, 2.0d, 3.0d,
                0.0d, 0.0d, 0.0d,
                -1.0d, 0.5d, 2.0d,
                2.0d, 1.0d, 0.0d
        }, new int[]{4, 3}, null, "denseCeParallelF64Logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0.0d, 0.0d, 1.0d,
                0.25d, 0.75d, 0.0d,
                0.5d, 0.0d, 0.5d,
                0.0d, 1.0d, 0.0d
        }, new int[]{4, 3}, null, "denseCeParallelF64Targets", DataType.FLOAT64);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        Tensor actual = execute(fixture, artifact);

        double expected = (
                denseLossRow(new double[]{1.0d, 2.0d, 3.0d}, new double[]{0.0d, 0.0d, 1.0d})
                        + denseLossRow(new double[]{0.0d, 0.0d, 0.0d}, new double[]{0.25d, 0.75d, 0.0d})
                        + denseLossRow(new double[]{-1.0d, 0.5d, 2.0d}, new double[]{0.5d, 0.0d, 0.5d})
                        + denseLossRow(new double[]{2.0d, 1.0d, 0.0d}, new double[]{0.0d, 1.0d, 0.0d})
        ) / 4.0d;
        assertArrayEquals(new double[]{expected}, actual.toDoubleArrayCopy(), 1.0e-12);
        StepTraceContribution trace = trace(fixture, artifact);
        assertEquals(2, trace.attributes().get("cpu1LossLaunchWorkers"));
        assertTrue((Integer) trace.attributes().get("cpu1LossScratchF64") > 0);
    }

    @Test
    void executesF32DenseTargetsOnNativeSegment() {
        Tensor logits = new Tensor(new float[]{
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        }, new int[]{2, 3}, null, "denseCeNativeRejectedLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        }, new int[]{2, 3}, null, "denseCeNativeRejectedTargets", DataType.FLOAT32);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedDenseCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        double expected = (denseLossRow(
                new double[]{1.0d, 2.0d, 3.0d},
                new double[]{0.0d, 0.0d, 1.0d}
        ) + denseLossRow(
                new double[]{0.0d, 0.0d, 0.0d},
                new double[]{1.0d, 0.0d, 0.0d}
        )) / 2.0d;
        assertArrayEquals(new float[]{(float) expected}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT.name(), trace(fixture, artifact).attributes().get("cpu1StorageKind"));
    }

    @Test
    void executesBf16DenseTargetsOnNativeSegment() {
        Tensor logits = new Tensor(bf16Bits(
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "denseCeBf16NativeLogits", DataType.BFLOAT16);
        Tensor targets = new Tensor(bf16Bits(
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "denseCeBf16NativeTargets", DataType.BFLOAT16);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedDenseCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        float expectedValue = (float) ((denseLossRow(
                new double[]{1.0d, 2.0d, 3.0d},
                new double[]{0.0d, 0.0d, 1.0d}
        ) + denseLossRow(
                new double[]{0.0d, 0.0d, 0.0d},
                new double[]{1.0d, 0.0d, 0.0d}
        )) / 2.0d);
        float expected = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(expectedValue));
        assertEquals(expected, TensorDTypeOps.fromBFloat16Bits(actual.toBFloat16BitsArrayCopy()[0]), 0.0f);
    }

    @Test
    void rejectsMixedDenseTargetDTypeForFirstVersion() {
        Tensor logits = new Tensor(new double[]{
                1.0d, 2.0d, 3.0d,
                0.0d, 0.0d, 0.0d
        }, new int[]{2, 3}, null, "denseCeMixedDTypeLogits", DataType.FLOAT64);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        }, new int[]{2, 3}, null, "denseCeMixedDTypeTargets", DataType.FLOAT32);
        Fixture fixture = fixture(logits.crossEntropyLoss(targets, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertEquals("cpu1 CROSS_ENTROPY_LOSS requires dense targets dtype to match logits, "
                + "logits=FLOAT64, targets=FLOAT32", exception.getMessage());
    }

    private static Cpu1PreparedArtifact prepare(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                config
        );
    }

    private static Tensor execute(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        context.requireCpuReadable(fixture.node().id(), CpuMaterializationReason.CPU_CONSUMER);
        return context.runtimeTensorForNodeId(fixture.node().id());
    }

    private static void assertLossTrace(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            int expectedClassAxis,
            int expectedAxisSize,
            int expectedAxisStride,
            int expectedGroupCount,
            DataType expectedDType
    ) {
        Map<String, Object> attrs = trace(fixture, artifact).attributes();
        assertEquals(artifact.preparedDenseCrossEntropyLossUnit().kernelId().name(),
                attrs.get("cpu1DenseCrossEntropyLossKernelId"));
        assertEquals(Operation.OpType.CROSS_ENTROPY_LOSS.name(), attrs.get("cpu1LossOpType"));
        assertEquals("MEAN", attrs.get("cpu1LossReduction"));
        assertEquals(expectedClassAxis, attrs.get("cpu1LossClassAxis"));
        assertEquals(expectedAxisSize, attrs.get("cpu1LossAxisSize"));
        assertEquals(expectedAxisStride, attrs.get("cpu1LossAxisStride"));
        assertEquals(expectedGroupCount, attrs.get("cpu1LossGroupCount"));
        assertEquals(expectedDType.name(), attrs.get("cpu1LossDType"));
    }

    private static StepTraceContribution trace(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        return artifact.traceContribution(fixture.node(), metadata, context(fixture, metadata));
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(Fixture fixture, CompiledNodeExecutionMetadata metadata) {
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = Map.of(fixture.node().id(), metadata);
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfigHolder.RUNTIME_CONFIG,
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata cpu1Metadata(
            CompiledNode node,
            Cpu1PreparedArtifact artifact
    ) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static double denseLossRow(double[] logits, double[] targets) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : logits) {
            max = Math.max(max, value);
        }
        double sumExp = 0.0d;
        double weightedLogits = 0.0d;
        double targetSum = 0.0d;
        for (int i = 0; i < logits.length; i++) {
            sumExp += Math.exp(logits[i] - max);
            weightedLogits += targets[i] * logits[i];
            targetSum += targets[i];
        }
        return targetSum * (max + Math.log(sumExp)) - weightedLogits;
    }

    private static short[] bf16Bits(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static final class RuntimeConfigHolder {
        private static final config.runtime.RuntimeConfig RUNTIME_CONFIG = config.runtime.RuntimeConfig.inferenceDefaults();
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
