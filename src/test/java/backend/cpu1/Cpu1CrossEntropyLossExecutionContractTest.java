package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.loss.crossentropy.Cpu1CrossEntropyKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.StepTraceContribution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.loss.LossReduction;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1CrossEntropyLossExecutionContractTest {
    @Test
    void executesF64MeanWithInt32Targets() {
        Tensor logits = new Tensor(new double[]{
                1.0d, 2.0d, 3.0d,
                0.0d, 0.0d, 0.0d
        }, new int[]{2, 3}, null, "ceF64Logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "ceI32Targets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 1, LossReduction.MEAN);
        Fixture fixture = fixture(loss);

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(
                Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F64_I32_ARRAY_DENSE_SCALAR,
                artifact.preparedCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new double[]{(lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2)
                + lossRow(new double[]{0.0d, 0.0d, 0.0d}, 0)) / 2.0d}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertLossTrace(fixture, artifact, "MEAN", 2, DataType.FLOAT64, DataType.INT32);
    }

    @Test
    void executesF32SumWithInt64Targets() {
        Tensor logits = new Tensor(new float[]{
                1.0f, 2.0f, 3.0f,
                0.0f, 1.0f, 0.0f
        }, new int[]{2, 3}, null, "ceF32Logits", DataType.FLOAT32);
        Tensor targets = new Tensor(new long[]{2L, 1L}, new int[]{2}, null, "ceI64Targets", DataType.INT64);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 1, LossReduction.SUM);
        Fixture fixture = fixture(loss);

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(
                Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I64_ARRAY_DENSE_SCALAR,
                artifact.preparedCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        double expected = lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2)
                + lossRow(new double[]{0.0d, 1.0d, 0.0d}, 1);
        assertArrayEquals(new float[]{(float) expected}, actual.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void executesF64NoneWithIgnoreIndexAndClassAxisZero() {
        Tensor logits = new Tensor(new double[]{
                1.0d, 0.0d,
                2.0d, 0.0d,
                3.0d, 0.0d
        }, new int[]{3, 2}, null, "ceAxis0Logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new int[]{2, -1}, new int[]{2}, null, "ceAxis0Targets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 0, -1, LossReduction.NONE);
        Fixture fixture = fixture(loss);

        Cpu1PreparedArtifact artifact = prepare(fixture);
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{2}, actual.getShape());
        assertArrayEquals(new double[]{lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2), 0.0d},
                actual.toDoubleArrayCopy(), 1.0e-12);
        assertLossTrace(fixture, artifact, "NONE", 2, DataType.FLOAT64, DataType.INT32);
    }

    @Test
    void executesBf16MeanWithIgnoredSamples() {
        Tensor logits = new Tensor(bf16Bits(
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "ceBf16Logits", DataType.BFLOAT16);
        Tensor targets = new Tensor(new int[]{2, -1}, new int[]{2}, null, "ceBf16Targets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 1, -1, LossReduction.MEAN);
        Fixture fixture = fixture(loss);

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(
                Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I32_ARRAY_DENSE_SCALAR,
                artifact.preparedCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        float expected = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(
                (float) lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2)
        ));
        assertEquals(expected, TensorDTypeOps.fromBFloat16Bits(actual.toBFloat16BitsArrayCopy()[0]), 0.0f);
    }

    @Test
    void executesF32NativeSegmentMeanWithInt32Targets() {
        Tensor logits = new Tensor(new float[]{
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        }, new int[]{2, 3}, null, "ceNativeRejectedLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "ceNativeRejectedTargets", DataType.INT32);
        Fixture fixture = fixture(logits.crossEntropyLossFromIndices(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I32_SEGMENT_DENSE_SCALAR,
                artifact.preparedCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        double expected = (lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2)
                + lossRow(new double[]{0.0d, 0.0d, 0.0d}, 0)) / 2.0d;
        assertArrayEquals(new float[]{(float) expected}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT.name(), traceAttrs(fixture, artifact).get("cpu1StorageKind"));
    }

    @Test
    void executesBf16NativeSegmentNoneWithInt64TargetsAndIgnoreIndex() {
        Tensor logits = new Tensor(bf16Bits(
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        ), new int[]{2, 3}, null, "ceBf16NativeLogits", DataType.BFLOAT16);
        Tensor targets = new Tensor(new long[]{2L, -1L}, new int[]{2}, null, "ceBf16NativeTargets", DataType.INT64);
        Tensor loss = logits.crossEntropyLossFromIndices(targets, 1, -1, LossReduction.NONE);
        Fixture fixture = fixture(loss);

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I64_SEGMENT_DENSE_SCALAR,
                artifact.preparedCrossEntropyLossUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        short expected = TensorDTypeOps.toBFloat16Bits((float) lossRow(new double[]{1.0d, 2.0d, 3.0d}, 2));
        assertArrayEquals(new short[]{expected, TensorDTypeOps.toBFloat16Bits(0.0f)},
                actual.toBFloat16BitsArrayCopy());
    }

    private static Cpu1PreparedArtifact prepare(Fixture fixture) {
        return prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
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
            String expectedReduction,
            int expectedGroupCount,
            DataType expectedLogitsDType,
            DataType expectedTargetDType
    ) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        Map<String, Object> attrs = traceAttrs(fixture, artifact);
        assertEquals(artifact.preparedCrossEntropyLossUnit().kernelId().name(), attrs.get("cpu1CrossEntropyLossKernelId"));
        assertEquals(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES.name(), attrs.get("cpu1LossOpType"));
        assertEquals(expectedReduction, attrs.get("cpu1LossReduction"));
        assertEquals(expectedGroupCount, attrs.get("cpu1LossGroupCount"));
        assertEquals(expectedLogitsDType.name(), attrs.get("cpu1LossLogitsDType"));
        assertEquals(expectedTargetDType.name(), attrs.get("cpu1LossTargetDType"));
    }

    private static Map<String, Object> traceAttrs(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        StepTraceContribution trace = artifact.traceContribution(fixture.node(), metadata, context(fixture, metadata));
        return trace.attributes();
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

    private static double lossRow(double[] row, int target) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : row) {
            max = Math.max(max, value);
        }
        double sumExp = 0.0d;
        for (double value : row) {
            sumExp += Math.exp(value - max);
        }
        return max + Math.log(sumExp) - row[target];
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
