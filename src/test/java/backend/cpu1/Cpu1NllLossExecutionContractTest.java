package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernelId;
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

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1NllLossExecutionContractTest {
    @Test
    void executesF64DenseTargetsMeanLoss() {
        Tensor logProbs = new Tensor(new double[]{
                -2.0d, -1.0d, -0.1d,
                -0.7d, -0.2d, -1.4d
        }, new int[]{2, 3}, null, "nllF64LogProbs", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0.0d, 0.0d, 1.0d,
                0.25d, 0.75d, 0.0d
        }, new int[]{2, 3}, null, "nllF64Targets", DataType.FLOAT64);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_F64_ARRAY_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new double[]{0.2125d}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertLossTrace(fixture, artifact, 1, 3, 1, 2, DataType.FLOAT64);
    }

    @Test
    void executesF32DenseTargetsWithClassAxisZero() {
        Tensor logProbs = new Tensor(new float[]{
                -2.0f, -1.0f,
                -0.1f, -0.7f,
                -0.2f, -1.4f
        }, new int[]{3, 2}, null, "nllF32Axis0LogProbs", DataType.FLOAT32);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.25f,
                1.0f, 0.75f,
                0.0f, 0.0f
        }, new int[]{3, 2}, null, "nllF32Axis0Targets", DataType.FLOAT32);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 0));

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new float[]{0.4375f}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertLossTrace(fixture, artifact, 0, 3, 2, 2, DataType.FLOAT32);
    }

    @Test
    void executesBf16DenseTargets() {
        Tensor logProbs = new Tensor(bf16Bits(
                -1.0f, -2.0f, -3.0f,
                -0.5f, -1.5f, -2.5f
        ), new int[]{2, 3}, null, "nllBf16LogProbs", DataType.BFLOAT16);
        Tensor targets = new Tensor(bf16Bits(
                0.0f, 1.0f, 0.0f,
                0.5f, 0.5f, 0.0f
        ), new int[]{2, 3}, null, "nllBf16Targets", DataType.BFLOAT16);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture);
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_BF16_ARRAY_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertEquals(1.5f, TensorDTypeOps.fromBFloat16Bits(actual.toBFloat16BitsArrayCopy()[0]), 0.0f);
    }

    @Test
    void executesF32DenseTargetsOnNativeSegment() {
        Tensor logProbs = new Tensor(new float[]{
                -2.0f, -1.0f, -0.1f,
                -0.7f, -0.2f, -1.4f
        }, new int[]{2, 3}, null, "nllNativeRejectedLogProbs", DataType.FLOAT32);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.0f, 1.0f,
                0.25f, 0.75f, 0.0f
        }, new int[]{2, 3}, null, "nllNativeRejectedTargets", DataType.FLOAT32);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new float[]{0.2125f}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT.name(), traceAttrs(fixture, artifact).get("cpu1StorageKind"));
    }

    @Test
    void executesF64DenseTargetsOnNativeSegment() {
        Tensor logProbs = new Tensor(new double[]{
                -2.0d, -1.0d, -0.1d,
                -0.7d, -0.2d, -1.4d
        }, new int[]{2, 3}, null, "nllF64NativeLogProbs", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0.0d, 0.0d, 1.0d,
                0.25d, 0.75d, 0.0d
        }, new int[]{2, 3}, null, "nllF64NativeTargets", DataType.FLOAT64);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_F64_SEGMENT_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new double[]{0.2125d}, actual.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void executesBf16DenseTargetsOnNativeSegment() {
        Tensor logProbs = new Tensor(bf16Bits(
                -1.0f, -2.0f, -3.0f,
                -0.5f, -1.5f, -2.5f
        ), new int[]{2, 3}, null, "nllBf16NativeLogProbs", DataType.BFLOAT16);
        Tensor targets = new Tensor(bf16Bits(
                0.0f, 1.0f, 0.0f,
                0.5f, 0.5f, 0.0f
        ), new int[]{2, 3}, null, "nllBf16NativeTargets", DataType.BFLOAT16);
        Fixture fixture = fixture(logProbs.nllLoss(targets, 1));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(Cpu1NllLossKernelId.NLL_DENSE_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedNllLossUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertEquals(1.5f, TensorDTypeOps.fromBFloat16Bits(actual.toBFloat16BitsArrayCopy()[0]), 0.0f);
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
            int expectedClassAxis,
            int expectedAxisSize,
            int expectedAxisStride,
            int expectedGroupCount,
            DataType expectedDType
    ) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        Map<String, Object> attrs = traceAttrs(fixture, artifact);
        assertEquals(artifact.preparedNllLossUnit().kernelId().name(), attrs.get("cpu1NllLossKernelId"));
        assertEquals(Operation.OpType.NLL_LOSS.name(), attrs.get("cpu1LossOpType"));
        assertEquals("MEAN", attrs.get("cpu1LossReduction"));
        assertEquals(expectedClassAxis, attrs.get("cpu1LossClassAxis"));
        assertEquals(expectedAxisSize, attrs.get("cpu1LossAxisSize"));
        assertEquals(expectedAxisStride, attrs.get("cpu1LossAxisStride"));
        assertEquals(expectedGroupCount, attrs.get("cpu1LossGroupCount"));
        assertEquals(expectedDType.name(), attrs.get("cpu1LossDType"));
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
