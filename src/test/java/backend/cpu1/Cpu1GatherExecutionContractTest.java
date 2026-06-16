package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.exec.Cpu1IndexExecutableUnit;
import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.contrib.StepExecutionTracer;
import operations.index.gather;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1GatherExecutionContractTest {
    @Test
    void preparedF32GatherDimOneReadsInt32Indices() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new float[]{3.0f, 4.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
        assertIndexPolicy(artifact, Cpu1StorageKind.JAVA_ARRAY);
    }

    @Test
    void preparedF64GatherDimZeroReadsInt64Indices() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new long[]{1L, 0L, 1L}, new int[]{3}, null, "indices", DataType.INT64);
        Fixture fixture = fixture(input.gather(indices, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_F64_I64_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{4.0d, 2.0d, 6.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedBf16GatherCopiesRawBfloat16Values() {
        Tensor input = new Tensor(
                new double[]{
                        1.25d, 2.5d, 3.75d,
                        4.5d, 5.25d, 6.5d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.BFLOAT16
        );
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_BF16_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new float[]{2.5f, 6.5f}, bf16ToF32(output), 1.0e-3f);
    }

    @Test
    void preparedGatherSupportsIntegerAndBoolValues() {
        Tensor ints = new Tensor(new int[]{10, 20, 30}, new int[]{3}, null, "ints", DataType.INT32);
        Tensor intIndices = new Tensor(new long[]{2L}, new int[]{1}, null, "intIndices", DataType.INT64);
        Tensor intOutput = execute(
                fixture(ints.gather(intIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new int[]{30}, intOutput.toInt32ArrayCopy());

        Tensor bools = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "bools", DataType.BOOL);
        Tensor boolIndices = new Tensor(new int[]{0}, new int[]{1}, null, "boolIndices", DataType.INT32);
        Tensor boolOutput = execute(
                fixture(bools.gather(boolIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new boolean[]{true}, boolOutput.toBooleanArrayCopy());

        Tensor longs = new Tensor(new long[]{100L, 200L, 300L}, new int[]{3}, null, "longs", DataType.INT64);
        Tensor longOutput = execute(
                fixture(longs.gather(boolIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new long[]{100L}, longOutput.toInt64ArrayCopy());
    }

    @Test
    void preparedGatherRejectsOutOfRangeIndexAtExecution() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> execute(fixture, artifact));

        assertTrue(exception.getMessage().contains("Gather index out of bounds"));
    }

    @Test
    void prepareRejectsNonContiguousGatherInput() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(view.gather(indices, 0));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void prepareRejectsNonContiguousGatherOutput() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2},
                new int[]{2},
                List.of(input, indices),
                new gather(1),
                "gather_strided_output",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(output);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("output access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void prepareRejectsMemorySegmentStorageKind() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread())
        );

        assertTrue(exception.getMessage().contains("JAVA_ARRAY"));
        assertTrue(exception.getMessage().contains("MEMORY_SEGMENT"));
    }

    @Test
    void prepareRejectsFloatingIndexDType() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new float[]{2.0f, 0.0f}, new int[]{2}, null, "indices", DataType.FLOAT32);
        Fixture fixture = fixture(input.gather(indices, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("INT32/INT64 indices"));
        assertTrue(exception.getMessage().contains("FLOAT32"));
    }

    @Test
    void artifactAndTraceExposeGatherKernelAndStorageKind() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertEquals(Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY, artifact.preparedIndexUnit().kernelId());
        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY.name(), trace.kernel());
        assertEquals(
                Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY.name(),
                trace.metadata().attributes().get("cpu1IndexKernelId")
        );
        assertEquals(Cpu1StorageKind.JAVA_ARRAY.name(), trace.metadata().attributes().get("cpu1StorageKind"));
        assertEquals("GATHER", trace.metadata().attributes().get("cpu1IndexOpType"));
        assertEquals("FLOAT32", trace.metadata().attributes().get("cpu1IndexValueDType"));
        assertEquals("INT32", trace.metadata().attributes().get("cpu1IndexDType"));
        assertEquals(1, trace.metadata().attributes().get("cpu1IndexDimension"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexLaunchWorkers"));
    }

    private static Tensor execute(Fixture fixture, Cpu1PrepareConfig config) {
        return execute(fixture, prepareRoot(fixture, config));
    }

    private static Tensor execute(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context.runtimeTensorForNodeId(fixture.node().id());
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertIndexKernel(Cpu1PreparedArtifact artifact, Cpu1IndexKernelId expected) {
        Cpu1IndexExecutableUnit executable = assertInstanceOf(Cpu1IndexExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedIndexUnit().kernelId());
        assertSame(artifact.preparedIndexUnit(), executable.preparedUnit());
    }

    private static void assertIndexPolicy(Cpu1PreparedArtifact artifact, Cpu1StorageKind storageKind) {
        Cpu1PreparedIndexUnit unit = artifact.preparedIndexUnit();
        assertEquals(storageKind, unit.storageKind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.inputAccessPlan().kind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.indexAccessPlan().kind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.outputAccessPlan().kind());
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(
            Fixture fixture,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex
    ) {
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] source = TensorInternalAccess.bfloat16Data(tensor);
        float[] out = new float[source.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(source[i]);
        }
        return out;
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
