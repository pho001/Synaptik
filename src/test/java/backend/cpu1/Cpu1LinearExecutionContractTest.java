package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1MatmulPreparer;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.CompiledGraph;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import trace.execution.ExecutionStepTrace;
import trace.execution.RunTrace;
import runtime.runner.StepExecutionTracer;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1LinearExecutionContractTest {
    @Test
    void preparedF32LinearWithRowBiasRunsDenseJavaArrayRouteWithoutRelu() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f,
                        10.0f, 11.0f, 12.0f
                },
                new int[]{2, 2, 3},
                null,
                "linearInput",
                DataType.FLOAT32
        );
        Tensor weight = new Tensor(
                new float[]{
                        1.0f, 0.0f,
                        -1.0f, 2.0f,
                        0.5f, -0.5f
                },
                new int[]{3, 2},
                null,
                "linearWeight",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{0.25f, -20.0f}, new int[]{1, 2}, null, "linearBias", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight, bias));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, artifact.preparedMatmulUnit().postOp());
        assertTrue(artifact.preparedMatmulUnit().hasBias());
        assertEquals(nodeId(fixture.nodes(), "linearBias"), artifact.preparedMatmulUnit().biasNodeId());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(new int[]{2, 2, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new float[]{
                        0.75f, -17.5f,
                        2.25f, -13.0f,
                        3.75f, -8.5f,
                        5.25f, -4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedBf16LinearWithBiasUsesJavaScalarMatmulRoute() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "bf16LinearInput",
                DataType.BFLOAT16
        );
        Tensor weight = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "bf16LinearWeight",
                DataType.BFLOAT16
        );
        Tensor bias = new Tensor(new float[]{-1.0f, 2.0f}, new int[]{2}, null, "bf16LinearBias", DataType.BFLOAT16);
        Fixture fixture = fixture(input.linear(weight, bias));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, artifact.preparedMatmulUnit().postOp());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{18.0f, 24.0f, 42.0f, 52.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-6f
        );
    }

    @Test
    void preparedGraphSpecializesLinearBiasAsSingleCpu1MatmulPostOpStep() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "graphLinearInput",
                DataType.FLOAT32
        );
        Tensor weight = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "graphLinearWeight",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{1.0f, -1.0f}, new int[]{2}, null, "graphLinearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        PreparedExecution execution = CompiledGraph.compile(linear, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults(DataType.FLOAT32));

        PreparedExecutionStep step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.metadata().artifact() instanceof Cpu1PreparedArtifact artifact
                        && artifact.executableUnit() instanceof Cpu1MatmulExecutableUnit)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, step.metadata().artifact());

        assertEquals(Operation.OpType.LINEAR, step.compiledNode().operation().opType());
        assertEquals(1, step.orderedNodeIds().size());
        assertEquals(List.of(step.compiledNode().id()), step.boundaryOutputNodeIds());
        assertEquals(step.compiledNode().id(), artifact.preparedMatmulUnit().nodeId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, artifact.preparedMatmulUnit().postOp());
        assertTrue(artifact.preparedMatmulUnit().hasBias());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertEquals(3, step.metadata().executionInputNodeIds().size());

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{20.0f, 21.0f, 44.0f, 49.0f}, linear.toFloat32ArrayCopy(), 1.0e-6f);
        ExecutionStepTrace tracedStep = trace.steps().stream()
                .filter(candidate -> "ADD_BIAS".equals(candidate.metadata().attributes().get("cpu1MatmulPostOp")))
                .findFirst()
                .orElseThrow();
        assertEquals("ADD_BIAS", tracedStep.metadata().attributes().get("cpu1MatmulPostOp"));
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR.name(), tracedStep.metadata().attributes().get("cpu1MatmulRoute"));
    }

    @Test
    void preparedF32LinearWithoutBiasRunsNativeMemorySegmentRouteWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        float[] inputData = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        float[] weightData = new float[]{
                7.0f, 8.0f,
                9.0f, 10.0f,
                11.0f, 12.0f
        };
        Tensor input = new Tensor(inputData, new int[]{2, 3}, null, "nativeLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(weightData, new int[]{3, 2}, null, "nativeLinearWeight", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulPostOp.NONE, artifact.preparedMatmulUnit().postOp());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().get(0), inputData);
        attachNativeF32Input(context, fixture.node().inputIds().get(1), weightData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        assertArrayEquals(
                new float[]{58.0f, 64.0f, 139.0f, 154.0f},
                nativeF32Values(assertInstanceOf(NativeFloat32Storage.class, output)),
                1.0e-4f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF32LinearWithBiasRunsNativeMemorySegmentRouteWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        float[] inputData = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        float[] weightData = new float[]{
                7.0f, 8.0f,
                9.0f, 10.0f,
                11.0f, 12.0f
        };
        float[] biasData = new float[]{0.5f, -10.0f};
        Tensor input = new Tensor(inputData, new int[]{2, 3}, null, "nativeLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(weightData, new int[]{3, 2}, null, "nativeLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(biasData, new int[]{2}, null, "nativeLinearBias", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight, bias));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, artifact.preparedMatmulUnit().postOp());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().get(0), inputData);
        attachNativeF32Input(context, fixture.node().inputIds().get(1), weightData);
        attachNativeF32Input(context, fixture.node().inputIds().get(2), biasData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        assertArrayEquals(
                new float[]{58.5f, 54.0f, 139.5f, 144.0f},
                nativeF32Values(assertInstanceOf(NativeFloat32Storage.class, output)),
                1.0e-4f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF64LinearWithBiasRunsNativeMemorySegmentRouteWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat64GemmAvailable(), OpenBlasRuntime.unavailableReason());
        double[] inputData = new double[]{
                1.0d, 2.0d,
                3.0d, 4.0d
        };
        double[] weightData = new double[]{
                2.0d, -1.0d,
                3.0d, 0.5d
        };
        double[] biasData = new double[]{-2.0d, 5.0d};
        Tensor input = new Tensor(inputData, new int[]{2, 2}, null, "nativeF64LinearInput", DataType.FLOAT64);
        Tensor weight = new Tensor(weightData, new int[]{2, 2}, null, "nativeF64LinearWeight", DataType.FLOAT64);
        Tensor bias = new Tensor(biasData, new int[]{2}, null, "nativeF64LinearBias", DataType.FLOAT64);
        Fixture fixture = fixture(input.linear(weight, bias));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, artifact.preparedMatmulUnit().postOp());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF64Input(context, fixture.node().inputIds().get(0), inputData);
        attachNativeF64Input(context, fixture.node().inputIds().get(1), weightData);
        attachNativeF64Input(context, fixture.node().inputIds().get(2), biasData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        assertArrayEquals(
                new double[]{6.0d, 5.0d, 16.0d, 4.0d},
                nativeF64Values(assertInstanceOf(NativeFloat64Storage.class, output)),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF32LinearReluWithBiasRunsNativeMemorySegmentRouteWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        float[] inputData = new float[]{
                1.0f, -2.0f,
                3.0f, 4.0f
        };
        float[] weightData = new float[]{
                2.0f, -1.0f,
                3.0f, -5.0f
        };
        float[] biasData = new float[]{5.0f, -20.0f};
        Tensor input = new Tensor(inputData, new int[]{2, 2}, null, "nativeReluLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(weightData, new int[]{2, 2}, null, "nativeReluLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(biasData, new int[]{2}, null, "nativeReluLinearBias", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight, bias).relu(), Operation.OpType.RELU);
        CompiledNode linearNode = node(fixture.nodes(), Operation.OpType.LINEAR);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = new Cpu1MatmulPreparer().prepareLinearEpilogue(
                linearNode,
                fixture.node(),
                fixture.descriptorIndex(),
                config,
                Cpu1MatmulPostOp.ADD_BIAS_RELU
        );
        assertLinearMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS_RELU, artifact.preparedMatmulUnit().postOp());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                linearNode.inputIds(),
                artifact
        );
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, linearNode.inputIds().get(0), inputData);
        attachNativeF32Input(context, linearNode.inputIds().get(1), weightData);
        attachNativeF32Input(context, linearNode.inputIds().get(2), biasData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        assertArrayEquals(
                new float[]{1.0f, 0.0f, 23.0f, 0.0f},
                nativeF32Values(assertInstanceOf(NativeFloat32Storage.class, output)),
                1.0e-4f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void linearWithBiasRejectsOpenBlasArrayCopyingAtPrepare() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{5.0f, 6.0f, 7.0f, 8.0f}, new int[]{2, 2}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1.0f, -1.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight, bias));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(
                        fixture,
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING)
                )
        );

        assertTrue(exception.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING.name()));
        assertTrue(exception.getMessage().contains(Cpu1MatmulPostOp.ADD_BIAS.name()));
    }

    @Test
    void bf16LinearWithBiasRejectsOpenBlasNativeSegmentAtPrepare() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new float[]{5.0f, 6.0f, 7.0f, 8.0f}, new int[]{2, 2}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new float[]{1.0f, -1.0f}, new int[]{2}, null, "bias", DataType.BFLOAT16);
        Fixture fixture = fixture(input.linear(weight, bias));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(
                        fixture,
                        Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT)
                )
        );

        assertTrue(exception.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT.name()));
        assertTrue(exception.getMessage().contains(DataType.BFLOAT16.name()));
    }

    @Test
    void linearRejectsStorageOffsetInputAtPrepare() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f,
                        10.0f, 11.0f, 12.0f
                },
                new int[]{2, 2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor inputView = base.select(0, 1);
        Tensor weight = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, new int[]{3, 2}, null, "weight", DataType.FLOAT32);
        Fixture fixture = fixture(inputView.linear(weight));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("dense contiguous inputs without storage offset"));
    }

    @Test
    void linearTraceReportsDirectAddBiasMatmulRoute() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{5.0f, 6.0f, 7.0f, 8.0f}, new int[]{2, 2}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1.0f, -1.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(input.linear(weight, bias));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );

        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR.name(), trace.kernel());
        assertEquals("JAVA_SCALAR", trace.metadata().matMul().route());
        assertEquals("ADD_BIAS", trace.metadata().attributes().get("cpu1MatmulPostOp"));
        assertEquals("JAVA_SCALAR", trace.metadata().attributes().get("cpu1MatmulRoute"));
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertLinearMatmulKernel(Cpu1PreparedArtifact artifact, Cpu1MatmulKernelId expected) {
        Cpu1MatmulExecutableUnit executable = assertInstanceOf(Cpu1MatmulExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedMatmulUnit().kernelId());
        assertSame(artifact.preparedMatmulUnit(), executable.preparedUnit());
    }

    private static Fixture fixture(Tensor out) {
        return fixture(out, Operation.OpType.LINEAR);
    }

    private static Fixture fixture(Tensor out, Operation.OpType expectedRootOp) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        CompiledNode node = nodes.getLast();
        assertEquals(expectedRootOp, node.operation().opType());
        return new Fixture(out, nodes, descriptorIndex, node);
    }

    private static CompiledNode node(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow();
    }

    private static int nodeId(List<CompiledNode> nodes, String label) {
        return nodes.stream()
                .filter(node -> label.equals(node.label()))
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
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

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1-linear-test-f32-input-" + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 LINEAR test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1-linear-test-f64-input-" + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 LINEAR test native F64 input");
    }

    private static float[] nativeF32Values(NativeFloat32Storage storage) {
        float[] out = new float[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat32At(i);
        }
        return out;
    }

    private static double[] nativeF64Values(NativeFloat64Storage storage) {
        double[] out = new double[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat64At(i);
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
