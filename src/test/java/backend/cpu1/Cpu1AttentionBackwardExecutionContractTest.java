package backend.cpu1;

import backend.cpu1.kernels.linalg.attention.backward.Cpu1AttentionBackwardKernelId;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1AttentionBackwardPreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.model.CompiledNode;
import graph.compile.CompiledProgram;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.InputResidencyRequirement;
import graph.execution.plan.OutputResidencyEffect;
import graph.execution.state.ExecutionState;
import graph.compile.planning.region.specialization.SdpaBackwardOutputKind;
import graph.compile.planning.region.specialization.RegionSpecializationCandidate;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import graph.compile.planning.region.specialization.SdpaBackwardSpecializationPayload;
import graph.compile.planning.value.GraphValueRef;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.options.AttentionOptions;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1AttentionBackwardExecutionContractTest {
    @Test
    void f32SpecializedSdpaBackwardMatchesPrimitiveBaselineForQueryKeyAndValueGradients() {
        assertSpecializedMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.QUERY, 2.0e-5);
        assertSpecializedMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.KEY, 2.0e-5);
        assertSpecializedMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.VALUE, 2.0e-5);
    }

    @Test
    void f64SpecializedSdpaBackwardMatchesPrimitiveBaselineForQueryKeyAndValueGradients() {
        assertSpecializedMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.QUERY, 1.0e-12);
        assertSpecializedMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.KEY, 1.0e-12);
        assertSpecializedMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.VALUE, 1.0e-12);
    }

    private static void assertSpecializedMatchesBaseline(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            double tolerance
    ) {
        GradientRun baseline = execute(dataType, outputKind, CompileConfig.cpuOnlyBaseline());
        GradientRun specialized = execute(dataType, outputKind, CompileConfig.training());

        Cpu1PreparedArtifact artifact = requireAttentionBackwardArtifact(specialized.execution(), outputKind);
        assertEquals(expectedKernelId(dataType, outputKind, Cpu1StorageKind.JAVA_ARRAY),
                artifact.preparedAttentionBackwardUnit().kernelId());
        assertEquals(outputKind, artifact.preparedAttentionBackwardUnit().outputKind());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, artifact.preparedAttentionBackwardUnit().storageKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedAttentionBackwardUnit().vectorizationKind());
        assertEquals(Cpu1StorageAccessKind.BROADCAST,
                artifact.preparedAttentionBackwardUnit().outGradAccessPlan().kind());
        assertArrayEquals(baseline.gradient(), specialized.gradient(), tolerance);
        assertFalse(specialized.compiledGraph().program().compiledNodes().stream()
                .map(CompiledNode::operation)
                .filter(java.util.Objects::nonNull)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    @Test
    void f32SpecializedSdpaBackwardMemorySegmentMatchesPrimitiveBaselineForQueryKeyAndValueGradients() {
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.QUERY, 2.0e-5);
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.KEY, 2.0e-5);
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT32, SdpaBackwardOutputKind.VALUE, 2.0e-5);
    }

    @Test
    void f64SpecializedSdpaBackwardMemorySegmentMatchesPrimitiveBaselineForQueryKeyAndValueGradients() {
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.QUERY, 1.0e-12);
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.KEY, 1.0e-12);
        assertSpecializedMemorySegmentMatchesBaseline(DataType.FLOAT64, SdpaBackwardOutputKind.VALUE, 1.0e-12);
    }

    private static void assertSpecializedMemorySegmentMatchesBaseline(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            double tolerance
    ) {
        GradientRun baseline = execute(dataType, outputKind, CompileConfig.cpuOnlyBaseline());
        RuntimeConfig nativeRuntime = RuntimeConfig.trainingDefaults(dataType)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        GradientRun specialized = executeMemorySegmentStep(dataType, outputKind, nativeRuntime);

        Cpu1PreparedArtifact artifact = requireAttentionBackwardArtifact(specialized.execution(), outputKind);
        assertEquals(expectedKernelId(dataType, outputKind, Cpu1StorageKind.MEMORY_SEGMENT),
                artifact.preparedAttentionBackwardUnit().kernelId());
        assertEquals(outputKind, artifact.preparedAttentionBackwardUnit().outputKind());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedAttentionBackwardUnit().storageKind());
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedAttentionBackwardUnit().vectorizationKind());
        assertEquals(Cpu1StorageAccessKind.BROADCAST,
                artifact.preparedAttentionBackwardUnit().outGradAccessPlan().kind());
        PreparedExecutionStep step = requireAttentionBackwardStep(specialized.execution(), outputKind);
        assertEquals(InputResidencyRequirement.Mode.NONE, step.metadata().inputResidencyRequirement().mode());
        assertEquals(OutputResidencyEffect.Mode.NONE, step.metadata().outputResidencyEffect().mode());
        assertArrayEquals(baseline.gradient(), specialized.gradient(), tolerance);
        assertFalse(specialized.compiledGraph().program().compiledNodes().stream()
                .map(CompiledNode::operation)
                .filter(java.util.Objects::nonNull)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    @Test
    void memorySegmentSpecializedSdpaBackwardMaterializesCpuArrayInputsThroughRuntimeBoundary() {
        RuntimeConfig nativeRuntime = RuntimeConfig.trainingDefaults(DataType.FLOAT32)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        AttentionFixture fixture = defaultFixture();
        PreparedCase prepared = prepareDenseOutGradCase(
                DataType.FLOAT32,
                SdpaBackwardOutputKind.QUERY,
                CompileConfig.training(),
                nativeRuntime,
                fixture
        );
        PreparedExecutionStep step = requireAttentionBackwardStep(prepared.execution(), SdpaBackwardOutputKind.QUERY);
        Cpu1PreparedArtifact artifact = (Cpu1PreparedArtifact) step.metadata().artifact();
        Cpu1PreparedAttentionBackwardUnit unit = artifact.preparedAttentionBackwardUnit();
        ExecutionContext context = isolatedContext(prepared);
        attachCpuArray(context, unit.weightsNodeId(), attentionWeights(fixture), DataType.FLOAT32);
        attachCpuArray(context, unit.outGradNodeId(), denseOutGradValues(fixture), DataType.FLOAT32);
        if (unit.queryNodeId() >= 0) {
            attachCpuArray(context, unit.queryNodeId(), fixture.query(), DataType.FLOAT32);
        }
        if (unit.keyNodeId() >= 0) {
            attachCpuArray(context, unit.keyNodeId(), fixture.key(), DataType.FLOAT32);
        }
        if (unit.valueNodeId() >= 0) {
            attachCpuArray(context, unit.valueNodeId(), fixture.value(), DataType.FLOAT32);
        }

        new Cpu1Backend().execute(step.compiledNode(), step.metadata(), context);

        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, unit.storageKind());
        assertTrue(context.residencyForNodeId(unit.weightsNodeId()).nativeCurrent());
        assertTrue(context.residencyForNodeId(unit.nodeId()).nativeCurrent());
        assertFalse(context.residencyForNodeId(unit.nodeId()).cpuCurrent());
        assertEquals(inputRefs(unit).size(), context.cpuMaterializationTraceCount());
    }

    @Test
    void scalarConfigSelectsScalarMemorySegmentSdpaBackwardKernels() {
        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64}) {
            RuntimeConfig nativeRuntime = RuntimeConfig.trainingDefaults(dataType)
                    .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
            for (SdpaBackwardOutputKind outputKind : SdpaBackwardOutputKind.values()) {
                Cpu1PreparedAttentionBackwardUnit unit = preparedUnitWithConfig(
                        dataType,
                        outputKind,
                        nativeRuntime,
                        defaultFixture(),
                        Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                );
                assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, unit.storageKind());
                assertEquals(Cpu1VectorizationKind.SCALAR, unit.vectorizationKind());
                assertEquals(
                        expectedKernelId(dataType, outputKind, Cpu1StorageKind.MEMORY_SEGMENT, Cpu1VectorizationKind.SCALAR),
                        unit.kernelId()
                );
            }
        }
    }

    @Test
    void preparedScratchSizingDependsOnOutputKindAndDataType() {
        assertScratchSizing(DataType.FLOAT32, SdpaBackwardOutputKind.QUERY);
        assertScratchSizing(DataType.FLOAT32, SdpaBackwardOutputKind.KEY);
        assertScratchSizing(DataType.FLOAT32, SdpaBackwardOutputKind.VALUE);
        assertScratchSizing(DataType.FLOAT64, SdpaBackwardOutputKind.QUERY);
        assertScratchSizing(DataType.FLOAT64, SdpaBackwardOutputKind.KEY);
        assertScratchSizing(DataType.FLOAT64, SdpaBackwardOutputKind.VALUE);
    }

    @Test
    void largeDkPreparedLaunchUsesWorkBasedChunking() {
        AttentionFixture fixture = fixture(32, 64, 64, 32);
        Cpu1PreparedAttentionBackwardUnit unit = preparedUnit(
                DataType.FLOAT32,
                SdpaBackwardOutputKind.KEY,
                runtimeConfig(CpuKernelConfig.defaultsTraining()),
                fixture
        );

        assertEquals(Math.multiplyExact(fixture.batchCount(), fixture.keyLen()), unit.rowCount());
        assertTrue(unit.launchConfig().workerCount() > 1);
        assertTrue(unit.launchConfig().chunkSize() < unit.rowCount(),
                () -> "expected dK chunkSize < rowCount, chunkSize=" + unit.launchConfig().chunkSize()
                        + ", rowCount=" + unit.rowCount());
        assertTrue(Cpu1RangeLauncher.slotCount(unit.rowCount(), unit.launchConfig()) > 1);
    }

    @Test
    void smallDqAndDvPreparedLaunchStaySingleThreadUnderAttentionThreshold() {
        RuntimeConfig runtime = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE));

        Cpu1PreparedAttentionBackwardUnit dq = preparedUnit(
                DataType.FLOAT32,
                SdpaBackwardOutputKind.QUERY,
                runtime,
                defaultFixture()
        );
        Cpu1PreparedAttentionBackwardUnit dv = preparedUnit(
                DataType.FLOAT32,
                SdpaBackwardOutputKind.VALUE,
                runtime,
                defaultFixture()
        );

        assertEquals(1, dq.launchConfig().workerCount());
        assertEquals(0, dq.launchConfig().chunkSize());
        assertEquals(1, dv.launchConfig().workerCount());
        assertEquals(0, dv.launchConfig().chunkSize());
    }

    @Test
    void optimizedDkMatchesPrimitiveBaselineForLargerArrayAndMemorySegmentCases() {
        AttentionFixture fixture = fixture(2, 5, 5, 4);
        assertLargerDkMatchesBaseline(DataType.FLOAT32, fixture, 2.0e-5);
        assertLargerDkMatchesBaseline(DataType.FLOAT64, fixture, 1.0e-12);
    }

    @Test
    void vectorSpecializedSdpaBackwardMatchesPrimitiveBaselineForDenseOutGrad() {
        AttentionFixture f32Fixture = fixture(2, 5, 5, FloatVector.SPECIES_PREFERRED.length() + 3);
        assertDenseOutGradVectorMatchesBaseline(DataType.FLOAT32, f32Fixture, 3.0e-5);

        AttentionFixture f64Fixture = fixture(2, 5, 5, DoubleVector.SPECIES_PREFERRED.length() + 3);
        assertDenseOutGradVectorMatchesBaseline(DataType.FLOAT64, f64Fixture, 1.0e-12);
    }

    private static void assertScratchSizing(DataType dataType, SdpaBackwardOutputKind outputKind) {
        Cpu1PreparedAttentionBackwardUnit unit = preparedUnit(
                dataType,
                outputKind,
                RuntimeConfig.trainingDefaults(dataType),
                defaultFixture()
        );
        Cpu1ScratchBufferSpec scratch = unit.scratchBufferSpec();
        int expectedPerSlot = switch (outputKind) {
            case QUERY -> Math.multiplyExact(unit.keyLen(), 2);
            case KEY -> Math.addExact(
                    Math.multiplyExact(unit.keyLen(), 2),
                    Math.multiplyExact(unit.queryLen(), unit.keyLen())
            );
            case VALUE -> 0;
        };
        assertEquals(expectedPerSlot, unit.scratchElementsPerSlot());
        if (outputKind == SdpaBackwardOutputKind.VALUE) {
            assertTrue(scratch.isEmpty());
            assertEquals(0, unit.scratchSlotCount());
            return;
        }
        int expectedElements = Math.multiplyExact(unit.scratchSlotCount(), expectedPerSlot);
        assertTrue(unit.scratchSlotCount() > 0);
        if (dataType == DataType.FLOAT64) {
            assertEquals(0, scratch.f32ArrayElements());
            assertEquals(expectedElements, scratch.f64ArrayElements());
        } else {
            assertEquals(expectedElements, scratch.f32ArrayElements());
            assertEquals(0, scratch.f64ArrayElements());
        }
        if (outputKind == SdpaBackwardOutputKind.KEY) {
            assertEquals(Math.multiplyExact(unit.keyLen(), 2), unit.dScoresScratchOffset(0));
        } else {
            assertThrows(IllegalStateException.class, () -> unit.dScoresScratchOffset(0));
        }
    }

    private static void assertLargerDkMatchesBaseline(
            DataType dataType,
            AttentionFixture fixture,
            double tolerance
    ) {
        GradientRun baseline = execute(
                dataType,
                SdpaBackwardOutputKind.KEY,
                CompileConfig.cpuOnlyBaseline(),
                RuntimeConfig.trainingDefaults(dataType),
                fixture
        );
        GradientRun array = execute(
                dataType,
                SdpaBackwardOutputKind.KEY,
                CompileConfig.training(),
                RuntimeConfig.trainingDefaults(dataType),
                fixture
        );
        RuntimeConfig nativeRuntime = RuntimeConfig.trainingDefaults(dataType)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        GradientRun segment = executeMemorySegmentStep(dataType, SdpaBackwardOutputKind.KEY, nativeRuntime, fixture);

        assertArrayEquals(baseline.gradient(), array.gradient(), tolerance);
        assertArrayEquals(baseline.gradient(), segment.gradient(), tolerance);
    }

    private static void assertDenseOutGradVectorMatchesBaseline(
            DataType dataType,
            AttentionFixture fixture,
            double tolerance
    ) {
        RuntimeConfig nativeRuntime = RuntimeConfig.trainingDefaults(dataType)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        for (SdpaBackwardOutputKind outputKind : SdpaBackwardOutputKind.values()) {
            GradientRun baseline = executeDenseOutGrad(
                    dataType,
                    outputKind,
                    CompileConfig.cpuOnlyBaseline(),
                    RuntimeConfig.trainingDefaults(dataType),
                    fixture
            );
            GradientRun specialized = executeDenseOutGrad(
                    dataType,
                    outputKind,
                    CompileConfig.training(),
                    RuntimeConfig.trainingDefaults(dataType),
                    fixture
            );
            Cpu1PreparedAttentionBackwardUnit unit = requireAttentionBackwardArtifact(
                    specialized.execution(),
                    outputKind
            ).preparedAttentionBackwardUnit();
            assertEquals(Cpu1VectorizationKind.VECTOR, unit.vectorizationKind());
            assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.outGradAccessPlan().kind());
            assertArrayEquals(baseline.gradient(), specialized.gradient(), tolerance);

            GradientRun segment = executeDenseOutGradMemorySegmentStep(dataType, outputKind, nativeRuntime, fixture);
            Cpu1PreparedAttentionBackwardUnit segmentUnit = requireAttentionBackwardArtifact(
                    segment.execution(),
                    outputKind
            ).preparedAttentionBackwardUnit();
            assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, segmentUnit.storageKind());
            assertEquals(Cpu1VectorizationKind.VECTOR, segmentUnit.vectorizationKind());
            assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, segmentUnit.outGradAccessPlan().kind());
            assertEquals(expectedKernelId(dataType, outputKind, Cpu1StorageKind.MEMORY_SEGMENT), segmentUnit.kernelId());
            assertArrayEquals(baseline.gradient(), segment.gradient(), tolerance);
        }
    }

    private static GradientRun execute(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig
    ) {
        return execute(dataType, outputKind, compileConfig, RuntimeConfig.trainingDefaults(dataType));
    }

    private static GradientRun execute(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig
    ) {
        return execute(dataType, outputKind, compileConfig, runtimeConfig, defaultFixture());
    }

    private static GradientRun execute(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        Tensor q = tensor(fixture.query(), "attentionBackwardQ", dataType, fixture.queryShape());
        Tensor k = tensor(fixture.key(), "attentionBackwardK", dataType, fixture.keyShape());
        Tensor v = tensor(fixture.value(), "attentionBackwardV", dataType, fixture.valueShape());
        Tensor target = switch (outputKind) {
            case QUERY -> q;
            case KEY -> k;
            case VALUE -> v;
        };
        target.setRequiresGrad(true);
        Tensor loss = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5d)).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, compileConfig);
        PreparedExecution execution = graph.prepare(runtimeConfig);
        execution.execute(ExecutionMode.FORWARD_BACKWARD);
        assertNotNull(target.getGradient());
        return new GradientRun(graph, execution, target.getGradient().toDoubleArrayCopy());
    }

    private static GradientRun executeDenseOutGrad(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        Tensor q = tensor(fixture.query(), "attentionBackwardQ", dataType, fixture.queryShape());
        Tensor k = tensor(fixture.key(), "attentionBackwardK", dataType, fixture.keyShape());
        Tensor v = tensor(fixture.value(), "attentionBackwardV", dataType, fixture.valueShape());
        Tensor target = switch (outputKind) {
            case QUERY -> q;
            case KEY -> k;
            case VALUE -> v;
        };
        target.setRequiresGrad(true);
        Tensor outGrad = tensor(
                denseOutGradValues(fixture),
                "attentionBackwardDenseOutGrad",
                dataType,
                new int[]{fixture.batchCount(), fixture.queryLen(), fixture.valueDim()}
        );
        Tensor attention = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5d));
        Tensor loss = attention.mul(outGrad).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, compileConfig);
        PreparedExecution execution = graph.prepare(runtimeConfig);
        execution.execute(ExecutionMode.FORWARD_BACKWARD);
        assertNotNull(target.getGradient());
        return new GradientRun(graph, execution, target.getGradient().toDoubleArrayCopy());
    }

    private static GradientRun executeMemorySegmentStep(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig
    ) {
        return executeMemorySegmentStep(dataType, outputKind, runtimeConfig, defaultFixture());
    }

    private static GradientRun executeMemorySegmentStep(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        PreparedCase prepared = prepareCase(dataType, outputKind, CompileConfig.training(), runtimeConfig, fixture);
        PreparedExecutionStep step = requireAttentionBackwardStep(prepared.execution(), outputKind);
        Cpu1PreparedArtifact artifact = (Cpu1PreparedArtifact) step.metadata().artifact();
        Cpu1PreparedAttentionBackwardUnit unit = artifact.preparedAttentionBackwardUnit();
        ExecutionContext context = isolatedContext(prepared);

        attachNative(context, unit.weightsNodeId(), attentionWeights(fixture), dataType);
        attachNative(context, unit.outGradNodeId(), outGradValues(fixture), dataType);
        if (unit.queryNodeId() >= 0) {
            attachNative(context, unit.queryNodeId(), fixture.query(), dataType);
        }
        if (unit.keyNodeId() >= 0) {
            attachNative(context, unit.keyNodeId(), fixture.key(), dataType);
        }
        if (unit.valueNodeId() >= 0) {
            attachNative(context, unit.valueNodeId(), fixture.value(), dataType);
        }

        new Cpu1Backend().execute(step.compiledNode(), step.metadata(), context);
        NativeTensorStorage nativeOutput = context.nativeStorageForNodeId(unit.nodeId());
        assertNotNull(nativeOutput);
        assertTrue(context.residencyForNodeId(unit.nodeId()).nativeCurrent());
        assertFalse(context.residencyForNodeId(unit.nodeId()).cpuCurrent());
        assertEquals(0, context.cpuMaterializationTraceCount());
        return new GradientRun(prepared.compiledGraph(), prepared.execution(), readNative(nativeOutput, dataType));
    }

    private static GradientRun executeDenseOutGradMemorySegmentStep(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        PreparedCase prepared = prepareDenseOutGradCase(dataType, outputKind, CompileConfig.training(), runtimeConfig, fixture);
        PreparedExecutionStep step = requireAttentionBackwardStep(prepared.execution(), outputKind);
        Cpu1PreparedArtifact artifact = (Cpu1PreparedArtifact) step.metadata().artifact();
        Cpu1PreparedAttentionBackwardUnit unit = artifact.preparedAttentionBackwardUnit();
        ExecutionContext context = isolatedContext(prepared);

        attachNative(context, unit.weightsNodeId(), attentionWeights(fixture), dataType);
        attachNative(context, unit.outGradNodeId(), denseOutGradValues(fixture), dataType);
        if (unit.queryNodeId() >= 0) {
            attachNative(context, unit.queryNodeId(), fixture.query(), dataType);
        }
        if (unit.keyNodeId() >= 0) {
            attachNative(context, unit.keyNodeId(), fixture.key(), dataType);
        }
        if (unit.valueNodeId() >= 0) {
            attachNative(context, unit.valueNodeId(), fixture.value(), dataType);
        }

        new Cpu1Backend().execute(step.compiledNode(), step.metadata(), context);
        NativeTensorStorage nativeOutput = context.nativeStorageForNodeId(unit.nodeId());
        assertNotNull(nativeOutput);
        assertTrue(context.residencyForNodeId(unit.nodeId()).nativeCurrent());
        assertFalse(context.residencyForNodeId(unit.nodeId()).cpuCurrent());
        assertEquals(0, context.cpuMaterializationTraceCount());
        return new GradientRun(prepared.compiledGraph(), prepared.execution(), readNative(nativeOutput, dataType));
    }

    private static PreparedCase prepareCase(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig
    ) {
        return prepareCase(dataType, outputKind, compileConfig, runtimeConfig, defaultFixture());
    }

    private static PreparedCase prepareCase(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        Tensor q = tensor(fixture.query(), "attentionBackwardQ", dataType, fixture.queryShape());
        Tensor k = tensor(fixture.key(), "attentionBackwardK", dataType, fixture.keyShape());
        Tensor v = tensor(fixture.value(), "attentionBackwardV", dataType, fixture.valueShape());
        Tensor target = switch (outputKind) {
            case QUERY -> q;
            case KEY -> k;
            case VALUE -> v;
        };
        target.setRequiresGrad(true);
        Tensor loss = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5d)).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, compileConfig);
        return new PreparedCase(loss, graph, graph.prepare(runtimeConfig));
    }

    private static PreparedCase prepareDenseOutGradCase(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            CompileConfig compileConfig,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        Tensor q = tensor(fixture.query(), "attentionBackwardQ", dataType, fixture.queryShape());
        Tensor k = tensor(fixture.key(), "attentionBackwardK", dataType, fixture.keyShape());
        Tensor v = tensor(fixture.value(), "attentionBackwardV", dataType, fixture.valueShape());
        Tensor target = switch (outputKind) {
            case QUERY -> q;
            case KEY -> k;
            case VALUE -> v;
        };
        target.setRequiresGrad(true);
        Tensor outGrad = tensor(
                denseOutGradValues(fixture),
                "attentionBackwardDenseOutGrad",
                dataType,
                new int[]{fixture.batchCount(), fixture.queryLen(), fixture.valueDim()}
        );
        Tensor attention = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5d));
        Tensor loss = attention.mul(outGrad).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, compileConfig);
        return new PreparedCase(loss, graph, graph.prepare(runtimeConfig));
    }

    private static Tensor tensor(double[] values, String label, DataType dataType) {
        return tensor(values, label, dataType, defaultFixture().queryShape());
    }

    private static Tensor tensor(double[] values, String label, DataType dataType, int[] shape) {
        return new Tensor(values.clone(), shape.clone(), null, label, dataType);
    }

    private static Cpu1PreparedAttentionBackwardUnit preparedUnit(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture
    ) {
        return requireAttentionBackwardArtifact(
                prepareCase(dataType, outputKind, CompileConfig.training(), runtimeConfig, fixture).execution(),
                outputKind
        ).preparedAttentionBackwardUnit();
    }

    private static Cpu1PreparedAttentionBackwardUnit preparedUnitWithConfig(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig,
            AttentionFixture fixture,
            Cpu1PrepareConfig config
    ) {
        PreparedCase prepared = prepareCase(dataType, outputKind, CompileConfig.training(), runtimeConfig, fixture);
        PreparedExecutionStep originalStep = requireAttentionBackwardStep(prepared.execution(), outputKind);
        Cpu1PreparedAttentionBackwardUnit original = ((Cpu1PreparedArtifact) originalStep.metadata().artifact())
                .preparedAttentionBackwardUnit();
        Cpu1PreparedArtifact artifact = new Cpu1AttentionBackwardPreparer().prepare(
                originalStep.compiledNode(),
                candidateFromUnit(original, originalStep.orderedNodeIds()),
                prepared.compiledGraph().program().descriptorIndex(),
                config
        );
        return artifact.preparedAttentionBackwardUnit();
    }

    private static RegionSpecializationCandidate candidateFromUnit(
            Cpu1PreparedAttentionBackwardUnit unit,
            List<Integer> orderedNodeIds
    ) {
        SdpaBackwardSpecializationPayload payload = new SdpaBackwardSpecializationPayload(
                unit.outputKind(),
                unit.scale(),
                unit.hasMask(),
                unit.weightsNodeId(),
                unit.outGradNodeId(),
                unit.queryNodeId(),
                unit.keyNodeId(),
                unit.valueNodeId(),
                unit.maskNodeId()
        );
        return new RegionSpecializationCandidate(
                RegionSpecializationKind.SDPA_BACKWARD,
                orderedNodeIds,
                inputRefs(unit),
                GraphValueRef.node(unit.nodeId()),
                unit.nodeId(),
                "test SDPA backward " + unit.outputKind(),
                payload
        );
    }

    private static List<GraphValueRef> inputRefs(Cpu1PreparedAttentionBackwardUnit unit) {
        LinkedHashSet<Integer> nodeIds = new LinkedHashSet<>();
        nodeIds.add(unit.weightsNodeId());
        nodeIds.add(unit.outGradNodeId());
        if (unit.queryNodeId() >= 0) {
            nodeIds.add(unit.queryNodeId());
        }
        if (unit.keyNodeId() >= 0) {
            nodeIds.add(unit.keyNodeId());
        }
        if (unit.valueNodeId() >= 0) {
            nodeIds.add(unit.valueNodeId());
        }
        if (unit.maskNodeId() >= 0) {
            nodeIds.add(unit.maskNodeId());
        }
        ArrayList<GraphValueRef> refs = new ArrayList<>();
        for (int nodeId : nodeIds) {
            refs.add(GraphValueRef.node(nodeId));
        }
        return refs;
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, ApproximationConfig.defaults(), BlasConfig.disabled());
    }

    private static Cpu1PreparedArtifact requireAttentionBackwardArtifact(
            PreparedExecution execution,
            SdpaBackwardOutputKind outputKind
    ) {
        for (PreparedExecutionStep step : execution.backwardSteps()) {
            if (!(step.metadata().artifact() instanceof Cpu1PreparedArtifact artifact)) {
                continue;
            }
            try {
                if (artifact.preparedAttentionBackwardUnit().outputKind() == outputKind) {
                    assertFalse(step.orderedNodeIds().isEmpty());
                    return artifact;
                }
            } catch (IllegalStateException ignored) {
                // Other cpu1 artifacts in the same backward graph are not attention backward units.
            }
        }
        throw new AssertionError("Missing cpu1 SDPA backward artifact for " + outputKind);
    }

    private static PreparedExecutionStep requireAttentionBackwardStep(
            PreparedExecution execution,
            SdpaBackwardOutputKind outputKind
    ) {
        for (PreparedExecutionStep step : execution.backwardSteps()) {
            if (!(step.metadata().artifact() instanceof Cpu1PreparedArtifact artifact)) {
                continue;
            }
            try {
                if (artifact.preparedAttentionBackwardUnit().outputKind() == outputKind) {
                    assertFalse(step.orderedNodeIds().isEmpty());
                    return step;
                }
            } catch (IllegalStateException ignored) {
                // Other cpu1 artifacts in the same backward graph are not attention backward units.
            }
        }
        throw new AssertionError("Missing cpu1 SDPA backward step for " + outputKind);
    }

    private static ExecutionContext isolatedContext(PreparedCase prepared) {
        CompiledProgram program = prepared.compiledGraph().program();
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = new HashMap<>();
        for (PreparedExecutionStep step : prepared.execution().executionSteps()) {
            metadataIndex.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                program.compiledNodes(),
                program.descriptorIndex(),
                metadataIndex,
                program.forwardBoundaryNodeId(),
                testsupport.PublicationPlans.forRoot(
                        prepared.loss(),
                        program.compiledNodes(),
                        program.forwardOutputNodeId()
                )
        );
        return ExecutionContext.fromRuntimeConfig(
                prepared.execution().runtimeConfig(),
                ExecutionMode.FORWARD_BACKWARD,
                metadataIndex,
                state
        );
    }

    private static void attachNative(
            ExecutionContext context,
            int nodeId,
            double[] values,
            DataType dataType
    ) {
        NativeTensorStorage storage = context.allocateNativeStorage(
                dataType,
                values.length,
                "cpu1-sdpa-backward-test-native-" + nodeId
        );
        if (dataType == DataType.FLOAT32) {
            NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
            for (int i = 0; i < values.length; i++) {
                f32.setFloat32At(i, (float) values[i]);
            }
        } else if (dataType == DataType.FLOAT64) {
            NativeFloat64Storage f64 = (NativeFloat64Storage) storage;
            for (int i = 0; i < values.length; i++) {
                f64.setFloat64At(i, values[i]);
            }
        } else {
            throw new IllegalArgumentException("Unsupported native test dtype " + dataType);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 SDPA backward test native input");
    }

    private static void attachCpuArray(
            ExecutionContext context,
            int nodeId,
            double[] values,
            DataType dataType
    ) {
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (dataType == DataType.FLOAT32) {
            float[] target = TensorInternalAccess.float32Data(tensor);
            int copyLength = cpuArrayCopyLength(nodeId, values, target.length);
            for (int i = 0; i < copyLength; i++) {
                target[i] = (float) values[i];
            }
        } else if (dataType == DataType.FLOAT64) {
            double[] target = TensorInternalAccess.float64Data(tensor);
            System.arraycopy(values, 0, target, 0, cpuArrayCopyLength(nodeId, values, target.length));
        } else {
            throw new IllegalArgumentException("Unsupported CPU array test dtype " + dataType);
        }
        TensorInternalAccess.markStorageModified(tensor);
        context.markCpuCurrent(nodeId, "cpu1 SDPA backward test CPU array input");
    }

    private static int cpuArrayCopyLength(int nodeId, double[] values, int targetLength) {
        if (values.length == targetLength) {
            return targetLength;
        }
        if (targetLength == 1) {
            return 1;
        }
        throw new IllegalArgumentException("Test CPU array seed size mismatch for nodeId=" + nodeId
                + ". values=" + values.length + ", target=" + targetLength);
    }

    private static double[] readNative(NativeTensorStorage storage, DataType dataType) {
        double[] out = new double[storage.getSize()];
        if (dataType == DataType.FLOAT32) {
            NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
            for (int i = 0; i < out.length; i++) {
                out[i] = f32.getFloat32At(i);
            }
            return out;
        }
        if (dataType == DataType.FLOAT64) {
            NativeFloat64Storage f64 = (NativeFloat64Storage) storage;
            for (int i = 0; i < out.length; i++) {
                out[i] = f64.getFloat64At(i);
            }
            return out;
        }
        throw new IllegalArgumentException("Unsupported native test dtype " + dataType);
    }

    private static double[] attentionWeights() {
        return attentionWeights(defaultFixture());
    }

    private static double[] attentionWeights(AttentionFixture fixture) {
        double[] weights = new double[Math.multiplyExact(
                fixture.batchCount(),
                Math.multiplyExact(fixture.queryLen(), fixture.keyLen())
        )];
        for (int batch = 0; batch < fixture.batchCount(); batch++) {
            int queryBatchBase = batch * fixture.queryLen() * fixture.depth();
            int keyBatchBase = batch * fixture.keyLen() * fixture.depth();
            int weightsBatchBase = batch * fixture.queryLen() * fixture.keyLen();
            for (int queryIndex = 0; queryIndex < fixture.queryLen(); queryIndex++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    double score = dot(
                            fixture.query(),
                            queryBatchBase + queryIndex * fixture.depth(),
                            fixture.key(),
                            keyBatchBase + keyIndex * fixture.depth(),
                            fixture.depth()
                    ) * fixture.scale();
                    max = Math.max(max, score);
                }
                double sum = 0.0d;
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    double score = dot(
                            fixture.query(),
                            queryBatchBase + queryIndex * fixture.depth(),
                            fixture.key(),
                            keyBatchBase + keyIndex * fixture.depth(),
                            fixture.depth()
                    ) * fixture.scale();
                    double exp = Math.exp(score - max);
                    weights[weightsBatchBase + queryIndex * fixture.keyLen() + keyIndex] = exp;
                    sum += exp;
                }
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    weights[weightsBatchBase + queryIndex * fixture.keyLen() + keyIndex] /= sum;
                }
            }
        }
        return weights;
    }

    private static double dot2(double[] left, int leftBase, double[] right, int rightBase) {
        return left[leftBase] * right[rightBase] + left[leftBase + 1] * right[rightBase + 1];
    }

    private static double dot(double[] left, int leftBase, double[] right, int rightBase, int depth) {
        double sum = 0.0d;
        for (int i = 0; i < depth; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static double[] outGradValues(AttentionFixture fixture) {
        double[] out = new double[Math.multiplyExact(
                fixture.batchCount(),
                Math.multiplyExact(fixture.queryLen(), fixture.valueDim())
        )];
        java.util.Arrays.fill(out, 1.0d);
        return out;
    }

    private static double[] denseOutGradValues(AttentionFixture fixture) {
        return patternedValues(
                Math.multiplyExact(fixture.batchCount(), Math.multiplyExact(fixture.queryLen(), fixture.valueDim())),
                0.07d,
                -0.015d
        );
    }

    private static AttentionFixture defaultFixture() {
        return new AttentionFixture(
                1,
                2,
                2,
                2,
                2,
                new int[]{1, 2, 2},
                new int[]{1, 2, 2},
                new int[]{1, 2, 2},
                qValues(),
                kValues(),
                vValues(),
                0.5d
        );
    }

    private static AttentionFixture fixture(int batchCount, int queryLen, int keyLen, int depth) {
        int valueDim = depth;
        double[] query = patternedValues(Math.multiplyExact(batchCount, Math.multiplyExact(queryLen, depth)), 0.17d, 0.03d);
        double[] key = patternedValues(Math.multiplyExact(batchCount, Math.multiplyExact(keyLen, depth)), -0.13d, 0.02d);
        double[] value = patternedValues(Math.multiplyExact(batchCount, Math.multiplyExact(keyLen, valueDim)), 0.11d, -0.04d);
        return new AttentionFixture(
                batchCount,
                queryLen,
                keyLen,
                depth,
                valueDim,
                new int[]{batchCount, queryLen, depth},
                new int[]{batchCount, keyLen, depth},
                new int[]{batchCount, keyLen, valueDim},
                query,
                key,
                value,
                0.5d
        );
    }

    private static double[] patternedValues(int size, double angleScale, double offsetScale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin((i + 1) * angleScale) * 0.5d + ((i % 7) - 3) * offsetScale;
        }
        return out;
    }

    private static double[] qValues() {
        return new double[]{0.25d, -0.5d, 1.25d, 0.75d};
    }

    private static double[] kValues() {
        return new double[]{-0.25d, 1.0d, 0.5d, -1.5d};
    }

    private static double[] vValues() {
        return new double[]{1.5d, -0.75d, 0.25d, 2.0d};
    }

    private static Cpu1AttentionBackwardKernelId expectedKernelId(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            Cpu1StorageKind storageKind
    ) {
        return expectedKernelId(dataType, outputKind, storageKind, Cpu1VectorizationKind.VECTOR);
    }

    private static Cpu1AttentionBackwardKernelId expectedKernelId(
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return switch (dataType) {
            case FLOAT32 -> switch (outputKind) {
                case QUERY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_SCALAR);
                case KEY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_ARRAY_DENSE_SCALAR);
                case VALUE -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_ARRAY_DENSE_SCALAR);
            };
            case FLOAT64 -> switch (outputKind) {
                case QUERY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_SCALAR);
                case KEY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_ARRAY_DENSE_SCALAR);
                case VALUE -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_ARRAY_DENSE_SCALAR);
            };
            case BFLOAT16, INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported test dtype " + dataType);
        };
    }

    private record GradientRun(
            CompiledGraph compiledGraph,
            PreparedExecution execution,
            double[] gradient
    ) {
    }

    private record PreparedCase(
            Tensor loss,
            CompiledGraph compiledGraph,
            PreparedExecution execution
    ) {
    }

    private record AttentionFixture(
            int batchCount,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            int[] queryShape,
            int[] keyShape,
            int[] valueShape,
            double[] query,
            double[] key,
            double[] value,
            double scale
    ) {
    }
}
