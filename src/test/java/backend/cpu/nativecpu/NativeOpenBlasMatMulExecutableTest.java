package backend.cpu.nativecpu;

import backend.ComputeBackend;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.bf16.BF16NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f64.F64NativeBlasMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeOpenBlasMatMulExecutableTest {
    @Test
    void float32NativeSegmentMatmulLeavesOutputNativeUntilPublication() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(out);
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        ExecutionState state = fixture.state();
        attachNative(state, matmulStep.compiledNode().inputIds().get(0));
        attachNative(state, matmulStep.compiledNode().inputIds().get(1));

        PreparedMatMulExecutable executable = new F32NativeBlasMatMulExecutable(nativeHints(matmulStep));
        executable.execute(
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                context(fixture, matmulStep, executable)
        );

        var residency = state.residencyForNodeId(matmulStep.compiledNode().id());
        assertEquals(StorageResidency.CPU_NATIVE, residency.residency());
        assertTrue(residency.nativeCurrent());
        assertFalse(residency.cpuCurrent());
        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, executable.lastExecutionRoute());
        assertEquals(0L, executable.lastCopyInBytes());
        assertEquals(0L, executable.lastCopyOutBytes());

        NativeFloat32Storage nativeOut = (NativeFloat32Storage) state.nativeStorageForNodeId(matmulStep.compiledNode().id());
        assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, read(nativeOut), 1e-6f);

        state.requireCpuReadable(matmulStep.compiledNode().id(), CpuMaterializationReason.GRAPH_OUTPUT);

        assertArrayEquals(new float[]{19f, 22f, 43f, 50f},
                state.runtimeTensorForNodeId(matmulStep.compiledNode().id()).getFloat32Data(),
                1e-6f);
        assertEquals(StorageResidency.CPU_ARRAY, state.residencyForNodeId(matmulStep.compiledNode().id()).residency());
    }

    @Test
    void float64NativeSegmentMatmulLeavesOutputNativeUntilPublication() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5d, 6d, 7d, 8d}, new int[]{2, 2}, null, "b", DataType.FLOAT64);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(out);
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        ExecutionState state = fixture.state();
        attachNative(state, matmulStep.compiledNode().inputIds().get(0));
        attachNative(state, matmulStep.compiledNode().inputIds().get(1));

        PreparedMatMulExecutable executable = new F64NativeBlasMatMulExecutable(nativeHints(matmulStep));
        executable.execute(
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                context(fixture, matmulStep, executable)
        );

        var residency = state.residencyForNodeId(matmulStep.compiledNode().id());
        assertEquals(StorageResidency.CPU_NATIVE, residency.residency());
        assertTrue(residency.nativeCurrent());
        assertFalse(residency.cpuCurrent());
        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, executable.lastExecutionRoute());

        NativeFloat64Storage nativeOut = (NativeFloat64Storage) state.nativeStorageForNodeId(matmulStep.compiledNode().id());
        assertArrayEquals(new double[]{19d, 22d, 43d, 50d}, read(nativeOut), 1e-12);
    }

    @Test
    void bfloat16NativeSegmentMatmulUsesBgemmAndLeavesOutputNativeUntilPublication() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d}, new int[]{2, 2}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{5d, 6d, 7d, 8d}, new int[]{2, 2}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(out);
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        ExecutionState state = fixture.state();
        attachNative(state, matmulStep.compiledNode().inputIds().get(0));
        attachNative(state, matmulStep.compiledNode().inputIds().get(1));

        PreparedMatMulExecutable executable = new BF16NativeBlasMatMulExecutable(nativeHints(matmulStep));
        executable.execute(
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                state.runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                context(fixture, matmulStep, executable)
        );

        var residency = state.residencyForNodeId(matmulStep.compiledNode().id());
        assertEquals(StorageResidency.CPU_NATIVE, residency.residency());
        assertTrue(residency.nativeCurrent());
        assertFalse(residency.cpuCurrent());
        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, executable.lastExecutionRoute());
        assertEquals("cblas_bgemm", executable.lastBlasSymbol());

        NativeBFloat16Storage nativeOut = (NativeBFloat16Storage) state.nativeStorageForNodeId(matmulStep.compiledNode().id());
        assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, read(nativeOut), 0.0f);
    }

    @Test
    void nativeSegmentMatmulReportsArrayToNativeInputCopiesWhenInputsStartAsCpuArrays() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(out);
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        PreparedMatMulExecutable executable = new F32NativeBlasMatMulExecutable(nativeHints(matmulStep));

        executable.execute(
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                context(fixture, matmulStep, executable)
        );

        assertEquals(32L, executable.lastCopyInBytes());
        assertEquals(0L, executable.lastCopyOutBytes());
        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, executable.lastExecutionRoute());
        assertEquals(2, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void nativeSegmentFallbackPolicyAllowsJavaRouteAndRecordsReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{1, 2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{1, 2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(out);
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        PreparedMatMulExecutable executable = new F32NativeBlasMatMulExecutable(nativeHints(matmulStep));

        executable.execute(
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                context(fixture, matmulStep, executable)
        );

        assertEquals(MatMulExecutionRoute.JAVA_DIRECT, executable.lastExecutionRoute());
        assertTrue(executable.lastFallbackReason().contains("rank-2 matmul"));
        assertArrayEquals(
                new float[]{19f, 22f, 43f, 50f},
                fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().id()).getFloat32Data(),
                1e-6f
        );
    }

    @Test
    void nativeSegmentRequireNativePolicyRejectsJavaFallback() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{1, 2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{1, 2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        Fixture fixture = fixture(
                out,
                RuntimeConfig.inferenceDefaults().withNativeCpuFailurePolicy(NativeCpuFailurePolicy.REQUIRE_NATIVE)
        );
        PreparedNodeExecution matmulStep = fixture.matmulStep();
        PreparedMatMulExecutable executable = new F32NativeBlasMatMulExecutable(nativeHints(matmulStep));

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> executable.execute(
                        fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(0)),
                        fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().inputIds().get(1)),
                        fixture.state().runtimeTensorForNodeId(matmulStep.compiledNode().id()),
                        context(fixture, matmulStep, executable)
                )
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertEquals(MatMulExecutionRoute.JAVA_DIRECT, executable.lastExecutionRoute());
        assertTrue(executable.lastFallbackReason().contains("rank-2 matmul"));
    }

    private static Fixture fixture(Tensor out) {
        return fixture(out, RuntimeConfig.inferenceDefaults());
    }

    private static Fixture fixture(Tensor out, RuntimeConfig runtime) {
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        PreparedExecution prepared = compiled.prepare(runtime);
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = prepared.executionSteps().stream()
                .collect(Collectors.toMap(step -> step.compiledNode().id(), PreparedNodeExecution::metadata));
        ExecutionState state = ExecutionState.create(
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                metadataIndex,
                compiled.program().forwardBoundaryNodeId(),
                compiled.publication()
        );
        PreparedNodeExecution matmul = prepared.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        return new Fixture(prepared, matmul, state, metadataIndex);
    }

    private static CpuKernelContext context(Fixture fixture, PreparedNodeExecution step, PreparedMatMulExecutable executable) {
        CpuNodeExecutionPlan base = testsupport.MetadataArtifacts.cpuPlan(step.metadata());
        CpuNodeExecutionPlan nativePlan = new CpuNodeExecutionPlan(
                base.layoutPlan(),
                base.computeContract(),
                base.publishFloatContinuation(),
                base.plannedWorkers(),
                base.contiguousMaterializeThreshold(),
                base.dispatchHints(),
                base.reductionHints(),
                nativeHints(step),
                executable,
                base.conv2dHints(),
                base.attentionPlan()
        );
        ExecutionContext executionContext = ExecutionContext.fromRuntimeConfig(
                fixture.prepared().runtimeConfig(),
                ExecutionMode.FORWARD,
                fixture.metadataIndex(),
                fixture.state()
        );
        CompiledNode node = step.compiledNode();
        return new CpuKernelContext(
                node.id(),
                node.inputIds(),
                nativePlan,
                executionContext,
                step.metadata(),
                node.inputIds().stream()
                        .map(fixture.metadataIndex()::get)
                        .toList()
        );
    }

    private static ResolvedMatMulHints nativeHints(PreparedNodeExecution step) {
        ResolvedMatMulHints base = testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints();
        return new ResolvedMatMulHints(
                true,
                false,
                MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT,
                base.parallel(),
                base.tileM(),
                base.tileN(),
                base.tileK(),
                base.plannedWorkers(),
                base.work(),
                base.microKernel()
        );
    }

    private static void attachNative(ExecutionState state, int nodeId) {
        Tensor tensor = state.runtimeTensorForNodeId(nodeId);
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(
                tensor.getDataType(),
                tensor.getFlatDataSize(),
                "test-node-" + nodeId
        );
        NativeCpuMaterializer.arrayToNative(tensor, storage);
        state.attachNativeStorage(nodeId, storage, "test native input");
    }

    private static float[] read(NativeFloat32Storage storage) {
        float[] out = new float[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat32At(i);
        }
        return out;
    }

    private static double[] read(NativeFloat64Storage storage) {
        double[] out = new double[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat64At(i);
        }
        return out;
    }

    private static float[] read(NativeBFloat16Storage storage) {
        float[] out = new float[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = CpuDTypeOps.fromBFloat16Bits(storage.getBFloat16BitsAt(i));
        }
        return out;
    }

    private record Fixture(
            PreparedExecution prepared,
            PreparedNodeExecution matmulStep,
            ExecutionState state,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex
    ) {
    }
}
