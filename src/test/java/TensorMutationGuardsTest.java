import backend.contract.ComputeBackend;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.model.CompiledNode;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import operations.elementwise.binary.mul;
import tensor.DataType;
import tensor.storage.Float64Storage;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorMutationGuardsTest {
    @Test
    void prevTensorsViewIsUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        List<Tensor> prev = out.getPrevTensors();

        assertThrows(UnsupportedOperationException.class, () -> prev.add(a));
    }

    @Test
    void compiledGraphExportIsUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());

        List<CompiledNode> nodes = compiled.program().compiledNodes();

        assertThrows(UnsupportedOperationException.class, () -> nodes.add(nodes.getFirst()));
    }

    @Test
    void preparedExecutionRejectsMutatedSemanticNodeTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD));
    }

    @Test
    void preparedExecutionRejectsMutatedStorageOwnerAfterPrepare() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        Tensor replacement = new Tensor(new double[]{11.0, 13.0}, new int[]{2}, null, "replacement", DataType.FLOAT64);
        TensorInternalAccess.aliasRuntimeFrom(out, replacement);

        assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD));
    }

    @Test
    void preparedExecutionRejectsMutatedShapeAfterPrepare() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        out.getShapeUnsafe()[0] = 1;

        assertStaleGraphContract(() -> execution.execute(ExecutionMode.FORWARD), "shape changed");
    }

    @Test
    void preparedExecutionRejectsMutatedLayoutStridesAfterPrepare() {
        Tensor x = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor selected = x.select(1, 1);
        Tensor out = selected.add(Tensor.onesLike(selected));

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        selected.getStridesUnsafe()[0] = 1;

        assertStaleGraphContract(() -> execution.execute(ExecutionMode.FORWARD), "strides changed");
    }

    @Test
    void preparedExecutionRejectsMutatedDTypeAfterPrepare() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        out.setDataType(DataType.FLOAT32);

        assertStaleGraphContract(() -> execution.execute(ExecutionMode.FORWARD), "dtype changed");
    }

    @Test
    void preparedExecutionUsesCompileLocalBackendIntentWithoutTensorContractMutation() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.of(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        BackendIntentPlan changedPlan = backendIntentPlan.withBackend(out, ComputeBackend.CPU);
        execution.execute(ExecutionMode.FORWARD);
        org.junit.jupiter.api.Assertions.assertEquals(ComputeBackend.CPU, changedPlan.backend(out));
    }

    @Test
    void preparedExecutionRejectsMutatedRequiresGradAfterPrepare() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        a.setRequiresGrad(true);

        assertStaleGraphContract(() -> execution.execute(ExecutionMode.FORWARD), "requiresGrad changed");
    }

    @Test
    void preparedExecutionRejectsMutatedTrainableParameterAfterPrepare() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(
                        out,
                        CompileConfig.noGraphOptimizationBaseline(),
                        tensor.CompileMode.INFERENCE_ONLY
                )
                .prepare(RuntimeConfig.inferenceDefaults());

        a.setTrainableParameter(true);

        assertStaleGraphContract(() -> execution.execute(ExecutionMode.FORWARD), "trainableParameter changed");
    }

    @Test
    void preparedExecutionUsesCompiledSnapshotForForwardOutputBinding() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        CompiledNode forwardOutput = compiled.program().compiledNodes().stream()
                .filter(t -> Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(t.label()))
                .findFirst()
                .orElseThrow();

        assertDoesNotThrow(() -> execution.execute(ExecutionMode.FORWARD));
        assertArrayEquals(new double[]{7.0, 10.0}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void prepareRejectsMutatedSemanticTopology() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());

        TensorInternalAccess.setPrevTensors(out, List.of(a));
        TensorInternalAccess.setOperation(out, new mul());

        assertThrows(IllegalStateException.class, () -> compiled.prepare(RuntimeConfig.inferenceDefaults()));
    }

    @Test
    void preparedExecutionAllowsInputValueChanges() {
        Tensor a = new Tensor(new double[]{2.0, 3.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5.0, 7.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        a.setData(new double[]{11.0, 13.0});
        b.setData(new double[]{17.0, 19.0});

        assertDoesNotThrow(() -> execution.execute(ExecutionMode.FORWARD));
        assertArrayEquals(new double[]{28.0, 32.0}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedExecutionRejectsPlanFromDifferentCompiledGraph() {
        Tensor a = new Tensor(new double[]{1.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0}, new int[]{1}, null, "b", DataType.FLOAT64);
        CompiledGraph first = CompiledGraph.compile(a.add(b), CompileConfig.noGraphOptimizationBaseline());

        Tensor x = new Tensor(new double[]{3.0}, new int[]{1}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(new double[]{4.0}, new int[]{1}, null, "y", DataType.FLOAT64);
        CompiledGraph second = CompiledGraph.compile(x.add(y), CompileConfig.noGraphOptimizationBaseline());
        PreparedExecution firstPlan = first.prepare(RuntimeConfig.inferenceDefaults());

        assertThrows(IllegalArgumentException.class,
                () -> firstPlan.requireCompatibleGraph(
                        second.publication().rootTensor(),
                        second.publication().graphContract()
                ));
    }

    @Test
    void replaceStorageRejectsViewLayout() {
        Tensor x = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor selected = x.select(1, 1);

        assertThrows(UnsupportedOperationException.class,
                () -> TensorInternalAccess.replaceStorage(selected, new Float64Storage(2)));
    }

    private static void assertStaleGraphContract(Runnable action, String expectedDetail) {
        IllegalStateException error = assertThrows(IllegalStateException.class, action::run);
        org.junit.jupiter.api.Assertions.assertTrue(
                error.getMessage().contains("Prepared execution graph contract is stale")
                        && error.getMessage().contains(expectedDetail),
                () -> "Expected stale graph contract detail '" + expectedDetail + "', got: " + error.getMessage()
        );
    }
}
