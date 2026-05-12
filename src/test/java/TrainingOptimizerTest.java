import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import training.optimizer.AdamOptimizer;
import training.optimizer.SgdOptimizer;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainingOptimizerTest {
    @Test
    void trainableFlagEnablesGradButIsSeparateFromRequiresGrad() {
        Tensor parameter = new Tensor(new float[]{1.0f}, new int[]{1}, null, "parameter", DataType.FLOAT32);

        parameter.setTrainableParameter(true);

        assertTrue(parameter.getRequiresGrad());
        assertTrue(parameter.isTrainableParameter());

        parameter.setTrainableParameter(false);

        assertTrue(parameter.getRequiresGrad());
        assertEquals(false, parameter.isTrainableParameter());
    }

    @Test
    void sgdUpdatesOnlyTrainableParametersAndDoesNotPublishAllGradients() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);

        prepared.executeOptimizerStep(optimizer);

        assertArrayEquals(new float[]{0.8f, 1.7f}, w.getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.getFloat32Data(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void adamUpdatesOnlyTrainableParameters() {
        Tensor w = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, -3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        AdamOptimizer optimizer = new AdamOptimizer(0.1f);

        graph.executeOptimizerStep(RuntimeConfig.trainingDefaults(), optimizer);

        assertArrayEquals(new float[]{0.9f, -1.9f}, w.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new float[]{2.0f, -3.0f}, x.getFloat32Data(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void explicitOptimizerParameterListCannotUpdateNonTrainableGradientTensors() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        SgdOptimizer optimizer = new SgdOptimizer(List.of(x), 0.1f);

        assertEquals(List.of(w), graph.trainableParameters());

        graph.executeOptimizerStep(RuntimeConfig.trainingDefaults(), optimizer);

        assertArrayEquals(new float[]{1.0f, 2.0f}, w.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{2.0f, 3.0f}, x.getFloat32Data(), 0.0f);
        assertNull(w.getGradient());
        assertNull(x.getGradient());
    }

    @Test
    void normalForwardBackwardStillPublishesGradients() {
        Tensor w = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "w", DataType.FLOAT32);
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, new int[]{2}, null, "x", DataType.FLOAT32);
        w.setTrainableParameter(true);
        x.setRequiresGrad(true);
        Tensor loss = w.mul(x).sum();

        CompiledGraph.compile(loss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new float[]{2.0f, 3.0f}, w.getGradient().getFloat32Data(), 1.0e-6f);
        assertArrayEquals(new float[]{1.0f, 2.0f}, x.getGradient().getFloat32Data(), 1.0e-6f);
    }
}
