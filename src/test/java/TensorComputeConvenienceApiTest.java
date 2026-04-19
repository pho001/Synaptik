import backend.runtime.ExecutionMode;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Test;
import tensor.CompileMode;
import tensor.ComputeOptions;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorComputeConvenienceApiTest {
    @Test
    void computeDefaultsToInferenceOnlyAndReturnsSameTensor() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        Tensor returned = loss.compute();

        assertSame(loss, returned);
        assertEquals(4.0d, loss.scalarAsDouble(), 1e-9);
        assertNull(a.getGradient());
    }

    @Test
    void compileInferenceOnlyProducesForwardOnlyArtifact() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph compiled = loss.compile(CompileMode.INFERENCE_ONLY);

        assertFalse(compiled.supportsBackward());
        PreparedExecution prepared = compiled.prepare();
        prepared.execute(ExecutionMode.FORWARD);

        assertEquals(4.0d, loss.scalarAsDouble(), 1e-9);
        assertNull(a.getGradient());
    }

    @Test
    void compileTrainingProducesBackwardCapableArtifact() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        CompiledGraph compiled = loss.compile(CompileMode.TRAINING);

        assertTrue(compiled.supportsBackward());
        PreparedExecution prepared = compiled.prepare();
        prepared.execute(ExecutionMode.FORWARD_BACKWARD);

        assertEquals(4.0d, loss.scalarAsDouble(), 1e-9);
        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }

    @Test
    void computeAutoUsesTrainingPathWhenGraphHasTrainableLeaves() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        loss.compute(CompileMode.AUTO);

        assertEquals(4.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }

    @Test
    void computeOptionsCanRequestTrainingExecution() {
        Tensor a = new Tensor(new double[]{3.0}, new int[]{1}, null, "a", DataType.FLOAT64);
        a.setRequiresGrad(true);
        Tensor loss = a.mul(a);

        Tensor returned = loss.compute(new ComputeOptions().compileMode(CompileMode.TRAINING));

        assertSame(loss, returned);
        assertEquals(9.0d, loss.scalarAsDouble(), 1e-9);
        assertEquals(6.0d, a.getGradient().scalarAsDouble(), 1e-9);
    }
}
