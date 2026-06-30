package backend.cuda.exec;

import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import operations.layout.noop;
import org.junit.jupiter.api.Test;
import planning.intent.BackendIntentPlan;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class CudaDirectPreparedExecutableTest {
    @Test
    void prepareResolvesKernelBeforeExecution() {
        Tensor input = new Tensor(new float[]{1.0f}, new int[]{1}, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{1}, List.of(input), new noop(), "output", DataType.FLOAT32);
        CompiledNode node = CompiledNodeSnapshotter.snapshot(
                output.topologicalSort(),
                BackendIntentPlan.empty()
        ).getLast();

        CudaDirectPreparedExecutable executable = CudaDirectPreparedExecutable.prepare(node);

        assertNotNull(executable.kernel());
    }
}
