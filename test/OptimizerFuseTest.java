import Graph.optimizer.GraphOptimizer;
import Graph.optimizer.rules.FuseElementWiseRule;
import Tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class OptimizerFuseTest {

    @Test
    public void testFuseElementWiseOptimization() {
        // Baseline graph size without optimization.
        Tensor a0 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a0");
        Tensor b0 = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b0");
        Tensor c0 = new Tensor(new double[]{5.0, 6.0}, new int[]{2}, null, "c0");
        a0.setRequiresGrad(true);
        b0.setRequiresGrad(true);
        c0.setRequiresGrad(true);
        Tensor e0 = a0.add(b0).add(c0);
        e0.compute(new GraphOptimizer());
        int baselineGraphSize = e0.getCompiledGraph().getCompiledGraphAsList().size();

        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a");
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b");
        Tensor c = new Tensor(new double[]{5.0, 6.0}, new int[]{2}, null, "c");
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        c.setRequiresGrad(true);

        Tensor d = a.add(b);
        Tensor e = d.add(c);

        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(new FuseElementWiseRule());

        e.compute(optimizer);
        e.getCompiledGraph().setTrainingModeOff();
        e.compute(optimizer);

        assertArrayEquals(new double[]{9.0, 12.0}, e.getData(), 1e-9);

        assertNotNull(e.getOperation(), "Final tensor should have an operation");
        int fusedGraphSize = e.getCompiledGraph().getCompiledGraphAsList().size();
        assertTrue(fusedGraphSize < baselineGraphSize,
                "Compiled graph should be smaller after fusion optimization");

        e.getCompiledGraph().setTrainingModeOn();
        e.compute(optimizer);

        assertArrayEquals(new double[]{1.0, 1.0}, a.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, b.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, c.getGradient().getData(), 1e-9);
    }
}
