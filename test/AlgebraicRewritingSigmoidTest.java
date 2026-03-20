import Graph.optimizer.GraphOptimizer;
import Graph.optimizer.rules.AlgebraicRewritingRule;
import Operations.Operation;
import Tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AlgebraicRewritingSigmoidTest {

    @Test
    void rewritesCanonicalSigmoidNegFormInInference() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base");
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        baselineOutput.compute(new GraphOptimizer());

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt");
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(new AlgebraicRewritingRule());
        optimizedOutput.compute(optimizer);

        assertTrue(containsSigmoid(optimizedOutput),
                "Algebraic rewriting should replace canonical sigmoid form in inference");
        assertArrayEquals(baselineOutput.getData(), optimizedOutput.getData(), 1e-9);
    }

    @Test
    void rewritesCanonicalSigmoidMulScalarFormInInference() {
        Tensor baselineInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_base");
        Tensor baselineOutput = baselineInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        baselineOutput.compute(new GraphOptimizer());

        Tensor optimizedInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_opt");
        Tensor optimizedOutput = optimizedInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(new AlgebraicRewritingRule());
        optimizedOutput.compute(optimizer);

        assertTrue(containsSigmoid(optimizedOutput),
                "Algebraic rewriting should rewrite mulScalar(-1) sigmoid form in inference");
        assertArrayEquals(baselineOutput.getData(), optimizedOutput.getData(), 1e-9);
    }

    @Test
    void doesNotRewriteSigmoidWhenGradientsAreRequired() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base");
        baselineInput.setRequiresGrad(true);
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        baselineOutput.compute(new GraphOptimizer());

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt");
        optimizedInput.setRequiresGrad(true);
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.addRule(new AlgebraicRewritingRule());
        optimizedOutput.compute(optimizer);

        assertTrue(!containsSigmoid(optimizedOutput),
                "Sigmoid rewrite should be skipped when gradients are required");
        assertArrayEquals(baselineOutput.getData(), optimizedOutput.getData(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().getData(), optimizedInput.getGradient().getData(), 1e-9);
    }

    private static boolean containsSigmoid(Tensor output) {
        return output.getCompiledGraph().getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SIGMOID);
    }
}
