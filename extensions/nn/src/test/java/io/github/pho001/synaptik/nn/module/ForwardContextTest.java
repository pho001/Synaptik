package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ForwardContextTest {
    @Test
    void moduleStartsInTrainingAndProducesIndependentImmutableModeSnapshots() {
        Module module = new EmptyModule();

        ForwardContext trainingContext = module.forwardContext();
        assertEquals(ForwardMode.TRAINING, module.mode());
        assertEquals(ForwardMode.TRAINING, trainingContext.mode());

        module.eval();
        ForwardContext evaluationContext = module.forwardContext();
        assertEquals(ForwardMode.EVALUATION, module.mode());
        assertEquals(ForwardMode.TRAINING, trainingContext.mode());
        assertEquals(ForwardMode.EVALUATION, evaluationContext.mode());
        assertNotSame(trainingContext, evaluationContext);

        module.train();
        assertEquals(ForwardMode.TRAINING, module.mode());
        assertEquals(ForwardMode.EVALUATION, evaluationContext.mode());
    }

    @Test
    void forwardContextRejectsNullMode() {
        assertThrows(NullPointerException.class, () -> new ForwardContext(null));
    }

    private static final class EmptyModule extends Module {
    }
}
