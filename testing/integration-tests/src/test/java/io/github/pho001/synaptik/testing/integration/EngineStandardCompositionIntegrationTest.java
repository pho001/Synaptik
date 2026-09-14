package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import org.junit.jupiter.api.Test;

/** Crosses the real public standard Engine construction and CPU composition boundary. */
final class EngineStandardCompositionIntegrationTest {
    @Test
    void opensFreshIndependentStandardEnginesAndClosesThem() {
        Engine first = Engine.standard();
        Engine second = Engine.standard();
        try {
            assertNotSame(first, second);
            assertFalse(first.isClosed());
            assertFalse(second.isClosed());

            first.close();
            assertTrue(first.isClosed());
            assertFalse(second.isClosed());
        } finally {
            first.close();
            second.close();
        }
        assertTrue(second.isClosed());
    }
}
