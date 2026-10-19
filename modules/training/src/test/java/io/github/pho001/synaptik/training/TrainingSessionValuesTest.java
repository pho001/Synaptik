package io.github.pho001.synaptik.training;

import static io.github.pho001.synaptik.training.TrainingValueFixture.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Numerical, positional and continuation contracts for fresh stored batches and targets. */
final class TrainingSessionValuesTest {
    /**
     * Checks independent losses, gradients, updates and call-local storage across all modes.
     * @throws Exception if existing private capture fields cannot be inspected
     */
    @Test
    void freshBatchesAndTargetsReuseCapturesAndAllGradientModes() throws Exception {
        TrainingStep retained;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor batch = scalar(arena, 1, false);
            Tensor target = scalar(arena, 0, false);
            var source = new TestParameterSource();
            var binding = source.parameter("weight", weight);
            var parameterStorage = weight.hostStorage().orElseThrow();
            var batchStorage = batch.hostStorage().orElseThrow();
            var targetStorage = target.hostStorage().orElseThrow();
            try (TrainingSession session = TrainingSession.open(engine, source,
                    objective(weight, batch, target), new Sgd(0.1))) {
                var graph = session.compiledGraph();
                var prepared = (PreparedExecution) field(session, "preparedExecution");
                Tensor seed = (Tensor) field(session, "cotangentSeed");
                assertEquals(graph.inputs().stream()
                        .filter(input -> !input.tensorId().equals(weight.id())
                                && !input.tensorId().equals(seed.id())).toList(), session.inputs());

                // w=2,x=3,y=1: loss=25, gradient=30; accumulation does not update w.
                try (Arena firstBatch = Arena.ofShared()) {
                    retained = session.runValues(values(session, batch, target,
                            scalar(firstBatch, 3, false), scalar(firstBatch, 1, false)),
                            GradientMode.ACCUMULATE);
                }
                assertEquals(25, retained.objective().bytes().getFloat());
                assertTrue(retained.optimizerStep().isEmpty());
                assertEquals(30, session.state().parameters().getFirst()
                        .accumulatedGradientBytes().getFloat());
                assertEquals(2, value(weight));
                // Fresh x=2,y=3: loss=1, gradient=4; accumulated update uses 30+4.
                TrainingStep second = session.runValues(values(session, batch, target,
                        scalar(arena, 2, false), scalar(arena, 3, false)),
                        GradientMode.ACCUMULATE_AND_STEP);
                assertEquals(1, second.objective().bytes().getFloat());
                assertEquals(-1.4f, value(weight), 1e-6f);
                assertEquals(0, session.accumulatedGradientRuns());
                // Pending gradients are discarded by RESET_AND_STEP, not included in its update.
                session.runValues(values(session, batch, target,
                        scalar(arena, 1, false), scalar(arena, 0, false)), GradientMode.ACCUMULATE);
                session.runValues(values(session, batch, target,
                        scalar(arena, 2, false), scalar(arena, -1, false)), GradientMode.RESET_AND_STEP);
                assertEquals(-0.68f, value(weight), 1e-6f); // gradient = -7.2, not -7.2-2.8
                assertEquals(4, session.executions());
                assertEquals(2, session.optimizerSteps());
                assertEquals(0, session.accumulatedGradientRuns());
                assertSame(graph, session.compiledGraph());
                assertSame(prepared, field(session, "preparedExecution"));
                assertSame(seed, field(session, "cotangentSeed"));
                assertSame(weight, binding.value());
                assertSame(parameterStorage, weight.hostStorage().orElseThrow());
                assertSame(batchStorage, batch.hostStorage().orElseThrow());
                assertSame(targetStorage, target.hostStorage().orElseThrow());
                assertEquals(1, value(batch));
                assertEquals(0, value(target));
                assertEquals(0, openResults(engine));
            }
        }
        assertEquals(25, retained.objective().bytes().getFloat());
    }

    /** Checks positional swapping and preserves exact-ID diagnostics on the original path. */
    @Test
    void sameDescriptorValuesArePositionalWhileOriginalRunRemainsIdentityBased() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor batch = scalar(arena, 1, false);
            Tensor target = scalar(arena, 0, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    objective(weight, batch, target), new Sgd(0.1))) {
                var initial = session.state();
                Tensor freshBatch = scalar(arena, 3, false);
                Tensor freshTarget = scalar(arena, 1, false);
                assertEquals(25, session.runValues(values(session, batch, target,
                        freshBatch, freshTarget), GradientMode.RESET_AND_STEP)
                        .objective().bytes().getFloat());
                assertEquals(-1, value(weight), 1e-6f);
                session.restore(initial);
                assertEquals(1, session.runValues(values(session, batch, target,
                        freshTarget, freshBatch), GradientMode.RESET_AND_STEP)
                        .objective().bytes().getFloat());
                assertEquals(2.2f, value(weight), 1e-6f);
                session.restore(initial);
                assertEquals(4, session.run(List.of(target, batch), GradientMode.RESET_AND_STEP)
                        .objective().bytes().getFloat());
                session.restore(initial);
                assertTrue(assertThrows(IllegalArgumentException.class, () -> session.run(
                        List.of(freshBatch, freshTarget), GradientMode.RESET_AND_STEP))
                        .getMessage().contains("foreign"));
                assertTrue(assertThrows(IllegalArgumentException.class, () -> session.run(
                        List.of(batch, batch), GradientMode.RESET_AND_STEP))
                        .getMessage().contains("duplicate"));
                assertThrows(IllegalArgumentException.class,
                        () -> session.run(List.of(batch), GradientMode.RESET_AND_STEP));
                assertState(initial, session.state());
            }
        }
    }

    /** Resumes adaptive optimizer and pending gradient state across both admission modes. */
    @Test
    void adamContinuationMixesBothEntryPointsAndRestoresPendingState() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor batch = scalar(arena, 1, false);
            Tensor target = scalar(arena, 0, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    objective(weight, batch, target), new Adam(0.1, 0.5, 0.25, 0.01, 0))) {
                session.run(List.of(target, batch), GradientMode.RESET_AND_STEP);
                session.runValues(values(session, batch, target,
                        scalar(arena, 3, false), scalar(arena, 1, false)), GradientMode.ACCUMULATE);
                TrainingState pending = session.state();
                Tensor nextBatch = scalar(arena, 2, false);
                Tensor nextTarget = scalar(arena, 3, false);
                session.runValues(values(session, batch, target, nextBatch, nextTarget),
                        GradientMode.ACCUMULATE_AND_STEP);
                TrainingState expected = session.state();
                assertEquals(3, expected.executions());
                assertEquals(2, expected.optimizerSteps());
                assertTrue(expected.parameters().getFirst().secondMomentBytes().isPresent());
                session.restore(pending);
                assertState(pending, session.state());
                session.runValues(values(session, batch, target,
                        scalar(arena, 2, false), scalar(arena, 3, false)),
                        GradientMode.ACCUMULATE_AND_STEP);
                assertState(expected, session.state());
                // Identity execution continues the very same adaptive history.
                session.run(List.of(batch, target), GradientMode.RESET_AND_STEP);
                assertEquals(3, session.optimizerSteps());
            }
        }
    }
}
