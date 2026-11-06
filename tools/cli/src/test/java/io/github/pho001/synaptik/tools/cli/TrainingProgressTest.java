package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TrainingProgressTest {
    @Test
    void everyFastFinalOutcomeCapturesTheLastCommittedStepWithoutWaitingForRefresh() {
        for (TrainingProgress.Outcome outcome : TrainingProgress.Outcome.values()) {
            AtomicLong clock = new AtomicLong();
            Capture capture = new Capture();
            try (TrainingProgress progress = new TrainingProgress(clock::get, capture, 500, 1)) {
                progress.beginEpoch(1);
                progress.step(Double.POSITIVE_INFINITY, 3, 60);
                clock.set(10);
                if (outcome == TrainingProgress.Outcome.COMPLETED) {
                    progress.endEpoch();
                }
                progress.finish(outcome, null);
                progress.finish(TrainingProgress.Outcome.FAILED, "must not replace final outcome");
                progress.refresh();
                assertEquals(1, capture.frames.size());
                assertSame(outcome, capture.outcomes.get(0));
                var state = capture.frames.get(0);
                assertEquals(1, state.steps());
                assertEquals(3, state.examples());
                assertEquals(60, state.bytes());
                assertEquals(Double.POSITIVE_INFINITY, state.loss());
                assertEquals(1, state.losses().length);
                assertEquals(10, state.elapsedNanos());
            }
            assertEquals(1, capture.closes);
        }
    }

    @Test
    void cancellationRequestsDoNotInterruptOrDiscardAnAdmittedSuccessfulStep() {
        Capture capture = new Capture();
        try (TrainingProgress progress = new TrainingProgress(() -> 0, capture, 500, 1)) {
            progress.beginEpoch(1);
            assertFalse(progress.cancellationRequested());
            progress.requestCancellation();
            assertTrue(progress.cancellationRequested());
            assertEquals(0, capture.closes);
            // Runner remains responsible for deciding whether to admit another step.
            progress.step(Double.NaN, 2, 32);
            progress.finish(TrainingProgress.Outcome.CANCELLED, null);
            assertSame(TrainingProgress.Outcome.CANCELLED, capture.outcomes.get(0));
            assertEquals(1, capture.frames.get(0).steps());
            assertEquals(32, capture.frames.get(0).bytes());
            assertTrue(Double.isNaN(capture.frames.get(0).loss()));
        }
    }

    @Test
    void longCallRefreshSamplesOnlyDetachedCompletedStateAndRealElapsedTime() {
        AtomicLong clock = new AtomicLong();
        Capture capture = new Capture();
        try (TrainingProgress progress = new TrainingProgress(clock::get, capture, 500, 1)) {
            progress.beginEpoch(1);
            for (int i = 1; i <= 3; i++) {
                clock.set(i * 500_000_000L);
                progress.refresh();
            }
            assertEquals(3, capture.frames.size());
            for (var frame : capture.frames) {
                assertEquals(0, frame.steps());
                assertEquals(0, frame.bytes());
                assertEquals(0, frame.losses().length);
            }
            assertEquals(1_500_000_000L, capture.frames.get(2).epochNanos());
            progress.step(4, 1, 8);
            progress.finish(TrainingProgress.Outcome.COMPLETED, null);
            assertEquals(1, capture.frames.get(3).steps());
            assertEquals(1, capture.frames.get(3).losses().length);
        }
    }

    @Test
    void renderingAndCleanupFailuresRemainSuppressedUnderThePrimaryTrainingFailure() {
        RuntimeException training = new RuntimeException("training");
        RuntimeException rendering = new RuntimeException("rendering");
        RuntimeException cleanup = new RuntimeException("cleanup");
        Capture capture = new Capture() {
            @Override
            public void render(TrainingProgressStats.Snapshot snapshot, TrainingProgress.Outcome outcome, String detail) {
                super.render(snapshot, outcome, detail);
                throw rendering;
            }

            @Override
            public void close() {
                super.close();
                throw cleanup;
            }
        };
        RuntimeException actual = assertThrows(RuntimeException.class, () -> {
            try (TrainingProgress progress = new TrainingProgress(() -> 0, capture, 500, 1)) {
                progress.beginEpoch(1);
                progress.step(1, 1, 8);
                progress.refresh();
                progress.finish(TrainingProgress.Outcome.FAILED, "training");
                throw training;
            }
        });
        assertSame(training, actual);
        assertEquals(1, actual.getSuppressed().length);
        assertSame(rendering, actual.getSuppressed()[0]);
        assertSame(cleanup, rendering.getSuppressed()[0]);
        assertEquals(2, capture.frames.size());
        assertSame(TrainingProgress.Outcome.FAILED, capture.outcomes.get(1));
        assertEquals(1, capture.closes);
    }

    @Test
    void implicitCloseHasOneFinalFailureOrCancellationAndRestoresOnlyOnce() {
        for (boolean cancel : new boolean[] {false, true}) {
            Capture capture = new Capture();
            TrainingProgress progress = new TrainingProgress(() -> 0, capture, 500, 1);
            if (cancel) {
                progress.requestCancellation();
            }
            progress.close();
            progress.close();
            progress.refresh();
            assertEquals(1, capture.frames.size());
            assertSame(cancel ? TrainingProgress.Outcome.CANCELLED : TrainingProgress.Outcome.FAILED,
                    capture.outcomes.get(0));
            assertEquals(1, capture.closes);
        }
    }

    @Test
    void knownPresentationFailuresStopAdmissionsAndCleanupCannotSelfSuppressThePrimary() {
        for (Throwable rendering : new Throwable[] {
                new IllegalStateException("renderer"),
                new java.io.IOError(new java.io.IOException("terminal capability"))
        }) {
            var cleanup = new java.io.IOError(new java.io.IOException("terminal close"));
            Capture capture = new Capture() {
                @Override
                public void render(TrainingProgressStats.Snapshot snapshot, TrainingProgress.Outcome outcome, String detail) {
                    if (rendering instanceof Error error) throw error;
                    throw (RuntimeException) rendering;
                }

                @Override
                public void close() {
                    super.close();
                    throw cleanup;
                }
            };
            Throwable actual = assertThrows(rendering.getClass(), () -> {
                try (TrainingProgress progress = new TrainingProgress(() -> 0, capture, 500, 1)) {
                    progress.refresh();
                    progress.cancellationRequested();
                }
            });
            assertSame(rendering, actual);
            assertEquals(1, capture.closes);
            assertEquals(1, actual.getSuppressed().length);
            assertSame(cleanup, actual.getSuppressed()[0]);
        }
    }

    private static class Capture implements TrainingProgress.Sink {
        final List<TrainingProgressStats.Snapshot> frames = new ArrayList<>();
        final List<TrainingProgress.Outcome> outcomes = new ArrayList<>();
        int closes;

        @Override
        public void render(TrainingProgressStats.Snapshot snapshot, TrainingProgress.Outcome outcome, String detail) {
            frames.add(snapshot);
            outcomes.add(outcome);
        }

        @Override
        public void close() {
            closes++;
        }
    }
}
