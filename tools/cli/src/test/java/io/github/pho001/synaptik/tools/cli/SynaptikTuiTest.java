package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Actual detached jobs, data admission, numeric results and cooperative controller lifetimes. */
final class SynaptikTuiTest {
    @TempDir Path directory;

    @Test
    void detachedTrainingPublishesOnlyCommittedNumericGeometryAndFilePrediction() throws Exception {
        Path initial = linear("initial", 2, new float[] {0, 0}, new float[] {0, 0});
        Path trained = directory.resolve("trained");
        var diagnostics = new ByteArrayOutputStream();
        try (var error = new PrintStream(diagnostics)) {
            var training = new SynaptikTui.ActiveJob(options("train", "--model", initial.toString(),
                    "--data", csv("train.csv", "1,2,4\n2,6,8\n").toString(),
                    "--output", trained.toString(), "--epochs", "1", "--learning-rate", "0.1"), error);
            run(training);
            var result = training.view();
            assertFalse(result.running());
            assertEquals(0, result.status());
            assertEquals(TrainingProgress.Outcome.COMPLETED, result.progress().outcome());
            assertEquals(1, result.progress().stepsPerEpoch());
            assertEquals(2, result.progress().examplesPerEpoch());
            var sample = result.progress().snapshot();
            assertEquals(1, sample.steps());
            assertEquals(2, sample.examples());
            assertEquals(24, sample.bytes());
            assertEquals(30.0, sample.loss(), 1e-6);
            assertArrayEquals(new double[] {30}, sample.losses(), 1e-6);
            assertEquals(1, sample.recentEpochs().length);
            assertTrue(result.elapsedNanos() >= sample.elapsedNanos());
            long elapsed = result.elapsedNanos();
            training.refresh();
            assertEquals(elapsed, training.view().elapsedNanos());

            Path predictions = directory.resolve("predictions.csv");
            var prediction = new SynaptikTui.ActiveJob(options("predict", "--model", trained.toString(),
                    "--data", csv("features.csv", "1\n2\n").toString(), "--output", predictions.toString()), error);
            run(prediction);
            assertEquals(0, prediction.view().status());
            float[][] values = CliRegressionTest.parse(Files.readString(predictions));
            assertArrayEquals(new float[] {1.1f, 1.6f}, values[0], 1e-5f);
            assertArrayEquals(new float[] {1.8f, 2.6f}, values[1], 1e-5f);
            assertEquals("", diagnostics.toString());
        }
    }

    @Test
    void stopBeforeMonitorAppearsPreventsNumericalAdmissionAndSaving() throws Exception {
        Path output = directory.resolve("not-saved");
        var options = options("train", "--data", csv("cancel.csv", "1,3\n2,5\n").toString(),
                "--output", output.toString(), "--backend", "metal", "--metal-library",
                directory.resolve("must-not-be-loaded").toString());
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var job = new SynaptikTui.ActiveJob(options, error);
            job.requestStop("test stop during validation");
            run(job);
            assertEquals(130, job.view().status());
            assertEquals(TrainingProgress.Outcome.CANCELLED, job.view().progress().outcome());
            assertEquals(0, job.view().progress().snapshot().steps());
            assertFalse(Files.exists(output));
            assertTrue(job.view().stopRequested());
        }
    }

    @Test
    void requestRacingFactoryPublicationIsAppliedBeforeFirstAdmission() throws Exception {
        Path output = directory.resolve("racing-not-saved");
        var args = options("train", "--data", csv("race.csv", "1,3\n").toString(),
                "--output", output.toString());
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var job = new SynaptikTui.ActiveJob(args, error);
            Thread worker = Thread.ofPlatform().start(() -> {
                try {
                    int status = CsvRegressionWorkflow.train(args, error, (stream, plain, refresh,
                            epochs, steps, examples, backend) -> {
                        entered.countDown();
                        await(release);
                        return job.openProgress(stream, plain, refresh, epochs, steps, examples, backend);
                    });
                    assertEquals(130, status);
                } catch (Throwable problem) { failure.set(problem); }
            });
            try {
                assertTrue(entered.await(10, TimeUnit.SECONDS));
                job.requestStop("stop before monitor publication");
            } finally { release.countDown(); }
            worker.join(10_000);
            assertFalse(worker.isAlive());
            if (failure.get() != null) throw new AssertionError(failure.get());
            assertEquals(TrainingProgress.Outcome.CANCELLED, job.view().progress().outcome());
            assertEquals(0, job.view().progress().snapshot().steps());
            assertFalse(Files.exists(output));
        }
    }

    @Test
    void invalidDataDoesNotCreateMonitorOrCheckpointAndCanBeCorrected() throws Exception {
        Path data = csv("bad.csv", "1,NaN\n");
        Path output = directory.resolve("after-edit");
        var args = options("train", "--data", data.toString(), "--output", output.toString(), "--epochs", "1");
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var failed = new SynaptikTui.ActiveJob(args, error);
            run(failed);
            assertEquals(2, failed.view().status());
            assertNull(failed.view().progress());
            assertFalse(Files.exists(output));
            Files.writeString(data, "1,3\n");
            var corrected = new SynaptikTui.ActiveJob(args, error);
            run(corrected);
            assertEquals(0, corrected.view().status());
            assertTrue(Files.exists(output));
        }
    }

    @Test
    void predictionBackAndExitWaitForNormalAtomicPublication() throws Exception {
        Path initial = linear("predict-source", 1, new float[] {2}, new float[] {1});
        Path data = csv("predict.csv", "0\n1\n2\n");
        for (var key : List.of(key(TuiConsole.Kind.ESCAPE), character('q'), key(TuiConsole.Kind.EOF),
                key(TuiConsole.Kind.INTERRUPT))) {
            Path output = directory.resolve("result-" + key.kind() + "-" + key.character() + ".csv");
            try (var error = new PrintStream(new ByteArrayOutputStream());
                    var session = new SynaptikTui.Session(error)) {
                session.submit(options("predict", "--model", initial.toString(), "--data", data.toString(),
                        "--output", output.toString()));
                session.handle(key);
                session.close();
                session.tick();
                assertArrayEquals(new float[] {1, 3, 5},
                        CliRegressionTest.flatten(CliRegressionTest.parse(Files.readString(output))), 0);
                assertEquals(key.kind() != TuiConsole.Kind.ESCAPE, session.finished());
            }
        }
    }

    @Test
    void interruptedUiWaitsWithoutInterruptingAdmittedPrediction() throws Exception {
        Path initial = linear("interrupt-source", 1, new float[] {2}, new float[] {1});
        Path output = directory.resolve("interrupt-result.csv");
        try (var error = new PrintStream(new ByteArrayOutputStream());
                var session = new SynaptikTui.Session(error)) {
            session.submit(options("predict", "--model", initial.toString(), "--data",
                    csv("interrupt-data.csv", "0\n1\n2\n").toString(), "--output", output.toString()));
            Thread.currentThread().interrupt();
            try {
                session.close();
                assertTrue(Thread.currentThread().isInterrupted());
            } finally {
                Thread.interrupted();
            }
            assertArrayEquals(new float[] {1, 3, 5},
                    CliRegressionTest.flatten(CliRegressionTest.parse(Files.readString(output))), 0);
        }
    }

    @Test
    void sharedFactoryPreservesDataFailureAndDistinctPresentationCleanupErrors() throws Exception {
        Path data = csv("changed.csv", "1,3\n");
        Path output = directory.resolve("must-not-save");
        var rendering = new AssertionError("render failure");
        var cleanup = new AssertionError("sink close failure");
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var thrown = assertThrows(IllegalArgumentException.class, () ->
                    CsvRegressionWorkflow.train(options("train", "--data", data.toString(),
                            "--output", output.toString(), "--epochs", "1"), error,
                            (stream, plain, refresh, epochs, steps, examples, backend) -> {
                                try { Files.writeString(data, "1,NaN\n"); }
                                catch (java.io.IOException failure) { throw new java.io.UncheckedIOException(failure); }
                                return new TrainingProgress(System::nanoTime, new TrainingProgress.Sink() {
                                    @Override public void render(TrainingProgressStats.Snapshot snapshot,
                                            TrainingProgress.Outcome outcome, String detail) { throw rendering; }
                                    @Override public void close() { throw cleanup; }
                                }, refresh, epochs);
                            }));
            assertArrayEquals(new Throwable[] {rendering}, thrown.getSuppressed());
            assertArrayEquals(new Throwable[] {cleanup}, rendering.getSuppressed());
            assertFalse(Files.exists(output));
        }
    }

    @Test
    void realMenuFormDataFailureReturnsEditableFormAndRetrySaves() throws Exception {
        Path bad = csv("malformed.csv", "1,NaN\n");
        Path good = csv("corrected.csv", "1,3\n");
        Path output = directory.resolve("retried-model");
        try (var error = new PrintStream(new ByteArrayOutputStream());
                var session = new SynaptikTui.Session(error)) {
            assertNull(session.currentJob());
            assertNull(session.frame(100, 40).cursor());
            session.handle(character('t'));
            assertNotNull(session.frame(100, 40).cursor());
            bad.toString().codePoints().forEach(code -> session.handle(character(code)));
            session.handle(key(TuiConsole.Kind.TAB));
            output.toString().codePoints().forEach(code -> session.handle(character(code)));
            session.handle(key(TuiConsole.Kind.ENTER));
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (session.currentJob().running() && System.nanoTime() < deadline) {
                session.tick();
                Thread.sleep(5);
            }
            assertFalse(session.currentJob().running());
            assertEquals(2, session.currentJob().status());
            session.handle(key(TuiConsole.Kind.ENTER));
            assertNull(session.currentJob());
            assertNotNull(session.frame(100, 40).cursor());
            session.handle(key(TuiConsole.Kind.BACK_TAB));
            session.handle(key(TuiConsole.Kind.HOME));
            for (int i = 0; i < bad.toString().codePointCount(0, bad.toString().length()); i++)
                session.handle(key(TuiConsole.Kind.DELETE));
            good.toString().codePoints().forEach(code -> session.handle(character(code)));
            session.handle(key(TuiConsole.Kind.ENTER));
            deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (session.currentJob().running() && System.nanoTime() < deadline) {
                session.tick();
                Thread.sleep(5);
            }
            assertFalse(session.currentJob().running());
            assertEquals(0, session.currentJob().status());
            assertTrue(Files.exists(output));
            session.handle(key(TuiConsole.Kind.ESCAPE));
            assertNull(session.currentJob());
            assertNull(session.frame(100, 40).cursor());
            session.handle(character('q'));
            assertTrue(session.finished());
        }
    }

    @Test
    void sessionRejectsScreenCorruptingStdoutPredictionBeforeAdmission() {
        try (var error = new PrintStream(new ByteArrayOutputStream());
                var session = new SynaptikTui.Session(error)) {
            assertThrows(IllegalArgumentException.class, () -> session.submit(options("predict",
                    "--model", "unused", "--data", "unused")));
            session.handle(character('q'));
            assertTrue(session.finished());
        }
    }

    @Test
    void inspectionDetachesPublicMetadataWithoutRegressionSignatureNarrowing() throws Exception {
        var scalar = TensorFactory.scalar(7f, Optional.empty(), false);
        Path path = directory.resolve("scalar-graph");
        GraphCheckpoint.save(path, new GraphCheckpoint.Source(List.of(), List.of(scalar),
                List.of(new GraphCheckpoint.Leaf("constant", GraphCheckpoint.Role.CONSTANT, scalar))), CliModel.LIMITS);
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var inspect = new SynaptikTui.ActiveJob(options("inspect", "--model", path.toString()), error);
            run(inspect);
            assertEquals(0, inspect.view().status());
            var detached = inspect.view().inspection();
            assertTrue(detached.stream().anyMatch(line -> line.contains(scalar.descriptor().toString())));
            assertTrue(detached.stream().anyMatch(line -> line.contains("storedBytes=4")));
            Files.delete(path);
            assertFalse(detached.isEmpty());
            assertThrows(UnsupportedOperationException.class, () -> detached.add("mutation"));
        }
    }

    @Test
    void eventHistoryRemainsBoundedAndFinishedMonitorCannotAdmitMoreSamples() {
        try (var error = new PrintStream(new ByteArrayOutputStream())) {
            var job = new SynaptikTui.ActiveJob(options("train"), error);
            var monitor = job.openProgress(error, false, 1, 50, 1, 1, "test detached geometry");
            for (int epoch = 1; epoch <= 50; epoch++) {
                monitor.beginEpoch(epoch);
                monitor.step(epoch, 1, 8);
                monitor.endEpoch();
                monitor.refresh();
            }
            monitor.finish(TrainingProgress.Outcome.COMPLETED, "actual detached completion");
            monitor.close();
            assertEquals(50, job.view().progress().snapshot().steps());
            assertEquals(50, job.view().progress().snapshot().recentEpochs()[7].number());
            assertTrue(job.events().size() <= 16);
            var snapshot = job.view().progress().snapshot();
            job.refresh();
            assertSame(snapshot, job.view().progress().snapshot());
        }
    }

    private Path linear(String name, int batch, float[] weights, float[] biases) throws Exception {
        var weight = TensorFactory.fromFlatArray(CliModel.f32(true, biases.length, 1), Optional.empty(), weights);
        var bias = TensorFactory.fromFlatArray(CliModel.f32(true, biases.length), Optional.empty(), biases);
        Path path = directory.resolve(name);
        try (var model = Model.builder().input(DataType.FLOAT32, batch, 1).linear(weight, bias).build()) {
            model.save(path);
        }
        return path;
    }

    private Path csv(String name, String content) throws Exception {
        return Files.writeString(directory.resolve(name), content);
    }

    private static CliArguments options(String... values) { return CliArguments.parse(values); }
    private static TuiConsole.Key key(TuiConsole.Kind kind) { return new TuiConsole.Key(kind, 0); }
    private static TuiConsole.Key character(int value) { return new TuiConsole.Key(TuiConsole.Kind.CHARACTER, value); }

    private static void run(SynaptikTui.ActiveJob job) throws Exception {
        Thread worker = Thread.ofPlatform().start(job);
        worker.join(10_000);
        assertFalse(worker.isAlive(), "real workflow did not close");
        assertFalse(job.view().running());
    }

    private static void await(CountDownLatch latch) {
        try { assertTrue(latch.await(10, TimeUnit.SECONDS)); }
        catch (InterruptedException failure) { throw new AssertionError(failure); }
    }
}
