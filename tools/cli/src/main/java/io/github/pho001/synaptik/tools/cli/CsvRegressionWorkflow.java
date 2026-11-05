package io.github.pho001.synaptik.tools.cli;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.training.Adam;
import io.github.pho001.synaptik.training.Optimizer;
import io.github.pho001.synaptik.training.Sgd;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Sequential outer public-API caller. Model owns all trainable state, Training owns every update,
 * Engine owns execution, and the CLI owns stable-file traversal and external batch storage.
 */
final class CsvRegressionWorkflow {
    private CsvRegressionWorkflow() { }

    static int train(CliArguments args, PrintStream error) throws IOException {
        Path dataPath = args.requiredPath("data");
        Path outputPath = args.requiredPath("output");
        requireDistinct(outputPath, dataPath, "model output must not replace training CSV");
        CliBackend backend = CliBackend.parse(args);
        int epochs = args.positiveInt("epochs", 20);
        long refresh = args.integer("refresh-ms", 500);
        if (refresh <= 0) throw new IllegalArgumentException("--refresh-ms must be positive");
        int requestedBatch = args.positiveInt("batch-size", 1);
        int requestedOutputs = args.positiveInt("outputs", 1);
        long seed = args.integer("seed", 0);
        Optimizer optimizer = optimizer(args);

        GraphCheckpoint.Source source = args.has("model") ? CliModel.load(args.requiredPath("model")) : null;
        CliModel.Signature loaded = source == null ? null : CliModel.signature(source);
        if (loaded != null) {
            if (args.has("seed")) throw new IllegalArgumentException("--seed applies only to new Models");
            if (args.has("batch-size") && requestedBatch != loaded.batch()) {
                throw new IllegalArgumentException("--batch-size differs from the loaded Model signature");
            }
            if (args.has("outputs") && requestedOutputs != loaded.outputs()) {
                throw new IllegalArgumentException("--outputs differs from the loaded Model signature");
            }
        }
        int columns = loaded == null ? 0 : Math.addExact(loaded.features(), loaded.outputs());
        NumericCsv data = NumericCsv.scan(dataPath, columns);
        int batch = loaded == null ? requestedBatch : loaded.batch();
        int outputs = loaded == null ? requestedOutputs : loaded.outputs();
        int features = loaded == null ? data.columns() - outputs : loaded.features();
        if (features <= 0) throw new IllegalArgumentException("training CSV must contain input columns before final targets");
        data.requireBatches(batch);
        long stepsPerEpoch = data.rows() / batch;
        // Check logical telemetry arithmetic before the first update too.
        long bytesPerBatch = Math.multiplyExact(4L, Math.multiplyExact((long) batch, (long) features + outputs));

        try (TrainingProgress progress = TrainingProgress.open(error, args.has("plain"), refresh,
                epochs, stepsPerEpoch, data.rows(), backend.label())) {
            try {
                boolean cancelled;
                try (var engine = backend.open();
                     var acquired = source == null
                             ? CliModel.create(batch, features, outputs, seed, engine)
                             : CliModel.acquire(source, engine);
                     var values = new NativeBatch(acquired.signature(), true)) {
                    var model = acquired.model();
                    cancelled = progress.cancellationRequested();
                    if (!cancelled) {
                        model.configureTraining(acquired.signature().target(),
                                (prediction, target) -> prediction.meanSquaredError(target, LossReduction.MEAN),
                                optimizer);
                    }
                    training:
                    for (int epoch = 0; epoch < epochs && !cancelled; epoch++) {
                        if (progress.cancellationRequested()) {
                            cancelled = true;
                            break;
                        }
                        progress.beginEpoch(epoch + 1);
                        try (var cursor = data.open()) {
                            for (long step = 0; step < stepsPerEpoch; step++) {
                                if (progress.cancellationRequested()) {
                                    cancelled = true;
                                    break training;
                                }
                                values.fill(cursor);
                                if (progress.cancellationRequested()) {
                                    cancelled = true;
                                    break training;
                                }
                                // No UI thread touches Model. Once admitted, this call finishes normally.
                                var result = model.train(values.input(), values.target());
                                result.optimizerStep().orElseThrow(
                                        () -> new IllegalStateException("training returned no committed optimizer step"));
                                progress.step(scalar(result.objective()), batch, bytesPerBatch);
                                if (progress.cancellationRequested()) {
                                    cancelled = true;
                                    break training;
                                }
                            }
                            if (cursor.next()) throw new IllegalArgumentException("CSV grew after validation");
                        }
                        progress.endEpoch();
                    }
                    cancelled |= progress.cancellationRequested();
                    if (!cancelled) model.save(outputPath);
                }
                progress.finish(cancelled ? TrainingProgress.Outcome.CANCELLED : TrainingProgress.Outcome.COMPLETED,
                        cancelled ? "Stopped after the last completed call; no model saved."
                                : "Saved current graph/weights to " + outputPath + "; no optimizer/cursor resume state.");
                return cancelled ? 130 : 0;
            } catch (IOException | RuntimeException | Error failure) {
                try {
                    progress.finish(TrainingProgress.Outcome.FAILED, failure.getMessage());
                } catch (RuntimeException | Error displayFailure) {
                    if (displayFailure != failure) failure.addSuppressed(displayFailure);
                }
                throw failure;
            }
        }
    }

    static int predict(CliArguments args, PrintStream output) throws IOException {
        Path modelPath = args.requiredPath("model");
        Path dataPath = args.requiredPath("data");
        Path outputPath = args.has("output") ? args.requiredPath("output") : null;
        if (outputPath != null) {
            requireDistinct(outputPath, dataPath, "prediction output must not replace input CSV");
            requireDistinct(outputPath, modelPath, "prediction output must not replace Model");
        }
        CliBackend backend = CliBackend.parse(args);
        var source = CliModel.load(modelPath);
        var signature = CliModel.signature(source);
        NumericCsv data = NumericCsv.scan(dataPath, signature.features());
        data.requireBatches(signature.batch());
        try (var destination = new PredictionOutput(outputPath, output)) {
            try (var engine = backend.open();
                 var acquired = CliModel.acquire(source, engine);
                 var values = new NativeBatch(signature, false);
                 var cursor = data.open()) {
                long steps = data.rows() / signature.batch();
                for (long step = 0; step < steps; step++) {
                    values.fill(cursor);
                    destination.batch(acquired.model().compute(values.input()), signature);
                }
                if (cursor.next()) throw new IllegalArgumentException("CSV grew after validation");
            }
            destination.commit();
        }
        return 0;
    }

    private static Optimizer optimizer(CliArguments args) {
        return switch (args.value("optimizer", "sgd")) {
            case "sgd" -> new Sgd(args.positiveFloatRate("learning-rate", 0.01));
            case "adam" -> new Adam(args.positiveFloatRate("learning-rate", 0.001), 0.9, 0.999, 1e-8, 0);
            default -> throw new IllegalArgumentException("--optimizer must be sgd or adam");
        };
    }

    static double scalar(HostTensorValue value) {
        if (value.shape().rank() != 0 || value.elementCount() != 1) {
            throw new IllegalStateException("training objective was not scalar");
        }
        ByteBuffer bytes = value.bytes();
        return switch (value.dataType()) {
            case FLOAT32 -> bytes.getFloat();
            case FLOAT64 -> bytes.getDouble();
            default -> throw new IllegalStateException("training objective was not floating point");
        };
    }

    private static void requireDistinct(Path destination, Path source, String detail) throws IOException {
        if (destination.equals(source) || (Files.exists(destination) && Files.isSameFile(destination, source))) {
            throw new IllegalArgumentException(detail);
        }
    }
}
