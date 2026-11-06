package io.github.pho001.synaptik.tools.cli;

import static io.github.pho001.synaptik.tools.cli.CliRegressionTest.run;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Failed commands preserve source data and never publish a misleading successful model/result. */
final class CliBoundaryTest {
    @TempDir Path directory;

    @ParameterizedTest
    @ValueSource(strings = {"1,2\n3,NaN\n", "1,2\n3,Infinity\n", "1,2\n3,1e39\n", "x,y\n",
            "1,2\n3,\n", "1,2\n3,4,5\n", "1,2\n\"3\",4\n", "1,2\n0x1p0,3\n", "\n \n"})
    void validatesWholeCsvBeforeAnySuccessfulCheckpoint(String content) throws Exception {
        Path csv = Files.writeString(directory.resolve("invalid.csv"), content);
        Path destination = Files.writeString(directory.resolve("existing-output"), "preserve");
        var result = run("train", "--data", csv.toString(), "--output", destination.toString(), "--plain");
        assertEquals(2, result.status(), result.error());
        assertEquals("preserve", Files.readString(destination));
        assertEquals(content, Files.readString(csv));
        assertEquals("", result.output());
    }

    @Test
    void invalidUtf8AndMissingCheckpointNeverCreateReplacementModels() throws Exception {
        Path csv = directory.resolve("invalid-utf8.csv");
        Files.write(csv, new byte[] {'1', ',', '2', '\n', (byte) 0xc3, (byte) 0x28});
        Path destination = directory.resolve("not-created");
        assertNotEquals(0, run("train", "--data", csv.toString(), "--output", destination.toString(), "--plain").status());
        assertFalse(Files.exists(destination));
        Files.writeString(csv, "1,2\n");
        assertNotEquals(0, run("train", "--data", csv.toString(), "--model", directory.resolve("missing").toString(),
                "--output", destination.toString(), "--plain").status());
        assertFalse(Files.exists(destination));
    }

    @Test
    void rejectedLaterOptimizerTransactionDoesNotSaveEarlierCommittedStepsAsSuccess() throws Exception {
        Path initial = zeroLinear("initial", 1);
        byte[] original = Files.readAllBytes(initial);
        Path destination = Files.writeString(directory.resolve("destination"), "old successful artifact");
        Path csv = Files.writeString(directory.resolve("overflow.csv"), "0,1\n1e30,1e30\n");
        var result = run("train", "--model", initial.toString(), "--data", csv.toString(),
                "--output", destination.toString(), "--epochs", "1", "--plain");
        assertNotEquals(0, result.status());
        assertEquals("old successful artifact", Files.readString(destination));
        assertArrayEquals(original, Files.readAllBytes(initial));
        assertEquals("", result.output());
    }

    @Test
    void partialStaticBatchesRejectBothCommandsWithoutReplacingDestinations() throws Exception {
        Path model = zeroLinear("batch-two", 2);
        byte[] original = Files.readAllBytes(model);
        Path data = Files.writeString(directory.resolve("three.csv"), "0,1\n1,2\n2,3\n");
        var training = run("train", "--model", model.toString(), "--data", data.toString(),
                "--output", model.toString(), "--plain");
        assertEquals(2, training.status(), training.error());
        assertArrayEquals(original, Files.readAllBytes(model));
        Path features = Files.writeString(directory.resolve("three-x.csv"), "0\n1\n2\n");
        Path destination = Files.writeString(directory.resolve("prediction.csv"), "keep");
        var prediction = run("predict", "--model", model.toString(), "--data", features.toString(),
                "--output", destination.toString());
        assertEquals(2, prediction.status(), prediction.error());
        assertEquals("keep", Files.readString(destination));
        assertEquals("", prediction.output());
    }

    @Test
    void aliasesCannotOverwriteTrainingDataOrPredictionInputs() throws Exception {
        Path csv = Files.writeString(directory.resolve("train.csv"), "0,1\n");
        Path hard = directory.resolve("hard.csv");
        Path symbolic = directory.resolve("symbolic.csv");
        Files.createLink(hard, csv);
        Files.createSymbolicLink(symbolic, csv.getFileName());
        for (Path alias : List.of(csv, directory.resolve("child/../train.csv"), hard, symbolic)) {
            var result = run("train", "--data", csv.toString(), "--output", alias.toString(), "--plain");
            assertEquals(2, result.status(), result.error());
            assertEquals("0,1\n", Files.readString(csv));
        }
        Path model = zeroLinear("predict-model", 1);
        Path modelAlias = directory.resolve("model-hardlink");
        Files.createLink(modelAlias, model);
        byte[] modelBytes = Files.readAllBytes(model);
        Path features = Files.writeString(directory.resolve("predict.csv"), "4\n");
        Path dataAlias = directory.resolve("predict-link");
        Files.createSymbolicLink(dataAlias, features.getFileName());
        for (Path alias : List.of(model, modelAlias, features, dataAlias)) {
            var result = run("predict", "--model", model.toString(), "--data", features.toString(),
                    "--output", alias.toString());
            assertEquals(2, result.status(), result.error());
            assertArrayEquals(modelBytes, Files.readAllBytes(model));
            assertEquals("4\n", Files.readString(features));
        }
    }

    @Test
    void sameModelPathIsReplacedOnlyAfterSuccessfulTraining() throws Exception {
        Path model = zeroLinear("replace", 1);
        byte[] old = Files.readAllBytes(model);
        var result = run("train", "--model", model.toString(), "--data",
                Files.writeString(directory.resolve("replace.csv"), "0,1\n").toString(),
                "--output", model.toString(), "--epochs", "1", "--plain");
        assertEquals(0, result.status(), result.error());
        assertFalse(java.util.Arrays.equals(old, Files.readAllBytes(model)));
        var prediction = run("predict", "--model", model.toString(), "--data",
                Files.writeString(directory.resolve("replace-x.csv"), "0\n").toString());
        assertEquals(0, prediction.status(), prediction.error());
        assertEquals(0.02f, Float.parseFloat(prediction.output().strip()), 1e-7);
    }

    @ParameterizedTest
    @ValueSource(strings = {"epochs=0", "batch-size=0", "outputs=0", "learning-rate=NaN", "learning-rate=1e-100",
            "optimizer=bogus", "refresh-ms=0", "backend=cuda", "metal-library=/tmp/unused", "onednn-api=graph"})
    void invalidOptionsDoNotModifyExistingOutput(String invalid) throws Exception {
        Path data = Files.writeString(directory.resolve("valid.csv"), "0,1\n");
        Path destination = Files.writeString(directory.resolve("unchanged"), "original");
        var result = run("train", "--data", data.toString(), "--output", destination.toString(), "--" + invalid, "--plain");
        assertEquals(2, result.status(), result.error());
        assertEquals("original", Files.readString(destination));
        assertEquals("0,1\n", Files.readString(data));
    }

    @Test
    void loadedOptionsCannotSilentlyChangeGeometryOrPretendToReseed() throws Exception {
        Path model = zeroLinear("fixed", 2);
        Path data = Files.writeString(directory.resolve("fixed.csv"), "0,1\n1,2\n");
        Path output = directory.resolve("unused");
        for (String option : List.of("--batch-size=1", "--outputs=2", "--seed=1", "--unknown=1", "--epochs=1 --epochs=2")) {
            String[] extras = option.split(" ");
            var args = new java.util.ArrayList<>(List.of("train", "--model", model.toString(), "--data", data.toString(),
                    "--output", output.toString(), "--plain"));
            args.addAll(List.of(extras));
            assertEquals(2, run(args.toArray(String[]::new)).status());
            assertFalse(Files.exists(output));
        }
    }

    @Test
    void incompatibleSourceSignaturesRejectRatherThanInventingAnAlternateModel() throws Exception {
        Path csv = Files.writeString(directory.resolve("one.csv"), "1\n");
        Tensor input = TensorFactory.create(CliModel.f32(false, 1, 1));
        Tensor second = TensorFactory.create(CliModel.f32(false, 1, 1));
        Tensor scalar = TensorFactory.scalar(2f, Optional.empty(), true);
        var parameter = new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, scalar);
        var arity = new GraphCheckpoint.Source(List.of(input, second), List.of(input.mul(scalar)), List.of(parameter));
        Tensor vector = TensorFactory.create(CliModel.f32(false, 1));
        var rank = new GraphCheckpoint.Source(List.of(vector), List.of(vector.mul(scalar)), List.of(parameter));
        Tensor doubles = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, Shape.of(1, 1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(1, 1))), false));
        Tensor offset = TensorFactory.scalar(2d, Optional.empty(), false);
        var dtype = new GraphCheckpoint.Source(List.of(doubles), List.of(doubles.add(offset)),
                List.of(new GraphCheckpoint.Leaf("offset", GraphCheckpoint.Role.CONSTANT, offset)));
        int index = 0;
        for (var source : List.of(arity, rank, dtype)) {
            Path model = directory.resolve("unsupported-" + index++);
            GraphCheckpoint.save(model, source, CliModel.LIMITS);
            byte[] bytes = Files.readAllBytes(model);
            Path destination = Files.writeString(directory.resolve("out-" + index), "keep");
            var prediction = run("predict", "--model", model.toString(), "--data", csv.toString(),
                    "--output", destination.toString());
            assertEquals(2, prediction.status(), prediction.error());
            assertEquals("keep", Files.readString(destination));
            assertArrayEquals(bytes, Files.readAllBytes(model));
        }
    }

    @Test
    void wrongPredictionWidthLeavesExistingFileAndStdoutUntouched() throws Exception {
        Path model = zeroLinear("width", 1);
        Path csv = Files.writeString(directory.resolve("wrong.csv"), "1\n2,3\n");
        Path output = Files.writeString(directory.resolve("predictions"), "before");
        var result = run("predict", "--model", model.toString(), "--data", csv.toString(), "--output", output.toString());
        assertEquals(2, result.status(), result.error());
        assertEquals("before", Files.readString(output));
        assertEquals("", result.output());
    }

    @Test
    void stdoutWriteFailureIsNotReportedAsSuccessfulPrediction() throws Exception {
        Path model = zeroLinear("stdout", 1);
        Path csv = Files.writeString(directory.resolve("stdout.csv"), "1\n");
        PrintStream broken = new PrintStream(new OutputStream() {
            @Override public void write(int value) throws IOException { throw new IOException("broken pipe"); }
        });
        int result = SynaptikCli.run(new String[] {"predict", "--model", model.toString(), "--data", csv.toString()},
                broken, new PrintStream(new ByteArrayOutputStream()));
        assertEquals(1, result);
    }

    @Test
    void knownBrokenProgressCannotTrainAndPublishAnApparentlySuccessfulModel() throws Exception {
        Path csv = Files.writeString(directory.resolve("broken-progress.csv"), "0,1\n");
        Path output = directory.resolve("must-not-publish");
        try (PrintStream broken = new PrintStream(new OutputStream() {
            @Override public void write(int value) throws IOException { throw new IOException("closed diagnostics"); }
        })) {
            int status = SynaptikCli.run(new String[] {"train", "--data", csv.toString(),
                    "--output", output.toString(), "--epochs", "1", "--plain"},
                    new PrintStream(new ByteArrayOutputStream()), broken);
            assertEquals(1, status);
            assertFalse(Files.exists(output));
            assertEquals("0,1\n", Files.readString(csv));
        }
    }

    private Path zeroLinear(String name, int batch) throws Exception {
        Tensor weight = TensorFactory.fromFlatArray(CliModel.f32(true, 1, 1), Optional.empty(), new float[] {0});
        Tensor bias = TensorFactory.fromFlatArray(CliModel.f32(true, 1), Optional.empty(), new float[] {0});
        Path path = directory.resolve(name);
        try (var model = Model.builder().input(DataType.FLOAT32, batch, 1).linear(weight, bias).build()) {
            model.save(path);
        }
        return path;
    }
}
