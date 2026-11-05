package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** File-backed numerical commands, public checkpoint round trips and successful-step boundaries. */
final class CliRegressionTest {
    @TempDir Path directory;

    @Test
    void loadedBiasedLinearMseStepSavesActualUpdatedWeightsAndPredictsDenseRows() throws Exception {
        Path initial = linear("initial", 2, new float[] {1}, new float[] {0});
        Path data = csv("training.csv", "1,3\n2,5\n");
        Path trained = directory.resolve("trained");
        Result training = run("train", "--model", initial.toString(), "--data", data.toString(),
                "--output", trained.toString(), "--epochs", "1", "--learning-rate", "0.1", "--plain");
        assertEquals(0, training.status(), training.error());
        assertEquals("", training.output());
        Path features = csv("features.csv", "1\n2\n");
        float[][] result = predict(trained, features);
        assertEquals(2.3f, result[0][0], 1e-5);
        assertEquals(4.1f, result[1][0], 1e-5);
        assertArrayEquals(new float[] {1, 2}, flatten(predict(initial, features)), 0);
        // The public loaded owner sees exactly the values emitted by the CLI.
        try (var model = Model.load(trained)) {
            var input = TensorFactory.fromFlatArray(CliModel.f32(false, 2, 1), Optional.empty(), new float[] {1, 2});
            var values = model.compute(input).bytes();
            assertEquals(result[0][0], values.getFloat());
            assertEquals(result[1][0], values.getFloat());
        }
    }

    @Test
    void multipleTargetsUseLastColumnsAndMeanOverAllBatchOutputElements() throws Exception {
        Path initial = linear("multi", 2, new float[] {0, 0}, new float[] {0, 0});
        Path data = csv("multi.csv", "1,2,4\n2,6,8\n");
        Path trained = directory.resolve("multi-trained");
        Result training = run("train", "--model", initial.toString(), "--data", data.toString(),
                "--output", trained.toString(), "--epochs", "1", "--learning-rate", "0.1", "--plain");
        assertEquals(0, training.status(), training.error());
        // dW=[-7,-10], db=[-4,-6] for MEAN across all four residuals.
        float[][] actual = predict(trained, csv("multi-features.csv", "1\n2\n"));
        assertArrayEquals(new float[] {1.1f, 1.6f}, actual[0], 1e-5f);
        assertArrayEquals(new float[] {1.8f, 2.6f}, actual[1], 1e-5f);
        Path file = directory.resolve("predictions.csv");
        Result output = run("predict", "--model", trained.toString(), "--data",
                directory.resolve("multi-features.csv").toString(), "--output", file.toString());
        assertEquals(0, output.status(), output.error());
        assertEquals("", output.output());
        assertArrayEquals(flatten(actual), flatten(parse(Files.readString(file))), 0);
    }

    @Test
    void newSeededModelLearnsRealCsvAndRepeatingSeedReproducesPredictions() throws Exception {
        Path data = csv("line.csv", "0,1\n1,3\n2,5\n");
        Path first = directory.resolve("first");
        Path second = directory.resolve("second");
        for (Path output : List.of(first, second)) {
            Result result = run("train", "--data", data.toString(), "--output", output.toString(),
                    "--epochs", "100", "--learning-rate", "0.05", "--seed", "19", "--plain");
            assertEquals(0, result.status(), result.error());
        }
        Path features = csv("test.csv", "0\n1\n2\n3\n");
        float[] a = flatten(predict(first, features));
        assertArrayEquals(a, flatten(predict(second, features)), 0);
        assertArrayEquals(new float[] {1, 3, 5, 7}, a, 0.01f);
    }

    @Test
    void adamCommitsActualFirstMomentUpdate() throws Exception {
        Path initial = linear("adam-initial", 1, new float[] {0}, new float[] {0});
        Path trained = directory.resolve("adam-trained");
        Result result = run("train", "--model", initial.toString(), "--data", csv("adam.csv", "0,1\n").toString(),
                "--output", trained.toString(), "--epochs", "1", "--optimizer", "adam", "--plain");
        assertEquals(0, result.status(), result.error());
        assertEquals(0.001f, predict(trained, csv("adam-x.csv", "0\n"))[0][0], 1e-7);
    }

    @Test
    void successfulInfiniteObjectiveIsNotMistakenForFailedTraining() throws Exception {
        Path initial = linear("overflow-loss-initial", 1, new float[] {0}, new float[] {0});
        Path trained = directory.resolve("overflow-loss-trained");
        Result result = run("train", "--model", initial.toString(), "--data",
                csv("overflow-loss.csv", "0,2e19\n").toString(), "--output", trained.toString(),
                "--epochs", "1", "--plain");
        assertEquals(0, result.status(), result.error());
        float prediction = predict(trained, csv("overflow-loss-x.csv", "0\n"))[0][0];
        assertEquals(4e17f, prediction, 4e11f);
        assertTrue(Float.isFinite(prediction));
    }

    @Test
    void loadedInputRetainsAbsentLayoutAndGradientEligibility() throws Exception {
        var shape = Shape.of(1, 2);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), true);
        Tensor input = TensorFactory.create(descriptor);
        Tensor weight = TensorFactory.scalar(2f, Optional.empty(), true);
        Path saved = directory.resolve("unresolved-input");
        GraphCheckpoint.save(saved, new GraphCheckpoint.Source(List.of(input), List.of(input.mul(weight)),
                List.of(new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight))), CliModel.LIMITS);
        float[][] actual = predict(saved, csv("unresolved-input.csv", "3,4\n"));
        assertArrayEquals(new float[] {6, 8}, actual[0], 0);
        Path trained = directory.resolve("unresolved-trained");
        Result result = run("train", "--model", saved.toString(), "--data",
                csv("unresolved-train.csv", "3,4,0,0\n").toString(), "--output", trained.toString(),
                "--epochs", "1", "--plain");
        assertEquals(0, result.status(), result.error());
        var retained = GraphCheckpoint.load(trained, CliModel.LIMITS).inputs().getFirst().descriptor();
        assertEquals(descriptor, retained);
        // MSE=(36+64)/2; scalar weight gradient=50, w'=1.5.
        assertArrayEquals(new float[] {4.5f, 6}, predict(trained, directory.resolve("unresolved-input.csv"))[0], 1e-5f);
    }

    @Test
    void inspectReportsActualPublicMetadataEvenForNonCliPredictionGeometry() throws Exception {
        Tensor input = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, Shape.scalar(), Optional.empty(), false));
        Tensor constant = TensorFactory.scalar(3d, Optional.empty(), false);
        Path model = directory.resolve("scalar-model");
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.add(constant)),
                List.of(new GraphCheckpoint.Leaf("offset", GraphCheckpoint.Role.CONSTANT, constant)));
        GraphCheckpoint.save(model, source, CliModel.LIMITS);
        Result inspection = run("inspect", "--model", model.toString());
        assertEquals(0, inspection.status(), inspection.error());
        assertTrue(inspection.output().contains(input.descriptor().toString()));
        assertTrue(inspection.output().contains(source.outputs().getFirst().descriptor().toString()));
        assertTrue(inspection.output().contains("offset"));
        assertTrue(inspection.output().contains(GraphCheckpoint.Role.CONSTANT.toString()));
        assertEquals("", inspection.error());
    }

    private Path linear(String name, int batch, float[] weights, float[] biases) throws Exception {
        Tensor weight = TensorFactory.fromFlatArray(CliModel.f32(true, biases.length, 1), Optional.empty(), weights);
        Tensor bias = TensorFactory.fromFlatArray(CliModel.f32(true, biases.length), Optional.empty(), biases);
        Path path = directory.resolve(name);
        try (var model = Model.builder().input(DataType.FLOAT32, batch, 1).linear(weight, bias).build()) {
            model.save(path);
        }
        return path;
    }

    private Path csv(String name, String text) throws Exception {
        return Files.writeString(directory.resolve(name), text);
    }

    private float[][] predict(Path model, Path csv) {
        Result result = run("predict", "--model", model.toString(), "--data", csv.toString());
        assertEquals(0, result.status(), result.error());
        assertEquals("", result.error());
        return parse(result.output());
    }

    static Result run(String... args) {
        var output = new ByteArrayOutputStream();
        var error = new ByteArrayOutputStream();
        int status = SynaptikCli.run(args, new PrintStream(output, true, StandardCharsets.UTF_8),
                new PrintStream(error, true, StandardCharsets.UTF_8));
        return new Result(status, output.toString(StandardCharsets.UTF_8), error.toString(StandardCharsets.UTF_8));
    }

    static float[][] parse(String csv) {
        List<float[]> rows = new ArrayList<>();
        for (String line : csv.lines().toList()) {
            String[] fields = line.split(",", -1);
            float[] row = new float[fields.length];
            for (int i = 0; i < fields.length; i++) row[i] = Float.parseFloat(fields[i]);
            rows.add(row);
        }
        return rows.toArray(float[][]::new);
    }

    static float[] flatten(float[][] rows) {
        int size = 0;
        for (float[] row : rows) size += row.length;
        float[] flat = new float[size];
        int offset = 0;
        for (float[] row : rows) {
            System.arraycopy(row, 0, flat, offset, row.length);
            offset += row.length;
        }
        return flat;
    }

    record Result(int status, String output, String error) { }
}
