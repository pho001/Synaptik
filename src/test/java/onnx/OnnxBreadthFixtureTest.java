package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class OnnxBreadthFixtureTest {
    @Test
    void checkedInBreadthFixturesMatchGeneratedModelsAndExecute() throws Exception {
        for (OnnxBreadthFixtureModels.Fixture fixture : OnnxBreadthFixtureModels.fixtures().values()) {
            Path resource = resourcePath(fixture.fileName());
            byte[] generated = fixture.model().proto().toByteArray();
            byte[] checkedIn = Files.readAllBytes(resource);
            assertArrayEquals(generated, checkedIn, fixture.fileName() + " must match the programmatic fixture builder.");

            ImportedOnnxModel imported = Onnx.read(resource);
            applyInputs(imported, fixture.inputs());
            for (Map.Entry<String, OnnxBreadthFixtureModels.ExpectedOutput> entry : fixture.outputs().entrySet()) {
                String outputName = entry.getKey();
                OnnxBreadthFixtureModels.ExpectedOutput expected = entry.getValue();
                imported.compile(outputName, CompileConfig.inference())
                        .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

                assertEquals(expected.dataType(), imported.output(outputName).getDataType(), fixture.fileName() + ":" + outputName);
                assertArrayEquals(expected.shape(), imported.output(outputName).getShape(), fixture.fileName() + ":" + outputName);
                assertArrayEquals(expected.values(), imported.output(outputName).toDoubleArrayCopy(), 1e-6, fixture.fileName() + ":" + outputName);
            }
        }
    }

    private static Path resourcePath(String fileName) throws Exception {
        URL resource = Thread.currentThread().getContextClassLoader().getResource("onnx/breadth/" + fileName);
        assertNotNull(resource, "Missing ONNX fixture resource: " + fileName);
        return Path.of(resource.toURI());
    }

    private static void applyInputs(ImportedOnnxModel imported, Map<String, Object> inputs) {
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof float[] floats) {
                imported.input(entry.getKey()).setData(floats);
            } else if (value instanceof int[] ints) {
                imported.input(entry.getKey()).setData(ints);
            } else {
                throw new IllegalArgumentException("Unsupported fixture input type for " + entry.getKey() + ": " + value.getClass());
            }
        }
    }
}
