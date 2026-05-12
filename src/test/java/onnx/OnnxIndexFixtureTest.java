package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class OnnxIndexFixtureTest {
    @Test
    void checkedInIndexFixturesMatchGeneratedModelsAndExecute() throws Exception {
        for (OnnxIndexFixtureModels.Fixture fixture : OnnxIndexFixtureModels.fixtures().values()) {
            Path resource = resourcePath(fixture.fileName());
            byte[] generated = fixture.model().proto().toByteArray();
            byte[] checkedIn = Files.readAllBytes(resource);
            assertArrayEquals(generated, checkedIn, fixture.fileName() + " must match the programmatic fixture builder.");

            ImportedOnnxModel imported = Onnx.read(resource);
            applyInputs(imported, fixture.inputs());
            imported.compile(fixture.outputName(), CompileConfig.inference())
                    .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

            assertArrayEquals(fixture.expectedShape(), imported.output(fixture.outputName()).getShape(), fixture.fileName());
            assertArrayEquals(fixture.expectedOutput(), imported.output(fixture.outputName()).toDoubleArrayCopy(), 1e-6, fixture.fileName());
        }
    }

    private static Path resourcePath(String fileName) throws Exception {
        URL resource = Thread.currentThread().getContextClassLoader().getResource("onnx/index/" + fileName);
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
