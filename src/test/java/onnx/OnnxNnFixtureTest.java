package onnx;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class OnnxNnFixtureTest {
    @Test
    void checkedInNnFixturesMatchGeneratedModelsAndExecute() throws Exception {
        for (OnnxNnFixtureModels.Fixture fixture : OnnxNnFixtureModels.fixtures().values()) {
            Path resource = resourcePath(fixture.fileName());
            byte[] generated = fixture.model().proto().toByteArray();
            byte[] checkedIn = Files.readAllBytes(resource);
            assertArrayEquals(generated, checkedIn, fixture.fileName() + " must match the programmatic fixture builder.");

            ImportedOnnxModel imported = Onnx.read(resource);
            applyInputs(imported, fixture.inputs());
            imported.compile(fixture.outputName(), CompileConfig.inference())
                    .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

            assertArrayEquals(fixture.expectedShape(), imported.output(fixture.outputName()).getShape(), fixture.fileName());
            assertArrayEquals(fixture.expectedOutput(), imported.output(fixture.outputName()).toDoubleArrayCopy(), 1e-5, fixture.fileName());
        }
    }

    private static Path resourcePath(String fileName) throws Exception {
        URL resource = Thread.currentThread().getContextClassLoader().getResource("onnx/nn/" + fileName);
        assertNotNull(resource, "Missing ONNX fixture resource: " + fileName);
        return Path.of(resource.toURI());
    }

    private static void applyInputs(ImportedOnnxModel imported, Map<String, Object> inputs) {
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof float[] floats) {
                imported.input(entry.getKey()).setData(floats);
            } else {
                throw new IllegalArgumentException("Unsupported fixture input type for " + entry.getKey() + ": " + value.getClass());
            }
        }
    }
}
