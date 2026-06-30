package onnx;

import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import runtime.contract.ExecutionMode;
import org.junit.jupiter.api.Test;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxCompatibilityHarnessTest {
    @Test
    void checkedInCompatibilityFixturesMatchGeneratedModelsAndExpectedStatus() throws Exception {
        for (OnnxCompatibilityFixtureModels.Fixture fixture : OnnxCompatibilityFixtureModels.fixtures().values()) {
            Path resource = resourcePath(fixture.fileName());
            assertArrayEquals(fixture.model().proto().toByteArray(), Files.readAllBytes(resource),
                    fixture.fileName() + " must match the programmatic fixture builder.");

            if (fixture.status() == OnnxCompatibilityFixtureModels.ExpectedStatus.REJECTED_WITH_REASON) {
                OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.read(resource));
                assertTrue(ex.getMessage().contains(fixture.rejectionReason()), fixture.fileName());
                continue;
            }

            ImportedOnnxModel imported = Onnx.read(resource);
            applyInputs(imported, fixture.inputs());
            if (fixture.status() == OnnxCompatibilityFixtureModels.ExpectedStatus.IMPORTED) {
                assertEquals(fixture.outputs().keySet(), imported.outputs().keySet());
                continue;
            }
            for (Map.Entry<String, OnnxCompatibilityFixtureModels.ExpectedOutput> entry : fixture.outputs().entrySet()) {
                String outputName = entry.getKey();
                OnnxCompatibilityFixtureModels.ExpectedOutput expected = entry.getValue();
                imported.compile(outputName, CompileConfig.inference())
                        .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

                assertEquals(expected.dataType(), imported.output(outputName).getDataType(), fixture.fileName() + ":" + outputName);
                assertArrayEquals(expected.shape(), imported.output(outputName).getShape(), fixture.fileName() + ":" + outputName);
                assertArrayEquals(expected.values(), imported.output(outputName).toDoubleArrayCopy(), 1e-5, fixture.fileName() + ":" + outputName);
            }
        }
    }

    private static Path resourcePath(String fileName) throws Exception {
        URL resource = Thread.currentThread().getContextClassLoader().getResource("onnx/compat/" + fileName);
        assertNotNull(resource, "Missing ONNX compat fixture resource: " + fileName);
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
