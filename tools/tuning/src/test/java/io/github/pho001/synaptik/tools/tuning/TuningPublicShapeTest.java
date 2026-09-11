package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TuningPublicShapeTest {
    @Test
    void exposesExactlyFivePublicTopLevelProductionTypesAndOnePackagePrivateCache() throws Exception {
        List<Class<?>> publicTypes = List.of(
                WorkloadTuning.class,
                WorkloadTuningRequest.class,
                BackendWorkloadTuning.class,
                ColdCandidateMeasurement.class,
                WorkloadTuningResult.class);

        assertAll(
                () -> publicTypes.forEach(type -> assertTrue(Modifier.isPublic(type.getModifiers()))),
                () -> assertFalse(Modifier.isPublic(WorkloadCacheFile.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(WorkloadTuning.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(WorkloadTuningRequest.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(WorkloadTuningResult.class.getModifiers())),
                () -> assertTrue(BackendWorkloadTuning.class.isInterface()),
                () -> assertTrue(ColdCandidateMeasurement.class.isInterface()),
                () -> assertEquals(6, BackendWorkloadTuning.class.getDeclaredMethods().length),
                () -> assertEquals(1, ColdCandidateMeasurement.class.getDeclaredMethods().length));
    }

    @Test
    void genericBoundsRemainTheMethodFreePrepareRoles() throws Exception {
        var collaborationParameters = BackendWorkloadTuning.class.getTypeParameters();
        var measurementParameters = ColdCandidateMeasurement.class.getTypeParameters();
        var tune = WorkloadTuning.class.getDeclaredMethod(
                "tune", WorkloadTuningRequest.class, BackendWorkloadTuning.class,
                ColdCandidateMeasurement.class);

        assertAll(
                () -> assertEquals(BackendTuningCandidateBatch.class,
                        collaborationParameters[0].getBounds()[0]),
                () -> assertEquals(BackendTuningDecision.class,
                        collaborationParameters[1].getBounds()[0]),
                () -> assertEquals(BackendTuningCandidateBatch.class,
                        measurementParameters[0].getBounds()[0]),
                () -> assertTrue(Modifier.isPublic(tune.getModifiers())),
                () -> assertTrue(Modifier.isStatic(tune.getModifiers())),
                () -> assertEquals(3, tune.getTypeParameters().length),
                () -> assertEquals(BackendTuningCandidateBatch.class,
                        tune.getTypeParameters()[0].getBounds()[0]),
                () -> assertEquals(BackendTuningDecision.class,
                        tune.getTypeParameters()[1].getBounds()[0]),
                () -> assertEquals(WorkloadTuningResult.class,
                        ((ParameterizedType) tune.getGenericReturnType()).getRawType()));
    }

    @Test
    void productionImportsContainNoForbiddenLayerOrConcreteBackend() throws Exception {
        Path source = Path.of("src/main/java/io/github/pho001/synaptik/tools/tuning");
        Set<String> text;
        try (var files = Files.list(source)) {
            text = files.filter(path -> path.toString().endsWith(".java"))
                    .map(path -> {
                        try { return Files.readString(path); }
                        catch (java.io.IOException exception) { throw new RuntimeException(exception); }
                    })
                    .collect(Collectors.toSet());
        }
        String joined = String.join("\n", text);
        assertAll(
                () -> assertFalse(joined.contains("synaptik.backend.cpu")),
                () -> assertFalse(joined.contains("synaptik.runtime")),
                () -> assertFalse(joined.contains("synaptik.engine")),
                () -> assertFalse(joined.contains("synaptik.config")),
                () -> assertFalse(joined.contains("java.io.Serializable")),
                () -> assertFalse(joined.contains("java.lang.reflect")));
    }
}
