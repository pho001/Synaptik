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

class CompletePlanTuningPublicShapeTest {
    @Test
    void exposesExactlySixNewPublicTopLevelTypesAndPackagePrivateCache() {
        List<Class<?>> publicTypes =
                List.of(
                        CompletePlanTuning.class,
                        CompletePlanTuningRequest.class,
                        BackendCompletePlanTuning.class,
                        CompletePlanCorrectness.class,
                        CompleteCandidateMeasurement.class,
                        CompletePlanTuningResult.class);

        assertAll(
                () ->
                        publicTypes.forEach(
                                type ->
                                        assertTrue(
                                                Modifier.isPublic(type.getModifiers()),
                                                type.getName())),
                () -> assertFalse(Modifier.isPublic(ModelPlanCacheFile.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CompletePlanTuning.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CompletePlanTuningRequest.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CompletePlanTuningResult.class.getModifiers())),
                () -> assertEquals(6, BackendCompletePlanTuning.class.getDeclaredMethods().length),
                () -> assertEquals(2, CompletePlanCorrectness.class.getDeclaredMethods().length),
                () -> assertEquals(1, CompleteCandidateMeasurement.class.getDeclaredMethods().length));
    }

    @Test
    void publicGenericBoundsRemainPrepareRolesAndToolLocalValues() throws Exception {
        var backendParameters = BackendCompletePlanTuning.class.getTypeParameters();
        var correctnessParameters = CompletePlanCorrectness.class.getTypeParameters();
        var measurementParameters = CompleteCandidateMeasurement.class.getTypeParameters();
        var tune =
                CompletePlanTuning.class.getDeclaredMethod(
                        "tune",
                        CompletePlanTuningRequest.class,
                        BackendCompletePlanTuning.class,
                        CompletePlanCorrectness.class,
                        CompleteCandidateMeasurement.class);

        assertAll(
                () ->
                        assertEquals(
                                BackendTuningCandidateBatch.class,
                                backendParameters[0].getBounds()[0]),
                () ->
                        assertEquals(
                                BackendTuningDecision.class,
                                backendParameters[1].getBounds()[0]),
                () ->
                        assertEquals(
                                BackendTuningCandidateBatch.class,
                                correctnessParameters[0].getBounds()[0]),
                () ->
                        assertEquals(
                                BackendTuningCandidateBatch.class,
                                measurementParameters[0].getBounds()[0]),
                () -> assertTrue(Modifier.isPublic(tune.getModifiers())),
                () -> assertTrue(Modifier.isStatic(tune.getModifiers())),
                () -> assertEquals(4, tune.getTypeParameters().length),
                () ->
                        assertEquals(
                                CompletePlanTuningResult.class,
                                ((ParameterizedType) tune.getGenericReturnType()).getRawType()));
    }

    @Test
    void nestedPublicSurfaceContainsOnlyPlannedConceptsAndNoExtensibleFingerprintBase() {
        Set<String> requestNested =
                publicNestedSimpleNames(CompletePlanTuningRequest.class);
        Set<String> resultNested =
                publicNestedSimpleNames(CompletePlanTuningResult.class);
        Set<String> tuningNested = publicNestedSimpleNames(CompletePlanTuning.class);

        assertAll(
                () ->
                        assertEquals(
                                Set.of(
                                        "Objective",
                                        "CorrectnessPolicy",
                                        "ReuseScope",
                                        "ModelFingerprint",
                                        "ProfileFingerprint",
                                        "TargetFingerprint",
                                        "PolicyIdentity",
                                        "ProducerIdentity",
                                        "DecisionCodecIdentity",
                                        "CandidateIdentity",
                                        "PlanCompatibility"),
                                requestNested),
                () -> assertFalse(requestNested.contains("Fingerprint")),
                () ->
                        assertEquals(
                                Set.of(
                                        "Source",
                                        "CorrectnessAction",
                                        "SampleSummary",
                                        "CandidateEvidence",
                                        "SelectedPlanRecord",
                                        "Evidence"),
                                resultNested),
                () -> assertEquals(Set.of("CorrectnessMismatchException"), tuningNested),
                () ->
                        List.of(
                                        CompletePlanTuningRequest.ModelFingerprint.class,
                                        CompletePlanTuningRequest.ProfileFingerprint.class,
                                        CompletePlanTuningRequest.TargetFingerprint.class,
                                        CompletePlanTuningRequest.PolicyIdentity.class,
                                        CompletePlanTuningRequest.ProducerIdentity.class,
                                        CompletePlanTuningRequest.DecisionCodecIdentity.class,
                                        CompletePlanTuningRequest.CandidateIdentity.class,
                                        CompletePlanTuningRequest.PlanCompatibility.class)
                                .forEach(
                                        type ->
                                                assertTrue(
                                                        Modifier.isFinal(type.getModifiers()),
                                                        type.getName())));
    }

    @Test
    void productionHasNoForbiddenImportsOrMechanisms() throws Exception {
        Path source = Path.of("src/main/java/io/github/pho001/synaptik/tools/tuning");
        String joined;
        try (var files = Files.list(source)) {
            joined =
                    files.filter(path -> path.toString().endsWith(".java"))
                            .map(
                                    path -> {
                                        try {
                                            return Files.readString(path);
                                        } catch (java.io.IOException exception) {
                                            throw new RuntimeException(exception);
                                        }
                                    })
                            .collect(Collectors.joining("\n"));
        }

        for (String forbidden :
                List.of(
                        "synaptik.backend.cpu",
                        "synaptik.engine",
                        "synaptik.runtime",
                        "synaptik.config",
                        "java.lang.reflect",
                        "Map<String, Object>",
                        "@SuppressWarnings",
                        "java.io.Serializable")) {
            assertFalse(joined.contains(forbidden), forbidden);
        }
    }

    private static Set<String> publicNestedSimpleNames(Class<?> owner) {
        return Arrays.stream(owner.getDeclaredClasses())
                .filter(type -> Modifier.isPublic(type.getModifiers()))
                .map(Class::getSimpleName)
                .collect(Collectors.toSet());
    }
}
