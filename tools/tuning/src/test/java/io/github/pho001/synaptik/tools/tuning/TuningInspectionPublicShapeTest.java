package io.github.pho001.synaptik.tools.tuning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

final class TuningInspectionPublicShapeTest {
    private static final Set<String> NESTED_TYPES = Set.of(
            "ArtifactStatus", "ArtifactSource", "InvalidReason", "CompatibilityStatus",
            "MismatchReason", "OpaqueSummary", "VersionedOpaqueSummary", "WorkloadExpectation",
            "ModelPlanExpectation", "WorkloadCacheInspection", "WorkloadEntryInspection",
            "ModelPlanCacheInspection", "ModelPlanEntryInspection", "OccurrenceEvidenceSummary",
            "CandidateMeasurementSummary", "WorkloadEvidenceSummary", "WorkloadEvidenceInspection",
            "CompleteCandidateMeasurementSummary", "CompletePlanEvidenceInspection");

    @Test
    void namespaceHasExactPublicShape() {
        assertTrue(Modifier.isPublic(TuningInspection.class.getModifiers()));
        assertTrue(Modifier.isFinal(TuningInspection.class.getModifiers()));
        assertEquals(0, Arrays.stream(TuningInspection.class.getConstructors()).count());
        assertEquals(NESTED_TYPES,
                Arrays.stream(TuningInspection.class.getDeclaredClasses())
                        .filter(type -> Modifier.isPublic(type.getModifiers()))
                        .map(Class::getSimpleName)
                        .collect(Collectors.toSet()));
        for (Class<?> type : TuningInspection.class.getDeclaredClasses()) {
            if (Modifier.isPublic(type.getModifiers())) {
                assertTrue(type.isEnum() || type.isRecord() || Modifier.isFinal(type.getModifiers()),
                        type.getName());
            }
        }
        var operations = Arrays.stream(TuningInspection.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> Modifier.isStatic(method.getModifiers()))
                .toList();
        assertEquals(10, operations.size());
        assertEquals(8, operations.stream().filter(method -> method.getName().startsWith("inspect"))
                .count());
        assertEquals(2, operations.stream().filter(method -> method.getName().equals("summarize"))
                .count());
        assertEquals(0, TuningInspection.class.getDeclaredFields().length);
    }

    @Test
    void productionSourceContainsNoForbiddenLayerOrMechanism() throws Exception {
        Path source = Path.of("src/main/java/io/github/pho001/synaptik/tools/tuning/TuningInspection.java");
        String text = Files.readString(source);
        for (String forbidden : Set.of(
                "synaptik.engine", "synaptik.runtime", "synaptik.config", "synaptik.compiler",
                "synaptik.backend.cpu", "java.lang.reflect", "ObjectInputStream",
                "ObjectOutputStream", "Map<String, Object>", "ServiceLoader", "Registry")) {
            assertFalse(text.contains(forbidden), forbidden);
        }
        for (String mutating : Set.of("StandardOpenOption.WRITE", "StandardOpenOption.CREATE",
                "Files.move", "Files.delete", ".force(", "createTemp")) {
            assertFalse(text.contains(mutating), mutating);
        }
    }
}
