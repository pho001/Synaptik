package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Binds every current generated pointwise inventory row to an executable semantic owner.
 *
 * <p>The manifest is deliberately an inventory projection, not a capability declaration. Its
 * digest and counts make an added, removed, relabelled, or stale generated row fail closed until
 * the applicable direct-entry oracle is explicitly assigned.</p>
 */
class CpuPointwiseSemanticClosureManifestTest {
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";
    private static final String INVENTORY = "generated-coverage-inventory.tsv";
    private static final String MANIFEST = "generated-pointwise-semantic-closure.tsv";
    private static final String INVENTORY_SHA256 =
            "1dcb69796c00fe3793d86f3f4cc3e816176062a45312ddbbaadfba9f8036cf20";

    @Test void manifestIsCurrentExactAndAccountsForEveryGeneratedPointwiseRow() throws Exception {
        Closure closure = parseManifest(resource(MANIFEST));
        String inventory = resource(INVENTORY);
        assertEquals(INVENTORY_SHA256, sha256(inventory));
        assertEquals(INVENTORY_SHA256, closure.inventorySha256());

        Map<CpuPointwiseOpcode, Integer> actual = generatedPointwiseCounts(inventory);
        assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class), actual.keySet());
        assertEquals(actual, closure.counts(), "orphan, stale, or unowned generated pointwise row");
        assertEquals(1_108, actual.values().stream().mapToInt(Integer::intValue).sum());
        assertEquals(845, closure.countFor("pointwise") + closure.countFor("cast"));
        assertEquals(263, closure.countFor("scalar-immediate"));
        assertEquals(1_108, closure.declaredGeneratedRows());
    }

    @Test void manifestParserFailsClosedForDuplicateOrphanStaleAndMutatedEntries() throws Exception {
        String canonical = resource(MANIFEST);
        assertThrows(AssertionError.class, () -> parseManifest(canonical.replaceFirst(
                "# source-inventory-sha256=[0-9a-f]{64}", "# source-inventory-sha256=" + "0".repeat(64))));
        assertThrows(AssertionError.class, () -> parseManifest(canonical.replaceFirst(
                "ABS\\t15\\tpointwise", "ABS\\t16\\tpointwise")));
        assertThrows(AssertionError.class, () -> parseManifest(canonical.replaceFirst(
                "WHERE\\t15\\tpointwise\\n", "WHERE\\t15\\tpointwise\\nWHERE\\t15\\tpointwise\\n")));
        assertThrows(AssertionError.class, () -> parseManifest(canonical.replaceFirst(
                "ABS\\t15\\tpointwise", "UNKNOWN\\t15\\tpointwise")));
        assertThrows(AssertionError.class, () -> parseManifest(canonical.replaceFirst(
                "CAST\\t180\\tcast", "CAST\\t180\\tunknown-owner")));
    }

    @Test void everyManifestOwnerExecutesItsIndependentGeneratedEntryOracle() throws Throwable {
        // These are direct generated-entry tests, not registry or Class-File-only evidence.
        new CpuPointwiseGeneratedKernelTest().generatedAndReferenceAgreeForEveryAdmittedOpcodeAndType();
        new CpuCastGeneratedKernelTest().generatedTypedCastMatrixCoversAll1296LegalCells();
        var scalar = new CpuScalarImmediateClampMatrixTest();
        scalar.everyExactArtifactExecutesAgainstItsIndependentTypedCleanJavaLoop();
        scalar.everyCompositionalFormExecutesAgainstItsResolvedLayoutOracle();
    }

    private static Map<CpuPointwiseOpcode, Integer> generatedPointwiseCounts(String inventory) {
        String[] lines = inventory.split("\\n", -1);
        assertTrue(lines.length > 1 && lines[0].startsWith("owner-id\tevidence-key\toperation-form\t"));
        Map<CpuPointwiseOpcode, Integer> counts = new EnumMap<>(CpuPointwiseOpcode.class);
        for (int index = 1; index < lines.length - 1; index++) {
            String[] row = lines[index].split("\\t", -1);
            assertEquals(27, row.length, "inventory line " + (index + 1));
            if (!row[21].equals("GENERATED")) continue;
            try {
                CpuPointwiseOpcode opcode = CpuPointwiseOpcode.valueOf(row[2]);
                counts.merge(opcode, 1, Integer::sum);
            } catch (IllegalArgumentException ignored) {
                // Other generated families are intentionally outside this pointwise closure.
            }
        }
        return Map.copyOf(counts);
    }

    private static Closure parseManifest(String text) {
        assertTrue(text.endsWith("\n") && !text.contains("\r"), "canonical LF manifest");
        String[] lines = text.split("\\n", -1);
        assertEquals("# source-inventory-sha256=" + INVENTORY_SHA256, lines[0]);
        assertEquals("# source-inventory-generated-rows=1108", lines[1]);
        assertEquals("operation\traw-generated-rows\tsemantic-owner", lines[2]);
        Map<CpuPointwiseOpcode, Integer> counts = new EnumMap<>(CpuPointwiseOpcode.class);
        Map<CpuPointwiseOpcode, String> owners = new EnumMap<>(CpuPointwiseOpcode.class);
        for (int index = 3; index < lines.length - 1; index++) {
            String[] row = lines[index].split("\\t", -1);
            assertEquals(3, row.length, "manifest line " + (index + 1));
            CpuPointwiseOpcode opcode;
            try {
                opcode = CpuPointwiseOpcode.valueOf(row[0]);
            } catch (IllegalArgumentException failure) {
                throw new AssertionError("orphan manifest operation " + row[0], failure);
            }
            assertTrue(Integer.parseInt(row[1]) > 0, "non-positive row count for " + opcode);
            assertTrue(EnumSet.of(CpuPointwiseOpcode.SCALAR_ADD, CpuPointwiseOpcode.SCALAR_SUB,
                    CpuPointwiseOpcode.SCALAR_MUL, CpuPointwiseOpcode.SCALAR_DIV,
                    CpuPointwiseOpcode.SCALAR_POW, CpuPointwiseOpcode.SCALAR_MIN,
                    CpuPointwiseOpcode.SCALAR_MAX, CpuPointwiseOpcode.SCALAR_CLAMP).contains(opcode)
                    ? row[2].equals("scalar-immediate") : opcode == CpuPointwiseOpcode.CAST
                    ? row[2].equals("cast") : row[2].equals("pointwise"), "wrong semantic owner for " + opcode);
            assertNull(counts.put(opcode, Integer.parseInt(row[1])), "duplicate manifest operation " + opcode);
            owners.put(opcode, row[2]);
        }
        assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class), counts.keySet(), "missing manifest operation");
        return new Closure(INVENTORY_SHA256, 1_108, Map.copyOf(counts), Map.copyOf(owners));
    }

    private static String resource(String name) throws Exception {
        try (InputStream stream = CpuPointwiseSemanticClosureManifestTest.class.getResourceAsStream(BASE + name)) {
            assertNotNull(stream, name);
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static String sha256(String text) throws Exception {
        return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(text.getBytes(StandardCharsets.UTF_8)));
    }

    private record Closure(String inventorySha256, int declaredGeneratedRows,
            Map<CpuPointwiseOpcode, Integer> counts, Map<CpuPointwiseOpcode, String> owners) {
        int countFor(String owner) {
            return counts.entrySet().stream().filter(entry -> owners.get(entry.getKey()).equals(owner))
                    .mapToInt(Map.Entry::getValue).sum();
        }
    }
}
