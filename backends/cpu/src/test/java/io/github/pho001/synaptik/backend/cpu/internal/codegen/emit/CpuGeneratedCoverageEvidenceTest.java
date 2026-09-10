package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Keeps retained performance evidence fail-closed when its immutable protocol did not pass. */
class CpuGeneratedCoverageEvidenceTest {
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";

    @Test void retainedLossFailureIsExactIncompleteAndNeverSelectable() throws Exception {
        Map<String, String[]> ledger = parseLedger(resource("generated-coverage-evidence-ledger.tsv"));
        assertEquals(8, ledger.size());
        assertArrayEquals(new String[] {"cpu-0009-inventory-17643", "CURRENT", "EXACT_EXECUTION_FIXTURE_MATRIX", "17643", "0",
                "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "CHECKED_SELECTED",
                "generated-coverage-inventory.tsv:17470-generated-plus-173-rejected-exact-owner-evidence-keys"}, ledger.get("cpu-0009-inventory-17643"));
        assertArrayEquals(new String[] {"cpu-0009c-oracle-proved-2252", "ORACLE_PROVED", "EXACT_CLEAN_JAVA_STRUCTURAL_ORACLE", "2252", "0",
                "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "CHECKED_SELECTED",
                "oracle-0009c-*-v1;exact-affine-movement-indexing-scatter-random-owner-selectors"}, ledger.get("cpu-0009c-oracle-proved-2252"));
        assertArrayEquals(new String[] {"cpu-0009-oracle-partial-15218", "PARTIAL", "EXACT_EXECUTION_FIXTURE_MATRIX", "15218", "15218",
                "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_SELECTED",
                "oracle-exact-fixture-classfile-v1;all-other-generated-owners-remain-unproved"}, ledger.get("cpu-0009-oracle-partial-15218"));
        assertArrayEquals(new String[] {"cpu-0009-performance-partial-17470", "PARTIAL", "EXACT_EXECUTION_FIXTURE_MATRIX", "17470", "17470",
                "NOT_TIMED_BY_0009", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_SELECTED",
                "performance-exact-fixture-unmeasured-v1;no-representative-benchmark-projection"}, ledger.get("cpu-0009-performance-partial-17470"));
        assertArrayEquals(new String[] {"cpu-0008i-loss-792", "NON_PASSING", "INCOMPLETE_FIVE_FORK", "792", "19",
                "1.3861164205039096", "536", "INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS-BFLOAT16-INT32-MEAN-false-roles0_1-0",
                "e0bef60ca1d84e31c8fc8d712cdd1accf365ba9be07e17315ae6198da28b8348", "FAIL_CLOSED",
                "/private/tmp/synaptik-cpu-0009-PIHSrD;fork-0-only;no-seal-forged"}, ledger.get("cpu-0008i-loss-792"));
        assertArrayEquals(new String[] {"cpu-0008q1-finite-basis", "REPRESENTATIVE_ONLY", "SEALED_FINITE_MATRIX", "256", "0",
                "NOT_TIMED_BY_0009", "NOT_APPLICABLE", "NOT_APPLICABLE", "01c4c8b1a077c5a0acaee59c50210ba5d9b5b730d760c45d17d31267a33a7cd9",
                "NOT_SELECTED", "CPU-0008Q1-schema-64-member-provenance-only"}, ledger.get("cpu-0008q1-finite-basis"));
        assertArrayEquals(new String[] {"cpu-0008o-stable-vector", "NON_PASSING", "FORK_0_ONLY", "180", "178", "NOT_PASSING",
                "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_RECORDED", "KEEP_SCALAR", "CPU-0008O-retained"}, ledger.get("cpu-0008o-stable-vector"));
        assertArrayEquals(new String[] {"cpu-0008p-partial-reduction", "NON_PASSING", "PRE_SEAL", "0", "0", "NOT_PASSING",
                "NOT_APPLICABLE", "NOT_APPLICABLE", "NOT_RECORDED", "KEEP_WHOLE_CELL", "CPU-0008P-retained"}, ledger.get("cpu-0008p-partial-reduction"));
        assertTrue(ledger.values().stream().noneMatch(row -> "PASSING".equals(row[1])));
    }

    @Test void gapMatrixIsCanonicalFailClosedJson() throws Exception {
        String json = resource("generated-coverage-gap-matrix.json");
        assertTrue(json.endsWith("\n") && !json.contains("\r"), "canonical LF JSON");
        Map<String, Object> root = object(new JsonReader(json).parse());
        assertEquals(List.of("schema", "status", "materializationCandidates", "materializationsSelected", "inventory", "gaps", "nonPassing", "projections"), List.copyOf(root.keySet()));
        assertEquals(65L, root.get("schema"));
        assertEquals("FAIL_CLOSED", root.get("status"));
        assertEquals(4L, root.get("materializationCandidates"));
        assertEquals(0L, root.get("materializationsSelected"));
        Map<String, Object> inventory = object(root.get("inventory"));
        assertEquals(Map.of("rows", 17643L, "current", 17643L, "partialOracle", 17470L,
                "partialPerformance", 17470L, "nonPassing", 3L, "status", "CHECKED"), inventory);
        Map<String, Object> gap = object(list(root.get("gaps")).getFirst());
        assertEquals(List.of("unit", "family", "form", "reason", "root", "rawFork0Sha256", "followUp"), List.copyOf(gap.keySet()));
        assertEquals("cpu-0008i-loss-792", gap.get("unit"));
        assertEquals("loss", gap.get("family"));
        assertEquals("scalar-generated-loss-792", gap.get("form"));
        assertEquals("fresh fork 0 has 19 ratios above 1.15 and the mandatory five-fork protocol stopped after fork 0", gap.get("reason"));
        assertEquals("/private/tmp/synaptik-cpu-0009-PIHSrD", gap.get("root"));
        assertEquals("e0bef60ca1d84e31c8fc8d712cdd1accf365ba9be07e17315ae6198da28b8348", gap.get("rawFork0Sha256"));
        assertEquals("Draft a narrow CPU loss performance remediation; preserve scalar selection until it seals five forks.", gap.get("followUp"));
        assertEquals(3, list(root.get("gaps")).size());
        List<Object> nonPassing = list(root.get("nonPassing"));
        assertEquals(2, nonPassing.size());
        assertEquals(List.of("unit", "selection"), List.copyOf(object(nonPassing.get(0)).keySet()));
        assertEquals("cpu-0008o-stable-vector", object(nonPassing.get(0)).get("unit"));
        assertEquals("KEEP_SCALAR", object(nonPassing.get(0)).get("selection"));
        assertEquals(List.of("unit", "selection"), List.copyOf(object(nonPassing.get(1)).keySet()));
        assertEquals("cpu-0008p-partial-reduction", object(nonPassing.get(1)).get("unit"));
        assertEquals("KEEP_WHOLE_CELL", object(nonPassing.get(1)).get("selection"));
        assertEquals("finite schema-64 member hashes only; no arbitrary-constant or fusion-topology projection",
                object(root.get("projections")).get("cpu-0008q1"));
    }

    private String resource(String name) throws Exception {
        try (InputStream stream = getClass().getResourceAsStream(BASE + name)) {
            assertNotNull(stream, name);
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static Map<String, String[]> parseLedger(String text) {
        assertTrue(text.endsWith("\n") && !text.contains("\r"));
        String[] lines = text.split("\\n", -1);
        assertEquals("unit\tdisposition\tprotocol\trows\tfailures\tmaximum-ratio\tmaximum-row\tmaximum-key\traw-fork-0-sha256\tselection\tprovenance", lines[0]);
        assertEquals(10, lines.length);
        Map<String, String[]> rows = new LinkedHashMap<>();
        for (int line = 1; line < lines.length - 1; line++) {
            String[] row = lines[line].split("\\t", -1);
            assertEquals(11, row.length, "ledger line " + (line + 1));
            for (String field : row) assertFalse(field.isBlank(), "blank ledger field at line " + (line + 1));
            assertTrue(List.of("CURRENT", "PARTIAL", "ORACLE_PROVED", "REPRESENTATIVE_ONLY", "NON_PASSING").contains(row[1]), row[0]);
            assertTrue(List.of("CHECKED_SELECTED", "NOT_SELECTED", "FAIL_CLOSED", "KEEP_SCALAR", "KEEP_WHOLE_CELL").contains(row[9]), row[0]);
            assertNull(rows.put(row[0], row), "duplicate ledger unit: " + row[0]);
        }
        assertEquals(List.of("cpu-0009-inventory-17643", "cpu-0009c-oracle-proved-2252", "cpu-0009-oracle-partial-15218", "cpu-0009-performance-partial-17470", "cpu-0008q1-finite-basis", "cpu-0008i-loss-792", "cpu-0008o-stable-vector", "cpu-0008p-partial-reduction"), List.copyOf(rows.keySet()));
        return rows;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> object(Object value) {
        assertInstanceOf(Map.class, value);
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        assertInstanceOf(List.class, value);
        return (List<Object>) value;
    }

    /** Parses the fixed JSON resource without adding a general JSON dependency. */
    private static final class JsonReader {
        private final String text;
        private int index;

        private JsonReader(String text) { this.text = text; }

        private Object parse() {
            Object value = value();
            if (index != text.length() - 1 || text.charAt(index) != '\n') throw new IllegalArgumentException("trailing JSON content");
            return value;
        }

        private Object value() {
            return switch (text.charAt(index)) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                default -> number();
            };
        }

        private Map<String, Object> object() {
            expect('{'); Map<String, Object> result = new LinkedHashMap<>();
            if (take('}')) return result;
            do { String key = string(); expect(':'); if (result.put(key, value()) != null) throw new IllegalArgumentException("duplicate JSON key"); } while (take(','));
            expect('}'); return result;
        }

        private List<Object> array() {
            expect('['); List<Object> result = new java.util.ArrayList<>();
            if (take(']')) return result;
            do result.add(value()); while (take(','));
            expect(']'); return result;
        }

        private String string() {
            expect('"'); StringBuilder result = new StringBuilder();
            while (index < text.length() && text.charAt(index) != '"') {
                char next = text.charAt(index++);
                if (next == '\\') throw new IllegalArgumentException("JSON escapes are not part of this fixed schema");
                result.append(next);
            }
            expect('"'); return result.toString();
        }

        private Long number() {
            int start = index;
            while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
            if (start == index) throw new IllegalArgumentException("expected JSON value");
            return Long.valueOf(text.substring(start, index));
        }

        private boolean take(char expected) {
            if (index < text.length() && text.charAt(index) == expected) { index++; return true; }
            return false;
        }

        private void expect(char expected) {
            if (!take(expected)) throw new IllegalArgumentException("expected '" + expected + "' at " + index);
        }
    }
}
