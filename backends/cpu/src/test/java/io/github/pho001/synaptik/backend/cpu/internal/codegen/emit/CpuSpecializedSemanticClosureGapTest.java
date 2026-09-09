package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Makes the specialized-family semantic-closure boundary reproducible.
 *
 * <p>The checked inventory may be sealed only by a named direct invocation-and-oracle owner for
 * every exact row. Its byte hash, raw-row count, prepared-behavior grouping, and owner identity
 * prevent a representative fixture from projecting coverage to an uninvoked row.</p>
 */
class CpuSpecializedSemanticClosureGapTest {
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";
    private static final Set<String> FORMS = Set.of("SCALED_DOT_PRODUCT_ATTENTION", "BATCH_NORM_TRAINING",
            "BATCH_NORM_INFERENCE", "CONV1D_COMPOSITION", "CONV2D", "CONV3D", "MATMUL",
            "MEAN_SQUARED_ERROR", "DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS",
            "INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS", "ARG_MIN", "ARG_MAX", "MASKED_SUM",
            "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM", "VARIANCE", "STANDARD_DEVIATION",
            "SOFTMAX", "LOG_SOFTMAX", "LAYER", "LAYER_AFFINE", "RMS", "RMS_SCALED",
            "AVERAGE_POOL2D", "MAX_POOL2D", "AVERAGE_POOL3D", "MAX_POOL3D");

    @Test void checkedGapManifestIsAByteExactCompleteNonProjectingJoin() throws Exception {
        String inventory = resource("generated-coverage-inventory.tsv");
        String manifest = resource("specialized-semantic-closure-gaps.tsv");
        assertTrue(inventory.endsWith("\n") && manifest.endsWith("\n"));
        assertFalse(inventory.contains("\r") || manifest.contains("\r"));
        String hash = sha256(inventory);
        Map<String, Counts> actual = inventoryCounts(inventory);
        Map<String, String[]> rows = manifestRows(manifest);
        assertEquals(FORMS, rows.keySet(), "duplicate or orphan manifest form");
        assertEquals(FORMS, actual.keySet(), "inventory scope changed without a manifest row");
        for (String form : FORMS) {
            String[] row = rows.get(form); Counts counts = actual.get(form);
            requireHash(hash, row[0]);
            assertEquals(counts.rows, Integer.parseInt(row[2]), form + " complete raw-row join");
            assertEquals(counts.groups, Integer.parseInt(row[3]), form + " exact prepared-behavior grouping");
            assertFalse(row[4].isBlank() || row[5].isBlank(), form + " concrete existing witness identity");
            if (form.equals("MATMUL")) {
                assertEquals("CpuMatmulSemanticClosureTest#everyMatmulInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals("MATMUL_390_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(390, counts.rows); assertEquals(325, counts.groups);
            } else if (form.equals("BATCH_NORM_INFERENCE") || form.equals("BATCH_NORM_TRAINING")) {
                String method = form.equals("BATCH_NORM_INFERENCE")
                        ? "everyBatchNormInferenceInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle"
                        : "everyBatchNormTrainingInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
                assertEquals("CpuBatchNormSemanticClosureTest#" + method, row[4]);
                assertEquals(form + "_2430_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(2430, counts.rows); assertEquals(2430, counts.groups);
            } else if (form.endsWith("CROSS_ENTROPY_WITH_LOGITS") || form.equals("MEAN_SQUARED_ERROR")) {
                assertEquals("CpuLossSemanticClosureTest#everyLossInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals("LOSS_180_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(180, counts.rows); assertEquals(180, counts.groups);
            } else if (form.equals("CONV1D_COMPOSITION") || form.equals("CONV2D")
                    || form.equals("CONV3D")) {
                assertEquals("CpuConvolutionSemanticClosureTest#every" + switch (form) {
                    case "CONV1D_COMPOSITION" -> "Conv1dComposition";
                    case "CONV2D" -> "Conv2d";
                    case "CONV3D" -> "Conv3d";
                    default -> throw new AssertionError(form);
                } + "InventoryRowDefinesAndInvokesItsActual" + (form.equals("CONV1D_COMPOSITION")
                        ? "LoweredEntry" : "GeneratedEntry"), row[4]);
                assertEquals(form + "_540_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(540, counts.rows);
                assertEquals(form.equals("CONV1D_COMPOSITION") ? 540 : 360, counts.groups);
            } else if (form.equals("SCALED_DOT_PRODUCT_ATTENTION")) {
                assertEquals("CpuAttentionSemanticClosureTest#everyAttentionInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals("SCALED_DOT_PRODUCT_ATTENTION_4560_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
            } else if (form.equals("ARG_MIN") || form.equals("ARG_MAX")) {
                assertEquals("CpuArgExtremaSemanticClosureTest#everyArgExtremaInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals(form + "_200_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(200, counts.rows); assertEquals(200, counts.groups);
            } else if (Set.of("MASKED_SUM", "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM",
                    "VARIANCE", "STANDARD_DEVIATION").contains(form)) {
                assertEquals("CpuMaskedAdvancedReductionSemanticClosureTest#everyMaskedAndAdvancedInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals(form + "_90_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(90, counts.rows); assertEquals(90, counts.groups);
            } else if (Set.of("SOFTMAX", "LOG_SOFTMAX", "LAYER", "LAYER_AFFINE", "RMS",
                    "RMS_SCALED").contains(form)) {
                assertEquals("CpuNormalizationSemanticClosureTest#everyNormalizationInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle", row[4]);
                assertEquals(form + "_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(switch (form) {
                    case "SOFTMAX", "LOG_SOFTMAX" -> 30;
                    case "LAYER", "RMS" -> 15;
                    case "LAYER_AFFINE" -> 135;
                    case "RMS_SCALED" -> 45;
                    default -> throw new AssertionError(form);
                }, counts.rows);
                assertEquals(counts.rows, counts.groups, form + " has no projected prepared group");
            } else if (Set.of("AVERAGE_POOL2D", "MAX_POOL2D", "AVERAGE_POOL3D", "MAX_POOL3D").contains(form)) {
                assertEquals("CpuPool2d3dSemanticClosureTest#everyExactGeneratedPoolOwnerDefinesAndInvokesItsSpecializedEntry", row[4]);
                assertEquals(form + "_30_EXACT_ROW_WITNESSES", row[5]);
                assertEquals("SEALED_EXACT_ROW_WITNESSES", row[6]);
                assertEquals(30, counts.rows);
                assertEquals(15, counts.groups, form + " has byte-identical floor/ceil projections but keeps exact row witnesses");
            } else assertEquals("EXPLICIT_GAP_NO_EXACT_GROUP_WITNESS", row[6], form);
        }
        Counts attention = actual.get("SCALED_DOT_PRODUCT_ATTENTION");
        assertEquals(4560, attention.rows); assertEquals(2280, attention.groups);
        assertThrows(AssertionError.class, () -> manifestRows(manifest + manifest.lines().skip(1)
                .findFirst().orElseThrow() + "\n"),
                "mutation negative: a duplicate owner must be rejected");
        assertThrows(AssertionError.class, () -> requireExactForms(manifestRows(manifest.replaceFirst(
                "SCALED_DOT_PRODUCT_ATTENTION", "ORPHAN_CONVOLUTION_FORM"))),
                "mutation negative: an orphan form must be rejected");
        assertThrows(AssertionError.class, () -> requireSame(attention, new Counts(attention.rows, attention.groups + 1)),
                "mutation negative: a changed group count must not silently seal closure");
        assertThrows(AssertionError.class, () -> requireHash(hash,
                "0000000000000000000000000000000000000000000000000000000000000000"),
                "stale inventory hash must not silently seal closure");
    }

    private static void requireExactForms(Map<String, String[]> rows) {
        if (!rows.keySet().equals(FORMS)) throw new AssertionError("orphan or missing manifest form");
    }

    private static void requireSame(Counts actual, Counts claimed) {
        if (!actual.equals(claimed)) throw new AssertionError("semantic group mutation");
    }

    private static void requireHash(String actual, String claimed) {
        if (!actual.equals(claimed)) throw new AssertionError("stale inventory reproduction hash");
    }

    private static Map<String, Counts> inventoryCounts(String text) {
        String[] lines = text.split("\\n", -1); assertEquals(27, lines[0].split("\\t", -1).length);
        Map<String, Integer> rows = new TreeMap<>(); Map<String, Set<String>> groups = new TreeMap<>();
        for (int line = 1; line < lines.length - 1; line++) {
            String[] fields = lines[line].split("\\t", -1); assertEquals(27, fields.length, "inventory line " + (line + 1));
            String form = fields[2]; if (!FORMS.contains(form)
                    || ((form.equals("ARG_MIN") || form.equals("ARG_MAX"))
                    && !fields[0].startsWith("specialized:"))
                    || (Set.of("MASKED_SUM", "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM",
                    "VARIANCE", "STANDARD_DEVIATION").contains(form) && !fields[0].startsWith("specialized:"))) continue;
            rows.merge(form, 1, Integer::sum);
            // Every behavior-affecting prepared fact: form, IR, descriptor roles/aliases, ordered
            // carriers/access, request/selection, materialization, structural/body/entry identity,
            // outcome, ordinary selection, fallback, oracle and performance disposition.
            groups.computeIfAbsent(form, ignored -> new HashSet<>()).add(String.join("\t",
                    Arrays.copyOfRange(fields, 2, 27)));
        }
        Map<String, Counts> result = new TreeMap<>();
        rows.forEach((form, count) -> result.put(form, new Counts(count, groups.get(form).size())));
        return result;
    }

    private static Map<String, String[]> manifestRows(String text) {
        String[] lines = text.split("\\n", -1);
        assertEquals("inventory-sha256\toperation-form\traw-row-count\texact-prepared-behavior-group-count\twitness-owner\twitness-evidence-key\tdisposition", lines[0]);
        Map<String, String[]> result = new TreeMap<>();
        for (int line = 1; line < lines.length - 1; line++) {
            String[] row = lines[line].split("\\t", -1); assertEquals(7, row.length, "manifest line " + (line + 1));
            assertNull(result.put(row[1], row), "duplicate manifest owner " + row[1]);
        }
        return result;
    }

    private String resource(String name) throws Exception {
        try (InputStream stream = getClass().getResourceAsStream(BASE + name)) {
            assertNotNull(stream, name); return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static String sha256(String text) throws Exception {
        return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(text.getBytes(StandardCharsets.UTF_8)));
    }

    private record Counts(int rows, int groups) { }
}
