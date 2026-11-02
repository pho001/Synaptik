package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * Renders current specialized-family row ownership from the execution-produced CPU inventory.
 *
 * <p>Generated and unsupported occurrences remain in the original complete raw-row join and
 * exact prepared-outcome grouping. This audit renderer neither executes the named owner tests
 * nor independently certifies semantic closure; their actual invocation/oracle and rejection
 * evidence remains owned by those tests.</p>
 */
public final class CpuSpecializedSemanticClosureInventory {
    private static final Set<String> FORMS = Set.of("SCALED_DOT_PRODUCT_ATTENTION", "BATCH_NORM_TRAINING",
            "BATCH_NORM_INFERENCE", "CONV1D_COMPOSITION", "CONV2D", "CONV3D", "MATMUL",
            "MEAN_SQUARED_ERROR", "DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS",
            "INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS", "ARG_MIN", "ARG_MAX", "MASKED_SUM",
            "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM", "VARIANCE", "STANDARD_DEVIATION",
            "SOFTMAX", "LOG_SOFTMAX", "LAYER", "LAYER_AFFINE", "RMS", "RMS_SCALED",
            "AVERAGE_POOL2D", "MAX_POOL2D", "AVERAGE_POOL3D", "MAX_POOL3D");

    private CpuSpecializedSemanticClosureInventory() { }

    /**
     * Writes the canonical seven-column manifest to standard output without changing resources.
     *
     * @param args exactly one path to the actual current UTF-8 execution-produced inventory
     * @throws IOException if the inventory cannot be read or output cannot be written
     * @throws NoSuchAlgorithmException if SHA-256 is unavailable
     * @throws IllegalArgumentException if the arguments or inventory schema are invalid
     */
    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
        if (args.length != 1)
            throw new IllegalArgumentException("usage: CpuSpecializedSemanticClosureInventory <inventory.tsv>");
        System.out.write(render(Files.readString(Path.of(args[0]), StandardCharsets.UTF_8))
                .getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Renders the existing seven-column manifest from all original inventory rows, including
     * provider rejections. No dtype, requested strategy or raw occurrence is removed from the join.
     *
     * @param inventory complete current execution-produced inventory with its header and final LF
     * @return canonical LF-terminated manifest with current hash and derived counts
     * @throws NoSuchAlgorithmException if SHA-256 is unavailable
     * @throws IllegalArgumentException if the inventory is not canonical or a specialized form is missing
     */
    public static String render(String inventory) throws NoSuchAlgorithmException {
        if (!inventory.endsWith("\n") || inventory.contains("\r"))
            throw new IllegalArgumentException("inventory must use LF with a final newline");
        var counts = inventoryCounts(inventory);
        if (!counts.keySet().equals(FORMS))
            throw new IllegalArgumentException("missing or orphan specialized form");
        String hash = sha256(inventory);
        var result = new StringBuilder("inventory-sha256\toperation-form\traw-row-count"
                + "\texact-prepared-behavior-group-count\twitness-owner\twitness-evidence-key\tdisposition\n");
        counts.forEach((form, count) -> result.append(hash).append('\t').append(form).append('\t')
                .append(count.rows).append('\t').append(count.groups).append('\t')
                .append(witnessOwner(form)).append('\t').append(form).append("_EXACT_ROW_OUTCOMES\t")
                .append("EXACT_ROW_OUTCOME_OWNERS\n"));
        return result.toString();
    }

    private static String witnessOwner(String form) {
        return switch (form) {
            case "MATMUL" -> "CpuMatmulSemanticClosureTest#everyMatmulInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "BATCH_NORM_INFERENCE" -> "CpuBatchNormSemanticClosureTest#everyBatchNormInferenceInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "BATCH_NORM_TRAINING" -> "CpuBatchNormSemanticClosureTest#everyBatchNormTrainingInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "MEAN_SQUARED_ERROR", "DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS",
                    "INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS" ->
                    "CpuLossSemanticClosureTest#everyLossInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CONV1D_COMPOSITION" -> "CpuConvolutionSemanticClosureTest#everyConv1dCompositionInventoryRowDefinesAndInvokesItsActualLoweredEntry";
            case "CONV2D" -> "CpuConvolutionSemanticClosureTest#everyConv2dInventoryRowDefinesAndInvokesItsActualGeneratedEntry";
            case "CONV3D" -> "CpuConvolutionSemanticClosureTest#everyConv3dInventoryRowDefinesAndInvokesItsActualGeneratedEntry";
            case "SCALED_DOT_PRODUCT_ATTENTION" -> "CpuAttentionSemanticClosureTest#everyAttentionInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "ARG_MIN", "ARG_MAX" -> "CpuArgExtremaSemanticClosureTest#everyArgExtremaInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "MASKED_SUM", "MASKED_MEAN", "LOG_SUM_EXP", "L1_NORM", "L2_NORM",
                    "VARIANCE", "STANDARD_DEVIATION" ->
                    "CpuMaskedAdvancedReductionSemanticClosureTest#everyMaskedAndAdvancedInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "SOFTMAX", "LOG_SOFTMAX", "LAYER", "LAYER_AFFINE", "RMS", "RMS_SCALED" ->
                    "CpuNormalizationSemanticClosureTest#everyNormalizationInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "AVERAGE_POOL2D", "MAX_POOL2D", "AVERAGE_POOL3D", "MAX_POOL3D" ->
                    "CpuPool2d3dSemanticClosureTest#everyExactGeneratedPoolOwnerDefinesAndInvokesItsSpecializedEntry";
            default -> throw new IllegalArgumentException("unowned specialized form " + form);
        };
    }

    private static Map<String, Counts> inventoryCounts(String text) {
        String[] lines = text.split("\\n", -1);
        if (lines[0].split("\\t", -1).length != 27)
            throw new IllegalArgumentException("inventory header must have 27 columns");
        Map<String, Integer> rows = new TreeMap<>(); Map<String, Set<String>> groups = new TreeMap<>();
        for (int line = 1; line < lines.length - 1; line++) {
            String[] fields = lines[line].split("\\t", -1);
            if (fields.length != 27)
                throw new IllegalArgumentException("inventory line " + (line + 1) + " must have 27 columns");
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

    private static String sha256(String text) throws NoSuchAlgorithmException {
        return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(text.getBytes(StandardCharsets.UTF_8)));
    }

    private record Counts(int rows, int groups) { }
}
