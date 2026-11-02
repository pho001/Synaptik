package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.MethodModel;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * Makes the generated structural-oracle boundary finite and accountable.
 *
 * <p>A category is intentionally a projection of code-shaping facts, not an owner id, class
 * hash, geometry, or constant.  It preserves the algorithm, represented input/output types,
 * scalar/vector realization, heap/segment/mixed access shape, dense/general address regime,
 * and scratch/materialization dataflow.  Constants and geometry are deliberately absent: the
 * existing normalized-body checks are the proof that those differences cannot silently turn
 * into a hot-loop projection.</p>
 */
class CpuGeneratedStructuralOracleCatalogTest {
    private static final Pattern VALUE_IDS = Pattern.compile("ValueId\\[value=\\d+\\]:");

    @Test void everyExactGeneratedOwnerProjectsOnceToAFiniteMaterialStructuralCategory() throws Throwable {
        // Inspect facts from current provider -> preparer -> generator executions.
        // Historical class-byte snapshots do not gate live structural evidence.
        String inventory = CpuGeneratedCoverageCheckpointTest.liveInventory();
        Map<String, String[]> rows = parse(inventory);
        Map<String, Category> catalog = catalog(rows);
        Map<String, Long> partitionCounts = new TreeMap<>();
        Map<String, String> ownerCategories = new LinkedHashMap<>();
        for (var entry : rows.entrySet()) {
            String owner = entry.getKey();
            String[] row = entry.getValue();
            if (!"GENERATED".equals(row[21])) continue;
            Category category = category(row);
            assertEquals(category.id(), catalog.get(category.id()).id(), owner);
            assertNull(ownerCategories.put(owner, category.id()), "duplicate owner projection " + owner);
            assertFalse(row[19].isBlank() || row[20].isBlank(), "missing class/body SHA " + owner);
            partitionCounts.merge(owner.substring(0, owner.indexOf(':')), 1L, Long::sum);
        }
        
        assertEquals(java.util.Set.of("specialized", "affine-matrix", "ordinary",
                "pointwise-matrix", "composition", "scalar-immediate"), partitionCounts.keySet());
        assertFalse(catalog.isEmpty());
        assertEquals(0, ownerCategories.entrySet().stream()
                .filter(entry -> catalog.get(entry.getValue()).proved()).count(),
                "catalog hygiene is discovery only; it cannot prove clean-Java equivalence");

        // A category is never keyed by owner or SHA and cannot claim an absent test owner.
        for (Category category : catalog.values()) {
            if (!category.proved()) continue;
            assertFalse(category.id().contains("sha") || category.id().contains(":"));
            assertTrue(ownerCategories.containsValue(category.id()), "orphan category " + category.id());
            assertOracleOwnerExists(category);
        }
    }

    @Test void eachCategoryRepresentativeHasTypedStaticEntryDirectLoopAndNoAvoidableDispatch()
            throws Throwable {
        Map<String, String[]> rows = parse(CpuGeneratedCoverageCheckpointTest.liveInventory());
        Map<String, Category> catalog = catalog(rows);
        assertFalse(catalog.isEmpty());
        for (Category category : catalog.values()) {
            if (!category.proved()) continue;
            String owner = category.representativeOwner();
            String[] row = rows.get(owner);
            byte[] bytes = CpuGeneratedCoverageEvidenceRegistry.classBytes(owner);
            assertEquals(row[19], sha256(bytes), "stale representative SHA " + owner);
            assertStructuralProfile(category, row, bytes);
        }
    }

    private static void assertStructuralProfile(Category category, String[] row, byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        assertEquals(1, model.methods().size(), category.id());
        MethodModel method = model.methods().getFirst();
        assertTrue(method.flags().has(java.lang.reflect.AccessFlag.STATIC), category.id());
        assertEquals(row[18], method.methodType().stringValue(), category.id());
        List<String> tokens = CpuScalarImmediateClampMatrixStructuralTest.normalize(bytes);
        String body = String.join("\n", tokens);
        for (String forbidden : List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY",
                "INVOKEDYNAMIC", "java/lang/reflect", "java/util/Map", "java/lang/String",
                "java/lang/Boolean.valueOf", "java/lang/Integer.valueOf", "java/lang/Long.valueOf",
                "java/lang/Float.valueOf", "java/lang/Double.valueOf", "io/github/pho001/synaptik")) {
            assertFalse(body.contains(forbidden), category.id() + " contains " + forbidden);
        }
        assertTrue(tokens.stream().anyMatch(token -> token.startsWith("GOTO|") || token.startsWith("IF")),
                category.id() + " has no explicit loop/branch shape");
        assertTrue(tokens.stream().anyMatch(token -> token.contains("STORE") || token.contains("MemorySegment.set")),
                category.id() + " has no direct output store");
        if (category.carrierShape().equals("SEGMENT")) {
            assertTrue(body.contains("java/lang/foreign/MemorySegment"), category.id());
        } else if (category.carrierShape().equals("MIXED")) {
            assertTrue(body.contains("java/lang/foreign/MemorySegment"), category.id());
            assertTrue(tokens.stream().anyMatch(token -> token.endsWith("ALOAD") || token.contains("ALOAD|")), category.id());
        }
        assertTrue(category.oracleOwner().contains("CpuScalarImmediateClampMatrix"), category.id());
    }

    private static Map<String, Category> catalog(Map<String, String[]> rows) {
        Map<String, Category> categories = new TreeMap<>();
        rows.forEach((owner, row) -> {
            if (!"GENERATED".equals(row[21])) return;
            Category category = category(row).withRepresentative(owner);
            categories.merge(category.id(), category, Category::merge);
        });
        return Map.copyOf(categories);
    }

    private static Category category(String[] row) {
        String types = VALUE_IDS.matcher(row[4]).replaceAll("");
        String realization = row[12].replace("parallel-", "parallel_");
        return new Category(row[2], row[3], types, realization, carrierShape(row[9]),
                addressShape(row[10]), "materializations=" + row[14], oracleOwner(row[2], row[3]), "");
    }

    private static String carrierShape(String carriers) {
        boolean segment = carriers.contains("MEMORY_SEGMENT");
        boolean heap = carriers.contains("_ARRAY");
        return segment && heap ? "MIXED" : segment ? "SEGMENT" : "HEAP";
    }

    private static String addressShape(String layout) {
        boolean general = layout.contains("GENERAL");
        boolean dense = layout.contains("DENSE");
        return general && dense ? "MIXED" : general ? "GENERAL" : "DENSE";
    }

    private static String oracleOwner(String form, String ir) {
        if (form.startsWith("SCALAR_"))
            return "CpuScalarImmediateClampMatrixStructuralTest#generatedBodiesHaveOneDirectEntryAndNoForbiddenAllocationOrDispatch";
        return switch (ir) {
            case "CpuAffineCopyIr" -> "CpuAffineGeneratedCoverageFixtureTest#everyAffineInventoryRowDefinesInvokesAndMatchesCoordinateOracle";
            case "CpuKernelIr" -> "CpuPointwiseSemanticClosureManifestTest#everyManifestOwnerExecutesItsIndependentGeneratedEntryOracle";
            case "CpuAggregateIr", "CpuScanIr" -> "CpuAggregateScanSemanticClosureTest#everyAggregateAndScanMatrixCandidateDefinesAndExecutesItsOwnArtifact";
            case "CpuDataMovementIr", "CpuFoldIr" -> "CpuOrdinaryMovementFoldSemanticClosureTest#everyExactOrdinaryMovementAndFoldOwnerExecutesAgainstIndependentOracle";
            case "CpuIndexingIr", "CpuOrderingIr" -> "CpuIndexingOrderingSemanticClosureTest#everyOrdinaryGatherAndOrderingRowExecutesAgainstAnIndependentOracle";
            case "CpuScatterIr" -> "CpuScatterSemanticClosureTest#everyOrdinaryScatterInventoryCandidateExecutesAgainstTheIndependentOracle";
            case "CpuRandomIr" -> "CpuRandomOneHotSemanticClosureTest#everyOrdinaryRandomAndOneHotOwnerDefinesAndExecutesItsActualEntry";
            case "CpuArgExtremaIr" -> "CpuArgExtremaSemanticClosureTest#everyArgExtremaInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuMaskedReductionIr", "CpuAdvancedReductionIr" -> "CpuMaskedAdvancedReductionSemanticClosureTest#everyMaskedAndAdvancedInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuLayerNormIr", "CpuRmsNormIr" -> "CpuNormalizationSemanticClosureTest#everyNormalizationInventoryOwnerDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuBatchNormInferenceIr", "CpuBatchNormTrainingIr" -> "CpuBatchNormSemanticClosureTest#everyBatchNormInferenceInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuMatmulIr" -> "CpuMatmulSemanticClosureTest#everyMatmulInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuConv2dIr", "CpuConv3dIr" -> "CpuConvolutionSemanticClosureTest#everyConv2dInventoryRowDefinesAndInvokesItsActualGeneratedEntry";
            case "CpuPool2dIr", "CpuPool3dIr" -> "CpuPool2d3dSemanticClosureTest#everyExactGeneratedPoolOwnerDefinesAndInvokesItsSpecializedEntry";
            case "CpuSoftmaxIr" -> "CpuSoftmaxGeneratedKernelTest#allTypesKindsAndHeapSegmentPairsProduceNormalizedSlices";
            case "CpuTrailingNormalizationIr" -> "CpuTrailingNormalizationEvidenceTest#mixedSegmentSpecializationsUseFrozenNativeLayouts";
            case "CpuAttentionIr" -> "CpuAttentionSemanticClosureTest#everyAttentionInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            case "CpuLossIr" -> "CpuLossSemanticClosureTest#everyLossInventoryRowDefinesAndInvokesItsActualGeneratedEntryAgainstIndependentOracle";
            default -> throw new AssertionError("no structural oracle owner for " + form + '/' + ir);
        };
    }

    private static void assertOracleOwnerExists(Category category) {
        String owner = category.oracleOwner();
        int marker = owner.indexOf('#');
        assertTrue(marker > 0, category.id());
        try {
            Class<?> type = Class.forName(CpuGeneratedStructuralOracleCatalogTest.class.getPackageName()
                    + '.' + owner.substring(0, marker));
            assertTrue(List.of(type.getDeclaredMethods()).stream()
                    .anyMatch(method -> method.getName().equals(owner.substring(marker + 1))), owner);
        } catch (ClassNotFoundException failure) { throw new AssertionError("missing oracle owner " + owner, failure); }
    }

    private static Map<String, String[]> parse(String text) {
        String[] lines = text.split("\\n", -1);
        Map<String, String[]> rows = new LinkedHashMap<>();
        for (int line = 1; line < lines.length - 1; line++) {
            String[] row = lines[line].split("\\t", -1);
            assertEquals(27, row.length, "inventory line " + (line + 1));
            assertNull(rows.put(row[0], row), "duplicate owner " + row[0]);
        }
        return rows;
    }


    private static String sha256(String text) throws Exception { return sha256(text.getBytes(StandardCharsets.UTF_8)); }
    private static String sha256(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder result = new StringBuilder();
        for (byte value : digest) result.append(String.format("%02x", value));
        return result.toString();
    }

    private record Category(String form, String ir, String types, String realization,
            String carrierShape, String addressShape, String materializationShape,
            String oracleOwner, String representativeOwner) {
        String id() {
            return String.join(";", "algorithm=" + form + '/' + ir, "types=" + types,
                    "realization=" + realization, "carrier=" + carrierShape,
                    "address=" + addressShape, "dataflow=" + materializationShape);
        }
        /** This catalog collapses ordered facts, so no category is structural-oracle evidence. */
        boolean proved() { return false; }
        Category withRepresentative(String owner) { return new Category(form, ir, types, realization,
                carrierShape, addressShape, materializationShape, oracleOwner, owner); }
        static Category merge(Category left, Category right) {
            assertEquals(left.id(), right.id());
            assertEquals(left.oracleOwner(), right.oracleOwner());
            return left.representativeOwner().compareTo(right.representativeOwner()) <= 0 ? left : right;
        }
    }
}
