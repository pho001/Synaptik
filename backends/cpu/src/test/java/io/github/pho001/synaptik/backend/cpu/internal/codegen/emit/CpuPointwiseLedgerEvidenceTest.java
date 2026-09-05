package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Validates the versioned CPU 0007A1O replacement for the immutable A1C evidence ledger. */
class CpuPointwiseLedgerEvidenceTest {
    private static final String RESOURCE = "operation-family-form-ledger-v2.tsv";
    private static final String ORIGINAL_SHA256 =
            "ab32cae447d85ee931d6cca1922266f9e0831363dfc2a5a7b0c0f919d6ef5e0a";
    private static final String LEDGER_HEADER = "operation\tfamily\tform\tsemantic_test_owner"
            + "\tclassfile_category\tperformance_category\ttypes\tcarriers\taccess\taddress"
            + "\trange\tvector\tscratch\tequivalence_group\trealization\trepresented_type"
            + "\tresult_type\tvalue_flow\timmediate_formula\tbranch_shape"
            + "\tcarrier_access_range\tstore_shape\tforbidden_hot_path\tevidence_provenance";
    private static final Set<CpuPointwiseOpcode> FRESH = EnumSet.of(
            CpuPointwiseOpcode.SUB, CpuPointwiseOpcode.MIN, CpuPointwiseOpcode.POW,
            CpuPointwiseOpcode.SCALAR_ADD, CpuPointwiseOpcode.SCALAR_SUB,
            CpuPointwiseOpcode.SCALAR_MUL, CpuPointwiseOpcode.SCALAR_DIV,
            CpuPointwiseOpcode.SCALAR_POW, CpuPointwiseOpcode.SCALAR_MIN,
            CpuPointwiseOpcode.SCALAR_MAX, CpuPointwiseOpcode.SCALAR_CLAMP,
            CpuPointwiseOpcode.NEG, CpuPointwiseOpcode.ABS, CpuPointwiseOpcode.RECIPROCAL,
            CpuPointwiseOpcode.LOG, CpuPointwiseOpcode.EXP, CpuPointwiseOpcode.EXPM1,
            CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.SQRT, CpuPointwiseOpcode.RSQRT,
            CpuPointwiseOpcode.FLOOR, CpuPointwiseOpcode.CEIL, CpuPointwiseOpcode.SIGN,
            CpuPointwiseOpcode.RELU, CpuPointwiseOpcode.GELU_EXACT,
            CpuPointwiseOpcode.GELU_TANH_APPROXIMATION, CpuPointwiseOpcode.SILU,
            CpuPointwiseOpcode.IS_FINITE, CpuPointwiseOpcode.IS_NAN, CpuPointwiseOpcode.IS_INF,
            CpuPointwiseOpcode.GREATER_THAN, CpuPointwiseOpcode.GREATER_OR_EQUAL,
            CpuPointwiseOpcode.LESS_THAN, CpuPointwiseOpcode.LESS_OR_EQUAL,
            CpuPointwiseOpcode.EQUAL, CpuPointwiseOpcode.NOT_EQUAL,
            CpuPointwiseOpcode.LOGICAL_AND, CpuPointwiseOpcode.LOGICAL_OR,
            CpuPointwiseOpcode.LOGICAL_NOT, CpuPointwiseOpcode.WHERE);
    private static final Map<CpuPointwiseOpcode, String> RETAINED = Map.of(
            CpuPointwiseOpcode.ADD, "P-INTEGRAL-MIXED/ADD",
            CpuPointwiseOpcode.MUL, "P-SCALAR-GENERAL/MUL",
            CpuPointwiseOpcode.DIV, "P-SCALAR-GENERAL/DIV",
            CpuPointwiseOpcode.MAX, "P-INTEGRAL-MIXED/MAX",
            CpuPointwiseOpcode.LOG1P, "P-VECTOR-SEGMENT/LOG1P",
            CpuPointwiseOpcode.SIGMOID, "P-SCALAR-GENERAL/SIGMOID",
            CpuPointwiseOpcode.TANH, "P-VECTOR-SEGMENT/TANH",
            CpuPointwiseOpcode.CAST, "P-INTEGRAL-MIXED/CAST");
    private static final Set<String> NON_POINTWISE_CATEGORIES = Set.of(
            "retained-affine", "A-GENERAL", "M-PAD", "retained-tile", "M-CONCAT",
            "M-STACK", "M-UNFOLD-AXIS", "M-UNFOLD2D", "retained-slice-update", "I-GATHER",
            "retained-gather-elements", "I-GATHER-ND", "retained-one-hot",
            "S-GENERAL-MIN+retained", "retained-scatter-add", "retained-fold-axis",
            "F-FOLD2D", "retained-sort", "O-ARGSORT", "retained-top-k",
            "STRUCTURAL_ONLY:fixed-two-word-work", "R-DROPOUT-GENERAL", "retained-scan",
            "C-SCAN-GENERAL", "X-MIN-MULTI", "X-ANY-SINGLE", "retained-numerical",
            "N-MEAN-GENERAL", "N-PROD-MULTI");

    @Test void replacementHasExactProvenanceInventoryAndSchema() throws Exception {
        Ledger ledger = read();
        assertEquals("2", ledger.metadata().get("ledger-version"));
        assertEquals(ORIGINAL_SHA256, ledger.metadata().get("replaces-sha256"));
        assertEquals("CPU-0007A1O", ledger.metadata().get("replacement-task"));
        int historicalSchema = Integer.parseInt(ledger.metadata().get("generated-schema"));
        assertEquals(42, historicalSchema);
        assertEquals(59, CpuGeneratorSchema.CURRENT_VERSION);
        assertTrue(historicalSchema <= CpuGeneratorSchema.CURRENT_VERSION);
        // The immutable artifact has 79 physical TSV lines: one header plus 78 inventory rows.
        assertEquals(79, ledger.rows().size() + 1);

        Set<CpuPointwiseOpcode> seen = EnumSet.noneOf(CpuPointwiseOpcode.class);
        Set<String> nonPointwiseForms = new HashSet<>();
        for (Row row : ledger.rows()) {
            if (row.family().equals("pointwise")) {
                CpuPointwiseOpcode opcode = CpuPointwiseOpcode.valueOf(row.operation());
                assertTrue(seen.add(opcode), () -> "duplicate pointwise opcode " + opcode);
                assertFalse(row.performanceCategory().startsWith("STRUCTURAL_ONLY:"));
                assertEquals(FRESH.contains(opcode) ? "P-" + opcode : RETAINED.get(opcode),
                        row.performanceCategory(), () -> "unknown evidence category for " + opcode);
                PointwiseExpectation expected = expected(opcode);
                assertEquals(expected.equivalenceGroup(), row.equivalenceGroup());
                assertEquals(expected.realization(), row.realization());
                assertEquals(expected.representedType(), row.representedType());
                assertEquals(expected.resultType(), row.resultType());
                assertEquals(expected.valueFlow(), row.valueFlow());
                assertEquals(expected.operationFormula(), row.operationFormula());
                assertEquals(expected.branchShape(), row.branchShape());
                assertEquals(expected.carrierAccessRange(), row.carrierAccessRange());
                assertEquals("one-direct-store", row.storeShape());
                assertEquals("allocation+boxing+reflection+dispatch+dynamic",
                        row.forbiddenHotPath());
                assertEquals(FRESH.contains(opcode) ? "a1o-fresh-five-fork"
                        : "a1n-schema42-regenerated", row.evidenceProvenance());
            } else {
                String formKey = row.operation() + ":" + row.family() + ":" + row.form();
                assertTrue(nonPointwiseForms.add(formKey),
                        () -> "duplicate non-pointwise form " + formKey);
                assertTrue(NON_POINTWISE_CATEGORIES.contains(row.performanceCategory()),
                        () -> "unknown non-pointwise category " + row.performanceCategory());
                if (row.performanceCategory().startsWith("STRUCTURAL_ONLY:")) {
                    assertEquals("INITIAL_STATE", row.operation());
                    assertEquals("authorized-structural", row.evidenceProvenance());
                }
            }
        }
        assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class), seen);
        assertEquals(30, nonPointwiseForms.size());
        assertEquals(40, FRESH.size());
        assertEquals(8, RETAINED.size());
    }

    private static Ledger read() throws Exception {
        Map<String, String> metadata = new HashMap<>();
        List<Row> rows = new ArrayList<>();
        try (var stream = CpuPointwiseLedgerEvidenceTest.class.getResourceAsStream(RESOURCE)) {
            assertTrue(stream != null, () -> "missing test resource " + RESOURCE);
            try (var reader = new BufferedReader(new InputStreamReader(stream,
                    StandardCharsets.UTF_8))) {
                String line;
                boolean header = false;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("# ")) {
                        String[] fact = line.substring(2).split("=", 2);
                        assertEquals(2, fact.length, "malformed provenance fact " + line);
                        assertTrue(metadata.put(fact[0], fact[1]) == null,
                                () -> "duplicate provenance fact " + fact[0]);
                    } else if (!header) {
                        assertEquals(LEDGER_HEADER, line);
                        header = true;
                    } else {
                        String[] value = line.split("\t", -1);
                        assertEquals(24, value.length, "malformed ledger row " + line);
                        rows.add(new Row(value[0], value[1], value[2], value[5], value[13],
                                value[14], value[15], value[16], value[17], value[18], value[19],
                                value[20], value[21], value[22], value[23]));
                    }
                }
                assertTrue(header, "missing ledger header");
            }
        }
        return new Ledger(Map.copyOf(metadata), List.copyOf(rows));
    }

    private record Ledger(Map<String, String> metadata, List<Row> rows) { }
    private record Row(String operation, String family, String form, String performanceCategory,
            String equivalenceGroup, String realization, String representedType, String resultType,
            String valueFlow, String operationFormula, String branchShape,
            String carrierAccessRange, String storeShape, String forbiddenHotPath,
            String evidenceProvenance) { }

    private static PointwiseExpectation expected(CpuPointwiseOpcode opcode) {
        String tag = "scalar-f32-dense-array";
        String realization = "scalar-f32-dense-heap-array";
        String representedType = "FLOAT32";
        String carrierAccessRange = "heap-array:dense:int:full-range";
        if (opcode == CpuPointwiseOpcode.ADD || opcode == CpuPointwiseOpcode.MAX
                || opcode == CpuPointwiseOpcode.CAST) {
            tag = "vector-i64-dense-mixed";
            realization = "vector-i64-mixed-array-segment";
            representedType = "INT64";
            carrierAccessRange = "mixed-array-segment:dense:int:full-range";
        } else if (opcode == CpuPointwiseOpcode.MUL || opcode == CpuPointwiseOpcode.DIV
                || opcode == CpuPointwiseOpcode.SIGMOID) {
            tag = "scalar-f32-general-mixed";
            realization = "scalar-f32-general-mixed-array-segment";
            carrierAccessRange = "mixed-array-segment:broadcast+strided:long:full-range";
        } else if (opcode == CpuPointwiseOpcode.LOG1P || opcode == CpuPointwiseOpcode.TANH) {
            tag = "vector-f32-dense-segment";
            realization = "vector-f32-segment";
            carrierAccessRange = "segment:dense:int:full-range";
        } else if (opcode.family() == CpuPointwiseOpcode.Family.LOGICAL) {
            tag = "scalar-bool-dense-array";
            realization = "scalar-bool-dense-heap-array";
            representedType = "BOOL";
        } else if (opcode == CpuPointwiseOpcode.WHERE) {
            tag = "scalar-bool-f32-dense-array";
            realization = "scalar-bool-f32-dense-heap-array";
            representedType = "BOOL+FLOAT32";
        }
        String resultType = switch (opcode.resultCategory()) {
            case BOOL -> "BOOL";
            case INPUT_TYPE -> representedType.equals("INT64") ? "INT64" : "FLOAT32";
            case BRANCH_TYPE -> "FLOAT32";
        };
        String valueFlow = switch (opcode.family()) {
            case CLASSIFICATION -> "value-to-bool";
            case COMPARISON -> "value-pair-to-bool";
            case LOGICAL -> opcode.arity() == 1 ? "bool-to-bool" : "bool-pair-to-bool";
            case SELECTION -> "bool-mask-to-value";
            default -> "value";
        };
        return new PointwiseExpectation(opcode + ":" + tag, realization, representedType,
                resultType, valueFlow, operationFormula(opcode), branchShape(opcode),
                carrierAccessRange);
    }

    private static String branchShape(CpuPointwiseOpcode opcode) {
        return opcode + ":" + switch (opcode) {
            case ERF, SIGMOID, GELU_EXACT, GELU_TANH_APPROXIMATION, SILU ->
                    "formula-conditional";
            case GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, EQUAL, NOT_EQUAL,
                    WHERE -> "value-selection";
            case ADD, SUB, MUL, DIV, MIN, MAX, POW,
                    SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW,
                    SCALAR_MIN, SCALAR_MAX, SCALAR_CLAMP,
                    NEG, ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, SQRT, RSQRT,
                    FLOOR, CEIL, SIGN, RELU, TANH,
                    IS_FINITE, IS_NAN, IS_INF,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, CAST -> "no-operation-branch";
        };
    }

    private static String operationFormula(CpuPointwiseOpcode opcode) {
        return switch (opcode) {
            case ADD -> "LongVector.add+scalar-tail:ladd";
            case SUB -> "primitive:fsub";
            case MUL -> "primitive:fmul";
            case DIV -> "primitive:fdiv";
            case MIN -> "invoke:Math.min(float,float)";
            case MAX ->
                    "generated:LongVector.max+tail:Long.max|oracle:LongVector.max+tail:Math.max";
            case POW -> "invoke:StrictMath.pow(double,double)+narrow-f32";
            case SCALAR_ADD -> "typed-immediate(0.75f)+primitive:fadd";
            case SCALAR_SUB -> "typed-immediate(0.75f)+primitive:fsub";
            case SCALAR_MUL -> "typed-immediate(0.75f)+primitive:fmul";
            case SCALAR_DIV -> "typed-immediate(0.75f)+primitive:fdiv";
            case SCALAR_POW ->
                    "typed-exponent(1.5f)+invoke:StrictMath.pow(double,double)+narrow-f32";
            case SCALAR_MIN -> "typed-immediate(0.75f)+invoke:Math.min(float,float)";
            case SCALAR_MAX -> "typed-immediate(0.75f)+invoke:Math.max(float,float)";
            case SCALAR_CLAMP -> "typed-bounds(-0.5f,0.5f)+invoke:Math.max+Math.min";
            case NEG -> "primitive:fneg";
            case ABS -> "invoke:Math.abs(float)";
            case RECIPROCAL -> "primitive:1.0f/fdiv";
            case LOG -> "invoke:StrictMath.log(double)+narrow-f32";
            case LOG1P -> "generated:CpuVectorMath.log1p+tail:StrictMath.log1p|oracle:"
                    + "FloatVector.lanewise(LOG1P)+tail:Math.log1p";
            case EXP -> "invoke:StrictMath.exp(double)+narrow-f32";
            case EXPM1 -> "invoke:StrictMath.expm1(double)+narrow-f32";
            case ERF -> "inline:erf-piecewise-polynomial+Math.exp+Math.copySign+narrow-f32";
            case SQRT -> "invoke:StrictMath.sqrt(double)+narrow-f32";
            case RSQRT -> "invoke:StrictMath.sqrt(double)+primitive:ddiv+narrow-f32";
            case FLOOR -> "invoke:StrictMath.floor(double)+narrow-f32";
            case CEIL -> "invoke:StrictMath.ceil(double)+narrow-f32";
            case SIGN -> "invoke:Math.signum(float)";
            case RELU -> "invoke:Math.max(float,float)";
            case SIGMOID -> "inline:stable-sign-selection+generated:StrictMath.exp+"
                    + "oracle:Math.exp+narrow-f32";
            case TANH -> "generated:CpuVectorMath.tanh+tail:StrictMath.tanh|oracle:"
                    + "FloatVector.lanewise(TANH)+tail:Math.tanh";
            case GELU_EXACT ->
                    "inline:negative-infinity-selection+erf-piecewise-polynomial+Math.sqrt+narrow-f32";
            case GELU_TANH_APPROXIMATION ->
                    "inline:negative-infinity-selection+cubic+Math.sqrt+StrictMath.tanh+narrow-f32";
            case SILU -> "inline:negative-infinity+sign-selection+StrictMath.exp+narrow-f32";
            case IS_FINITE -> "invoke:Float.isFinite(float)";
            case IS_NAN -> "invoke:Float.isNaN(float)";
            case IS_INF -> "invoke:Float.isInfinite(float)";
            case GREATER_THAN -> "primitive:fcmpl+GT-boolean-selection";
            case GREATER_OR_EQUAL -> "primitive:fcmpl+GE-boolean-selection";
            case LESS_THAN -> "primitive:fcmpg+LT-boolean-selection";
            case LESS_OR_EQUAL -> "primitive:fcmpg+LE-boolean-selection";
            case EQUAL -> "primitive:fcmpg+EQ-boolean-selection";
            case NOT_EQUAL -> "primitive:fcmpg+NE-boolean-selection";
            case LOGICAL_AND -> "primitive:iand";
            case LOGICAL_OR -> "primitive:ior";
            case LOGICAL_NOT -> "primitive:ixor-one";
            case WHERE -> "conditional:bool-select-f32";
            case CAST -> "LongVector.identity+scalar-tail:identity";
        };
    }

    private record PointwiseExpectation(String equivalenceGroup, String realization,
            String representedType, String resultType, String valueFlow, String operationFormula,
            String branchShape, String carrierAccessRange) { }
}
