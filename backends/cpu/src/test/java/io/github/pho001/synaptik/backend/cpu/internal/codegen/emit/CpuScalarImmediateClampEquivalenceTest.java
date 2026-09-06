package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Fail-closed finite fixture, execution-oracle, and Class-File projection contract. */
class CpuScalarImmediateClampEquivalenceTest {
    static final String NORMALIZER_VERSION = "instruction-normalizer-v2";
    private static final String FIXTURES = "scalar-immediate-clamp-fixtures.tsv";
    private static final String PROJECTION = "scalar-immediate-clamp-equivalence.tsv";
    private static final List<String> FIXTURE_HEADER = List.of("id", "class-file-sha256", "binary-name", "structural-key", "entry-descriptor", "schema", "operation", "type", "input-carrier", "output-carrier", "layout", "strategy", "shape", "immediate-category", "immediate-location", "fixture-resource-sha256");
    private static final List<String> PROJECTION_HEADER = List.of("unit", "normalizer-version", "disposition", "operation", "type", "input-carrier", "output-carrier", "layout", "strategy", "shape", "fixture-resource-sha256", "exact-fixture-members", "exact-member-sha256");

    @Test void fixturesAreExactCompleteForTheDeclaredBoundedMechanism() throws Exception {
        var table = table(FIXTURES, FIXTURE_HEADER);
        assertEquals(Set.of("bf16-mul-one", "bf16-mul-two"), table.rows().keySet());
        assertEquals(resourceDigest(table), table.rows().get("bf16-mul-one").get(15));
        assertEquals(resourceDigest(table), table.rows().get("bf16-mul-two").get(15));
        for (var fixture : CpuScalarImmediateClampEquivalenceOracle.fixtures()) {
            var artifact = CpuScalarImmediateClampEquivalenceOracle.artifact(fixture);
            var row = Objects.requireNonNull(table.rows().get(fixture.id()), fixture.id());
            assertEquals(List.of(artifact.hash(), artifact.binaryName(), artifact.structuralKey(),
                    artifact.specialization().entryType().descriptorString(), Integer.toString(artifact.specialization().classIdentitySchema()),
                    "MUL", "BFLOAT16", "SHORT_ARRAY", "SHORT_ARRAY", "CONTIGUOUS", "SCALAR", "[8]", "represented", fixture.immediateLocation()), row.subList(1, 15), fixture.id());
        }
    }

    @Test void generatedClassesExecuteAgainstIndependentTypedCleanJavaLoopsOnOrdinaryAndEdgeInputs() throws Throwable {
        short[] input = {b(0.0f), b(-0.0f), b(1.5f), b(-2.5f), b(Float.NaN), b(Float.POSITIVE_INFINITY), b(Float.NEGATIVE_INFINITY), b(Float.MAX_VALUE)};
        for (var fixture : CpuScalarImmediateClampEquivalenceOracle.fixtures()) {
            var artifact = CpuScalarImmediateClampEquivalenceOracle.artifact(fixture);
            assertArrayEquals(CpuScalarImmediateClampEquivalenceOracle.cleanJava(fixture, input),
                    CpuScalarImmediateClampEquivalenceOracle.invoke(artifact, input), fixture.id());
        }
    }

    @Test void projectionIsBidirectionalAndUsesExactMemberHashesAndFixtureProvenance() throws Exception {
        var fixtures = table(FIXTURES, FIXTURE_HEADER); var projection = table(PROJECTION, PROJECTION_HEADER);
        assertEquals(Set.of("bf16-scalar-mul-represented"), projection.rows().keySet());
        var row = projection.rows().get("bf16-scalar-mul-represented");
        assertEquals(NORMALIZER_VERSION, row.get(1)); assertEquals("PROVED_CONSTANTS_ONLY", row.get(2));
        assertEquals(List.of("MUL", "BFLOAT16", "SHORT_ARRAY", "SHORT_ARRAY", "CONTIGUOUS", "SCALAR", "[8]"), row.subList(3, 10));
        assertEquals(resourceDigest(fixtures), row.get(10));
        var members = List.of(row.get(11).split(",", -1)); var hashes = List.of(row.get(12).split(",", -1));
        assertEquals(List.of("bf16-mul-one", "bf16-mul-two"), members); assertEquals(2, hashes.size());
        var artifacts = members.stream().map(id -> CpuScalarImmediateClampEquivalenceOracle.artifact(
                CpuScalarImmediateClampEquivalenceOracle.fixtures().stream().filter(f -> f.id().equals(id)).findFirst().orElseThrow())).toList();
        assertEquals(artifacts.stream().map(CpuScalarImmediateClampEquivalenceOracle.Artifact::hash).toList(), hashes);
        assertTrue(constantsOnly(artifacts.getFirst(), artifacts.get(1)), instructionTokens(artifacts.getFirst().bytes()) + " vs " + instructionTokens(artifacts.get(1).bytes()));
        assertEquals(new HashSet<>(members), fixtures.rows().keySet(), "no stale, missing, or unprojected declared fixture");
    }

    static boolean constantsOnly(CpuScalarImmediateClampEquivalenceOracle.Artifact left,
            CpuScalarImmediateClampEquivalenceOracle.Artifact right) {
        if (!left.specialization().equals(right.specialization()) && (!left.specialization().carrierPattern().equals(right.specialization().carrierPattern())
                || left.specialization().executionStrategy() != right.specialization().executionStrategy())) return false;
        var leftTokens = instructionTokens(left.bytes()); var rightTokens = instructionTokens(right.bytes());
        if (leftTokens.size() != rightTokens.size()) return false;
        int differing = -1;
        for (int index = 0; index < leftTokens.size(); index++) if (!leftTokens.get(index).equals(rightTokens.get(index))) {
            if (differing != -1 || !leftTokens.get(index).substring(0, leftTokens.get(index).indexOf('|')).equals(rightTokens.get(index).substring(0, rightTokens.get(index).indexOf('|')))) return false;
            differing = index;
        }
        return differing >= 0 && ("entry:" + differing).equals(left.fixture().immediateLocation())
                && left.fixture().immediateLocation().equals(right.fixture().immediateLocation())
                && classStructure(left.bytes()).equals(classStructure(right.bytes()));
    }

    static List<String> instructionTokens(byte[] bytes) {
        List<CodeElement> elements = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow().elementStream().toList();
        var labels = new IdentityHashMap<java.lang.classfile.Label, Integer>(); int ordinal = 0;
        for (CodeElement element : elements) { if (element instanceof LabelTarget target) labels.put(target.label(), ordinal); if (element instanceof Instruction) ordinal++; }
        return elements.stream().filter(Instruction.class::isInstance).map(Instruction.class::cast)
                .map(instruction -> token(instruction, labels)).toList();
    }
    private static String token(Instruction instruction, IdentityHashMap<java.lang.classfile.Label, Integer> labels) {
        String text = instruction.opcode().name();
        if (instruction instanceof BranchInstruction branch) text += "|target=" + labels.get(branch.target());
        else if (instruction instanceof ConstantInstruction constant) text += "|constant=" + constant.constantValue();
        else if (instruction instanceof InvokeInstruction invoke) text += "|invoke=" + invoke.owner().asInternalName() + '.' + invoke.name() + invoke.type();
        else if (instruction instanceof FieldInstruction field) text += "|field=" + field.owner().asInternalName() + '.' + field.name() + ':' + field.type();
        else text += "|" + instruction;
        return text;
    }
    static List<String> classStructure(byte[] bytes) {
        var model = ClassFile.of().parse(bytes); var result = new ArrayList<String>();
        result.add("flags=" + model.flags()); result.add("interfaces=" + model.interfaces());
        model.fields().forEach(field -> result.add("field=" + field.fieldName().stringValue() + field.fieldType().stringValue() + field.flags()));
        model.methods().forEach(method -> result.add("method=" + method.methodName().stringValue() + method.methodType().stringValue() + method.flags()));
        return result;
    }
    static Table table(String file, List<String> header) throws Exception {
        var rows = new LinkedHashMap<String, List<String>>(); var raw = new ArrayList<String>();
        try (var input = CpuScalarImmediateClampEquivalenceTest.class.getResourceAsStream(file);
                var reader = new BufferedReader(new InputStreamReader(Objects.requireNonNull(input, file), StandardCharsets.UTF_8))) {
            String line; while ((line = reader.readLine()) != null) { if (!line.isBlank() && !line.startsWith("#")) raw.add(line); }
        }
        assertFalse(raw.isEmpty(), file); assertFalse(raw.getFirst().contains("\\t"), "literal escape is forbidden");
        assertEquals(header, List.of(raw.removeFirst().split("\t", -1)), file);
        for (String line : raw) { assertFalse(line.contains("\\t"), line); var fields = List.of(line.split("\t", -1));
            assertEquals(header.size(), fields.size(), line); assertTrue(fields.stream().noneMatch(String::isBlank), line);
            assertNull(rows.put(fields.getFirst(), fields), "duplicate " + fields.getFirst()); }
        return new Table(rows, raw);
    }
    static String resourceDigest(Table table) {
        String canonical = String.join("\n", table.raw().stream().map(line -> line.replaceAll("\t[0-9a-f]{64}$", "\t")).toList()) + "\n";
        try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(canonical.getBytes(StandardCharsets.UTF_8))); }
        catch (Exception failure) { throw new AssertionError(failure); }
    }
    private static short b(float value) { return CpuScalarImmediateClampEquivalenceOracle.bfloat(value); }
    record Table(Map<String, List<String>> rows, List<String> raw) { }
}
