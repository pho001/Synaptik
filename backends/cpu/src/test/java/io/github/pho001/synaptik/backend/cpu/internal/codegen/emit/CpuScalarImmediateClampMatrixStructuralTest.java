package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** Class-File-level, fail-closed proof used by the checked matrix projections. */
class CpuScalarImmediateClampMatrixStructuralTest {
    static final String NORMALIZER_VERSION = "classfile-member-normalizer-v3";

    @Test void generatedBodiesHaveOneDirectEntryAndNoForbiddenAllocationOrDispatch() {
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            byte[] bytes = CpuScalarImmediateClampMatrixOracle.formArtifact(form).artifact().bytes();
            var model = ClassFile.of().parse(bytes);
            assertEquals(1, model.methods().size(), form.id());
            List<String> tokens = normalize(bytes);
            String body = tokens.toString();
            for (String forbidden : List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY",
                    "INVOKEDYNAMIC", "INSTANCEOF", "java/lang/reflect", "java/util/Map",
                    "java/lang/Boolean.valueOf", "java/lang/Short.valueOf",
                    "java/lang/Integer.valueOf", "java/lang/Long.valueOf",
                    "java/lang/Float.valueOf", "java/lang/Double.valueOf",
                    "java/lang/String", "io/github/pho001/synaptik"))
                assertFalse(body.contains(forbidden), form.id() + " contains " + forbidden);
            for (String token : tokens) {
                if (token.startsWith("INVOKEINTERFACE|")) {
                    assertTrue(token.contains("invoke=java/lang/foreign/MemorySegment.")
                                    || token.contains("invoke=java/lang/foreign/ValueLayout.withOrder"),
                            form.id() + " avoidable interface dispatch: " + token);
                }
                if (token.startsWith("INVOKEVIRTUAL|")) {
                    assertTrue(token.contains("invoke=jdk/incubator/vector/"),
                            form.id() + " avoidable virtual dispatch: " + token);
                }
                if (token.startsWith("CHECKCAST|")) {
                    assertTrue(token.contains("type=java/lang/foreign/ValueLayout$Of"),
                            form.id() + " generic carrier cast: " + token);
                }
            }
            assertTrue(hasBackwardBranch(tokens), form.id() + " retains direct start/end loop");
            assertCarrierBody(form, body);
        }
    }

    @Test void retainsEveryExactClassAndNormalizedDossierWhenExplicitlyRequested() throws Exception {
        String requested = System.getenv("SYNAPTIK_CPU_SCALAR_MATRIX_INSPECTION");
        org.junit.jupiter.api.Assumptions.assumeTrue(requested != null && !requested.isBlank());
        Path root = Path.of(requested);
        Path classes = root.resolve("classes");
        Path dossiers = root.resolve("normalized");
        Files.createDirectories(classes);
        Files.createDirectories(dossiers);
        var manifest = new ArrayList<String>();
        manifest.add("id\tclass-file-sha256\tdescriptor\tstrategy\tgenerator-schema\tclass-identity-schema\tnormalizer\tnormalized-sha256");
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            var artifact = CpuScalarImmediateClampMatrixOracle.formArtifact(form).artifact();
            byte[] normalized = String.join("\n", normalize(artifact.bytes())).concat("\n")
                    .getBytes(StandardCharsets.UTF_8);
            Files.write(classes.resolve(form.id() + ".class"), artifact.bytes());
            Files.write(dossiers.resolve(form.id() + ".txt"), normalized);
            manifest.add(String.join("\t", form.id(), artifact.hash(), artifact.descriptor(),
                    artifact.strategy(), artifact.generatorSchema(), artifact.classIdentitySchema(),
                    NORMALIZER_VERSION,
                    CpuScalarImmediateClampMatrixOracle.sha256(normalized)));
        }
        Files.writeString(root.resolve("manifest.tsv"), String.join("\n", manifest) + "\n",
                StandardCharsets.UTF_8);
    }

    private static boolean hasBackwardBranch(List<String> tokens) {
        for (int index = 1; index < tokens.size(); index++) {
            int marker = tokens.get(index).indexOf("|target=");
            if (marker >= 0 && Integer.parseInt(tokens.get(index).substring(marker + 8)) < index - 1)
                return true;
        }
        return false;
    }

    private static void assertCarrierBody(CpuScalarImmediateClampMatrixOracle.Form form,
            String body) {
        var carriers = form.fixture().carriers();
        boolean constantPower = form.fixture().category()
                == CpuScalarImmediateClampMatrixOracle.Category.POW_POSITIVE_ZERO
                || form.fixture().category()
                == CpuScalarImmediateClampMatrixOracle.Category.POW_NEGATIVE_ZERO;
        if (!constantPower) {
            if (carriers.getFirst() == io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                assertTrue(body.contains("java/lang/foreign/MemorySegment.get"), form.id());
            else assertTrue(body.contains(arrayOpcode(form.fixture().type(), true)), form.id());
        }
        if (carriers.getLast() == io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
            assertTrue(body.contains("java/lang/foreign/MemorySegment.set"), form.id());
        else assertTrue(body.contains(arrayOpcode(form.fixture().type(), false)), form.id());
        if (form.fixture().operation()
                == io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind.CLAMP) {
            int maximum = Math.max(body.indexOf(".max"), body.indexOf("MAX|"));
            int minimum = Math.max(body.indexOf(".min"), body.indexOf("MIN|"));
            assertTrue(maximum >= 0 && minimum > maximum,
                    form.id() + " preserves lower-max then upper-min order");
        }
    }

    private static String arrayOpcode(
            io.github.pho001.synaptik.model.datatype.DataType type, boolean load) {
        return switch (type) {
            case BFLOAT16 -> load ? "SALOAD" : "SASTORE";
            case FLOAT32 -> load ? "FALOAD" : "FASTORE";
            case FLOAT64 -> load ? "DALOAD" : "DASTORE";
            case INT32 -> load ? "IALOAD" : "IASTORE";
            case INT64 -> load ? "LALOAD" : "LASTORE";
            default -> throw new AssertionError(type);
        };
    }

    @Test void normalizerRejectsOpcodeBranchInvokeCarrierAndLoopDifferences() {
        var base = List.of("ILOAD|local=0", "LDC|constant=17", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3");
        assertTrue(constantsOnly(base, List.of("ILOAD|local=0", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3"), List.of(1)));
        for (var changed : List.of(
                List.of("ILOAD|local=0", "LDC|constant=19", "FADD|FADD", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3"),
                List.of("ILOAD|local=0", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=5", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3"),
                List.of("ILOAD|local=0", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(DD)D", "FALOAD|FALOAD", "ISTORE|local=3"),
                List.of("ILOAD|local=0", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "SALOAD|SALOAD", "ISTORE|local=3"),
                List.of("ILOAD|local=1", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3"))) assertFalse(constantsOnly(base, changed, List.of(1)));
        assertFalse(constantsOnly(base, List.of("ILOAD|local=0", "LDC|constant=19", "FMUL|FMUL", "IFEQ|target=4", "INVOKESTATIC|invoke=java/lang/Math.max(FF)F", "FALOAD|FALOAD", "ISTORE|local=3"), List.of(2)));
    }

    static boolean constantsOnly(byte[] left, byte[] right, List<Integer> locations) { return constantsOnly(normalize(left), normalize(right), locations); }
    static boolean constantsOnly(List<String> left, List<String> right, List<Integer> locations) {
        if (left.size() != right.size()) return false; boolean changed = false;
        for (int i = 0; i < left.size(); i++) if (!left.get(i).equals(right.get(i))) {
            if (!locations.contains(i) || !constantOpcode(left.get(i)).equals(constantOpcode(right.get(i)))) return false;
            changed = true;
        }
        return changed;
    }
    private static String constantOpcode(String token) {
        int separator = token.indexOf('|');
        if (separator < 0 || !token.substring(separator + 1).startsWith("constant=")) return "";
        return token.substring(0, separator);
    }
    static List<String> normalize(byte[] bytes) {
        var method = ClassFile.of().parse(bytes).methods().getFirst(); List<CodeElement> elements = method.code().orElseThrow().elementStream().toList();
        var labels = new IdentityHashMap<Label, Integer>(); int pc = 0;
        for (var element : elements) { if (element instanceof LabelTarget target) labels.put(target.label(), pc); if (element instanceof Instruction) pc++; }
        var result = new ArrayList<String>(); result.add("MEMBER|" + method.methodName().stringValue() + method.methodType().stringValue());
        for (var element : elements) if (element instanceof Instruction instruction) result.add(token(instruction, labels)); return List.copyOf(result);
    }
    private static String token(Instruction i, IdentityHashMap<Label, Integer> labels) {
        String op = i.opcode().name();
        if (i instanceof BranchInstruction b) return op + "|target=" + labels.get(b.target());
        if (i instanceof ConstantInstruction c) return op + "|constant=" + c.constantValue();
        if (i instanceof InvokeInstruction call) return op + "|invoke=" + call.owner().asInternalName() + '.' + call.name() + call.type();
        if (i instanceof FieldInstruction field) return op + "|field=" + field.owner().asInternalName() + '.' + field.name() + ':' + field.type();
        return op + "|" + i;
    }
}
