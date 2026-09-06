package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.util.List;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;

/** Structural hot-path prohibition and normalizer mutation controls for the bounded fixture pair. */
class CpuScalarImmediateClampEquivalenceStructuralTest {
    @Test void generatedUnitsRejectForbiddenHotPathMechanisms() {
        for (var fixture : CpuScalarImmediateClampEquivalenceOracle.fixtures()) {
            byte[] bytes = CpuScalarImmediateClampEquivalenceOracle.artifact(fixture).bytes();
            var model = ClassFile.of().parse(bytes);
            assertTrue(model.fields().isEmpty(), fixture.id()); assertEquals(1, model.methods().size(), fixture.id());
            String body = model.methods().getFirst().code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                    .map(Object::toString).reduce("", (left, right) -> left + '\n' + right);
            String pool = StreamSupport.stream(model.constantPool().spliterator(), false)
                    .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                    .filter(member -> !member.owner().asInternalName().startsWith("io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/Generated_"))
                    .map(Object::toString)
                    .reduce("", (left, right) -> left + '\n' + right);
            String all = body + '\n' + pool;
            for (String forbidden : List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "INVOKEDYNAMIC", "java/lang/reflect", "MethodHandles", "ConstantDynamic", "java/util/Map", "java/util/HashMap", "java/lang/String", "INVOKEINTERFACE", "INVOKEVIRTUAL", "io/github/pho001/synaptik"))
                assertFalse(all.contains(forbidden), fixture.id() + " contains " + forbidden);
        }
    }

    @Test void normalizerFailsClosedForEverySyntheticNonImmediateDifference() {
        var base = List.of("ILOAD|0", "LDC|#17", "FMUL", "IFEQ|L1", "INVOKESTATIC|Math.max", "SALOAD");
        assertTrue(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#19", "FMUL", "IFEQ|L1", "INVOKESTATIC|Math.max", "SALOAD"), 1));
        assertFalse(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#17", "FADD", "IFEQ|L1", "INVOKESTATIC|Math.max", "SALOAD"), 1));
        assertFalse(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#17", "FMUL", "IFNE|L1", "INVOKESTATIC|Math.max", "SALOAD"), 1));
        assertFalse(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#17", "FMUL", "IFEQ|L1", "INVOKESTATIC|StrictMath.pow", "SALOAD"), 1));
        assertFalse(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#17", "FMUL", "IFEQ|L1", "INVOKESTATIC|Math.max", "FALOAD"), 1));
        assertFalse(normalizedTokensEqual(base, List.of("ILOAD|0", "LDC|#17", "FMUL", "IFEQ|L1", "INVOKESTATIC|Math.max", "SALOAD"), 2));
    }

    private static boolean normalizedTokensEqual(List<String> left, List<String> right, int immediateLocation) {
        if (left.size() != right.size()) return false;
        for (int index = 0; index < left.size(); index++) if (!left.get(index).equals(right.get(index))
                && (index != immediateLocation || !left.get(index).startsWith("LDC|") || !right.get(index).startsWith("LDC|"))) return false;
        return !left.equals(right);
    }
}
