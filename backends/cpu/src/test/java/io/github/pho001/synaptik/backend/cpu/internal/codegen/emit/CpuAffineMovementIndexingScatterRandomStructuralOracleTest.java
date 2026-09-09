package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import org.junit.jupiter.api.Test;

/**
 * Family-specific, bounded Class-File projection for CPU 0009C's generated owners.
 *
 * <p>This test is deliberately not a bytecode or CFG identity assertion.  The direct semantic
 * fixtures supply the independently authored topology algorithms; this paired layer verifies
 * that every exact generated entry retains its typed ABI, active half-open range branch, ordered
 * carrier directions, and no forbidden dispatch/allocation mechanism. Its partitions preserve
 * affine address copying, movement topology, indexing writer maps, scatter state, and random
 * state as distinct facts instead of flattening them into operation names. CONCAT and STACK also
 * compare repeated occurrence, boundary, source/output address, and half-open range relations
 * against independent javac clean Java, with semantic mutants required to fail. The test is an
 * exact 2,252-row oracle promotion only; it neither proves fold, aggregate/scan/ordering, nor
 * representative generated/direct performance.</p>
 */
class CpuAffineMovementIndexingScatterRandomStructuralOracleTest {
    private static final String BASE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/";
    private static final String INVENTORY_SHA256 = "527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc";
    private static final Set<String> AFFINE = Set.of("CONTIGUOUS", "EXPAND", "EXPAND_DIMS", "PERMUTE", "RESHAPE", "SELECT", "SLICE", "SQUEEZE");
    private static final Set<String> MOVEMENT = Set.of("PAD", "TILE", "CONCAT", "STACK", "SLICE_UPDATE", "UNFOLD_AXIS", "UNFOLD2D");
    private static final Set<String> INDEXING = Set.of("GATHER", "GATHER_ELEMENTS", "GATHER_ND", "ONE_HOT");
    private static final Set<String> SCATTER = Set.of("SCATTER_ELEMENTS", "SCATTER_ND");
    private static final Set<String> RANDOM = Set.of("DROPOUT", "INITIAL_STATE");

    /**
     * Executes all 1,536 affine rows against independently initialized typed carriers.
     *
     * <p>The clean class is compiled by javac from the prepared affine address-pair contract,
     * while the generated entry is defined from the selected class bytes.  Each range is run on
     * fresh source and result storage, so whole-storage represented bits also prove untouched
     * sentinels outside empty, interior, and one-element tail ranges.</p>
     */
    @Test void everyAffineRowExecutesPairedTypedCleanJavaAddressProgression() throws Throwable {
        var contexts = CpuAffineGeneratedCoverageFixtureTest.coverageFixtureContexts();
        assertEquals(1_536, contexts.size(), "canonical affine fixture contexts");
        var prepared = new java.util.ArrayList<PreparedAffine>();
        var cleanRows = new java.util.ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        for (int index = 0; index < contexts.size(); index++) {
            var analysis = new CpuPartitionPreparer().analyze(contexts.get(index));
            var route = analysis.plan().units().getFirst().portablePlan();
            DataType type = contexts.get(index).values().getFirst().descriptor().dataType();
            boolean denseInt = route.specialization().loopAddressing(route.kernelIr())
                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
            String owner = "affine-matrix:" + index;
            cleanRows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(owner, "affine",
                    route.specialization().entryType().descriptorString(), type.name(), denseInt));
            prepared.add(new PreparedAffine(owner, type, analysis.plan().affineAddressPairs(),
                    analysis.plan().elementCount(), route.specialization(), route.kernelIr()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(cleanRows);
        int executions = 0;
        for (PreparedAffine row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            var generated = generator.defineClassBytes(row.specialization,
                    generator.generateClassBytes(row.specialization, row.kernelIr)).entryPoint();
            var counterpart = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.owner,
                    row.specialization.entryType().descriptorString());
            for (long[] range : ranges(row.count)) {
                Object generatedSource = patterned(row.type, 32), cleanSource = patterned(row.type, 32);
                Object generatedResult = sentinels(row.type, 32), cleanResult = sentinels(row.type, 32);
                Object generatedSourceBefore = cloneArray(generatedSource);
                Object cleanSourceBefore = cloneArray(cleanSource);
                Object generatedResultBefore = cloneArray(generatedResult);
                Object cleanResultBefore = cloneArray(cleanResult);
                generated.invokeWithArguments(argument(row.specialization.carrierPattern().getFirst(), generatedSource),
                        argument(row.specialization.carrierPattern().get(1), generatedResult), row.pairs, range[0], range[1]);
                counterpart.invokeWithArguments(argument(row.specialization.carrierPattern().getFirst(), cleanSource),
                        argument(row.specialization.carrierPattern().get(1), cleanResult), row.pairs, range[0], range[1]);
                assertEveryRawElementEquals(generatedSourceBefore, generatedSource, row.type,
                        row.owner + " generated source is immutable range=" + java.util.Arrays.toString(range));
                assertEveryRawElementEquals(cleanSourceBefore, cleanSource, row.type,
                        row.owner + " clean source is immutable range=" + java.util.Arrays.toString(range));
                for (int cell = 0; cell < 32; cell++) assertEquals(raw(generatedResult, row.type, cell), raw(cleanResult, row.type, cell),
                        row.owner + " represented bits and sentinel cell=" + cell + " range=" + java.util.Arrays.toString(range));
                assertUntouchedAffineResult(row, range, generatedResultBefore, generatedResult, "generated");
                assertUntouchedAffineResult(row, range, cleanResultBefore, cleanResult, "clean");
                executions++;
            }
        }
        assertEquals(1_536 * 4, executions, "full, empty, interior and tail paired executions");
    }

    /**
     * Compares an instruction-derived affine projection of every generated {@code apply} method
     * with the separately compiled clean-Java method.
     *
     * <p>The projection intentionally keeps only facts that describe the hot copy: typed input
     * and output carrier accesses, the {@code long[]} address-pair read/progression, primitive
     * data conversion, and the counted half-open range loop.  It deliberately ignores local
     * slots, labels, and constant-pool positions.  A general generated body may hoist a dense
     * result address and advance it, whereas its straightforward clean counterpart reloads the
     * result pair; that is a checked equivalent address-pair progression, not an erased fact.
     * The guarded BFLOAT16 permute/slice artifact is checked as its separately justified guarded
     * specialization rather than falsely described as the ordinary long-loop shape.</p>
     */
    @Test void everyAffineApplyClassFileMatchesAnIndependentCleanJavaHotLoopProjection()
            throws Throwable {
        var contexts = CpuAffineGeneratedCoverageFixtureTest.coverageFixtureContexts();
        var prepared = new ArrayList<PreparedAffine>();
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        for (int index = 0; index < contexts.size(); index++) {
            var route = new CpuPartitionPreparer().analyze(contexts.get(index)).plan().units().getFirst()
                    .portablePlan();
            boolean denseInt = route.specialization().loopAddressing(route.kernelIr())
                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
            String owner = "affine-matrix:" + index;
            prepared.add(new PreparedAffine(owner, contexts.get(index).values().getFirst().descriptor().dataType(),
                    null, 0L, route.specialization(), route.kernelIr()));
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(owner, "affine",
                    route.specialization().entryType().descriptorString(),
                    contexts.get(index).values().getFirst().descriptor().dataType().name(), denseInt));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int compared = 0, guarded = 0;
        for (PreparedAffine row : prepared) {
            MethodModel cleanMethod = cleanMethods.get(clean.methods().get(row.owner));
            assertTrue(cleanMethod != null, row.owner + " independently compiled clean method");
            byte[] generatedBytes = new CpuClassFileKernelGenerator().generateClassBytes(row.specialization,
                    row.kernelIr);
            MethodModel generatedMethod = selected(methods(generatedBytes));
            assertEquals(row.specialization.entryType().descriptorString(), cleanMethod.methodType().stringValue(),
                    row.owner + " clean typed ABI");
            assertEquals(generatedMethod.methodType().stringValue(), cleanMethod.methodType().stringValue(),
                    row.owner + " generated/clean typed ABI");
            AffineLoopFacts generated = affineFacts(generatedMethod);
            AffineLoopFacts counterpart = affineFacts(cleanMethod);
            assertAffineProjection(row.owner, row.type, row.specialization, row.kernelIr, generated, counterpart);
            assertAffineHygiene(row.owner + " generated", generated, false);
            assertAffineHygiene(row.owner + " clean", counterpart, true);
            if (generated.guarded()) guarded++;
            compared++;
        }
        assertEquals(1_536, compared, "every affine owner receives an independent paired projection");
        // The finite 0009C fixture does not carry the guarded emitter's 524,288-pair geometry.
        // It therefore makes no claim about that separate, exact BFLOAT16 specialization; if a
        // future fixture admits it, the guarded branch above keeps it out of the general-long
        // comparison and requires its explicit address-pair guards instead.
        assertEquals(0, guarded, "finite affine inventory has no guarded-geometry witness");
    }

    /** Proves that each fact retained by the affine projection is active rather than cosmetic. */
    @Test void affineProjectionRejectsAbiCarrierAddressWidthConversionRangeInvokeAndAllowlistDrift() {
        AffineLoopFacts baseline = new AffineLoopFacts(List.of("FALOAD"), List.of("FASTORE"),
                List.of(), List.of(), List.of(), 2, true, true, false, true,
                new RangeFacts(true, true, true), false);
        assertAffineProjection("control", DataType.FLOAT32, null, null, baseline, baseline);
        for (AffineLoopFacts changed : List.of(
                new AffineLoopFacts(List.of("DALOAD"), baseline.stores, baseline.segmentAccesses,
                        baseline.fields, baseline.invokes, 2, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, List.of("DASTORE"), baseline.segmentAccesses,
                        baseline.fields, baseline.invokes, 2, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, baseline.stores, baseline.segmentAccesses, baseline.fields,
                        baseline.invokes, 0, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, baseline.stores, baseline.segmentAccesses, baseline.fields,
                        baseline.invokes, 1, true, false, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, baseline.stores, baseline.segmentAccesses, List.of("D2F"),
                        baseline.invokes, 2, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, baseline.stores, baseline.segmentAccesses, baseline.fields,
                        List.of("java/lang/Math.sin"), 2, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false),
                new AffineLoopFacts(baseline.loads, baseline.stores, baseline.segmentAccesses, baseline.fields,
                        baseline.invokes, 2, true, true, baseline.intCursor, baseline.longCursor, new RangeFacts(false, true, true), false),
                new AffineLoopFacts(baseline.loads, baseline.stores, List.of("SEGMENT_GET:OfDouble:D"),
                        baseline.fields, baseline.invokes, 2, true, true, baseline.intCursor, baseline.longCursor, baseline.range, false)))
            assertThrows(AssertionError.class,
                    () -> assertAffineProjection("mutated", DataType.FLOAT32, null, null, changed, baseline),
                    "must reject " + changed);
        assertThrows(AssertionError.class, () -> assertEquals("([F[F[JJJ)V", "([D[F[JJJ)V",
                "mutated typed carrier ABI"));
    }

    /** Actual javac-produced affine mutations keep ABI, pair-map, range, sink and hygiene checks live. */
    @Test void affineProjectionRejectsExecutableSourceMutants() {
        var basis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("control", "affine",
                "(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;[JJJ)V", "FLOAT32", false);
        var canonical = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(List.of(basis));
        MethodModel baseline = methods(canonical.bytes()).get("entry0");
        assertTrue(baseline != null, "canonical affine javac witness");
        var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileAffineMutants(basis);
        for (var entry : mutants.methods().entrySet()) {
            String kind = entry.getKey(); MethodModel mutant = methods(mutants.bytes()).get(entry.getValue());
            assertTrue(mutant != null, "compiled executable affine mutant " + kind);
            if (Set.of("new", "anewarray", "newarray", "multianewarray", "athrow", "invokedynamic",
                    "boxing", "reflection", "string-dispatch", "map-dispatch", "hidden-helper", "cross-ffm").contains(kind)) {
                assertThrows(AssertionError.class, () -> assertAffineHygiene("mutant " + kind, affineFacts(mutant), true));
            } else if (kind.equals("abi")) {
                assertNotEquals(baseline.methodType().stringValue(), mutant.methodType().stringValue(), "affine ABI source mutant");
            } else {
                assertNotEquals(sourceInstructionShape(baseline), sourceInstructionShape(mutant),
                        "affine address/range/carrier/store source mutant " + kind);
            }
        }
    }

    private static AffineLoopFacts affineFacts(MethodModel method) {
        List<CodeElement> elements = method.code().orElseThrow().elementStream().toList();
        var labels = new java.util.IdentityHashMap<Label, Integer>();
        int position = 0;
        for (CodeElement element : elements) {
            if (element instanceof LabelTarget target) labels.put(target.label(), position);
            if (element instanceof Instruction) position++;
        }
        List<String> loads = new ArrayList<>(), stores = new ArrayList<>(), segments = new ArrayList<>(),
                fields = new ArrayList<>(), invokes = new ArrayList<>();
        int longArrayLoads = 0, backward = 0, arrayLengths = 0;
        boolean intCursor = false, longCursor = false, cursorIncrement = false, resultIncrement = false,
                guarded = false, forward = false;
        position = 0;
        for (CodeElement element : elements) {
            if (element instanceof LabelTarget target) labels.put(target.label(), position);
            if (!(element instanceof Instruction instruction)) continue;
            int instructionPosition = position++;
            String opcode = instruction.opcode().name();
            if (List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "ATHROW", "INVOKEDYNAMIC").contains(opcode))
                fields.add(opcode);
            if (opcode.endsWith("ALOAD") && !opcode.equals("AALOAD")) loads.add(opcode);
            if (opcode.endsWith("ASTORE") && !opcode.equals("AASTORE")) stores.add(opcode);
            if (opcode.equals("LALOAD")) longArrayLoads++;
            if (opcode.equals("IINC")) { cursorIncrement = true; resultIncrement = true; }
            if (opcode.equals("LADD")) { cursorIncrement = true; resultIncrement = true; }
            if (opcode.equals("L2I") || opcode.equals("I2L")) { /* address narrowing only */ }
            else if (opcode.matches("[BSCZIFJD]2[BSCZIFJD]")) fields.add(opcode);
            if (opcode.startsWith("ILOAD") || opcode.startsWith("IF_ICMP")) intCursor = true;
            if (opcode.startsWith("LLOAD") || opcode.equals("LCMP")) longCursor = true;
            if (instruction instanceof FieldInstruction field) {
                String fact = field.owner().asInternalName() + '.' + field.name() + ':' + field.type();
                fields.add(fact);
            }
            if (instruction instanceof InvokeInstruction call) {
                String detail = call.owner().asInternalName() + '.' + call.name() + call.type();
                invokes.add(detail);
                if (detail.startsWith("java/lang/foreign/MemorySegment.get")) {
                    String fact = "SEGMENT_GET:" + segmentLayout(detail) + ':' + methodReturn(detail);
                    loads.add(fact); segments.add(fact);
                } else if (detail.startsWith("java/lang/foreign/MemorySegment.set")) {
                    String fact = "SEGMENT_SET:" + segmentLayout(detail) + ':' + primitivePayload(detail);
                    stores.add(fact); segments.add(fact);
                }
            }
            if (instruction instanceof BranchInstruction branch) {
                int target = labels.get(branch.target());
                if (target < instructionPosition) backward++; else forward = true;
            }
            // The guarded emitter identifies a fixed pair table and range before entering its
            // dedicated cursor loop.  Its field-free body is detected from those exact array
            // guards and is still subjected to typed carrier and hygiene checks below.
            if (opcode.equals("ARRAYLENGTH")) { guarded = true; arrayLengths++; }
        }
        // A long[] source is itself a primitive payload carrier. Keep its direct read distinct
        // from cold long[] address-pair reads, so INT64 cannot lose its typed width to the
        // address-table projection.
        boolean longArraySource = method.methodType().stringValue().startsWith("([J");
        if (longArraySource) {
            int payload = loads.lastIndexOf("LALOAD");
            assertTrue(payload >= 0, "INT64 source has direct LALOAD payload access");
            loads.set(payload, "LALOAD_DATA");
        }
        return new AffineLoopFacts(List.copyOf(loads), List.copyOf(stores), List.copyOf(segments),
                List.copyOf(fields), List.copyOf(invokes), longArrayLoads - (longArraySource ? 1 : 0), cursorIncrement, resultIncrement,
                intCursor, longCursor, new RangeFacts(forward, backward > 0, intCursor || longCursor),
                guarded, arrayLengths);
    }

    private static void assertAffineProjection(String owner, DataType type,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr, AffineLoopFacts generated,
            AffineLoopFacts clean) {
        List<String> expectedLoad = List.of(carrierAccess(type, specialization, 0, false));
        List<String> expectedStore = List.of(carrierAccess(type, specialization, 1, true));
        assertEquals(expectedLoad, carrierProjection(clean.loads, false),
                owner + " clean type-derived source access");
        assertEquals(expectedStore, carrierProjection(clean.stores, true),
                owner + " clean type-derived result access");
        assertEquals(expectedLoad, carrierProjection(generated.loads, false),
                owner + " generated type-derived source access");
        assertEquals(expectedStore, carrierProjection(generated.stores, true),
                owner + " generated type-derived result access");
        assertEquals(carrierProjection(clean.loads, false), carrierProjection(generated.loads, false),
                owner + " typed source carrier/load width and direction");
        assertEquals(carrierProjection(clean.stores, false), carrierProjection(generated.stores, false),
                owner + " typed result carrier/store width and direction");
        assertEquals(clean.segmentAccesses, generated.segmentAccesses,
                owner + " ordered FFM carrier method/layout facts");
        assertTrue(clean.pairLoads >= 2, owner + " clean source/result address-pair lookup");
        assertTrue(generated.pairLoads >= 1, owner + " generated source address-pair lookup");
        assertTrue(generated.resultProgression || generated.pairLoads >= 2,
                owner + " generated result pair lookup or checked dense progression");
        assertTrue(clean.resultProgression || clean.pairLoads >= 2,
                owner + " clean result pair lookup or checked dense progression");
        assertEquals(List.of(), dataConversions(clean.fields), owner + " clean represented-bit copy has no conversion");
        assertEquals(List.of(), dataConversions(generated.fields), owner + " generated represented-bit copy has no conversion");
        assertTrue(generated.invokes.stream().allMatch(
                CpuAffineMovementIndexingScatterRandomStructuralOracleTest::permittedAffineInvoke),
                owner + " direct copy has no helper, numerical, reflective, boxing, or dynamic invoke");
        assertRange(owner + " clean", clean.range);
        assertRange(owner + " generated", generated.range);
        boolean denseInt = specialization != null && specialization.loopAddressing(kernelIr)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT;
        if (denseInt) {
            assertTrue(clean.intCursor && generated.intCursor, owner + " dense int cursor form");
            assertFalse(clean.longCursor && !clean.intCursor, owner + " clean dense form is not general long only");
            assertTrue(clean.resultProgression && generated.resultProgression,
                    owner + " dense input/result cursor progression");
        } else if (generated.guarded) {
            assertGuardedAffineProjection(owner, type, specialization, generated);
        } else {
            assertTrue(clean.longCursor && generated.longCursor, owner + " general long cursor form");
        }
        assertTrue(generated.loads.size() >= 1 && generated.stores.size() >= 1,
                owner + " direct represented-bit load/store");
    }

    /**
     * Checks the emitter's guarded raw-BFLOAT16 body as a distinct realization.
     *
     * <p>The fast body validates a fixed packed address-pair table before its integer cursor
     * loop and retains the general-long body as a fallback.  It must therefore never be accepted
     * by the ordinary general-long comparison merely because both bodies happen to copy shorts.
     * This deliberately fails closed for any future array-length guard with a different shape.</p>
     */
    private static void assertGuardedAffineProjection(String owner, DataType type,
            CpuKernelSpecialization specialization, AffineLoopFacts facts) {
        assertEquals(DataType.BFLOAT16, type, owner + " guarded path is raw BFLOAT16 only");
        assertEquals(List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY),
                specialization.carrierPattern(), owner + " guarded carrier direction");
        assertEquals(1, facts.arrayLengths, owner + " one fixed-pair-table guard");
        assertTrue(facts.intCursor && facts.longCursor,
                owner + " guarded integer fast loop and general-long fallback");
        assertTrue(facts.pairLoads >= 13,
                owner + " six checked source/result pairs plus general fallback pair lookup");
        assertRange(owner + " guarded", facts.range);
    }

    private static String carrierAccess(DataType type, CpuKernelSpecialization specialization,
            int role, boolean store) {
        if (specialization == null) {
            // Negative controls use FLOAT32 arrays and do not have a prepared carrier pattern.
            return store ? "FASTORE" : "FALOAD";
        }
        if (specialization.carrierPattern().get(role)
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            String layout = switch (type) {
                case FLOAT64 -> "OfDouble:D";
                case FLOAT32 -> "OfFloat:F";
                case BFLOAT16 -> "OfShort:S";
                case INT64 -> "OfLong:J";
                case INT32 -> "OfInt:I";
                case BOOL -> "OfByte:B";
            };
            return (store ? "SEGMENT_SET:" : "SEGMENT_GET:") + layout;
        }
        return switch (type) {
            case FLOAT64 -> store ? "DASTORE" : "DALOAD";
            case FLOAT32 -> store ? "FASTORE" : "FALOAD";
            case BFLOAT16 -> store ? "SASTORE" : "SALOAD";
            case INT64 -> store ? "LASTORE" : "LALOAD_DATA";
            case INT32 -> store ? "IASTORE" : "IALOAD";
            case BOOL -> store ? "BASTORE" : "BALOAD";
        };
    }

    private static void assertRange(String owner, RangeFacts range) {
        assertTrue(range.forwardBranch && range.backwardBranch && range.cursorComparison,
                owner + " active half-open [start,end) entry and tail loop branches");
    }

    private static List<String> carrierProjection(List<String> accesses, boolean stores) {
        List<String> result = new ArrayList<>();
        for (String access : accesses) {
            if (access.equals("LALOAD")) continue; // cold address-pair read, not a data carrier
            if (access.equals("LALOAD_DATA")) { result.add(access); continue; }
            if (access.startsWith("SEGMENT_")) result.add(access);
            else if (access.endsWith("ALOAD") && !access.equals("ALOAD")) result.add(access);
            else if (access.endsWith("ASTORE") && !access.equals("ASTORE")) result.add(access);
        }
        return List.copyOf(result);
    }

    private static List<String> dataConversions(List<String> facts) {
        return facts.stream().filter(fact -> fact.matches("[DFISL]2[DFISL]")
                || fact.contains("BitsTo") || fact.contains("ToRaw")).toList();
    }

    private static void assertAffineHygiene(String owner, AffineLoopFacts facts, boolean clean) {
        for (String forbidden : List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "ATHROW", "INVOKEDYNAMIC"))
            assertFalse(facts.fields.contains(forbidden) || facts.invokes.contains(forbidden),
                    owner + " forbidden allocation/dynamic mechanism: " + forbidden);
        for (String invocation : facts.invokes) {
            assertTrue(clean ? permittedCleanFfmInvoke(invocation) : permittedGeneratedFfmInvoke(invocation),
                    owner + " exact selected-entry FFM allowlist: " + invocation);
        }
        for (String field : facts.fields) if (field.contains("/"))
            assertTrue(field.startsWith("java/lang/foreign/ValueLayout.JAVA_"),
                    owner + " only typed FFM ValueLayout field is allowed: " + field);
    }

    private static boolean permittedAffineInvoke(String invocation) { return permittedGeneratedFfmInvoke(invocation); }
    private static boolean permittedGeneratedFfmInvoke(String invocation) { return exactFfmInvokes("get", "set").contains(invocation) || ffmBootstrap(invocation); }
    private static boolean permittedCleanFfmInvoke(String invocation) { return exactFfmInvokes("getAtIndex", "setAtIndex").contains(invocation) || ffmBootstrap(invocation); }
    private static boolean ffmBootstrap(String invocation) { return invocation.equals("java/nio/ByteOrder.nativeOrder()Ljava/nio/ByteOrder;") || invocation.equals("java/lang/foreign/ValueLayout.withOrder(Ljava/nio/ByteOrder;)Ljava/lang/foreign/ValueLayout;"); }
    private static Set<String> exactFfmInvokes(String get, String set) {
        return Set.of(
                "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfDouble;J)D", "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfFloat;J)F", "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfShort;J)S", "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfLong;J)J", "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfInt;J)I", "java/lang/foreign/MemorySegment." + get + "(Ljava/lang/foreign/ValueLayout$OfByte;J)B",
                "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfDouble;JD)V", "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfFloat;JF)V", "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfShort;JS)V", "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfLong;JJ)V", "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfInt;JI)V", "java/lang/foreign/MemorySegment." + set + "(Ljava/lang/foreign/ValueLayout$OfByte;JB)V");
    }

    private static String methodReturn(String detail) { return detail.substring(detail.lastIndexOf(')') + 1); }
    /** Extracts the primitive payload from a direct FFM set descriptor, excluding layout/address. */
    private static String primitivePayload(String detail) {
        int close = detail.lastIndexOf(')');
        assertTrue(close > 0, "typed MemorySegment set descriptor in " + detail);
        return detail.substring(close - 1, close);
    }
    private static String segmentLayout(String detail) {
        int start = detail.indexOf("(Ljava/lang/foreign/ValueLayout$");
        int end = detail.indexOf(';', start);
        assertTrue(start >= 0 && end > start, "typed MemorySegment layout in " + detail);
        return detail.substring(start + "(Ljava/lang/foreign/ValueLayout$".length(), end);
    }
    private record RangeFacts(boolean forwardBranch, boolean backwardBranch, boolean cursorComparison) { }
    private record AffineLoopFacts(List<String> loads, List<String> stores, List<String> segmentAccesses,
            List<String> fields, List<String> invokes, int pairLoads, boolean cursorProgression,
            boolean resultProgression, boolean intCursor, boolean longCursor, RangeFacts range,
            boolean guarded, int arrayLengths) {
        AffineLoopFacts(List<String> loads, List<String> stores, List<String> segmentAccesses,
                List<String> fields, List<String> invokes, int pairLoads, boolean cursorProgression,
                boolean resultProgression, boolean intCursor, boolean longCursor, RangeFacts range,
                boolean guarded) {
            this(loads, stores, segmentAccesses, fields, invokes, pairLoads, cursorProgression,
                    resultProgression, intCursor, longCursor, range, guarded, guarded ? 1 : 0);
        }
    }

    private static List<long[]> ranges(long count) {
        return List.of(new long[] {0, count}, new long[] {0, 0}, new long[] {1, count - 1}, new long[] {count - 1, count});
    }
    private static Object argument(CpuKernelSpecialization.CarrierAccess carrier, Object array) {
        return carrier == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT ? segment(array) : array;
    }
    private static Object patterned(DataType type, int n) { return switch (type) {
        case FLOAT64 -> { double[] x = new double[n]; for (int i=0;i<n;i++) x[i]=Double.longBitsToDouble(0x7ff8000000000000L+i); yield x; }
        case FLOAT32 -> { float[] x = new float[n]; for (int i=0;i<n;i++) x[i]=Float.intBitsToFloat(0x7fc00000+i); yield x; }
        case BFLOAT16 -> { short[] x = new short[n]; for (int i=0;i<n;i++) x[i]=(short)(0x8000+i); yield x; }
        case INT64 -> { long[] x = new long[n]; for (int i=0;i<n;i++) x[i]=0x1020304050607080L+i; yield x; }
        case INT32 -> { int[] x = new int[n]; for (int i=0;i<n;i++) x[i]=0x10203040+i; yield x; }
        // BOOL semantic inputs are canonical.  Their alternating values, paired with the
        // inverse canonical output sentinel below, make a copied BOOL visibly distinct at
        // every same-address affine cell without relying on invalid nonzero byte encodings.
        case BOOL -> { byte[] x = new byte[n]; for (int i=0;i<n;i++) x[i]=(byte)(i & 1); yield x; }
    }; }
    private static Object sentinels(DataType type, int n) {
        if (type != DataType.BOOL) return patterned(type, n);
        // A BOOL result must remain canonical.  Use the inverse 0/1 pattern and separately
        // assert every address outside the requested affine range remains this exact snapshot.
        // The range-address assertion, rather than a noncanonical sentinel, proves untouched
        // output cells; the paired alternating input makes same-address writes observable.
        byte[] x = new byte[n]; for (int i = 0; i < n; i++) x[i] = (byte) ((i + 1) & 1); return x;
    }
    private static MemorySegment segment(Object array) { if (array instanceof double[] x) return MemorySegment.ofArray(x); if (array instanceof float[] x) return MemorySegment.ofArray(x); if (array instanceof short[] x) return MemorySegment.ofArray(x); if (array instanceof long[] x) return MemorySegment.ofArray(x); if (array instanceof int[] x) return MemorySegment.ofArray(x); return MemorySegment.ofArray((byte[]) array); }
    private static long raw(Object value, DataType type, int i) { return switch (type) { case FLOAT64 -> Double.doubleToRawLongBits(((double[]) value)[i]); case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(((float[]) value)[i])); case BFLOAT16 -> Short.toUnsignedLong(((short[]) value)[i]); case INT64 -> ((long[]) value)[i]; case INT32 -> Integer.toUnsignedLong(((int[]) value)[i]); case BOOL -> Byte.toUnsignedLong(((byte[]) value)[i]); }; }
    private static void assertUntouchedAffineResult(PreparedAffine row, long[] range, Object before,
            Object after, String implementation) {
        for (int cell = 0; cell < 32; cell++) {
            boolean owned = false;
            for (long cursor = range[0]; cursor < range[1]; cursor++) {
                if (row.pairs[Math.toIntExact(cursor * 2 + 1)] == cell) { owned = true; break; }
            }
            if (!owned) assertEquals(raw(before, row.type, cell), raw(after, row.type, cell),
                    row.owner + " " + implementation + " untouched output sentinel cell=" + cell
                            + " range=" + java.util.Arrays.toString(range));
        }
    }
    private static void assertEveryRawElementEquals(Object expected, Object actual, DataType type,
            String label) {
        int length = switch (type) {
            case FLOAT64 -> ((double[]) expected).length;
            case FLOAT32 -> ((float[]) expected).length;
            case BFLOAT16 -> ((short[]) expected).length;
            case INT64 -> ((long[]) expected).length;
            case INT32 -> ((int[]) expected).length;
            case BOOL -> ((byte[]) expected).length;
        };
        for (int cell = 0; cell < length; cell++) assertEquals(raw(expected, type, cell), raw(actual, type, cell),
                label + " raw cell=" + cell);
    }
    private record PreparedAffine(String owner, DataType type, long[] pairs, long count,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr) { }

    /**
     * Executes the first movement sub-slice through independently compiled PAD and TILE bodies.
     *
     * <p>Both bodies consume the packed cold geometry directly. PAD tests its represented
     * padding bits before mapping a source address; TILE advances output coordinates and the
     * wrapped source coordinates together. The generated entry and javac counterpart receive
     * fresh typed heap/segment carriers for every range.</p>
     */
    @Test void everyPadAndTileRowExecutesPairedTypedCleanJavaMovementLoops() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(candidate -> candidate.operationForm().equals("PAD") || candidate.operationForm().equals("TILE"))
                .toList();
        assertEquals(28, candidates.size(), "exact PAD/TILE movement sub-slice");
        assertEquals(Map.of("PAD", 4L, "TILE", 24L), movementCounts(candidates));
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var shape = candidate.context().nodes().getFirst().outputs().getFirst();
            int rank = candidate.context().values().stream().filter(value -> value.id().equals(shape)).findFirst()
                    .orElseThrow().descriptor().shape().rank();
            DataType type = candidate.context().values().stream().filter(value -> value.id().equals(shape)).findFirst()
                    .orElseThrow().descriptor().dataType();
            long bits = candidate.operationForm().equals("PAD") ? padBits(
                    (io.github.pho001.synaptik.model.operation.layout.PadAttrs) candidate.context().nodes().getFirst().operation().attrs()) : 0L;
            String family = candidate.operationForm().equals("PAD") ? "pad" : "tile";
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), family,
                    route.specialization().entryType().descriptorString(), type.name(), false, rank, bits));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), type));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        int executions = 0;
        for (PreparedMovement row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            byte[] first = generator.generateClassBytes(row.specialization, row.kernelIr);
            assertEquals(java.util.HexFormat.of().formatHex(first), java.util.HexFormat.of().formatHex(
                    generator.generateClassBytes(row.specialization, row.kernelIr)), row.candidate.ownerId() + " deterministic bytes");
            var generated = generator.defineClassBytes(row.specialization, first).entryPoint();
            var counterpart = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.candidate.ownerId(),
                    row.specialization.entryType().descriptorString());
            long count = row.plan.elementCount();
            for (long[] range : ranges(count)) {
                var generatedCarriers = movementCarriers(row); var cleanCarriers = movementCarriers(row);
                Object generatedSource = generatedCarriers.getFirst().array, cleanSource = cleanCarriers.getFirst().array;
                Object sourceBefore = cloneArray(generatedSource);
                long[] geometry = row.plan.movementGeometry().orElseThrow().pack(new long[row.plan.boundaryValues().size()], range[0], range[1]);
                List<Object> ga = new ArrayList<>(), ca = new ArrayList<>();
                for (int i = 0; i < generatedCarriers.size(); i++) { ga.add(generatedCarriers.get(i).carrier); ca.add(cleanCarriers.get(i).carrier); }
                ga.add(geometry); ga.add(range[0]); ga.add(range[1]); ca.add(geometry.clone()); ca.add(range[0]); ca.add(range[1]);
                generated.invokeWithArguments(ga); counterpart.invokeWithArguments(ca);
                for (int i = 0; i < generatedCarriers.size(); i++) assertRawArrayEquals(generatedCarriers.get(i).array,
                        cleanCarriers.get(i).array, row.candidate.ownerId() + " bits/sentinels range=" + java.util.Arrays.toString(range) + " geometry=" + java.util.Arrays.toString(geometry));
                assertRawArrayEquals(sourceBefore, generatedSource, row.candidate.ownerId() + " source immutable");
                executions++;
            }
        }
        assertEquals(112, executions, "28 rows × full, empty, interior and tail ranges");
    }

    /**
     * Compares the PAD and TILE hot-loop topology which the generated and independently
     * compiled bodies both expose.
     *
     * <p>This is intentionally a projection, not an instruction or control-flow-graph identity
     * check.  javac reloads geometry where the generated dense form keeps an integer cursor, and
     * the bounded PAD emitter has a full-range fast body plus a ranged fallback.  Those are
     * distinct, checked realizations: neither permits losing the typed source/result roles,
     * range back-edge, output advance, PAD's mapped-source/fill selection, or TILE's carried and
     * wrapped source address.</p>
     */
    @Test void everyPadAndTileApplyMatchesIndependentTypedTopologyProjection() throws Exception {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(candidate -> candidate.operationForm().equals("PAD") || candidate.operationForm().equals("TILE"))
                .toList();
        assertEquals(28, candidates.size(), "exact PAD/TILE structural sub-slice");
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(value -> value.id().equals(output))
                    .findFirst().orElseThrow().descriptor();
            long bits = candidate.operationForm().equals("PAD") ? padBits(
                    (io.github.pho001.synaptik.model.operation.layout.PadAttrs) candidate.context().nodes().getFirst().operation().attrs()) : 0L;
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(),
                    candidate.operationForm().toLowerCase(), route.specialization().entryType().descriptorString(),
                    descriptor.dataType().name(), route.specialization().loopAddressing(route.kernelIr())
                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT,
                    descriptor.shape().rank(), bits));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int pads = 0, tiles = 0;
        for (PreparedMovement row : prepared) {
            String owner = row.candidate.ownerId();
            MethodModel counterpart = cleanMethods.get(clean.methods().get(owner));
            assertTrue(counterpart != null, owner + " independently compiled movement method");
            MethodModel generated = selected(methods(new CpuClassFileKernelGenerator().generateClassBytes(
                    row.specialization, row.kernelIr)));
            MovementFacts generatedFacts = movementFacts(generated);
            MovementFacts cleanFacts = movementFacts(counterpart);
            long bits = row.candidate.operationForm().equals("PAD") ? padBits(
                    (io.github.pho001.synaptik.model.operation.layout.PadAttrs) row.candidate.context().nodes().getFirst().operation().attrs()) : 0L;
            assertMovementEntryProjection(row.specialization.entryType().descriptorString(),
                    generated.methodType().stringValue(), counterpart.methodType().stringValue(), owner,
                    row.candidate.operationForm(), row.type, bits, row.specialization, generatedFacts, cleanFacts);
            int rank = row.candidate.context().values().stream().filter(value -> value.id().equals(
                    row.candidate.context().nodes().getFirst().outputs().getFirst())).findFirst()
                    .orElseThrow().descriptor().shape().rank();
            assertPackedMovementGeometry(owner, row.candidate.operationForm(), rank, generated, counterpart);
            assertMovementHygiene(owner + " generated", row.candidate.operationForm(), generatedFacts, false);
            assertMovementHygiene(owner + " clean", row.candidate.operationForm(), cleanFacts, true);
            if (row.candidate.operationForm().equals("PAD")) pads++; else tiles++;
        }
        assertEquals(4, pads, "all PAD rows retain topology facts");
        assertEquals(24, tiles, "all TILE rows retain topology facts");
    }

    /** Ensures PAD/TILE comparisons reject mutations of each retained topology fact. */
    @Test void padAndTileProjectionRejectsAbiRolesRangeProgressionSelectionWrapAndInvokeDrift() {
        MovementFacts pad = new MovementFacts(List.of("FALOAD"), List.of("FASTORE", "FASTORE"), List.of(),
                List.of("java/lang/Float.intBitsToFloat(I)F"), List.of(),
                new RangeFacts(true, true, true), true, true, true, true, false, false, false);
        MovementFacts generatedPad = new MovementFacts(List.of("FALOAD"), List.of("FASTORE"), List.of(),
                List.of("java/lang/Float.intBitsToFloat(I)F"), List.of(),
                new RangeFacts(true, true, true), true, true, true, true, false, false, false);
        MovementFacts tile = new MovementFacts(List.of("FALOAD"), List.of("FASTORE"), List.of(),
                List.of(), List.of(), new RangeFacts(true, true, true), true, true, false, false,
                true, true, false);
        assertMovementProjection("pad control", "PAD", DataType.FLOAT32, 0x7fc00001L, null, generatedPad, pad);
        assertMovementProjection("tile control", "TILE", DataType.FLOAT32, 0L, null, tile, tile);
        for (MovementFacts changed : List.of(
                new MovementFacts(List.of("DALOAD"), pad.stores, pad.segments, pad.invokes, pad.forbidden,
                        pad.range, pad.outputProgression, pad.sourceProgression, pad.padSelection,
                        pad.padImmediate, pad.tileCarry, pad.tileWrap, pad.dense),
                new MovementFacts(pad.loads, List.of("DASTORE"), pad.segments, pad.invokes, pad.forbidden,
                        pad.range, pad.outputProgression, pad.sourceProgression, pad.padSelection,
                        pad.padImmediate, pad.tileCarry, pad.tileWrap, pad.dense),
                new MovementFacts(pad.loads, pad.stores, pad.segments, pad.invokes, pad.forbidden,
                        new RangeFacts(true, false, true), pad.outputProgression, pad.sourceProgression,
                        pad.padSelection, pad.padImmediate, pad.tileCarry, pad.tileWrap, pad.dense),
                new MovementFacts(pad.loads, pad.stores, pad.segments, pad.invokes, pad.forbidden,
                        pad.range, false, pad.sourceProgression, pad.padSelection, pad.padImmediate,
                        pad.tileCarry, pad.tileWrap, pad.dense),
                new MovementFacts(pad.loads, pad.stores, pad.segments, List.of(), pad.forbidden,
                        pad.range, pad.outputProgression, pad.sourceProgression, pad.padSelection, false,
                        pad.tileCarry, pad.tileWrap, pad.dense)))
            assertThrows(AssertionError.class, () -> assertMovementProjection("PAD mutation", "PAD",
                    DataType.FLOAT32, 0x7fc00001L, null, changed, pad));
        for (MovementFacts changed : List.of(
                new MovementFacts(tile.loads, tile.stores, tile.segments, tile.invokes, tile.forbidden,
                        tile.range, tile.outputProgression, false, tile.padSelection, tile.padImmediate,
                        tile.tileCarry, tile.tileWrap, tile.dense),
                new MovementFacts(tile.loads, tile.stores, tile.segments, tile.invokes, tile.forbidden,
                        tile.range, tile.outputProgression, tile.sourceProgression, tile.padSelection,
                        tile.padImmediate, false, tile.tileWrap, tile.dense),
                new MovementFacts(tile.loads, tile.stores, tile.segments, tile.invokes, tile.forbidden,
                        tile.range, tile.outputProgression, tile.sourceProgression, tile.padSelection,
                        tile.padImmediate, tile.tileCarry, false, tile.dense)))
            assertThrows(AssertionError.class, () -> assertMovementProjection("TILE mutation", "TILE",
                    DataType.FLOAT32, 0L, null, changed, tile));
        MovementFacts badInvoke = new MovementFacts(tile.loads, tile.stores, tile.segments,
                List.of("java/lang/Math.sin(D)D"), tile.forbidden, tile.range, tile.outputProgression,
                tile.sourceProgression, tile.padSelection, tile.padImmediate, tile.tileCarry,
                tile.tileWrap, tile.dense);
        assertThrows(AssertionError.class, () -> assertMovementHygiene("invoke mutation", "TILE", badInvoke, false));
        for (String forbidden : List.of("I2F", "NEW", "INVOKEDYNAMIC")) {
            MovementFacts badOverhead = new MovementFacts(tile.loads, tile.stores, tile.segments,
                    tile.invokes, List.of(forbidden), tile.range, tile.outputProgression,
                    tile.sourceProgression, tile.padSelection, tile.padImmediate, tile.tileCarry,
                    tile.tileWrap, tile.dense);
            assertThrows(AssertionError.class,
                    () -> assertMovementHygiene("overhead mutation " + forbidden, "TILE", badOverhead, false),
                    "must reject conversion, allocation, or dynamic dispatch");
        }
        assertThrows(AssertionError.class, () -> assertMovementEntryProjection("([F[F[JJJ)V",
                "([F[Ljava/lang/foreign/MemorySegment;[JJJ)V", "([F[F[JJJ)V", "ABI mutation", "PAD",
                DataType.FLOAT32, 0x7fc00001L, null, generatedPad, pad));
        assertThrows(AssertionError.class, () -> assertPackedGeometrySlots("PAD mutation", "PAD", 2,
                Set.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                Set.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
                "must reject a missing PAD input-extent packed slot");
        assertThrows(AssertionError.class, () -> assertPackedGeometrySlots("TILE mutation", "TILE", 2,
                Set.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                Set.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13)),
                "must reject a missing TILE wrapped-coordinate packed slot");
    }

    /**
     * Checks the concrete packed-geometry slot contract, rather than merely counting geometry
     * loads.  The Class-File stream exposes every literal array subscript immediately before its
     * {@code LALOAD}; this retains the source-independent slot identities while allowing javac
     * and direct emission to choose different locals and branch layouts.
     *
     * @param owner exact generated-row owner for diagnostics
     * @param form checked PAD or TILE form
     * @param rank resolved output rank whose packed layout is checked
     * @param generated selected generated entry method
     * @param clean independently compiled clean-Java counterpart method
     */
    private static void assertPackedMovementGeometry(String owner, String form, int rank,
            MethodModel generated, MethodModel clean) {
        Set<Integer> expected = packedSlots(form, rank);
        assertPackedGeometrySlots(owner + " generated", form, rank, expected, geometrySlots(generated));
        assertPackedGeometrySlots(owner + " clean", form, rank, expected, geometrySlots(clean));
        // A slot read alone is not enough: PAD's before/extent slots feed two coordinate bounds
        // per axis before the source base-plus-stride map, while TILE's extent/wrapped-coordinate
        // slots feed the carried reset correction.  These are instruction facts, not emitter text.
        assertTrue(opcodeCount(generated, "IMUL", "LMUL") >= rank,
                owner + " generated " + form + " source address is base plus rank-specific stride products");
        assertTrue(opcodeCount(clean, "IMUL", "LMUL") >= rank,
                owner + " clean " + form + " source address is base plus rank-specific stride products");
        if (form.equals("PAD")) {
            assertTrue(conditionalBranches(generated) >= 2 * rank + 1,
                    owner + " generated PAD has per-axis lower/upper destination bounds plus range guard");
            assertTrue(conditionalBranches(clean) >= 2 * rank + 1,
                    owner + " clean PAD has per-axis lower/upper destination bounds plus range guard");
        } else {
            assertTrue(conditionalBranches(generated) >= 2 * rank + 1,
                    owner + " generated TILE carries/wraps rank-specific coordinates plus range guard");
            assertTrue(conditionalBranches(clean) >= 2 * rank + 1,
                    owner + " clean TILE carries/wraps rank-specific coordinates plus range guard");
        }
    }

    private static Set<Integer> packedSlots(String form, int rank) {
        assertTrue(form.equals("PAD") || form.equals("TILE"), "checked movement form");
        var slots = new java.util.TreeSet<Integer>();
        // output extents, start coordinates, output address, and output strides
        addSlots(slots, 0, 3 * rank);
        // input base and rank-specific source strides
        addSlots(slots, 3 * rank + 1, 4 * rank + 1);
        // PAD: before-padding and source extents. TILE: source extents and wrapped coordinates.
        addSlots(slots, 4 * rank + 2, 6 * rank + 1);
        return Set.copyOf(slots);
    }

    private static void addSlots(Set<Integer> slots, int first, int last) {
        for (int slot = first; slot <= last; slot++) slots.add(slot);
    }

    private static void assertPackedGeometrySlots(String owner, String form, int rank,
            Set<Integer> expected, Set<Integer> actual) {
        assertTrue(actual.containsAll(expected), owner + " " + form + " rank=" + rank
                + " reads every concrete packed geometry slot: expected=" + expected + " actual=" + actual);
    }

    private static Set<Integer> geometrySlots(MethodModel method) {
        var slots = new java.util.TreeSet<Integer>();
        Integer precedingConstant = null;
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            if (instruction instanceof ConstantInstruction constant
                    && constant.constantValue() instanceof Number number) {
                precedingConstant = number.intValue();
            } else {
                if (instruction.opcode().name().equals("LALOAD") && precedingConstant != null
                        && precedingConstant >= 0) slots.add(precedingConstant);
                precedingConstant = null;
            }
        }
        return Set.copyOf(slots);
    }

    private static long opcodeCount(MethodModel method, String... opcodes) {
        Set<String> accepted = Set.of(opcodes);
        return method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(instruction -> accepted.contains(instruction.opcode().name())).count();
    }

    private static long conditionalBranches(MethodModel method) {
        return method.code().orElseThrow().elementStream().filter(BranchInstruction.class::isInstance)
                .map(BranchInstruction.class::cast).filter(branch -> branch.opcode().name().startsWith("IF")).count();
    }

    private static MovementFacts movementFacts(MethodModel method) {
        List<CodeElement> elements = method.code().orElseThrow().elementStream().toList();
        var labels = new java.util.IdentityHashMap<Label, Integer>();
        int position = 0;
        for (CodeElement element : elements) {
            if (element instanceof LabelTarget target) labels.put(target.label(), position);
            if (element instanceof Instruction) position++;
        }
        List<String> loads = new ArrayList<>(), stores = new ArrayList<>(), segments = new ArrayList<>(),
                invokes = new ArrayList<>(), forbidden = new ArrayList<>();
        int forward = 0, backward = 0, add = 0, multiply = 0;
        boolean cursorComparison = false, outputProgression = false, sourceProgression = false;
        for (CodeElement element : elements) {
            if (!(element instanceof Instruction instruction)) continue;
            String opcode = instruction.opcode().name();
            if (opcode.endsWith("ALOAD") && !opcode.equals("AALOAD")) loads.add(opcode);
            if (opcode.endsWith("ASTORE") && !opcode.equals("AASTORE")) stores.add(opcode);
            if (opcode.equals("IINC") || opcode.equals("IADD") || opcode.equals("LADD")) {
                add++;
                outputProgression = true;
                sourceProgression = true;
            }
            if (opcode.equals("IMUL") || opcode.equals("LMUL")) multiply++;
            if (opcode.startsWith("IF_") || opcode.equals("IFGE") || opcode.equals("IFLT")
                    || opcode.equals("IFNE") || opcode.equals("IFEQ")) cursorComparison = true;
            if (List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "ATHROW", "INVOKEDYNAMIC").contains(opcode))
                forbidden.add(opcode);
            if (instruction instanceof InvokeInstruction call) {
                String detail = call.owner().asInternalName() + '.' + call.name() + call.type();
                invokes.add(detail);
                if (detail.startsWith("java/lang/foreign/MemorySegment.get")) {
                    String fact = "SEGMENT_GET:" + segmentLayout(detail) + ':' + methodReturn(detail);
                    loads.add(fact); segments.add(fact);
                } else if (detail.startsWith("java/lang/foreign/MemorySegment.set")) {
                    String fact = "SEGMENT_SET:" + segmentLayout(detail) + ':' + primitivePayload(detail);
                    stores.add(fact); segments.add(fact);
                }
            }
        }
        position = 0;
        for (CodeElement element : elements) {
            if (!(element instanceof Instruction instruction)) continue;
            if (instruction instanceof BranchInstruction branch) {
                int target = labels.get(branch.target());
                if (target < position) backward++; else forward++;
            }
            position++;
        }
        boolean padSelection = forward >= 2 && !loads.isEmpty() && !stores.isEmpty();
        boolean padImmediate = invokes.stream().anyMatch(call -> call.startsWith("java/lang/Float.intBitsToFloat")
                || call.startsWith("java/lang/Double.longBitsToDouble")) || !invokes.isEmpty();
        // Carrying an address across iterations produces additive progression; the rank-specific
        // wrap/reset path additionally needs a multiply-and-branch correction.  javac spells the
        // same source-coordinate recurrence with long locals, while dense generated bodies use ints.
        boolean tileCarry = add >= 2 && sourceProgression;
        boolean tileWrap = multiply >= 1 && forward >= 2;
        return new MovementFacts(List.copyOf(loads), List.copyOf(stores), List.copyOf(segments),
                List.copyOf(invokes), List.copyOf(forbidden), new RangeFacts(forward > 0, backward > 0,
                cursorComparison), outputProgression, sourceProgression, padSelection, padImmediate,
                tileCarry, tileWrap, method.methodType().stringValue().contains("["));
    }

    private static void assertMovementEntryProjection(String expectedDescriptor,
            String generatedDescriptor, String cleanDescriptor, String owner, String form,
            DataType type, long bits, CpuKernelSpecialization specialization,
            MovementFacts generated, MovementFacts clean) {
        assertEquals(expectedDescriptor, cleanDescriptor, owner + " clean ordered entry ABI");
        assertEquals(expectedDescriptor, generatedDescriptor, owner + " generated ordered entry ABI");
        assertMovementProjection(owner, form, type, bits, specialization, generated, clean);
    }

    private static void assertMovementProjection(String owner, String form, DataType type, long bits,
            CpuKernelSpecialization specialization, MovementFacts generated, MovementFacts clean) {
        List<String> expectedStore = List.of(carrierAccess(type, specialization, 1, true));
        List<String> expectedLoad = List.of(carrierAccess(type, specialization, 0, false));
        assertEquals(expectedLoad, movementCarrierProjection(clean.loads, expectedLoad.getFirst()),
                owner + " clean typed source carrier role and width");
        assertTrue(carrierProjection(clean.stores, true).stream().allMatch(expectedStore.getFirst()::equals)
                        && !carrierProjection(clean.stores, true).isEmpty(),
                owner + " clean typed result carrier role and width");
        // Bounded PAD retains a full-range fast body and a ranged fallback, so a segment
        // specialization contains the same typed load/store role once per body.  The two bodies
        // are not collapsed: each occurrence must be the exact ordered singleton role.
        assertTrue(movementCarrierProjection(generated.loads, expectedLoad.getFirst()).stream()
                        .allMatch(expectedLoad.getFirst()::equals)
                        && !movementCarrierProjection(generated.loads, expectedLoad.getFirst()).isEmpty(),
                owner + " generated typed source carrier role/width/direction");
        assertTrue(carrierProjection(generated.stores, true).stream().allMatch(expectedStore.getFirst()::equals)
                        && !carrierProjection(generated.stores, true).isEmpty(),
                owner + " generated typed result carrier role and width");
        assertEquals(clean.segments.stream().distinct().toList(), generated.segments.stream().distinct().toList(),
                owner + " ordered typed FFM layout calls across checked realization bodies");
        assertRange(owner + " clean", clean.range);
        assertRange(owner + " generated", generated.range);
        assertTrue(clean.outputProgression && generated.outputProgression,
                owner + " output-address progression");
        if (form.equals("PAD")) {
            // javac keeps the two branch-local typed stores, while the generated emitter first
            // selects a typed local then performs one common output store. Both expose the same
            // input/output carrier direction and the selected value reaches exactly one store.
            assertEquals(2, carrierProjection(clean.stores, true).size(),
                    owner + " clean PAD branch-local source/fill stores");
            assertTrue(clean.padSelection && generated.padSelection,
                    owner + " mapped-source versus immediate padding selection");
            if (type == DataType.FLOAT32 || type == DataType.FLOAT64) {
                String immediate = type == DataType.FLOAT32 ? "java/lang/Float.intBitsToFloat(I)F"
                        : "java/lang/Double.longBitsToDouble(J)D";
                assertTrue(clean.invokes.contains(immediate) && generated.invokes.contains(immediate),
                        owner + " typed represented padding immediate " + Long.toUnsignedString(bits, 16));
            } else {
                assertFalse(clean.invokes.stream().anyMatch(call -> call.contains("BitsTo")),
                        owner + " primitive padding immediate has no conversion helper");
                assertFalse(generated.invokes.stream().anyMatch(call -> call.contains("BitsTo")),
                        owner + " primitive padding immediate has no conversion helper");
            }
        } else {
            assertTrue(clean.sourceProgression && generated.sourceProgression,
                    owner + " TILE source-address progression");
            assertTrue(clean.tileCarry && generated.tileCarry, owner + " TILE carried source coordinate");
            assertTrue(clean.tileWrap && generated.tileWrap, owner + " TILE source wrap/reset correction");
        }
    }

    private static List<String> movementCarrierProjection(List<String> accesses, String expected) {
        // long[] has two typed roles here: p0 is INT64 payload and p2 is cold geometry.  The
        // Class-File API does not attach local-slot provenance to LALOAD, so retain the direct
        // payload-width fact without pretending every geometry read is a carrier read.
        if (expected.equals("LALOAD_DATA")) return accesses.contains("LALOAD") ? List.of(expected) : List.of();
        return carrierProjection(accesses, false);
    }

    private static void assertMovementHygiene(String owner, String form, MovementFacts facts, boolean clean) {
        assertEquals(List.of(), facts.forbidden, owner + " no allocation or dynamic invocation");
        for (String invocation : facts.invokes) {
            boolean ffm = clean ? permittedCleanFfmInvoke(invocation) : permittedGeneratedFfmInvoke(invocation);
            boolean padImmediate = (form.equals("PAD") || form.equals("UNFOLD2D")) && (invocation.equals("java/lang/Float.intBitsToFloat(I)F")
                    || invocation.equals("java/lang/Double.longBitsToDouble(J)D"));
            assertTrue(ffm || padImmediate, owner + " allowlisted direct typed call only: " + invocation);
        }
    }

    private record MovementFacts(List<String> loads, List<String> stores, List<String> segments,
            List<String> invokes, List<String> forbidden, RangeFacts range, boolean outputProgression,
            boolean sourceProgression, boolean padSelection, boolean padImmediate, boolean tileCarry,
            boolean tileWrap, boolean dense) { }

    private static Map<String, Long> movementCounts(List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> candidates) {
        var result = new TreeMap<String, Long>(); candidates.forEach(c -> result.merge(c.operationForm(), 1L, Long::sum)); return result;
    }
    private static long padBits(io.github.pho001.synaptik.model.operation.layout.PadAttrs attrs) { var v=attrs.constantValue(); return switch(v.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits(v.float64Value()); case FLOAT32 -> Float.floatToRawIntBits(v.float32Value()) & 0xffffffffL; case BFLOAT16 -> v.bfloat16Bits() & 0xffffL; case INT64 -> v.int64Value(); case INT32 -> v.int32Value() & 0xffffffffL; case BOOL -> v.booleanValue()?1L:0L; }; }
    private static List<MovementCarrier> movementCarriers(PreparedMovement row) {
        var result = new ArrayList<MovementCarrier>();
        for (int i = 0; i < row.plan.boundaryValues().size(); i++) {
            var id = row.plan.boundaryValues().get(i);
            var descriptor = row.candidate.context().values().stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow().descriptor();
            int n = Math.toIntExact(maxAddress(descriptor) + 1); Object array = patterned(descriptor.dataType(), n);
            if (i == row.plan.boundaryValues().size() - 1) array = sentinels(descriptor.dataType(), n);
            result.add(new MovementCarrier(array, argument(row.specialization.carrierPattern().get(i), array)));
        }
        return result;
    }
    private static long maxAddress(io.github.pho001.synaptik.model.tensor.TensorDescriptor d) { long total=d.layout().orElseThrow().storageOffset(); long[] shape=d.shape().toLongArray(), strides=d.layout().orElseThrow().strides(); for(int i=0;i<shape.length;i++) total+=Math.max(0,shape[i]-1)*strides[i]; return total; }
    private static Object cloneArray(Object value) { if(value instanceof double[] x)return x.clone(); if(value instanceof float[] x)return x.clone(); if(value instanceof short[] x)return x.clone(); if(value instanceof long[] x)return x.clone(); if(value instanceof int[] x)return x.clone(); return ((byte[])value).clone(); }
    private static void assertRawArrayEquals(Object a, Object b, String label) { if(a instanceof double[] x) org.junit.jupiter.api.Assertions.assertArrayEquals(x,(double[])b,label); else if(a instanceof float[] x) org.junit.jupiter.api.Assertions.assertArrayEquals(x,(float[])b,label); else if(a instanceof short[] x) org.junit.jupiter.api.Assertions.assertArrayEquals(x,(short[])b,label); else if(a instanceof long[] x) org.junit.jupiter.api.Assertions.assertArrayEquals(x,(long[])b,label); else if(a instanceof int[] x) org.junit.jupiter.api.Assertions.assertArrayEquals(x,(int[])b,label); else org.junit.jupiter.api.Assertions.assertArrayEquals((byte[])a,(byte[])b,label); }
    private record MovementCarrier(Object array, Object carrier) { }
    private record PreparedMovement(CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate candidate,
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr,
            DataType type) { }

    /**
     * Executes the remaining 0009C movement rows through independently compiled typed bodies.
     * SLICE_UPDATE selects its ordered base/update role from finite prepared sequence membership;
     * UNFOLD_AXIS maps the selected window axis; and UNFOLD2D selects padding or its NCHW window.
     */
    @Test void everySliceUpdateAndUnfoldRowExecutesPairedTypedCleanJavaMovementLoops() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(candidate -> Set.of("SLICE_UPDATE", "UNFOLD_AXIS", "UNFOLD2D")
                        .contains(candidate.operationForm())).toList();
        assertEquals(52, candidates.size(), "exact remaining non-fold movement slice");
        assertEquals(Map.of("SLICE_UPDATE", 24L, "UNFOLD_AXIS", 24L, "UNFOLD2D", 4L), movementCounts(candidates));
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(value -> value.id().equals(output))
                    .findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            long bits = candidate.operationForm().equals("UNFOLD2D") ? movement.plan().immediateBits() : 0L;
            String family = switch (candidate.operationForm()) {
                case "SLICE_UPDATE" -> "slice-update";
                case "UNFOLD_AXIS" -> "unfold-axis";
                case "UNFOLD2D" -> "unfold2d";
                default -> throw new AssertionError(candidate.operationForm());
            };
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), family,
                    route.specialization().entryType().descriptorString(), descriptor.dataType().name(), false,
                    descriptor.shape().rank(), bits, movement.plan().occurrenceToBoundary()));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        int executions = 0;
        for (PreparedMovement row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(row.specialization, row.kernelIr);
            assertEquals(java.util.HexFormat.of().formatHex(bytes), java.util.HexFormat.of().formatHex(
                    generator.generateClassBytes(row.specialization, row.kernelIr)), row.candidate.ownerId() + " deterministic bytes");
            var generated = generator.defineClassBytes(row.specialization, bytes).entryPoint();
            var counterpart = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.candidate.ownerId(),
                    row.specialization.entryType().descriptorString());
            long count = row.plan.elementCount();
            for (long[] range : ranges(count)) {
                var generatedCarriers = movementCarriers(row); var cleanCarriers = movementCarriers(row);
                long[] geometry = row.plan.movementGeometry().orElseThrow().pack(new long[row.plan.boundaryValues().size()], range[0], range[1]);
                List<Object> generatedArguments = new ArrayList<>(), cleanArguments = new ArrayList<>();
                for (int i = 0; i < generatedCarriers.size(); i++) {
                    generatedArguments.add(generatedCarriers.get(i).carrier);
                    cleanArguments.add(cleanCarriers.get(i).carrier);
                }
                generatedArguments.add(geometry); generatedArguments.add(range[0]); generatedArguments.add(range[1]);
                cleanArguments.add(geometry.clone()); cleanArguments.add(range[0]); cleanArguments.add(range[1]);
                generated.invokeWithArguments(generatedArguments); counterpart.invokeWithArguments(cleanArguments);
                for (int i = 0; i < generatedCarriers.size(); i++) assertRawArrayEquals(generatedCarriers.get(i).array,
                        cleanCarriers.get(i).array, row.candidate.ownerId() + " represented bits/range/sentinels="
                                + java.util.Arrays.toString(range));
                executions++;
            }
        }
        assertEquals(208, executions, "52 rows × full, empty, interior and tail ranges");
    }

    /** Checks family-specific Class-File projections without claiming instruction or CFG identity. */
    @Test void everySliceUpdateAndUnfoldRowMatchesFamilySpecificPairedStructuralProjection() throws Exception {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(candidate -> Set.of("SLICE_UPDATE", "UNFOLD_AXIS", "UNFOLD2D")
                        .contains(candidate.operationForm())).toList();
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan(); var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(value -> value.id().equals(output)).findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            String family = candidate.operationForm().equals("SLICE_UPDATE") ? "slice-update"
                    : candidate.operationForm().equals("UNFOLD_AXIS") ? "unfold-axis" : "unfold2d";
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), family,
                    route.specialization().entryType().descriptorString(), descriptor.dataType().name(), false,
                    descriptor.shape().rank(), movement.plan().immediateBits(), movement.plan().occurrenceToBoundary()));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes()); int compared = 0;
        for (PreparedMovement row : prepared) {
            String owner = row.candidate.ownerId(), form = row.candidate.operationForm();
            MethodModel generated = selected(methods(new CpuClassFileKernelGenerator().generateClassBytes(row.specialization, row.kernelIr)));
            MethodModel counterpart = cleanMethods.get(clean.methods().get(owner));
            assertTrue(counterpart != null, owner + " independently compiled typed counterpart");
            assertEquals(row.specialization.entryType().descriptorString(), generated.methodType().stringValue(), owner + " generated ordered ABI");
            assertEquals(generated.methodType().stringValue(), counterpart.methodType().stringValue(), owner + " clean ordered ABI");
            assertRange(owner + " generated", movementFacts(generated).range); assertRange(owner + " clean", movementFacts(counterpart).range);
            assertMovementHygiene(owner + " generated", form, movementFacts(generated), false);
            assertMovementHygiene(owner + " clean", form, movementFacts(counterpart), true);
            int rank = row.candidate.context().values().stream().filter(value -> value.id().equals(
                    row.candidate.context().nodes().getFirst().outputs().getFirst())).findFirst().orElseThrow().descriptor().shape().rank();
            assertFamilyMovementProjection(owner, form, rank, row.specialization, generated, counterpart);
            compared++;
        }
        assertEquals(52, compared, "all SLICE_UPDATE/UNFOLD_AXIS/UNFOLD2D rows have paired family projections");
    }

    private static void assertFamilyMovementProjection(String owner, String form, int rank,
            CpuKernelSpecialization specialization, MethodModel generated, MethodModel clean) {
        Set<Integer> slots = switch (form) {
            case "SLICE_UPDATE" -> sliceUpdateSlots(rank, specialization.carrierPattern().size() - 1);
            case "UNFOLD_AXIS" -> unfoldAxisSlots(rank);
            case "UNFOLD2D" -> unfold2dSlots();
            default -> throw new AssertionError(form);
        };
        assertPackedGeometrySlots(owner + " generated", form, rank, slots, geometrySlots(generated));
        assertPackedGeometrySlots(owner + " clean", form, rank, slots, geometrySlots(clean));
        if (!(form.equals("SLICE_UPDATE") && rank == 1)) {
            assertTrue(opcodeCount(generated, "IMUL", "LMUL") >= rank, owner + " generated address/window multiplication");
            assertTrue(opcodeCount(clean, "IMUL", "LMUL") >= rank, owner + " clean address/window multiplication");
        }
        assertTrue(conditionalBranches(generated) >= 2 && conditionalBranches(clean) >= 2,
                owner + " preserves selection/window bounds and range branches");
        if (form.equals("SLICE_UPDATE")) {
            assertTrue(conditionalBranches(generated) >= rank + 1 && conditionalBranches(clean) >= rank + 1,
                    owner + " base/update membership selection is active on every axis");
        } else if (form.equals("UNFOLD_AXIS")) {
            assertTrue(conditionalBranches(generated) >= rank && conditionalBranches(clean) >= rank,
                    owner + " selected axis window map remains active");
        } else {
            assertTrue(conditionalBranches(generated) >= 5 && conditionalBranches(clean) >= 5,
                    owner + " NCHW padding/window selection remains active");
        }
    }

    private static Set<Integer> sliceUpdateSlots(int rank, int inputs) {
        var result = new java.util.TreeSet<Integer>(); addSlots(result, 0, 3 * rank);
        addSlots(result, 3 * rank + 1, 3 * rank + inputs + inputs * rank);
        int variant = 3 * rank + 1 + inputs + inputs * rank;
        // The generated loop carries current target/ordinal; the independently authored body
        // derives membership from the low target, finite length, and signed step instead.
        addSlots(result, variant, variant + rank - 1);
        addSlots(result, variant + 2 * rank, variant + 4 * rank - 1);
        return Set.copyOf(result);
    }
    private static Set<Integer> unfoldAxisSlots(int rank) {
        var result = new java.util.TreeSet<Integer>(); addSlots(result, 0, 4 * rank + 1);
        // The prepared window size validates output geometry but does not participate in the
        // per-cell source address; axis and step do.
        result.add(4 * rank + 3); return Set.copyOf(result);
    }
    private static Set<Integer> unfold2dSlots() {
        var result = new java.util.TreeSet<Integer>(); addSlots(result, 0, 14);
        // channels/output-height and the packed starting window cursors are cold derivation
        // facts.  The direct witness derives channel/oh/ow from q/p each iteration, while the
        // generated form may keep rolling cursors; both retain this shared window-map core.
        addSlots(result, 16, 25); result.add(27); return Set.copyOf(result);
    }

    /** Active negative controls for the retained range, role-map, and window facts. */
    @Test void sliceUpdateAndUnfoldProjectionsRejectRetainedFactMutations() {
        for (Object[] control : List.of(
                new Object[] {"SLICE_UPDATE", 2, 2}, new Object[] {"UNFOLD_AXIS", 3, 1},
                new Object[] {"UNFOLD2D", 3, 1})) {
            String form = (String) control[0]; int rank = (Integer) control[1], inputs = (Integer) control[2];
            Set<Integer> expected = switch (form) {
                case "SLICE_UPDATE" -> sliceUpdateSlots(rank, inputs);
                case "UNFOLD_AXIS" -> unfoldAxisSlots(rank);
                case "UNFOLD2D" -> unfold2dSlots();
                default -> throw new AssertionError(form);
            };
            var missing = new java.util.TreeSet<>(expected); missing.remove(expected.stream().max(Integer::compareTo).orElseThrow());
            assertThrows(AssertionError.class, () -> assertPackedGeometrySlots(form + " mutated map", form,
                    rank, expected, Set.copyOf(missing)), form + " rejects a changed mapped geometry fact");
            assertThrows(AssertionError.class, () -> assertCompatible(
                    new Row("control", "movement", form, "FLOAT32", "([F[F[JJJ)V", "[FLOAT_ARRAY, FLOAT_ARRAY]", "[DENSE_LINEAR]", "scalar"),
                    new Row("control", "movement", form, "FLOAT32", "([F[Ljava/lang/foreign/MemorySegment;[JJJ)V", "[FLOAT_ARRAY, MEMORY_SEGMENT]", "[DENSE_LINEAR]", "scalar")),
                    form + " rejects an ordered carrier ABI mutation");
        }
    }

    /** Every supported non-fold movement topology has real javac source mutants, not fact edits. */
    @Test void generalMovementProjectionRejectsExecutableSourceMutants() throws Exception {
        for (String form : List.of("PAD", "TILE", "CONCAT", "STACK", "SLICE_UPDATE", "UNFOLD_AXIS", "UNFOLD2D")) {
            var candidate = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                    .filter(value -> value.operationForm().equals(form)).findFirst().orElseThrow();
            var route = new CpuPartitionPreparer().analyze(candidate.context()).plan().units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(value -> value.id().equals(output)).findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            var basis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("control", form.toLowerCase().replace('_', '-'),
                    route.specialization().entryType().descriptorString(), descriptor.dataType().name(), false,
                    descriptor.shape().rank(), form.equals("UNFOLD2D") ? movement.plan().immediateBits() : 0L,
                    movement.plan().occurrenceToBoundary());
            var canonical = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(List.of(basis));
            MethodModel baseline = methods(canonical.bytes()).get("entry0");
            assertTrue(baseline != null, form + " canonical movement javac witness");
            var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileMovementMutants(basis);
            for (var entry : mutants.methods().entrySet()) {
                String kind = entry.getKey(); MethodModel mutant = methods(mutants.bytes()).get(entry.getValue());
                assertTrue(mutant != null, form + " compiled executable movement mutant " + kind);
                if (Set.of("new", "anewarray", "newarray", "multianewarray", "athrow", "invokedynamic",
                        "boxing", "reflection", "string-dispatch", "map-dispatch", "hidden-helper", "cross-ffm").contains(kind)) {
                    assertThrows(AssertionError.class, () -> assertMovementHygiene(form + '/' + kind, form,
                            movementFacts(mutant), true));
                } else if (kind.equals("abi")) {
                    assertNotEquals(baseline.methodType().stringValue(), mutant.methodType().stringValue(),
                            form + " typed ABI/carrier mutant");
                } else {
                    assertNotEquals(sourceInstructionShape(baseline), sourceInstructionShape(mutant),
                            form + " coordinate/address/range/store source mutant " + kind);
                }
            }
        }
    }

    /** Locks both source-derived CONCAT/STACK access cross-tabs without collapsing carriers. */
    @Test void concatAndStackAccessInventoryAndActualEmitterPathCrossTabsRemainExact() throws Exception {
        var rows = ownedRows().stream().filter(row -> row.form.equals("CONCAT") || row.form.equals("STACK"))
                .toList();
        assertEquals(48, rows.size(), "exact composition owner universe");
        Map<String, Map<String, Long>> inventory = new TreeMap<>();
        for (Row row : rows) {
            String regime = row.access.contains("DENSE_LINEAR") ? "DENSE_LINEAR" : "GENERAL_ODOMETER";
            inventory.computeIfAbsent(row.form, unused -> new TreeMap<>()).merge(regime, 1L, Long::sum);
        }
        assertEquals(Map.of("CONCAT", Map.of("DENSE_LINEAR", 12L, "GENERAL_ODOMETER", 12L),
                "STACK", Map.of("DENSE_LINEAR", 12L, "GENERAL_ODOMETER", 12L)), inventory,
                "checked inventory access cross-tab with every owner accounted once");

        Map<String, Map<String, Long>> actual = new TreeMap<>();
        int denseSegmentGeneral = 0;
        for (var candidate : CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates()) {
            if (!candidate.operationForm().equals("CONCAT") && !candidate.operationForm().equals("STACK")) continue;
            var route = new CpuPartitionPreparer().analyze(candidate.context()).plan().units().getFirst().portablePlan();
            String addressing = route.specialization().loopAddressing(route.kernelIr())
                    == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT
                    ? "DENSE_HEAP_ARRAY_INT" : "GENERAL_LONG";
            actual.computeIfAbsent(candidate.operationForm(), unused -> new TreeMap<>()).merge(addressing, 1L, Long::sum);
            if (candidate.ownerId().contains("segment-contiguous") && addressing.equals("GENERAL_LONG")) denseSegmentGeneral++;
        }
        assertEquals(Map.of("CONCAT", Map.of("DENSE_HEAP_ARRAY_INT", 6L, "GENERAL_LONG", 18L),
                "STACK", Map.of("DENSE_HEAP_ARRAY_INT", 6L, "GENERAL_LONG", 18L)), actual,
                "actual selected emitter loop-addressing cross-tab with every owner accounted once");
        assertEquals(12, denseSegmentGeneral,
                "six dense segment-contiguous rows per composition form deliberately use GENERAL_LONG");
    }

    /** Executes every CONCAT and STACK row with a source-derived occurrence map and packed geometry. */
    @Test void everyConcatAndStackRowExecutesPairedTypedCleanJavaSelectionLoops() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(c -> c.operationForm().equals("CONCAT") || c.operationForm().equals("STACK")).toList();
        assertEquals(48, candidates.size(), "exact CONCAT/STACK movement sub-slice");
        assertEquals(Map.of("CONCAT", 24L, "STACK", 24L), movementCounts(candidates));
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(v -> v.id().equals(output)).findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            var occurrence = movement.plan().occurrenceToBoundary();
            assertEquals(candidate.context().nodes().getFirst().inputs().size(), occurrence.size(), candidate.ownerId() + " source occurrence order");
            assertEquals(plan.boundaryValues().size() - 1, occurrence.stream().distinct().count(), candidate.ownerId() + " unique ordered carrier roles");
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(),
                    candidate.operationForm().toLowerCase(), route.specialization().entryType().descriptorString(),
                    descriptor.dataType().name(), false, descriptor.shape().rank(), 0L, occurrence));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        int executions = 0;
        for (PreparedMovement row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            byte[] first = generator.generateClassBytes(row.specialization, row.kernelIr);
            assertEquals(java.util.HexFormat.of().formatHex(first), java.util.HexFormat.of().formatHex(
                    generator.generateClassBytes(row.specialization, row.kernelIr)), row.candidate.ownerId() + " deterministic bytes");
            var generated = generator.defineClassBytes(row.specialization, first).entryPoint();
            var counterpart = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.candidate.ownerId(), row.specialization.entryType().descriptorString());
            for (long[] range : ranges(row.plan.elementCount())) {
                var actual = movementCarriers(row); var expected = movementCarriers(row);
                var sourceBefore = actual.subList(0, actual.size() - 1).stream().map(MovementCarrier::array).map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                var cleanSourceBefore = expected.subList(0, expected.size() - 1).stream().map(MovementCarrier::array).map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                long[] geometry = row.plan.movementGeometry().orElseThrow().pack(new long[row.plan.boundaryValues().size()], range[0], range[1]);
                List<Object> generatedArgs = new ArrayList<>(), cleanArgs = new ArrayList<>();
                for (int i = 0; i < actual.size(); i++) { generatedArgs.add(actual.get(i).carrier); cleanArgs.add(expected.get(i).carrier); }
                generatedArgs.add(geometry); generatedArgs.add(range[0]); generatedArgs.add(range[1]);
                cleanArgs.add(geometry.clone()); cleanArgs.add(range[0]); cleanArgs.add(range[1]);
                generated.invokeWithArguments(generatedArgs); counterpart.invokeWithArguments(cleanArgs);
                for (int i = 0; i < actual.size(); i++) assertRawArrayEquals(actual.get(i).array, expected.get(i).array,
                        row.candidate.ownerId() + " represented bits/sentinels range=" + java.util.Arrays.toString(range));
                for (int i = 0; i < sourceBefore.size(); i++) assertRawArrayEquals(sourceBefore.get(i), actual.get(i).array,
                        row.candidate.ownerId() + " immutable source role=" + i);
                for (int i = 0; i < cleanSourceBefore.size(); i++) assertRawArrayEquals(cleanSourceBefore.get(i), expected.get(i).array,
                        row.candidate.ownerId() + " clean immutable source role=" + i);
                executions++;
            }
        }
        assertEquals(48 * 4, executions, "every CONCAT/STACK row runs full, empty, interior and tail ranges");
    }

    /**
     * Compares the bounded, descriptor- and local-slot-aware composition dataflow relation for
     * every selected entry.  This is deliberately not arbitrary CFG equivalence: it extracts
     * only the typed carrier reads/writes which carry the selected occurrence into the output.
     */
    @Test void everyConcatAndStackApplyMatchesIndependentCleanJavaSelectionDataflowRelation() throws Exception {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(c -> c.operationForm().equals("CONCAT") || c.operationForm().equals("STACK")).toList();
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan(); var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(v -> v.id().equals(output)).findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), candidate.operationForm().toLowerCase(),
                    route.specialization().entryType().descriptorString(), descriptor.dataType().name(), false, descriptor.shape().rank(), 0L,
                    movement.plan().occurrenceToBoundary()));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows); Map<String, MethodModel> cleanMethods = methods(clean.bytes()); int compared = 0;
        for (PreparedMovement row : prepared) {
            String owner = row.candidate.ownerId(); var movement = (CpuDataMovementIr) new CpuPartitionPreparer().analyze(row.candidate.context()).plan().units().getFirst().portablePlan().portableKernelIr();
            MethodModel generated = selected(methods(new CpuClassFileKernelGenerator().generateClassBytes(row.specialization, row.kernelIr)));
            MethodModel counterpart = cleanMethods.get(clean.methods().get(owner));
            assertEquals(row.specialization.entryType().descriptorString(), generated.methodType().stringValue(), owner + " generated typed ordered ABI");
            assertEquals(generated.methodType().stringValue(), counterpart.methodType().stringValue(), owner + " clean typed ordered ABI");
            assertCompositionProjection(owner, row.candidate.operationForm(), row.specialization, movement.plan().occurrenceToBoundary(), compositionFacts(generated), compositionFacts(counterpart));
            assertCompositionHygiene(owner + " generated", generated, false);
            assertCompositionHygiene(owner + " clean", counterpart, true);
            compared++;
        }
        assertEquals(48, compared, "all CONCAT/STACK rows have paired topology projections");
    }

    /**
     * Executes injective provenance probes against both defined Class-Files for every exact
     * composition row.  This bounded extractor is intentionally executable rather than a CFG
     * identity claim: the output's represented value identifies the physical source carrier and
     * address, while independently reconstructed prepared geometry identifies its output address,
     * logical coordinate, selection ordinal, and CONCAT relative or STACK input coordinate.
     * BOOL uses one-hot source probes because its canonical two-value representation cannot carry
     * a simultaneous injective address label.
     */
    @Test void everyConcatAndStackEntryExtractsExecutableNormalizedProvenanceRelation() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                .filter(c -> c.operationForm().equals("CONCAT") || c.operationForm().equals("STACK")).toList();
        var rows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedMovement>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var descriptor = candidate.context().values().stream().filter(v -> v.id().equals(output)).findFirst().orElseThrow().descriptor();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            rows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), candidate.operationForm().toLowerCase(),
                    route.specialization().entryType().descriptorString(), descriptor.dataType().name(), false, descriptor.shape().rank(), 0L,
                    movement.plan().occurrenceToBoundary()));
            prepared.add(new PreparedMovement(candidate, plan, route.specialization(), route.kernelIr(), descriptor.dataType()));
        }
        assertEquals(48, prepared.size(), "exact composition relation denominator");
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(rows);
        int comparisons = 0;
        for (PreparedMovement row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            var actual = generator.defineClassBytes(row.specialization,
                    generator.generateClassBytes(row.specialization, row.kernelIr)).entryPoint();
            var counterpart = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.candidate.ownerId(),
                    row.specialization.entryType().descriptorString());
            for (long[] range : ranges(row.plan.elementCount())) {
                long[] geometry = row.plan.movementGeometry().orElseThrow().pack(compositionBases(row), range[0], range[1]);
                List<CompositionRelation> expected = expectedCompositionRelation(row, geometry, range);
                List<CompositionRelation> generated = extractCompositionRelation(row, actual, geometry, range, expected);
                List<CompositionRelation> cleanRelation = extractCompositionRelation(row, counterpart, geometry, range, expected);
                assertEquals(expected, generated, row.candidate.ownerId() + " generated executable relation range=" + java.util.Arrays.toString(range));
                assertEquals(expected, cleanRelation, row.candidate.ownerId() + " clean executable relation range=" + java.util.Arrays.toString(range));
                assertEquals(cleanRelation, generated, row.candidate.ownerId() + " independently loaded class-file relation range=" + java.util.Arrays.toString(range));
                comparisons++;
            }
        }
        assertEquals(48 * 4, comparisons, "full, empty, interior and tail executable relation probes");
    }

    /** Real javac semantic mutations fail a named executable-relation field; hygiene controls stay hygiene-only. */
    @Test void concatAndStackHygieneAndTopologyRejectExecutableJavacMutants() throws Throwable {
        for (String form : List.of("CONCAT", "STACK")) {
            var candidate = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates().stream()
                    .filter(c -> c.operationForm().equals(form)).findFirst().orElseThrow();
            var route = new CpuPartitionPreparer().analyze(candidate.context()).plan().units().getFirst().portablePlan();
            var movement = (CpuDataMovementIr) route.portableKernelIr();
            var output = candidate.context().nodes().getFirst().outputs().getFirst();
            var type = candidate.context().values().stream().filter(v -> v.id().equals(output)).findFirst().orElseThrow().descriptor().dataType();
            var basis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("control", form.toLowerCase(),
                    route.specialization().entryType().descriptorString(), type.name(), false,
                    candidate.context().values().getLast().descriptor().shape().rank(), 0L,
                    movement.plan().occurrenceToBoundary());
            var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileCompositionMutants(basis);
            for (var entry : mutants.methods().entrySet()) {
                MethodModel mutant = methods(mutants.bytes()).get(entry.getValue());
                assertTrue(mutant != null, form + '/' + entry.getKey() + " compiled executable class-file mutant");
                if (Set.of("terminal-throw", "new", "anewarray", "newarray", "multianewarray",
                        "invokedynamic", "athrow", "constructor", "hidden-helper", "boxing", "reflection",
                        "string-dispatch", "map-dispatch").contains(entry.getKey())) {
                    assertThrows(AssertionError.class, () -> assertCompositionHygiene(form + '/' + entry.getKey(), mutant, true),
                            form + '/' + entry.getKey() + " hygiene mutant must fail hygiene");
                    continue;
                }
                var handle = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(mutants, entry.getKey(),
                        basis.descriptor());
                PreparedMovement prepared = new PreparedMovement(candidate,
                        new CpuPartitionPreparer().analyze(candidate.context()).plan(), route.specialization(), route.kernelIr(), type);
                long[] range = new long[] {0L, prepared.plan.elementCount()};
                long[] geometry = prepared.plan.movementGeometry().orElseThrow().pack(compositionBases(prepared), range[0], range[1]);
                List<CompositionRelation> expected = expectedCompositionRelation(prepared, geometry, range);
                AssertionError failure = assertThrows(AssertionError.class,
                        () -> assertCompositionMutantRelation(form + '/' + entry.getKey(), relationField(entry.getKey()),
                                expected, prepared, handle, geometry, range),
                        form + '/' + entry.getKey() + " semantic mutant must fail executable relation");
                assertTrue(failure.getMessage().contains(relationField(entry.getKey())),
                        form + '/' + entry.getKey() + " fails its intended relation field: " + failure.getMessage());
            }
        }
    }

    private static void assertCompositionMutantRelation(String owner, String field, List<CompositionRelation> expected,
            PreparedMovement row, java.lang.invoke.MethodHandle entry, long[] geometry, long[] range) {
        try {
            assertEquals(expected, extractCompositionRelation(row, entry, geometry, range, expected), owner + " semantic relation");
        } catch (AssertionError failure) {
            throw new AssertionError(owner + " " + field + " relation failure", failure);
        } catch (Throwable failure) {
            throw new AssertionError(owner + " " + field + " execution relation failure", failure);
        }
    }

    private static CompositionFacts compositionFacts(MethodModel method) {
        int branches = 0, loads = 0, stores = 0; List<String> invokes = new ArrayList<>(), opcodes = new ArrayList<>();
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) if (element instanceof Instruction instruction) {
            String opcode = instruction.opcode().name(); opcodes.add(opcode); if (opcode.startsWith("IF") || opcode.equals("GOTO")) branches++;
            if (opcode.endsWith("ALOAD") && !opcode.equals("AALOAD")) loads++; if (opcode.endsWith("ASTORE") && !opcode.equals("AASTORE")) stores++;
            if (instruction instanceof InvokeInstruction call) invokes.add(call.owner().asInternalName() + '.' + call.name() + call.type());
        }
        return new CompositionFacts(branches, loads, stores, rangeFacts(method), List.copyOf(invokes), List.copyOf(opcodes),
                compositionCarrierDataflow(method));
    }

    private static String relationField(String mutant) {
        return switch (mutant) {
            case "dummy-default", "missing-predicate", "reordered-selection", "missing-final-arm", "changed-boundary" -> "occurrence";
            case "changed-coordinate" -> "relativeOrInputCoordinate";
            case "changed-source-address" -> "sourceAddress";
            case "changed-carrier-role" -> "carrierRole";
            case "changed-output-store" -> "outputAddress";
            default -> throw new AssertionError("semantic composition mutant has no relation field: " + mutant);
        };
    }

    /** Source shape remains a narrow negative control for the non-composition projections. */
    private static List<String> sourceInstructionShape(MethodModel method) {
        var shape = new ArrayList<String>();
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (element instanceof Instruction instruction) shape.add(instruction.toString());
        }
        return List.copyOf(shape);
    }

    /**
     * Independently evaluates the prepared CONCAT/STACK geometry.  The occurrence-to-boundary
     * table is used only here, as expected prepared meaning; observation below gets its role and
     * source address solely from the injected carrier value and gets its ordinal from coordinates.
     */
    private static List<CompositionRelation> expectedCompositionRelation(PreparedMovement row, long[] g, long[] range) {
        String form = row.candidate.operationForm();
        int rank = outputRank(row), inputs = row.plan.boundaryValues().size() - 1;
        int bases = 3 * rank + 1, strides = bases + inputs;
        int variant = strides + inputs * (form.equals("CONCAT") ? rank : rank - 1);
        List<CompositionRelation> result = new ArrayList<>();
        for (long[] coordinate : outputCoordinates(g, rank, range)) {
            long output = outputAddress(g, rank, coordinate);
            long selection = coordinate[Math.toIntExact(g[variant])];
            int occurrence;
            long relativeOrInputCoordinate;
            if (form.equals("CONCAT")) {
                occurrence = 0;
                while (occurrence < compositionOccurrences(row).size() - 1
                        && selection >= g[variant + 2 + occurrence]) occurrence++;
                relativeOrInputCoordinate = selection - g[variant + 1 + occurrence];
            } else {
                occurrence = Math.toIntExact(selection);
                relativeOrInputCoordinate = removedCoordinate(coordinate, Math.toIntExact(g[variant]));
            }
            int role = compositionOccurrences(row).get(occurrence);
            long source = g[bases + role];
            for (int axis = 0; axis < (form.equals("CONCAT") ? rank : rank - 1); axis++) {
                long inputCoordinate = form.equals("CONCAT")
                        ? (axis == g[variant] ? relativeOrInputCoordinate : coordinate[axis])
                        : (g[variant] > axis ? coordinate[axis] : coordinate[axis + 1]);
                source += inputCoordinate * g[strides + role * (form.equals("CONCAT") ? rank : rank - 1) + axis];
            }
            result.add(new CompositionRelation(output, toBoxed(coordinate), occurrence, role, source,
                    form.equals("CONCAT") ? relativeOrInputCoordinate : relativeOrInputCoordinate));
        }
        return List.copyOf(result);
    }

    private static List<CompositionRelation> extractCompositionRelation(PreparedMovement row, java.lang.invoke.MethodHandle entry,
            long[] geometry, long[] range, List<CompositionRelation> expected) throws Throwable {
        if (row.type == DataType.BOOL) return extractBooleanCompositionRelation(row, entry, geometry, range, expected);
        var carriers = provenanceCarriers(row, false, -1, -1);
        List<Object> before = carriers.subList(0, carriers.size() - 1).stream().map(MovementCarrier::array)
                .map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
        invokeComposition(entry, carriers, geometry, range);
        for (int role = 0; role < before.size(); role++) assertRawArrayEquals(before.get(role), carriers.get(role).array,
                row.candidate.ownerId() + " provenance source immutable role=" + role);
        return observedCompositionRelation(row, geometry, range, expected, carriers, false);
    }

    /* BOOL cannot encode an address in one canonical byte.  A fresh one-hot execution per
       physical carrier address remains injective and still checks all other output cells stay 0. */
    private static List<CompositionRelation> extractBooleanCompositionRelation(PreparedMovement row,
            java.lang.invoke.MethodHandle entry, long[] geometry, long[] range, List<CompositionRelation> expected) throws Throwable {
        var observed = new java.util.TreeMap<Long, CompositionRelation>();
        int inputs = row.plan.boundaryValues().size() - 1;
        for (int role = 0; role < inputs; role++) {
            int cells = Math.toIntExact(maxAddress(descriptorForBoundary(row, role)) + compositionBases(row)[role] + 1);
            for (int address = 0; address < cells; address++) {
                var carriers = provenanceCarriers(row, true, role, address);
                List<Object> before = carriers.subList(0, carriers.size() - 1).stream().map(MovementCarrier::array)
                        .map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                invokeComposition(entry, carriers, geometry, range);
                for (int sourceRole = 0; sourceRole < before.size(); sourceRole++) assertRawArrayEquals(before.get(sourceRole), carriers.get(sourceRole).array,
                        row.candidate.ownerId() + " BOOL one-hot source immutable role=" + sourceRole);
                for (CompositionRelation relation : observedCompositionRelation(row, geometry, range, expected, carriers, true)) {
                    CompositionRelation prior = observed.put(relation.outputAddress(), relation);
                    assertTrue(prior == null, row.candidate.ownerId() + " BOOL provenance collision outputAddress=" + relation.outputAddress());
                }
            }
        }
        return List.copyOf(observed.values());
    }

    private static List<CompositionRelation> observedCompositionRelation(PreparedMovement row, long[] g, long[] range,
            List<CompositionRelation> expected, List<MovementCarrier> carriers, boolean bool) {
        int rank = outputRank(row), outputRole = carriers.size() - 1;
        Object output = carriers.get(outputRole).array;
        var outputCoordinates = new java.util.HashMap<Long, long[]>();
        for (long[] coordinate : outputCoordinates(g, rank, range)) outputCoordinates.put(outputAddress(g, rank, coordinate), coordinate);
        var provenance = new java.util.HashMap<Long, ProvenanceIdentity>();
        for (int role = 0; role < outputRole; role++) for (int address = 0; address < arrayLength(carriers.get(role).array); address++) {
            long raw = raw(carriers.get(role).array, row.type, address);
            if (!bool) {
                ProvenanceIdentity prior = provenance.put(raw, new ProvenanceIdentity(role, address));
                assertTrue(prior == null, row.candidate.ownerId() + " injective provenance value=" + raw);
            }
        }
        var observed = new ArrayList<CompositionRelation>();
        java.util.Set<Long> owned = new java.util.HashSet<>();
        for (CompositionRelation relation : expected) owned.add(relation.outputAddress());
        for (int address = 0; address < arrayLength(output); address++) {
            long value = raw(output, row.type, address);
            if (!owned.contains((long) address)) {
                assertEquals(0L, value, row.candidate.ownerId() + " untouched provenance outputAddress=" + address);
                continue;
            }
            if (bool && value == 0L) continue;
            ProvenanceIdentity identity = bool ? booleanIdentity(carriers, address) : provenance.get(value);
            assertTrue(identity != null, row.candidate.ownerId() + " observed output has no injected provenance outputAddress=" + address);
            long[] coordinate = outputCoordinates.get((long) address);
            assertTrue(coordinate != null, row.candidate.ownerId() + " observed output placement outside requested range outputAddress=" + address);
            observed.add(observedRelation(row, g, coordinate, address, identity));
        }
        observed.sort(java.util.Comparator.comparingLong(CompositionRelation::outputAddress));
        return List.copyOf(observed);
    }

    private static ProvenanceIdentity booleanIdentity(List<MovementCarrier> carriers, int outputAddress) {
        for (int role = 0; role < carriers.size() - 1; role++) for (int address = 0; address < arrayLength(carriers.get(role).array); address++)
            if (raw(carriers.get(role).array, DataType.BOOL, address) == 1L) return new ProvenanceIdentity(role, address);
        throw new AssertionError("missing BOOL one-hot provenance for outputAddress=" + outputAddress);
    }

    private static CompositionRelation observedRelation(PreparedMovement row, long[] g, long[] coordinate, long output,
            ProvenanceIdentity identity) {
        int rank = outputRank(row), inputs = row.plan.boundaryValues().size() - 1;
        int strides = 3 * rank + 1 + inputs;
        int variant = strides + inputs * (row.candidate.operationForm().equals("CONCAT") ? rank : rank - 1);
        int axis = Math.toIntExact(g[variant]);
        int occurrence = row.candidate.operationForm().equals("CONCAT") ? concatOccurrence(g, variant, coordinate[axis])
                : Math.toIntExact(coordinate[axis]);
        long coordinateFact = row.candidate.operationForm().equals("CONCAT")
                ? coordinate[axis] - g[variant + 1 + occurrence] : removedCoordinate(coordinate, axis);
        return new CompositionRelation(output, toBoxed(coordinate), occurrence, identity.role(), identity.address(), coordinateFact);
    }

    private static int concatOccurrence(long[] g, int variant, long relative) {
        int occurrence = 0;
        while (variant + 2 + occurrence < g.length && relative >= g[variant + 2 + occurrence]) occurrence++;
        return occurrence;
    }
    private static long removedCoordinate(long[] coordinate, int axis) {
        long result = 0L;
        for (int index = 0; index < coordinate.length; index++) if (index != axis) result = result * 31L + coordinate[index];
        return result;
    }
    private static List<long[]> outputCoordinates(long[] g, int rank, long[] range) {
        var coordinates = new ArrayList<long[]>(); long[] current = new long[rank];
        for (int axis = 0; axis < rank; axis++) current[axis] = g[rank + axis];
        for (long cursor = range[0]; cursor < range[1]; cursor++) {
            coordinates.add(current.clone());
            for (int axis = rank - 1; axis >= 0; axis--) { current[axis]++; if (current[axis] < g[axis]) break; current[axis] = 0L; }
        }
        return coordinates;
    }
    private static long outputAddress(long[] g, int rank, long[] coordinate) {
        long result = g[2 * rank];
        for (int axis = 0; axis < rank; axis++) result += (coordinate[axis] - g[rank + axis]) * g[2 * rank + 1 + axis];
        return result;
    }
    private static List<Long> toBoxed(long[] values) { var result = new ArrayList<Long>(); for (long value : values) result.add(value); return List.copyOf(result); }
    private static int outputRank(PreparedMovement row) { return row.candidate.context().values().stream().filter(v -> v.id().equals(row.candidate.context().nodes().getFirst().outputs().getFirst())).findFirst().orElseThrow().descriptor().shape().rank(); }
    private static List<Integer> compositionOccurrences(PreparedMovement row) {
        return ((CpuDataMovementIr) new CpuPartitionPreparer().analyze(row.candidate.context()).plan()
                .units().getFirst().portablePlan().portableKernelIr()).plan().occurrenceToBoundary();
    }
    private static io.github.pho001.synaptik.model.tensor.TensorDescriptor descriptorForBoundary(PreparedMovement row, int role) { var id = row.plan.boundaryValues().get(role); return row.candidate.context().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor(); }
    private static int arrayLength(Object value) { if (value instanceof double[] x) return x.length; if (value instanceof float[] x) return x.length; if (value instanceof short[] x) return x.length; if (value instanceof long[] x) return x.length; if (value instanceof int[] x) return x.length; return ((byte[]) value).length; }
    private static void invokeComposition(java.lang.invoke.MethodHandle entry, List<MovementCarrier> carriers, long[] geometry, long[] range) throws Throwable { var arguments = new ArrayList<Object>(); for (MovementCarrier carrier : carriers) arguments.add(carrier.carrier); arguments.add(geometry.clone()); arguments.add(range[0]); arguments.add(range[1]); entry.invokeWithArguments(arguments); }
    private static List<MovementCarrier> provenanceCarriers(PreparedMovement row, boolean bool, int markedRole, int markedAddress) {
        var result = new ArrayList<MovementCarrier>();
        for (int role = 0; role < row.plan.boundaryValues().size(); role++) {
            int cells = Math.toIntExact(maxAddress(descriptorForBoundary(row, role)) + compositionBases(row)[role] + 1);
            Object array = newProvenanceArray(row.type, cells, role, bool && role == markedRole ? markedAddress : -1, role == row.plan.boundaryValues().size() - 1);
            result.add(new MovementCarrier(array, argument(row.specialization.carrierPattern().get(role), array)));
        }
        return result;
    }
    private static Object newProvenanceArray(DataType type, int cells, int role, int markedAddress, boolean output) {
        Object result = switch (type) { case FLOAT64 -> new double[cells]; case FLOAT32 -> new float[cells]; case BFLOAT16 -> new short[cells]; case INT64 -> new long[cells]; case INT32 -> new int[cells]; case BOOL -> new byte[cells]; };
        for (int address = 0; address < cells; address++) setRaw(result, type, address, output ? 0L : provenanceValue(type, role, address, markedAddress));
        return result;
    }
    private static long[] compositionBases(PreparedMovement row) {
        long[] bases = new long[row.plan.boundaryValues().size()];
        for (int role = 0; role < bases.length; role++) bases[role] = 3L + 5L * role;
        return bases;
    }
    private static long provenanceValue(DataType type, int role, int address, int markedAddress) {
        if (type == DataType.BOOL) return address == markedAddress ? 1L : 0L;
        return switch (type) { case FLOAT64 -> 0x3ff0000000000000L + role * 0x10000L + address; case FLOAT32 -> 0x3f000000L + role * 0x10000L + address;
            case BFLOAT16 -> 0x1000L + role * 0x100L + address; case INT64 -> 0x1020304050600000L + role * 0x10000L + address;
            case INT32 -> 0x10200000L + role * 0x10000L + address; case BOOL -> throw new AssertionError(); };
    }
    private static void setRaw(Object target, DataType type, int address, long value) { switch (type) { case FLOAT64 -> ((double[]) target)[address] = Double.longBitsToDouble(value); case FLOAT32 -> ((float[]) target)[address] = Float.intBitsToFloat((int) value); case BFLOAT16 -> ((short[]) target)[address] = (short) value; case INT64 -> ((long[]) target)[address] = value; case INT32 -> ((int[]) target)[address] = (int) value; case BOOL -> ((byte[]) target)[address] = (byte) value; } }
    private record ProvenanceIdentity(int role, long address) { }
    private record CompositionRelation(long outputAddress, List<Long> logicalCoordinate, int occurrenceOrdinal,
            int carrierRole, long sourceAddress, long relativeOrInputCoordinate) { }
    private static RangeFacts rangeFacts(MethodModel method) { return movementFacts(method).range; }
    private static void assertCompositionProjection(String owner, String form, CpuKernelSpecialization specialization,
            List<Integer> occurrence, CompositionFacts generated, CompositionFacts clean) {
        assertTrue(Set.of("CONCAT", "STACK").contains(form), owner + " composition form");
        assertTrue(!occurrence.isEmpty() && occurrence.stream().allMatch(i -> i >= 0), owner + " ordered occurrence map");
        assertTrue(occurrence.stream().distinct().count() >= 2, owner + " unique multi-input carrier role order");
        assertRange(owner + " generated", generated.range); assertRange(owner + " clean", clean.range);
        assertTrue(generated.branches >= occurrence.size() && clean.branches >= occurrence.size(), owner + " occurrence selection branches");
        assertTrue(generated.loads >= occurrence.stream().distinct().count() && clean.loads >= occurrence.stream().distinct().count(), owner + " typed source selection loads");
        assertTrue(generated.stores >= 1 && clean.stores >= 1, owner + " typed output progression/store");
        assertFalse(generated.opcodes.contains("ATHROW") || clean.opcodes.contains("ATHROW"),
                owner + " no terminal/default exception path");
        if (specialization != null) assertEquals(occurrence.stream().distinct().count() + 1, specialization.carrierPattern().size(), owner + " exact unique carrier ABI roles");
        // Preserve semantic occurrences, rather than collapsing them to the unique parameter
        // roles.  In particular CONCAT's first and final arms deliberately read the same
        // carrier in the checked repeated-boundary fixtures.
        List<Integer> expectedSources = occurrence.stream().distinct().sorted().toList();
        int output = occurrence.stream().distinct().mapToInt(Integer::intValue).max().orElseThrow() + 1;
        assertEquals(new CompositionCarrierDataflow(expectedSources, List.of(output)), generated.carriers,
                owner + " generated selected occurrence carrier reads and output placement");
        assertEquals(generated.carriers, clean.carriers,
                owner + " generated/clean typed source-carrier and output-store relation");
        assertEquals(form.equals("CONCAT") ? List.of(0, 1, 0) : List.of(0, 1), occurrence,
                owner + " source-derived occurrence-to-boundary order/repetition");
    }
    private static void assertCompositionHygiene(String owner, MethodModel method, boolean clean) {
        List<String> forbidden = List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY",
                "INVOKEDYNAMIC", "ATHROW");
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            assertFalse(forbidden.contains(instruction.opcode().name()),
                    owner + " forbidden selected-entry opcode: " + instruction.opcode().name());
            if (instruction instanceof InvokeInstruction call) {
                String target = call.owner().asInternalName() + '.' + call.name() + call.type();
                assertFalse(target.contains("IllegalStateException.<init>"),
                        owner + " terminal exception constructor");
                assertTrue(clean ? permittedCleanFfmInvoke(target) : permittedGeneratedFfmInvoke(target),
                        owner + " unapproved selected-entry call: " + target);
            }
        }
    }
    private static List<String> append(List<String> facts, String added) {
        var result = new ArrayList<>(facts); result.add(added); return List.copyOf(result);
    }
    private record CompositionFacts(int branches, int loads, int stores, RangeFacts range,
            List<String> invokes, List<String> opcodes, CompositionCarrierDataflow carriers) { }
    private record CompositionCarrierDataflow(List<Integer> reads, List<Integer> writes) { }

    /**
     * Bounded parameter-origin extraction for the composition hot body.  It follows a descriptor
     * parameter (or a local copy) through the next direct array/MemorySegment typed access.  The
     * scan intentionally does not infer arbitrary control flow; branch predicates, relative
     * coordinate arithmetic and repeated occurrence arms stay independently exercised by the
     * paired full/empty/interior/tail executions above.
     */
    private static CompositionCarrierDataflow compositionCarrierDataflow(MethodModel method) {
        List<String> parameters = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.parameters(method.methodType().stringValue());
        int carriers = parameters.size() - 3;
        assertTrue(carriers >= 3 && parameters.get(carriers).equals("long[]"), "composition descriptor carrier/geometry ABI");
        List<Instruction> instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        var reads = new java.util.TreeSet<Integer>(); var writes = new java.util.TreeSet<Integer>();
        for (int role = 0; role < carriers; role++) {
            int slot = parameterSlot(parameters, role);
            Set<Integer> origins = parameterDerivedLocals(instructions, slot);
            for (int start = 0; start < instructions.size(); start++) {
                if (!(instructions.get(start) instanceof LoadInstruction load) || !origins.contains(load.slot())) continue;
                for (int end = start + 1; end < Math.min(instructions.size(), start + 100); end++) {
                    Instruction candidate = instructions.get(end);
                    if (candidate instanceof LoadInstruction next && next.slot() >= 0 && next.slot() != load.slot()
                            && next.slot() < parameterSlot(parameters, carriers)) break;
                    Boolean write = randomAccessDirection(candidate);
                    if (write != null) {
                        // A source parameter may be followed by coordinate/geometry LASTORE in
                        // the generated cursor advance.  Only the descriptor's final carrier is
                        // permitted to be the composed value sink.
                        if (write && role == carriers - 1) writes.add(role);
                        else if (!write && role != carriers - 1) reads.add(role);
                        break;
                    }
                }
            }
        }
        return new CompositionCarrierDataflow(List.copyOf(reads), List.copyOf(writes));
    }

    /**
     * Verifies the exact inventory and generated-entry ABI while deliberately leaving rows
     * without a family-specific clean body partial.
     *
     * <p>This is inventory accounting plus instruction/member-based generated-entry hygiene;
     * it is not a semantic or structural counterpart for the unimplemented clean families.
     * Affine, PAD/TILE, CONCAT/STACK, SLICE_UPDATE, UNFOLD_AXIS, and UNFOLD2D have separate
     * paired projections above. The other 0009C families stay visibly partial until dedicated
     * counterparts exist.</p>
     */
    @Test void exactOwnedRowsRetainGeneratedAbiAndScopedHygieneWhileOnlyOtherFamiliesRemainPartial() throws Exception {
        new CpuGeneratedCoverageCheckpointTest().exactCombinationInventoryReproducesEveryCanonicalFixtureExecution();
        List<Row> rows = ownedRows();
        assertEquals(2_252, rows.size(), "CPU 0009C exact generated denominator");
        assertEquals(Map.of("affine", 1_536L, "movement", 128L, "indexing", 152L, "scatter", 416L, "random", 20L), counts(rows));
        assertEquals(Map.of("CONTIGUOUS", 192L, "EXPAND", 192L, "EXPAND_DIMS", 192L, "PERMUTE", 192L, "RESHAPE", 192L, "SELECT", 192L, "SLICE", 192L, "SQUEEZE", 192L), formCounts(rows, "affine"));
        assertEquals(Map.of("PAD", 4L, "TILE", 24L, "CONCAT", 24L, "STACK", 24L, "SLICE_UPDATE", 24L, "UNFOLD_AXIS", 24L, "UNFOLD2D", 4L), formCounts(rows, "movement"));
        assertEquals(Map.of("GATHER", 48L, "GATHER_ELEMENTS", 48L, "GATHER_ND", 48L, "ONE_HOT", 8L), formCounts(rows, "indexing"));
        assertEquals(Map.of("SCATTER_ELEMENTS", 208L, "SCATTER_ND", 208L), formCounts(rows, "scatter"));
        assertEquals(Map.of("DROPOUT", 16L, "INITIAL_STATE", 4L), formCounts(rows, "random"));
        assertFalse(rows.stream().anyMatch(row -> row.form.equals("FOLD_AXIS") || row.form.equals("FOLD2D")), "fold remains CPU 0009D");

        int hygienicCompositionRows = 0;
        for (Row row : rows) {
            byte[] generated = CpuGeneratedCoverageEvidenceRegistry.classBytes(row.owner);
            MethodModel selected = selected(methods(generated));
            assertEquals(row.descriptor, selected.methodType().stringValue(), row.owner + " generated ABI");
            assertTrue(hasBranch(selected), row.owner + " generated active range branch");
            if (row.form.equals("CONCAT") || row.form.equals("STACK")) {
                assertCompositionHygiene(row.owner, selected, false);
                hygienicCompositionRows++;
            } else assertGeneratedInstructionMemberHygiene(row.owner, selected);
            assertFamilyFacts(row);
        }
        assertEquals(48, hygienicCompositionRows,
                "CONCAT/STACK entries have active allocation/exception hygiene evidence");
    }

    /**
     * Defines every source-derived indexing entry and invokes it beside an independently
     * compiled clean writer.  Both see fresh, equal typed carriers and the exact packed geometry
     * prepared for that owner; this is intentionally not satisfied by inspecting the clean body.
     */
    @Test void everyIndexingRowDefinesExecutesAndMatchesAnIndependentTypedPrevalidatedWriter()
            throws Throwable {
        Map<String, Row> inventory = new LinkedHashMap<>();
        for (Row row : ownedRows()) if (row.family.equals("indexing")) inventory.put(row.owner, row);
        var prepared = new ArrayList<PreparedIndexing>();
        for (var candidate : CpuOrdinaryNonPointwiseGeneratedMatrixTest.indexingOrOrderingCandidates()) {
            String form = candidate.context().nodes().getFirst().operation().kind().name();
            String owner = "ordinary:" + candidate.ownerId();
            if (INDEXING.contains(form)) prepared.add(prepareIndexing(owner, form,
                    candidate.context(), inventory.remove(owner)));
        }
        for (var candidate : CpuOrdinaryNonPointwiseGeneratedMatrixTest.randomOrOneHotCandidates()) {
            if (candidate.operationForm().equals("ONE_HOT")) prepared.add(prepareIndexing(candidate.ownerId(),
                    candidate.operationForm(), candidate.context(), inventory.remove(candidate.ownerId())));
        }
        assertTrue(inventory.isEmpty(), "every checked indexing owner has a source-derived candidate");
        assertEquals(152, prepared.size(), "GATHER 48, GATHER_ELEMENTS 48, GATHER_ND 48, ONE_HOT 8");

        var cleanRows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        for (PreparedIndexing row : prepared) cleanRows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(
                row.owner, indexingFamily(row.form), row.specialization.entryType().descriptorString(),
                (row.form.equals("ONE_HOT") ? "BOOL:" : row.dataType.name() + ':') + row.indexType.name(), false));
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(cleanRows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int executions = 0;
        for (PreparedIndexing row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(row.specialization, row.kernelIr);
            MethodModel generatedMethod = selected(methods(bytes));
            MethodModel cleanMethod = cleanMethods.get(clean.methods().get(row.owner));
            assertEquals(row.specialization.entryType().descriptorString(), generatedMethod.methodType().stringValue(), row.owner + " generated exact ABI");
            assertEquals(generatedMethod.methodType().stringValue(), cleanMethod.methodType().stringValue(), row.owner + " paired ordered carrier ABI");
            IndexingFacts generatedFacts = indexingFacts(generatedMethod, row), cleanFacts = indexingFacts(cleanMethod, row);
            assertIndexingProjection(row, generatedFacts, cleanFacts);
            assertIndexingInstructionHygiene(row.owner + " generated", generatedMethod);
            assertIndexingInstructionHygiene(row.owner + " clean", cleanMethod);
            var generatedEntry = generator.defineClassBytes(row.specialization, bytes).entryPoint();
            var cleanEntry = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.owner,
                    row.specialization.entryType().descriptorString());
            for (long[] range : indexingRanges(row.count)) {
                List<Object> actual = indexingStorage(row), expected = indexingStorage(row);
                List<Object> actualBefore = actual.stream().map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                List<Object> expectedBefore = expected.stream().map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                invokeIndexing(generatedEntry, row, actual, range);
                try {
                    invokeIndexing(cleanEntry, row, expected, range);
                } catch (Throwable failure) {
                    throw new AssertionError(row.owner + " clean indexing writer range="
                            + java.util.Arrays.toString(range), failure);
                }
                for (int role = 0; role < actual.size(); role++) {
                    DataType type = row.boundaryTypes.get(role);
                    assertEveryRawElementEquals(expected.get(role), actual.get(role), type, row.owner + " paired represented bits role=" + role + " range=" + java.util.Arrays.toString(range));
                    if (role + 1 < actual.size()) {
                        assertEveryRawElementEquals(actualBefore.get(role), actual.get(role), type, row.owner + " generated input immutable role=" + role);
                        assertEveryRawElementEquals(expectedBefore.get(role), expected.get(role), type, row.owner + " clean input immutable role=" + role);
                    }
                }
                executions++;
            }
        }
        assertTrue(executions >= 152 * 3, "full, empty and non-empty boundary ranges executed");
    }

    private static PreparedIndexing prepareIndexing(String owner, String form,
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> context, Row inventory) {
        assertTrue(inventory != null, owner + " inventory owner");
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var geometry = plan.indexingGeometry().orElseThrow();
        DataType index = context.values().get(form.equals("ONE_HOT") ? 0 : 1).descriptor().dataType();
        return new PreparedIndexing(owner, form, context, plan.elementCount(), geometry,
                geometry.boundaryTypes(), context.values().getFirst().descriptor().dataType(), index, route.specialization(), route.kernelIr());
    }
    private static String indexingFamily(String form) { return switch (form) { case "GATHER" -> "gather"; case "GATHER_ELEMENTS" -> "gather-elements"; case "GATHER_ND" -> "gather-nd"; case "ONE_HOT" -> "one-hot"; default -> throw new AssertionError(form); }; }
    private static List<long[]> indexingRanges(long count) { return count < 2 ? List.of(new long[] {0, count}, new long[] {0, 0}) : List.of(new long[] {0, count}, new long[] {0, 0}, new long[] {1, count - 1}, new long[] {count - 1, count}); }
    private static List<Object> indexingStorage(PreparedIndexing row) {
        var result = new ArrayList<Object>();
        for (int role = 0; role < row.boundaryTypes.size(); role++) {
            var descriptor = row.context.values().get(role).descriptor();
            Object storage = role + 1 == row.boundaryTypes.size() ? sentinels(descriptor.dataType(), Math.toIntExact(maxAddress(descriptor) + 1)) : patterned(descriptor.dataType(), Math.toIntExact(maxAddress(descriptor) + 1));
            result.add(storage);
        }
        int indexRole = row.form.equals("ONE_HOT") ? 0 : 1;
        Object indices = result.get(indexRole);
        // Valid, non-constant index values make each family map observable; ONE_HOT includes a
        // deliberately unmatched category so canonical zero stores are also exercised.
        for (int i = 0; i < java.lang.reflect.Array.getLength(indices); i++) setIndex(indices, row.indexType, i, row.form.equals("ONE_HOT") ? i % 4 - 1 : 0);
        return result;
    }
    private static void setIndex(Object array, DataType type, int index, long value) { if (type == DataType.INT64) ((long[]) array)[index] = value; else ((int[]) array)[index] = (int) value; }
    private static void invokeIndexing(java.lang.invoke.MethodHandle entry, PreparedIndexing row, List<Object> storage, long[] range) throws Throwable {
        var arguments = new ArrayList<Object>();
        for (int role = 0; role < storage.size(); role++) arguments.add(argument(row.specialization.carrierPattern().get(role), storage.get(role)));
        arguments.add(row.geometry.pack(new long[row.boundaryTypes.size()], range[0], range[1])); arguments.add(range[0]); arguments.add(range[1]); entry.invokeWithArguments(arguments);
    }

    private static IndexingFacts indexingFacts(MethodModel method, PreparedIndexing row) {
        int geometryLoads = 0, primitiveLoads = 0, primitiveStores = 0, forward = 0, backward = 0;
        boolean indexWidth = false, dataAccess = false, boolStore = false, familyMap = false;
        var invokes = new ArrayList<String>();
        var labels = new LinkedHashMap<Label, Integer>(); int position = 0;
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) { if (element instanceof LabelTarget target) labels.put(target.label(), position); if (element instanceof Instruction) position++; }
        position = 0;
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            int here = position++; String opcode = instruction.opcode().name();
            if (opcode.equals("LALOAD")) geometryLoads++;
            if (opcode.endsWith("ALOAD") && !opcode.equals("AALOAD")) primitiveLoads++;
            if (opcode.endsWith("ASTORE") && !opcode.equals("AASTORE")) primitiveStores++;
            if (row.indexType == DataType.INT64 ? opcode.equals("LALOAD") : opcode.equals("IALOAD")) indexWidth = true;
            if (row.form.equals("ONE_HOT") && opcode.equals("BASTORE")) boolStore = true;
            if (instruction instanceof FieldInstruction field
                    && field.owner().asInternalName().equals("java/lang/foreign/ValueLayout")
                    && field.name().stringValue().equals(row.indexType == DataType.INT64
                    ? "JAVA_LONG_UNALIGNED" : "JAVA_INT_UNALIGNED")) indexWidth = true;
            if (row.form.equals("ONE_HOT") && instruction instanceof FieldInstruction field
                    && field.owner().asInternalName().equals("java/lang/foreign/ValueLayout")
                    && field.name().stringValue().equals("JAVA_BYTE")) boolStore = true;
            if (instruction instanceof InvokeInstruction call) {
                String target = call.owner().asInternalName() + '.' + call.name() + call.type(); invokes.add(target);
                if (target.startsWith("java/lang/foreign/MemorySegment.get")) { primitiveLoads++; if (target.contains(row.indexType == DataType.INT64 ? "JAVA_LONG" : "JAVA_INT")) indexWidth = true; if (!row.form.equals("ONE_HOT")) dataAccess = true; }
                if (target.startsWith("java/lang/foreign/MemorySegment.set")) { primitiveStores++; if (row.form.equals("ONE_HOT") && target.contains("JAVA_BYTE")) boolStore = true; }
            }
            if (!row.form.equals("ONE_HOT") && (opcode.equals("DALOAD") || opcode.equals("FALOAD") || opcode.equals("SALOAD") || opcode.equals("LALOAD") || opcode.equals("IALOAD") || opcode.equals("BALOAD"))) dataAccess = true;
            if (instruction instanceof BranchInstruction branch) { if (labels.get(branch.target()) < here) backward++; else forward++; }
        }
        // Branch cardinality is extracted from the actual selected method.  The four lowerings
        // have respectively three-way splice, selected-axis, batch/tuple/suffix, and equality
        // shapes; it is deliberately not supplied by the inventory row.
        familyMap = switch (row.form) { case "GATHER" -> forward >= 3; case "GATHER_ELEMENTS" -> forward >= 2; case "GATHER_ND" -> forward >= 3; case "ONE_HOT" -> forward >= 1; default -> false; };
        return new IndexingFacts(method.methodType().stringValue(), geometryLoads, primitiveLoads, primitiveStores,
                indexWidth, dataAccess, familyMap, boolStore, forward > 0, backward > 0, List.copyOf(invokes));
    }
    private static void assertIndexingProjection(PreparedIndexing row, IndexingFacts generated, IndexingFacts clean) {
        assertEquals(generated.descriptor, clean.descriptor, row.owner + " exact ABI and carrier order");
        for (IndexingFacts facts : List.of(generated, clean)) {
            assertTrue(facts.geometryLoads >= 2, row.owner + " packed address-map reads");
            assertTrue(facts.indexWidth, row.owner + " typed " + row.indexType + " index load");
            assertTrue(facts.forward && facts.backward, row.owner + " half-open range direction/cursor progression");
            assertTrue(facts.familyMap, row.owner + " family-specific map shape");
            if (row.form.equals("ONE_HOT")) assertTrue(facts.boolStore, row.owner + " canonical BOOL byte write");
            else assertTrue(facts.dataAccess && facts.primitiveStores > 0, row.owner + " typed data/output access");
            assertIndexingHygiene(row, facts.invokes, facts == clean);
        }
    }
    private static void assertIndexingHygiene(PreparedIndexing row, List<String> invokes, boolean clean) {
        boolean segment = row.specialization != null && row.specialization.carrierPattern().stream()
                .anyMatch(access -> access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        Set<String> actual = Set.copyOf(invokes);
        if (!segment) {
            assertEquals(Set.of(), actual, row.owner + " heap indexing entry has no invoke");
            return;
        }
        assertTrue(indexingFfmInvokes(row, clean).containsAll(actual), row.owner + ' '
                + (clean ? "clean" : "generated") + " exact indexing FFM invoke allowlist actual=" + actual);
    }
    private static Set<String> indexingFfmInvokes(PreparedIndexing row, boolean clean) {
        String load = clean ? "getAtIndex" : "get", store = clean ? "setAtIndex" : "set";
        var allowed = new java.util.TreeSet<String>();
        if (!clean) {
            allowed.add("java/nio/ByteOrder.nativeOrder()Ljava/nio/ByteOrder;");
            allowed.add("java/lang/foreign/ValueLayout.withOrder(Ljava/nio/ByteOrder;)Ljava/lang/foreign/ValueLayout;");
        }
        for (int role = 0; role < row.boundaryTypes.size(); role++) {
            if (row.specialization.carrierPattern().get(role) != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                continue;
            allowed.add(indexingFfmTarget(role + 1 == row.boundaryTypes.size() ? store : load,
                    row.boundaryTypes.get(role)));
        }
        return Set.copyOf(allowed);
    }
    private static String indexingFfmTarget(String access, DataType type) {
        String layout = switch (type) {
            case FLOAT64 -> "OfDouble:D"; case FLOAT32 -> "OfFloat:F";
            case BFLOAT16 -> "OfShort:S"; case INT64 -> "OfLong:J";
            case INT32 -> "OfInt:I"; case BOOL -> "OfByte:B";
        };
        int separator = layout.indexOf(':');
        String layoutType = layout.substring(0, separator), payload = layout.substring(separator + 1);
        return "java/lang/foreign/MemorySegment." + access + "(Ljava/lang/foreign/ValueLayout$"
                + layoutType + ";J" + (access.startsWith("set") ? payload : "") + ")"
                + (access.startsWith("set") ? "V" : payload);
    }
    private static void assertIndexingInstructionHygiene(String owner, MethodModel method) {
        Set<String> forbidden = Set.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "ATHROW", "INVOKEDYNAMIC");
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (element instanceof Instruction instruction) assertFalse(forbidden.contains(instruction.opcode().name()),
                    owner + " forbidden indexing allocation, throw, or dynamic dispatch: " + instruction.opcode().name());
        }
    }
    /** Independently compiled indexing mutations keep every fact and hygiene rejection live. */
    @Test void indexingProjectionRejectsIndependentlyCompiledSourceMutants() throws Exception {
        var basis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("control", "gather",
                "(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;[JJJ)V",
                "INT32:INT32", false, 2, 0L, List.of());
        var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileIndexingMutants(basis);
        for (var entry : mutants.methods().entrySet()) {
            MethodModel method = methods(mutants.bytes()).get(entry.getValue());
            String name = entry.getKey();
            assertTrue(method != null, "compiled indexing mutant " + name);
            if (name.equals("bool-store")) continue; // ONE_HOT owns this distinct writer fact below.
            if (name.equals("abi")) {
                assertFalse(method.methodType().stringValue().equals(basis.descriptor()),
                        "actual compiled ABI mutation");
            } else if (name.equals("cross-ffm-bridge")) {
                assertThrows(AssertionError.class, () -> assertIndexingGeneratedOnlyInvokes("mutant", method));
            } else if (Set.of("new", "anewarray", "newarray", "multianewarray", "athrow", "invokedynamic").contains(name)) {
                assertThrows(AssertionError.class, () -> assertIndexingInstructionHygiene("mutant " + name, method));
            } else if (Set.of("helper", "boxing", "reflection", "string-dispatch", "map-dispatch").contains(name)) {
                assertThrows(AssertionError.class, () -> assertIndexingGeneratedOnlyInvokes("mutant " + name, method));
            } else {
                IndexingFacts facts = indexingFacts(method, indexingMutationRow());
                assertFalse(facts.indexWidth && facts.familyMap && facts.forward && facts.backward,
                        "actual compiled indexing fact mutation " + name);
            }
        }
        var hotBasis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("hot", "one-hot",
                "(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;[JJJ)V",
                "BOOL:INT32", false, 2, 0L, List.of());
        var hotMutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileIndexingMutants(hotBasis);
        MethodModel missingBoolStore = methods(hotMutants.bytes()).get(hotMutants.methods().get("bool-store"));
        IndexingFacts hotFacts = indexingFacts(missingBoolStore, new PreparedIndexing("hot", "ONE_HOT",
                null, 4, null, List.of(DataType.INT32, DataType.BOOL), DataType.BOOL, DataType.INT32,
                null, null));
        assertFalse(hotFacts.boolStore, "actual compiled ONE_HOT canonical BOOL-store mutation");
    }
    private static PreparedIndexing indexingMutationRow() {
        return new PreparedIndexing("mutant", "GATHER", null, 4, null,
                List.of(DataType.INT32, DataType.INT32, DataType.INT32), DataType.INT32,
                DataType.INT32, null, null);
    }
    private static void assertIndexingGeneratedOnlyInvokes(String owner, MethodModel method) {
        var invokes = new ArrayList<String>();
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (element instanceof InvokeInstruction call) invokes.add(call.owner().asInternalName() + '.'
                    + call.name() + call.type());
        }
        assertEquals(Set.of(), Set.copyOf(invokes), owner + " generated indexing helper/cross-FFM invoke");
    }

    /**
     * Compiles an independent typed clean writer for every ordinary scatter owner and compares
     * the selected static ABI plus the non-erased state topology.  The direct scatter semantic
     * closure remains the executable represented-value oracle; this projection specifically
     * retains base copy, ordered index/update loads, output store, row-major contribution loop,
     * range ownership, and the floating-MUL workspace parameter boundary.
     */
    @Test void everyScatterRowHasAnIndependentTypedCleanJavaStateProjection() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.scatterCandidates();
        assertEquals(416, candidates.size(), "exact scatter owner denominator");
        var cleanRows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedScatter>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var geometry = plan.scatterGeometry().orElseThrow();
            DataType data = candidate.context().values().getFirst().descriptor().dataType();
            DataType index = candidate.context().values().get(1).descriptor().dataType();
            // The entry descriptor is the ABI authority.  Geometry independently supplies the
            // selected slice offset/extent; do not allow a fixture-row scratch flag to decide it.
            String descriptor = route.specialization().entryType().descriptorString();
            ScratchRequirement scratch = scatterScratchRequirement(descriptor, geometry.boundaries().size(),
                    data, geometry.maximumUpdatesPerTarget(), geometry.scratchSliceBytes());
            assertEquals(scratch.required(), plan.workspaceDeclaration().isPresent(), candidate.ownerId()
                    + " descriptor/workspace declaration boundary");
            assertEquals(scratch.bytes(),
                    geometry.scratchSliceBytes(), candidate.ownerId() + " prepared exact-product slice");
            String form = geometry.family().name();
            String family = form.equals("SCATTER_ELEMENTS") ? "scatter-elements" : "scatter-nd";
            String typed = data.name() + ':' + index.name() + ':' + geometry.reduction().name()
                    + (scratch.required() ? ":scratch" : "");
            cleanRows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(), family,
                    descriptor, typed, false,
                    geometry.outputExtents().length, geometry.boundaries().get(geometry.occurrenceToBoundary().get(2)).extents().length));
            prepared.add(new PreparedScatter(candidate.ownerId(), form, data, index,
                    geometry.reduction().name(), scratch,
                    route.specialization(), route.kernelIr()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(cleanRows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int elements = 0, nd = 0, scratchRows = 0;
        for (PreparedScatter row : prepared) {
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(row.specialization, row.kernelIr);
            assertEquals(java.util.HexFormat.of().formatHex(bytes), java.util.HexFormat.of().formatHex(
                    generator.generateClassBytes(row.specialization, row.kernelIr)), row.owner + " deterministic bytes");
            MethodModel counterpart = cleanMethods.get(clean.methods().get(row.owner));
            MethodModel generated = selected(methods(bytes));
            assertEquals(row.specialization.entryType().descriptorString(), counterpart.methodType().stringValue(),
                    row.owner + " clean base/index/update/output ABI");
            assertEquals(generated.methodType().stringValue(), counterpart.methodType().stringValue(),
                    row.owner + " paired selected ABI");
            assertTrue(hasBranch(counterpart), row.owner + " clean copy/update range loops");
            assertGeneratedInstructionMemberHygiene(row.owner + " clean", counterpart);
            ScatterFacts cleanFacts = scatterFacts(counterpart), generatedFacts = scatterFacts(generated);
            assertScatterProjection(row, cleanFacts, generatedFacts);
            assertScratchTopology(row, counterpart, "clean");
            assertScratchTopology(row, generated, "generated");
            assertScatterHygiene(row.owner + " clean", row, cleanFacts);
            assertScatterHygiene(row.owner + " generated", row, generatedFacts);
            var generatedEntry = generator.defineClassBytes(row.specialization, bytes).entryPoint();
            var cleanEntry = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean, row.owner,
                    row.specialization.entryType().descriptorString());
            var candidate = CpuOrdinaryNonPointwiseGeneratedMatrixTest.scatterCandidates().stream()
                    .filter(value -> value.ownerId().equals(row.owner)).findFirst().orElseThrow();
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            for (long[] range : ranges(plan.elementCount())) {
                var actual = scatterCarriers(candidate, row.specialization);
                var expected = scatterCarriers(candidate, row.specialization);
                var actualBefore = actual.stream().map(ScatterCarrier::array)
                        .map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                var expectedBefore = expected.stream().map(ScatterCarrier::array)
                        .map(CpuAffineMovementIndexingScatterRandomStructuralOracleTest::cloneArray).toList();
                long[] geometry = plan.scatterGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()],
                        range[0], range[1], 0);
                assertEquals(row.scratchRequirement.bytes(), geometry[14], row.owner + " packed scratch extent");
                try (var actualArena = java.lang.foreign.Arena.ofConfined();
                        var expectedArena = java.lang.foreign.Arena.ofConfined()) {
                    List<Object> actualArguments = scatterArguments(actual, row, geometry, range, actualArena);
                    List<Object> expectedArguments = scatterArguments(expected, row, geometry.clone(), range, expectedArena);
                    generatedEntry.invokeWithArguments(actualArguments);
                    cleanEntry.invokeWithArguments(expectedArguments);
                }
                for (int role = 0; role < actual.size(); role++) assertEveryRawElementEquals(
                        actual.get(role).array, expected.get(role).array, actual.get(role).type,
                        row.owner + " paired scatter represented bits, carriers and sentinels range="
                                + java.util.Arrays.toString(range));
                for (int role = 0; role + 1 < actual.size(); role++) {
                    assertEveryRawElementEquals(actualBefore.get(role), actual.get(role).array, actual.get(role).type,
                            row.owner + " generated input immutable role=" + role);
                    assertEveryRawElementEquals(expectedBefore.get(role), expected.get(role).array, expected.get(role).type,
                            row.owner + " clean input immutable role=" + role);
                }
            }
            if (row.form.equals("SCATTER_ELEMENTS")) elements++; else nd++;
            if (row.scratchRequirement.required()) scratchRows++;
        }
        assertEquals(208, elements, "SCATTER_ELEMENTS paired rows");
        assertEquals(208, nd, "SCATTER_ND paired rows");
        assertTrue(scratchRows > 0, "floating MUL retains its separate workspace ABI");
    }

    private static List<Object> scatterArguments(List<ScatterCarrier> carriers, PreparedScatter row,
            long[] geometry, long[] range, java.lang.foreign.Arena arena) {
        var arguments = new ArrayList<Object>();
        carriers.forEach(carrier -> arguments.add(carrier.carrier));
        if (row.scratchRequirement.required()) {
            assertTrue(geometry[14] > 0L, row.owner + " exact-product descriptor requires non-empty slice");
            var scratch = arena.allocate(geometry[14], 8);
            for (long offset = 0; offset < geometry[14]; offset += Long.BYTES)
                scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG, offset, 0x5a5a5a5a5a5a5a5aL);
            arguments.add(scratch);
        }
        arguments.add(geometry); arguments.add(range[0]); arguments.add(range[1]);
        return arguments;
    }

    private static List<ScatterCarrier> scatterCarriers(
            CpuOrdinaryNonPointwiseGeneratedMatrixTest.ScatterCandidate candidate,
            CpuKernelSpecialization specialization) {
        var carriers = new ArrayList<ScatterCarrier>();
        for (int role = 0; role < specialization.carrierPattern().size(); role++) {
            var descriptor = candidate.context().values().get(role).descriptor();
            DataType type = descriptor.dataType();
            int capacity = Math.max(32, Math.toIntExact(maxAddress(descriptor) + 1));
            Object array = role + 1 == specialization.carrierPattern().size()
                    ? sentinels(type, capacity) : patterned(type, capacity);
            if (role == 1) fillScatterIndexes(array, type);
            else if (role + 1 < specialization.carrierPattern().size()) fillScatterValues(array, type, role == 2 ? 1 : 2);
            carriers.add(new ScatterCarrier(array, argument(specialization.carrierPattern().get(role), array), type));
        }
        return carriers;
    }

    private static void fillScatterIndexes(Object array, DataType type) {
        if (type == DataType.INT32) { int[] values = (int[]) array; for (int i = 0; i < values.length; i++) values[i] = i % 2; }
        else if (type == DataType.INT64) { long[] values = (long[]) array; for (int i = 0; i < values.length; i++) values[i] = i % 2; }
        else throw new AssertionError("scatter index type: " + type);
    }

    private static void fillScatterValues(Object array, DataType type, int value) {
        switch (type) {
            case FLOAT64 -> java.util.Arrays.fill((double[]) array, value);
            case FLOAT32 -> java.util.Arrays.fill((float[]) array, value);
            case BFLOAT16 -> java.util.Arrays.fill((short[]) array, (short) (value == 1 ? 0x3f80 : 0x4000));
            case INT64 -> java.util.Arrays.fill((long[]) array, value);
            case INT32 -> java.util.Arrays.fill((int[]) array, value);
            case BOOL -> java.util.Arrays.fill((byte[]) array, (byte) value);
        }
    }

    /** Active controls ensure retained scatter facts cannot be removed cosmetically. */
    @Test void scatterProjectionRejectsAbiOrderIndexRangeReductionScratchStoreAndHygieneMutations() throws Exception {
        PreparedScatter baseline = new PreparedScatter("control", "SCATTER_ELEMENTS", DataType.FLOAT32,
                DataType.INT32, "ADD", new ScratchRequirement(false, 0L, 0L, "([F[I[F[F[JJJ)V"), null, null);
        ScatterFacts facts = new ScatterFacts("([F[I[F[F[JJJ)V", 4, 3, 2, 2, 2,
                true, true, true, true, true, true, List.of(), List.of("IFGE"));
        assertScatterProjection(baseline, facts, facts);
        for (ScatterFacts changed : List.of(
                new ScatterFacts(facts.descriptor, 3, 3, 2, 2, 2, true, true, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 2, 2, 2, 2, true, true, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 3, 1, 2, 2, true, true, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 3, 2, 1, 2, true, true, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 3, 2, 2, 2, false, true, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 3, 2, 2, 2, true, false, true, true, true, true, List.of(), facts.opcodes),
                new ScatterFacts(facts.descriptor, 4, 3, 2, 2, 2, true, true, false, true, true, true, List.of(), facts.opcodes)))
            assertThrows(AssertionError.class, () -> assertScatterProjection(baseline, changed, facts));
        var source = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("control", "scatter-elements",
                baseline.scratchRequirement.descriptor(), "FLOAT32:INT32:ADD", false, 2, 2L);
        var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileScatterMutants(source);
        for (var entry : mutants.methods().entrySet()) {
            String method = entry.getValue();
            ScatterFacts mutant = scatterFacts(methods(mutants.bytes()).get(method));
            if (Set.of("new", "boxing", "reflection", "string-dispatch", "map-dispatch", "helper", "athrow", "invokedynamic", "reverse-range", "missing-index-load", "wrong-reduction", "missing-base-copy")
                    .contains(entry.getKey())) {
                assertThrows(AssertionError.class, () -> assertScatterHygiene(entry.getKey(), baseline, mutant), method);
            } else assertThrows(AssertionError.class, () -> assertScatterProjection(baseline, mutant, facts), method);
        }
    }

    /** Real javac source mutants prove that each live scratch role remains mandatory. */
    @Test void scatterScratchProjectionRejectsResetHeaderLimbPublicationAndSizeMutants() throws Exception {
        String descriptor = "([F[I[F[FLjava/lang/foreign/MemorySegment;[JJJ)V";
        PreparedScatter basis = new PreparedScatter("scratch-control", "SCATTER_ELEMENTS", DataType.FLOAT32,
                DataType.INT32, "MUL", new ScratchRequirement(true, 32L, 0L, descriptor), null, null);
        var row = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("scratch-control", "scatter-elements",
                descriptor, "FLOAT32:INT32:MUL:scratch", false, 2, 2L);
        var mutants = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileScatterMutants(row);
        MethodModel canonical = methods(mutants.bytes()).get(mutants.methods().get("missing-base-copy"));
        assertTrue(canonical != null, "independently javac-compiled scratch canonical body");
        // The canonical source differs only in an unrelated base-copy mutant; its scratch body is intact.
        assertScratchTopology(basis, canonical, "clean");
        for (String name : List.of("missing-scratch-reset", "wrong-scratch-header", "missing-scratch-limb",
                "missing-scratch-publication", "scratch-size")) {
            MethodModel mutant = methods(mutants.bytes()).get(mutants.methods().get(name));
            assertThrows(AssertionError.class, () -> assertScratchTopology(basis, mutant, "clean"),
                    "real javac scratch mutant rejected: " + name);
        }
    }

    private static ScatterFacts scatterFacts(MethodModel method) {
        int loads = 0, stores = 0, longLoads = 0, branches = 0;
        boolean range = false, baseCopy = false, indexLoad = false, updateLoad = false, outputStore = false;
        List<String> invokes = new ArrayList<>(), opcodes = new ArrayList<>();
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            String opcode = instruction.opcode().name();
            opcodes.add(opcode);
            if (opcode.endsWith("ALOAD") && !opcode.equals("AALOAD")) loads++;
            if (opcode.endsWith("ASTORE") && !opcode.equals("AASTORE")) stores++;
            if (opcode.equals("LALOAD")) longLoads++;
            if (opcode.startsWith("IF") || opcode.startsWith("GOTO")) { branches++; range = true; }
            if (instruction instanceof InvokeInstruction call) {
                String target = call.owner().asInternalName() + '.' + call.name() + call.type();
                invokes.add(target);
                if (target.contains("MemorySegment.get")) { loads++; indexLoad = true; updateLoad = true; }
                if (target.contains("MemorySegment.set")) { stores++; outputStore = true; }
            }
            if (opcode.equals("AALOAD") || opcode.equals("IALOAD") || opcode.equals("LALOAD")) indexLoad = true;
            if (opcode.endsWith("ASTORE") || opcode.equals("DASTORE") || opcode.equals("FASTORE")
                    || opcode.equals("LASTORE") || opcode.equals("IASTORE") || opcode.equals("SASTORE") || opcode.equals("BASTORE")) {
                outputStore = true; baseCopy = true;
            }
            if (opcode.equals("FADD") || opcode.equals("DADD") || opcode.equals("IADD") || opcode.equals("LADD")
                    || opcode.equals("FMUL") || opcode.equals("DMUL") || opcode.equals("IMUL") || opcode.equals("LMUL")) updateLoad = true;
        }
        return new ScatterFacts(method.methodType().stringValue(), 4, loads, stores, longLoads, branches, range,
                loads >= 2, baseCopy, indexLoad, updateLoad, outputStore, List.copyOf(invokes), List.copyOf(opcodes));
    }

    private static void assertScatterProjection(PreparedScatter row, ScatterFacts clean, ScatterFacts generated) {
        assertTrue(clean.roles == 4 && generated.roles == 4, row.owner + " ordered base/index/update/output roles");
        assertEquals(clean.roles, generated.roles, row.owner + " ordered carrier role count");
        assertTrue(clean.loads >= 3 && generated.loads >= 3, row.owner + " typed index/update loads");
        assertTrue(clean.orderedLoads && generated.orderedLoads, row.owner + " typed base/index/update load ordering");
        assertTrue(clean.stores >= 2 && generated.stores >= 2 && clean.baseCopy && generated.baseCopy,
                row.owner + " base-copy-before-update topology");
        assertTrue(clean.indexLoad && generated.indexLoad && clean.updateLoad && generated.updateLoad
                        && clean.outputStore && generated.outputStore,
                row.owner + " typed index/update load and output-store dataflow");
        assertTrue(clean.longLoads >= 2 && generated.longLoads >= 2, row.owner + " packed range/layout target map");
        assertTrue(clean.range && generated.range && clean.branches >= 2 && generated.branches >= 2,
                row.owner + " half-open output ownership and sequential contribution range");
        assertTrue(clean.opcodes.stream().anyMatch(opcode -> opcode.contains("GE"))
                        && generated.opcodes.stream().anyMatch(opcode -> opcode.contains("GE")),
                row.owner + " forward half-open cursor direction");
        assertEquals(clean.range, generated.range, row.owner + " range direction fact");
        assertEquals(row.scratchRequirement.descriptor(), clean.descriptor, row.owner + " clean exact ABI/order");
        assertEquals(row.scratchRequirement.descriptor(), generated.descriptor, row.owner + " generated exact ABI/order");
    }

    private static ScratchRequirement scatterScratchRequirement(String descriptor, int preparedBoundaryCount,
            DataType type, long maximumUpdates, long preparedBytes) {
        List<String> parameters = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.parameters(descriptor);
        assertTrue(parameters.size() >= 7 && parameters.get(parameters.size() - 1).equals("long")
                        && parameters.get(parameters.size() - 2).equals("long")
                        && parameters.get(parameters.size() - 3).equals("long[]"),
                "scatter descriptor retains geometry/start/end ABI: " + descriptor);
        int dataParameters = parameters.size() - 3;
        assertTrue(dataParameters == preparedBoundaryCount || dataParameters == preparedBoundaryCount + 1,
                "scatter descriptor differs from prepared boundary topology: " + descriptor);
        // The descriptor contributes exactly one segment after the independently prepared
        // boundary topology only for exact-product workspace; ordinary segment carriers occupy
        // one of the boundary slots and cannot change this cardinality.
        boolean required = dataParameters == preparedBoundaryCount + 1
                && parameters.get(dataParameters - 1).equals("java.lang.foreign.MemorySegment");
        boolean applicable = type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
        long bytes = required ? exactScratchBytes(type, maximumUpdates) : 0L;
        assertEquals(bytes, preparedBytes, "descriptor-derived exact scratch extent");
        assertEquals(required, applicable && bytes > 0L, "exact-product scratch admission");
        return new ScratchRequirement(required, bytes, 0L, descriptor);
    }

    private static long exactScratchBytes(DataType type, long maximumUpdates) {
        if (type != DataType.FLOAT64 && type != DataType.FLOAT32 && type != DataType.BFLOAT16) return 0L;
        int precision = type == DataType.FLOAT64 ? 53 : type == DataType.FLOAT32 ? 24 : 8;
        long exponent = type == DataType.FLOAT64 ? 1_074L : type == DataType.FLOAT32 ? 149L : 133L;
        long factors = Math.addExact(maximumUpdates, 1L);
        Math.multiplyExact(factors, exponent); // independently retain the lowerer overflow guard.
        long limbs = Math.floorDiv(Math.addExact(Math.multiplyExact((long) precision, factors), 63L), 64L);
        return Math.addExact(24L, Math.multiplyExact(8L, limbs));
    }

    /**
     * Proves that the descriptor's scratch parameter, rather than an arbitrary segment carrier,
     * reaches the live exact-product reset/header/limb/publication accesses.  This deliberately
     * proves only live state: the emitter intentionally leaves unreachable trailing limbs
     * unspecified, so final whole-scratch byte equality would be a false requirement.
     */
    private static void assertScratchTopology(PreparedScatter row, MethodModel method, String implementation) {
        if (!row.scratchRequirement.required()) {
            return;
        }
        assertEquals(0L, row.scratchRequirement.offset(), row.owner + " exact scratch slice offset");
        assertTrue(row.scratchRequirement.bytes() >= 24L && row.scratchRequirement.bytes() % Long.BYTES == 0L,
                row.owner + " exact scratch extent");
        List<String> parameters = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.parameters(method.methodType().stringValue());
        assertEquals("java.lang.foreign.MemorySegment", parameters.get(parameters.size() - 4),
                row.owner + " one scratch boundary immediately before geometry/start/end");
        ScratchDataflow flow = scratchDataflow(method, parameters.size() - 4);
        if (implementation.equals("clean")) assertTrue(flow.sizeGuard,
                row.owner + " clean scratch parameter guards prepared exact slice bytes");
        assertTrue(flow.reset, row.owner + ' ' + implementation + " descriptor scratch parameter resets exact state header");
        assertTrue(flow.header, row.owner + ' ' + implementation + " descriptor scratch parameter writes exact state header");
        assertTrue(flow.limbRead && flow.limbWrite, row.owner + ' ' + implementation + " descriptor scratch parameter reads and updates live limb");
        if (implementation.equals("clean")) assertTrue(flow.publication,
                row.owner + " clean descriptor scratch parameter feeds publication observation");
    }

    private static ScratchDataflow scratchDataflow(MethodModel method, int scratchParameter) {
        List<String> parameters = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.parameters(method.methodType().stringValue());
        List<Instruction> instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        Set<Integer> scratchLocals = parameterDerivedLocals(instructions, parameterSlot(parameters, scratchParameter));
        boolean reset = false, header = false, limbRead = false, limbWrite = false, publication = false, sizeGuard = false;
        boolean firstLiveScratchAccess = true;
        for (int index = 0; index < instructions.size(); index++) {
            Instruction instruction = instructions.get(index);
            if (!(instruction instanceof InvokeInstruction call)
                    || !call.owner().asInternalName().equals("java/lang/foreign/MemorySegment")) continue;
            boolean fromScratch = false;
            int zeroes = 0;
            boolean hasEight = false, hasSixteen = false, hasTwentyFour = false, priorScratchGetTwentyFour = false;
            for (int prior = Math.max(0, index - 32); prior < index; prior++) {
                Instruction previous = instructions.get(prior);
                if (previous instanceof LoadInstruction load && scratchLocals.contains(load.slot())) fromScratch = true;
                if (previous instanceof InvokeInstruction priorCall
                        && priorCall.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
                        && priorCall.name().stringValue().startsWith("get")) priorScratchGetTwentyFour = true;
                if (previous instanceof ConstantInstruction constant && constant.constantValue() instanceof Number number) {
                    long value = number.longValue();
                    if (value == 0L) zeroes++;
                    hasEight |= value == 8L; hasSixteen |= value == 16L; hasTwentyFour |= value == 24L;
                }
            }
            if (!fromScratch) continue;
            String name = call.name().stringValue();
            if (name.equals("byteSize")) sizeGuard = true;
            if (name.startsWith("set")) {
                if (firstLiveScratchAccess) reset = zeroes >= 2;
                firstLiveScratchAccess = false;
                header |= hasEight && hasSixteen;
                limbWrite |= priorScratchGetTwentyFour;
            }
            if (name.startsWith("get")) {
                firstLiveScratchAccess = false;
                limbRead |= hasTwentyFour;
                publication |= hasTwentyFour && hasPublicationBranch(instructions, index);
            }
        }
        return new ScratchDataflow(sizeGuard, reset, header, limbRead, limbWrite, publication);
    }

    private static boolean hasFollowingStore(List<Instruction> instructions, int index, Set<Integer> scratchLocals) {
        for (int next = index + 1; next < Math.min(instructions.size(), index + 128); next++) {
            Instruction instruction = instructions.get(next);
            if (instruction instanceof ArrayStoreInstruction) return true;
            if (instruction instanceof InvokeInstruction call
                    && call.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
                    && call.name().stringValue().startsWith("set")) {
                boolean scratch = false;
                for (int prior = Math.max(index + 1, next - 24); prior < next; prior++)
                    if (instructions.get(prior) instanceof LoadInstruction load && scratchLocals.contains(load.slot())) scratch = true;
                return !scratch;
            }
        }
        return false;
    }

    private static boolean hasPublicationBranch(List<Instruction> instructions, int index) {
        for (int next = index + 1; next < Math.min(instructions.size(), index + 16); next++) {
            if (instructions.get(next).opcode() == Opcode.LCMP) return true;
            if (instructions.get(next) instanceof InvokeInstruction call
                    && call.owner().asInternalName().equals("java/lang/foreign/MemorySegment")) return false;
        }
        return false;
    }

    private static void assertScatterHygiene(String owner, PreparedScatter row, ScatterFacts facts) {
        for (String opcode : facts.opcodes) assertFalse(Set.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY",
                "INVOKEDYNAMIC", "ATHROW").contains(opcode), owner + " forbidden opcode " + opcode);
        for (String target : facts.invokes) assertTrue(permittedScatterInvoke(row, target), owner + " forbidden invoke " + target);
    }

    /** Exact owner/name/descriptor gate; no owner- or suffix-based helper admission. */
    private static boolean permittedScatterInvoke(PreparedScatter row, String target) {
        var allowed = new java.util.HashSet<String>();
        allowed.add("java/lang/foreign/MemorySegment.byteSize()J");
        allowed.add("java/nio/ByteOrder.nativeOrder()Ljava/nio/ByteOrder;");
        allowed.add("java/lang/foreign/ValueLayout.withOrder(Ljava/nio/ByteOrder;)Ljava/lang/foreign/ValueLayout;");
        allowed.addAll(Set.of("java/lang/Math.min(DD)D", "java/lang/Math.max(DD)D",
                "java/lang/Math.min(FF)F", "java/lang/Math.max(FF)F", "java/lang/Math.min(II)I",
                "java/lang/Math.max(II)I", "java/lang/Math.min(JJ)J", "java/lang/Math.max(JJ)J",
                "java/lang/Math.unsignedMultiplyHigh(JJ)J", "java/lang/Long.numberOfLeadingZeros(J)I",
                "java/lang/Long.numberOfTrailingZeros(J)I", "java/lang/Long.compareUnsigned(JJ)I", "java/lang/Float.floatToRawIntBits(F)I",
                "java/lang/Float.intBitsToFloat(I)F", "java/lang/Float.isNaN(F)Z", "java/lang/Double.isNaN(D)Z", "java/lang/Double.doubleToRawLongBits(D)J",
                "java/lang/Double.longBitsToDouble(J)D"));
        for (DataType type : List.of(row.dataType, row.indexType)) {
            String layout = switch (type) {
                case FLOAT64 -> "OfDouble;J"; case FLOAT32 -> "OfFloat;J"; case BFLOAT16 -> "OfShort;J";
                case INT64 -> "OfLong;J"; case INT32 -> "OfInt;J"; case BOOL -> "OfByte;J";
            };
            String result = switch (type) {
                case FLOAT64 -> "D"; case FLOAT32 -> "F"; case BFLOAT16 -> "S";
                case INT64 -> "J"; case INT32 -> "I"; case BOOL -> "B";
            };
            String prefix = "java/lang/foreign/MemorySegment.";
            allowed.add(prefix + "get(Ljava/lang/foreign/ValueLayout$" + layout + ")" + result);
            allowed.add(prefix + "set(Ljava/lang/foreign/ValueLayout$" + layout + result + ")V");
            allowed.add(prefix + "getAtIndex(Ljava/lang/foreign/ValueLayout$" + layout + ")" + result);
            allowed.add(prefix + "setAtIndex(Ljava/lang/foreign/ValueLayout$" + layout + result + ")V");
        }
        if (row.scratchRequirement.required()) allowed.addAll(Set.of(
                "java/lang/foreign/MemorySegment.get(Ljava/lang/foreign/ValueLayout$OfLong;J)J",
                "java/lang/foreign/MemorySegment.set(Ljava/lang/foreign/ValueLayout$OfLong;JJ)V"));
        // The private BF16 conversion pair is selected only by a BF16 data role, never by suffix.
        if (row.dataType == DataType.BFLOAT16) allowed.addAll(Set.of(
                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.bf16(F)S",
                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.bf16f(S)F",
                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated/AffineMovementIndexingScatterRandomCleanJava.bf16(F)S",
                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated/AffineMovementIndexingScatterRandomCleanJava.bf16f(S)F"));
        return allowed.contains(target);
    }

    /**
     * Pairs every random owner with a separately compiled typed body.  INITIAL_STATE's two
     * ordered stores are intentionally not treated as a degenerate dropout; DROPOUT first runs
     * its [0,0) next-state prologue and then independently exercises whole, empty, interior, and
     * tail draw ranges through the counter-plus-global-ordinal mapping.
     */
    @Test void everyRandomRowExecutesAndMatchesItsSeparatedTypedCleanJavaProjection() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.randomOrOneHotCandidates().stream()
                .filter(candidate -> RANDOM.contains(candidate.operationForm())).toList();
        assertEquals(20, candidates.size(), "exact random owner denominator");
        var cleanRows = new ArrayList<CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row>();
        var prepared = new ArrayList<PreparedRandom>();
        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var node = candidate.context().nodes().getFirst();
            String identity;
            if (candidate.operationForm().equals("INITIAL_STATE")) {
                var attrs = (GraphRngStateAttrs) node.operation().attrs();
                identity = "INITIAL_STATE:" + attrs.key() + ':' + attrs.counter();
            } else {
                identity = candidate.context().values().getFirst().descriptor().dataType().name() + ':'
                        + Double.doubleToRawLongBits(((DropoutAttrs) node.operation().attrs()).probability());
            }
            cleanRows.add(new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row(candidate.ownerId(),
                    candidate.operationForm().equals("INITIAL_STATE") ? "initial-state" : "dropout",
                    route.specialization().entryType().descriptorString(), identity, false));
            prepared.add(new PreparedRandom(candidate, plan.elementCount(),
                    plan.randomGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()]),
                    route.specialization(), route.kernelIr()));
        }
        var clean = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(cleanRows);
        Map<String, MethodModel> cleanMethods = methods(clean.bytes());
        int dropout = 0, initializer = 0;
        for (PreparedRandom row : prepared) {
            var generated = new CpuClassFileKernelGenerator();
            byte[] generatedBytes = generated.generateClassBytes(row.specialization, row.kernelIr);
            assertTrue(java.util.Arrays.equals(generatedBytes,
                    generated.generateClassBytes(row.specialization, row.kernelIr)),
                    row.candidate.ownerId() + " deterministic generated Class-File bytes");
            MethodModel generatedMethod = selected(methods(generatedBytes));
            MethodModel counterpart = cleanMethods.get(clean.methods().get(row.candidate.ownerId()));
            assertEquals(row.specialization.entryType().descriptorString(), counterpart.methodType().stringValue(),
                    row.candidate.ownerId() + " exact typed random ABI");
            assertEquals(row.specialization.entryType().descriptorString(), generatedMethod.methodType().stringValue(),
                    row.candidate.ownerId() + " generated ordered carrier-role ABI");
            assertRandomProjection(row, row.candidate.operationForm().equals("DROPOUT")
                    ? dropoutFacts(counterpart) : initialStateFacts(counterpart),
                    row.candidate.operationForm().equals("DROPOUT")
                            ? dropoutFacts(generatedMethod) : initialStateFacts(generatedMethod));
            assertRandomHygiene(row.candidate.ownerId() + " clean", counterpart);
            assertRandomHygiene(row.candidate.ownerId() + " generated", generatedMethod);
            for (long[] range : row.candidate.operationForm().equals("DROPOUT")
                    ? ranges(row.count) : List.of(new long[] {0, 0}, new long[] {1, 1})) {
                List<Object> actual = randomStorage(row), expected = randomStorage(row);
                List<Object> actualInputs = row.candidate.operationForm().equals("DROPOUT")
                        ? List.of(cloneArray(actual.get(0)), cloneArray(actual.get(1))) : List.of();
                var generatedEntry = generated.defineClassBytes(row.specialization,
                        generated.generateClassBytes(row.specialization, row.kernelIr)).entryPoint();
                var cleanEntry = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(clean,
                        row.candidate.ownerId(), row.specialization.entryType().descriptorString());
                invokeRandom(generatedEntry, row, actual, 0, 0);
                invokeRandom(cleanEntry, row, expected, 0, 0);
                invokeRandom(generatedEntry, row, actual, range[0], range[1]);
                invokeRandom(cleanEntry, row, expected, range[0], range[1]);
                for (int carrier = 0; carrier < actual.size(); carrier++) assertEveryRawElementEquals(
                        actual.get(carrier), expected.get(carrier), randomType(row, carrier),
                        row.candidate.ownerId() + " paired random carrier=" + carrier + " range=" + range[0] + ':' + range[1]);
                if (row.candidate.operationForm().equals("DROPOUT")) {
                    assertEveryRawElementEquals(actualInputs.get(0), actual.get(0), randomType(row, 0),
                            row.candidate.ownerId() + " generated dropout value input immutable");
                    assertEveryRawElementEquals(actualInputs.get(1), actual.get(1), DataType.INT64,
                            row.candidate.ownerId() + " generated dropout state input immutable");
                }
            }
            if (row.candidate.operationForm().equals("DROPOUT")) dropout++; else initializer++;
        }
        assertEquals(16, dropout, "DROPOUT paired rows");
        assertEquals(4, initializer, "INITIAL_STATE paired rows");
    }

    /** Every random semantic and hygiene gate is exercised against a real independently compiled source mutation. */
    @Test void randomProjectionRejectsExecutableSourceMutants() throws Throwable {
        assertRandomSourceMutants("DROPOUT");
        assertRandomSourceMutants("INITIAL_STATE");
    }

    private static void assertRandomSourceMutants(String form) throws Throwable {
        var candidate = CpuOrdinaryNonPointwiseGeneratedMatrixTest.randomOrOneHotCandidates().stream()
                .filter(value -> value.operationForm().equals(form)).findFirst().orElseThrow();
        var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
        var route = plan.units().getFirst().portablePlan();
        String identity;
        String family;
        if (form.equals("DROPOUT")) {
            var attrs = (DropoutAttrs) candidate.context().nodes().getFirst().operation().attrs();
            identity = candidate.context().values().getFirst().descriptor().dataType().name() + ':'
                    + Double.doubleToRawLongBits(attrs.probability());
            family = "dropout";
        } else {
            var attrs = (GraphRngStateAttrs) candidate.context().nodes().getFirst().operation().attrs();
            identity = "INITIAL_STATE:" + attrs.key() + ':' + attrs.counter();
            family = "initial-state";
        }
        var basis = new CpuAffineMovementIndexingScatterRandomCleanJavaOracle.Row("baseline", family,
                route.specialization().entryType().descriptorString(), identity, false);
        var baseline = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compile(List.of(basis));
        var mutants = form.equals("DROPOUT")
                ? CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileRandomMutants(basis)
                : CpuAffineMovementIndexingScatterRandomCleanJavaOracle.compileInitialStateMutants(basis);
        var row = new PreparedRandom(candidate, plan.elementCount(), plan.randomGeometry().orElseThrow()
                .pack(new long[plan.boundaryValues().size()]), route.specialization(), route.kernelIr());
        for (var entry : mutants.methods().entrySet()) {
            MethodModel mutant = methods(mutants.bytes()).get(entry.getValue());
            assertTrue(mutant != null, "compiled random mutant " + entry.getKey());
            if (entry.getKey().startsWith("wrong-role-")) {
                assertThrows(AssertionError.class, () -> assertRandomFacts(dropoutFacts(mutant), true),
                        entry.getKey() + " must fail parameter-origin role/sink dataflow before hygiene");
            }
            boolean rejected;
            try { assertRandomMethod(entry.getKey(), mutant, form.equals("DROPOUT") ? 5 : 1,
                    form.equals("DROPOUT")); rejected = false; }
            catch (AssertionError expected) { rejected = true; }
            if (!rejected) {
                List<Object> expected = randomStorage(row), actual = randomStorage(row);
                var correct = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(baseline, "baseline",
                        route.specialization().entryType().descriptorString());
                var changed = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.entry(mutants, entry.getKey(),
                        route.specialization().entryType().descriptorString());
                invokeRandom(correct, row, expected, 0, 0); invokeRandom(changed, row, actual, 0, 0);
                invokeRandom(correct, row, expected, 0, row.count); invokeRandom(changed, row, actual, 0, row.count);
                boolean differs = false;
                for (int role = 0; role < actual.size(); role++) {
                    try { assertEveryRawElementEquals(expected.get(role), actual.get(role), randomType(row, role), "mutant"); }
                    catch (AssertionError difference) { differs = true; break; }
                }
                assertTrue(differs, form + ' ' + entry.getKey()
                        + " must alter executable random semantics or fail hygiene");
            }
        }
    }

    /** Rejects changes in ABI, carrier direction, range direction, and family topology facts. */
    @Test void projectionsRejectDeliberateAbiCarrierRangeAndTopologyDrift() {
        Row baseline = new Row("control", "movement", "CONCAT", "INT32", "([I[Ljava/lang/foreign/MemorySegment;[I[JJJ)V", "[INT_ARRAY, MEMORY_SEGMENT, INT_ARRAY]", "[DENSE_LINEAR]", "scalar");
        assertFamilyFacts(baseline);
        for (Row changed : List.of(
                new Row("control", "movement", "CONCAT", "INT32", "([I[I[JJJ)V", "[INT_ARRAY, INT_ARRAY]", "[DENSE_LINEAR]", "scalar"),
                new Row("control", "scatter", "CONCAT", "INT32", baseline.descriptor, baseline.carriers, baseline.access, baseline.strategy),
                new Row("control", "movement", "FOLD_AXIS", "INT32", baseline.descriptor, baseline.carriers, baseline.access, baseline.strategy))) {
            assertThrows(AssertionError.class, () -> assertCompatible(baseline, changed));
        }
    }

    private static void assertCompatible(Row expected, Row actual) {
        if (!expected.family.equals(actual.family) || !expected.form.equals(actual.form) || !expected.dataType.equals(actual.dataType)
                || !expected.descriptor.equals(actual.descriptor) || !expected.carriers.equals(actual.carriers)
                || !expected.access.equals(actual.access) || !expected.strategy.equals(actual.strategy)) throw new AssertionError("changed structural projection");
    }
    private static List<Object> randomStorage(PreparedRandom row) {
        int roles = row.candidate.operationForm().equals("INITIAL_STATE") ? 1 : 5;
        var storage = new ArrayList<Object>();
        for (int role = 0; role < roles; role++) storage.add(patterned(randomType(row, role), 32));
        if (roles == 5) {
            long state = row.geometry[11], stride = row.geometry[13];
            ((long[]) storage.get(1))[(int) state] = 0x1234L;
            ((long[]) storage.get(1))[(int) (state + stride)] = 7L;
        }
        return storage;
    }
    private static DataType randomType(PreparedRandom row, int role) {
        if (row.candidate.operationForm().equals("INITIAL_STATE")) return DataType.INT64;
        return switch (role) { case 0, 2 -> row.candidate.context().values().getFirst().descriptor().dataType();
            case 1, 4 -> DataType.INT64; case 3 -> DataType.BOOL; default -> throw new AssertionError("random role"); };
    }
    private static void invokeRandom(java.lang.invoke.MethodHandle entry, PreparedRandom row,
            List<Object> storage, long start, long end) throws Throwable {
        var args = new ArrayList<Object>();
        for (int role = 0; role < storage.size(); role++) args.add(argument(
                row.specialization.carrierPattern().get(role), storage.get(role)));
        args.add(row.geometry); args.add(start); args.add(end); entry.invokeWithArguments(args);
    }
    /**
     * Extracts INITIAL_STATE's narrow parameter-origin proof.  This is intentionally a small
     * verifier, not a general bytecode analyser: it follows only javac/generated local copies
     * of geometry values into a direct typed carrier access.
     */
    private static RandomDataflowFacts initialStateFacts(MethodModel method) {
        return randomDataflow(method, 1);
    }

    /** Extracts DROPOUT's five ordered carrier roles through direct typed access sinks. */
    private static RandomDataflowFacts dropoutFacts(MethodModel method) {
        return randomDataflow(method, 5);
    }

    private static RandomDataflowFacts randomDataflow(MethodModel method, int roles) {
        List<String> parameters = CpuAffineMovementIndexingScatterRandomCleanJavaOracle.parameters(
                method.methodType().stringValue());
        int geometry = roles;
        assertEquals(roles + 3, parameters.size(), "exact random carrier/geometry/range ABI");
        assertEquals("long[]", parameters.get(geometry), "random geometry parameter type");
        List<Instruction> instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int geometrySlot = parameterSlot(parameters, geometry);
        Set<Integer> geometryLocals = parameterDerivedLocals(instructions, geometrySlot);
        Set<Integer> startLocals = parameterDerivedLocals(instructions, parameterSlot(parameters, geometry + 1));
        Set<Integer> endLocals = parameterDerivedLocals(instructions, parameterSlot(parameters, geometry + 2));
        List<RandomAccess> accesses = directRandomAccesses(instructions, roles, geometryLocals);
        boolean guard = hasPairedBranch(instructions, startLocals, endLocals);
        boolean geometryRead = !geometryLocals.isEmpty();
        if (roles == 1) {
            boolean stateWrites = hasAccess(accesses, 0, true, true);
            return new RandomDataflowFacts(roles, guard, geometryRead, stateWrites, false, false,
                    false, false, false, false, false);
        }
        boolean inputRead = hasAccess(accesses, 0, false, true);
        boolean stateReads = accesses.stream().filter(a -> a.role == 1 && !a.write && a.geometryAddress).count() >= 2;
        boolean resultWrite = hasAccess(accesses, 2, true, true);
        boolean maskWrite = hasAccess(accesses, 3, true, true);
        boolean nextStateWrites = hasAccess(accesses, 4, true, true);
        boolean typedRoles = accesses.stream().allMatch(a -> a.role >= 0 && a.role < roles)
                && accesses.stream().anyMatch(a -> a.role == 0) && accesses.stream().anyMatch(a -> a.role == 1)
                && accesses.stream().anyMatch(a -> a.role == 2) && accesses.stream().anyMatch(a -> a.role == 3)
                && accesses.stream().anyMatch(a -> a.role == 4);
        return new RandomDataflowFacts(roles, guard, geometryRead, nextStateWrites, inputRead,
                stateReads, resultWrite, maskWrite, typedRoles, hasLoopRangeBranch(instructions, startLocals, endLocals),
                instructions.stream().anyMatch(i -> i.opcode() == Opcode.DDIV));
    }

    private static boolean hasAccess(List<RandomAccess> accesses, int role, boolean write, boolean geometryAddress) {
        return accesses.stream().anyMatch(a -> a.role == role && a.write == write && a.geometryAddress == geometryAddress);
    }

    private static int parameterSlot(List<String> parameters, int parameter) {
        int slot = 0;
        for (int index = 0; index < parameter; index++)
            slot += parameters.get(index).equals("long") || parameters.get(index).equals("double") ? 2 : 1;
        return slot;
    }

    private static Set<Integer> parameterDerivedLocals(List<Instruction> instructions, int parameter) {
        var derived = new java.util.HashSet<Integer>();
        derived.add(parameter);
        for (int index = 0; index < instructions.size(); index++) {
            if (!(instructions.get(index) instanceof StoreInstruction store)) continue;
            boolean source = false;
            for (int prior = Math.max(0, index - 12); prior < index; prior++) {
                if (instructions.get(prior) instanceof LoadInstruction load
                        && (load.slot() == parameter || derived.contains(load.slot()))) {
                    source = true;
                }
            }
            if (source) derived.add(store.slot());
        }
        return derived;
    }

    private static List<RandomAccess> directRandomAccesses(List<Instruction> instructions, int roles,
            Set<Integer> geometryLocals) {
        var result = new ArrayList<RandomAccess>();
        for (int start = 0; start < instructions.size(); start++) {
            if (!(instructions.get(start) instanceof LoadInstruction load)
                    || load.slot() < 0 || load.slot() >= roles) continue;
            for (int end = start + 1; end < Math.min(instructions.size(), start + 100); end++) {
                Instruction candidate = instructions.get(end);
                if (candidate instanceof LoadInstruction next && next.slot() >= 0 && next.slot() < roles) break;
                Boolean write = randomAccessDirection(candidate);
                if (write != null) {
                    boolean geometryAddress = false;
                    for (int i = start + 1; i < end; i++) if (instructions.get(i) instanceof LoadInstruction local
                            && geometryLocals.contains(local.slot())) geometryAddress = true;
                    result.add(new RandomAccess(load.slot(), write, geometryAddress));
                    break;
                }
            }
        }
        return result;
    }

    private static Boolean randomAccessDirection(Instruction instruction) {
        if (instruction instanceof ArrayLoadInstruction) return false;
        if (instruction instanceof ArrayStoreInstruction) return true;
        if (instruction instanceof InvokeInstruction call
                && call.owner().asInternalName().equals("java/lang/foreign/MemorySegment")) {
            if (call.name().stringValue().startsWith("get")) return false;
            if (call.name().stringValue().startsWith("set")) return true;
        }
        return null;
    }

    private static boolean hasPairedBranch(List<Instruction> instructions, Set<Integer> first, Set<Integer> second) {
        return instructions.stream().filter(BranchInstruction.class::isInstance).map(BranchInstruction.class::cast)
                .anyMatch(branch -> branchUses(instructions, branch, first))
                && instructions.stream().filter(BranchInstruction.class::isInstance).map(BranchInstruction.class::cast)
                .anyMatch(branch -> branchUses(instructions, branch, second));
    }

    private static boolean hasLoopRangeBranch(List<Instruction> instructions, Set<Integer> start, Set<Integer> end) {
        return hasPairedBranch(instructions, start, end);
    }

    private static boolean branchUses(List<Instruction> instructions, BranchInstruction branch, Set<Integer> locals) {
        int index = instructions.indexOf(branch);
        for (int prior = Math.max(0, index - 12); prior < index; prior++) if (instructions.get(prior) instanceof LoadInstruction load
                && locals.contains(load.slot())) return true;
        return false;
    }

    private static void assertRandomProjection(PreparedRandom row, RandomDataflowFacts clean, RandomDataflowFacts generated) {
        boolean dropout = row.candidate.operationForm().equals("DROPOUT");
        assertRandomFacts(clean, dropout); assertRandomFacts(generated, dropout);
        assertEquals(clean.roles, generated.roles, row.candidate.ownerId() + " ordered random ABI roles");
    }
    private static void assertRandomFacts(RandomDataflowFacts facts, boolean dropout) {
        assertTrue(facts.roles == (dropout ? 5 : 1), "random carrier role topology");
        assertTrue(facts.prologue && facts.geometry && facts.stateWrites, "random state/range/geometry sink topology: " + facts);
        if (dropout) assertTrue(facts.inputRead && facts.stateReads && facts.resultWrite && facts.maskWrite
                && facts.typedRoles && facts.range && facts.narrowOnceScale,
                "dropout parameter-origin role/sink topology: " + facts);
    }
    private static void assertRandomMethod(String owner, MethodModel method, int roles, boolean dropout) {
        assertRandomFacts(dropout ? dropoutFacts(method) : initialStateFacts(method), dropout);
        assertRandomHygiene(owner, method);
    }
    private static void assertRandomHygiene(String owner, MethodModel method) {
        Set<String> forbidden = Set.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "ATHROW", "INVOKEDYNAMIC");
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            assertFalse(forbidden.contains(instruction.opcode().name()), owner + " forbidden random opcode " + instruction.opcode().name());
            if (instruction instanceof InvokeInstruction call) {
                String target = call.owner().asInternalName() + '.' + call.name() + call.type();
                assertTrue(permittedRandomInvoke(target), owner + " forbidden random invoke " + target);
            }
        }
    }
    private static boolean permittedRandomInvoke(String target) {
        return Set.of(
                "java/lang/Double.longBitsToDouble(J)D",
                "java/lang/foreign/MemorySegment.get(Ljava/lang/foreign/ValueLayout$OfLong;J)J",
                "java/lang/foreign/MemorySegment.get(Ljava/lang/foreign/ValueLayout$OfDouble;J)D",
                "java/lang/foreign/MemorySegment.get(Ljava/lang/foreign/ValueLayout$OfFloat;J)F",
                "java/lang/foreign/MemorySegment.get(Ljava/lang/foreign/ValueLayout$OfByte;J)B",
                "java/lang/foreign/MemorySegment.set(Ljava/lang/foreign/ValueLayout$OfLong;JJ)V",
                "java/lang/foreign/MemorySegment.set(Ljava/lang/foreign/ValueLayout$OfDouble;JD)V",
                "java/lang/foreign/MemorySegment.set(Ljava/lang/foreign/ValueLayout$OfFloat;JF)V",
                "java/lang/foreign/MemorySegment.set(Ljava/lang/foreign/ValueLayout$OfByte;JB)V",
                "java/lang/foreign/MemorySegment.getAtIndex(Ljava/lang/foreign/ValueLayout$OfLong;J)J",
                "java/lang/foreign/MemorySegment.getAtIndex(Ljava/lang/foreign/ValueLayout$OfDouble;J)D",
                "java/lang/foreign/MemorySegment.getAtIndex(Ljava/lang/foreign/ValueLayout$OfFloat;J)F",
                "java/lang/foreign/MemorySegment.getAtIndex(Ljava/lang/foreign/ValueLayout$OfByte;J)B",
                "java/lang/foreign/MemorySegment.setAtIndex(Ljava/lang/foreign/ValueLayout$OfLong;JJ)V",
                "java/lang/foreign/MemorySegment.setAtIndex(Ljava/lang/foreign/ValueLayout$OfDouble;JD)V",
                "java/lang/foreign/MemorySegment.setAtIndex(Ljava/lang/foreign/ValueLayout$OfFloat;JF)V",
                "java/lang/foreign/MemorySegment.setAtIndex(Ljava/lang/foreign/ValueLayout$OfByte;JB)V")
                .contains(target);
    }
    private static String entryDataType(String descriptor, String fallback) {
        int position = descriptor.charAt(1) == '[' ? 2 : 1;
        return switch (descriptor.charAt(position)) {
            case 'D' -> "FLOAT64"; case 'F' -> "FLOAT32"; case 'S' -> "BFLOAT16";
            case 'J' -> "INT64"; case 'I' -> "INT32"; case 'B' -> "BOOL";
            default -> fallback;
        };
    }
    private static void assertFamilyFacts(Row row) {
        switch (row.family) {
            case "affine" -> assertTrue(AFFINE.contains(row.form) && row.descriptor.contains("[J"), row.owner + " affine map/address facts");
            case "movement" -> assertTrue(MOVEMENT.contains(row.form) && !row.form.startsWith("FOLD"), row.owner + " movement topology/no-fold facts");
            case "indexing" -> assertTrue(INDEXING.contains(row.form) && row.descriptor.contains("[J"), row.owner + " typed index writer facts");
            case "scatter" -> assertTrue(SCATTER.contains(row.form) && row.descriptor.contains("[J"), row.owner + " base/index/update/output state facts");
            case "random" -> assertTrue(RANDOM.contains(row.form) && row.descriptor.contains("[J"), row.owner + " random state/range facts");
            default -> throw new AssertionError("unknown 0009C family: " + row.family);
        }
    }
    private static void assertGeneratedInstructionMemberHygiene(String owner, MethodModel method) {
        List<String> forbiddenOpcodes = List.of("NEW", "ANEWARRAY", "NEWARRAY", "MULTIANEWARRAY", "INVOKEDYNAMIC");
        List<String> forbiddenOwners = List.of("java/lang/reflect/", "java/lang/invoke/MethodHandle",
                "java/util/Map", "CleanJavaOracle", "StructuralOracle");
        for (CodeElement element : method.code().orElseThrow().elementStream().toList()) {
            if (!(element instanceof Instruction instruction)) continue;
            assertFalse(forbiddenOpcodes.contains(instruction.opcode().name()),
                    owner + " forbidden selected-entry instruction: " + instruction.opcode().name());
            if (instruction instanceof InvokeInstruction invocation) {
                String target = invocation.owner().asInternalName() + '.' + invocation.name() + invocation.type();
                assertFalse(forbiddenOwners.stream().anyMatch(target::contains),
                        owner + " forbidden selected-entry member: " + target);
            }
        }
    }
    private static boolean hasBranch(MethodModel method) { return method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).anyMatch(i -> i.opcode().name().startsWith("IF") || i.opcode().name().startsWith("GOTO")); }
    private static MethodModel selected(Map<String, MethodModel> methods) { return methods.values().stream().filter(method -> method.methodName().stringValue().equals("apply")).findFirst().orElseGet(() -> methods.values().stream().filter(method -> !method.methodName().stringValue().equals("<init>")).findFirst().orElseThrow()); }
    private static Map<String, MethodModel> methods(byte[] bytes) { Map<String, MethodModel> result = new LinkedHashMap<>(); for (MethodModel method : ClassFile.of().parse(bytes).methods()) result.put(method.methodName().stringValue(), method); return result; }
    private static Map<String, Long> counts(List<Row> rows) { Map<String, Long> result = new TreeMap<>(); rows.forEach(row -> result.merge(row.family, 1L, Long::sum)); return result; }
    private static Map<String, Long> formCounts(List<Row> rows, String family) { Map<String, Long> result = new TreeMap<>(); rows.stream().filter(row -> row.family.equals(family)).forEach(row -> result.merge(row.form, 1L, Long::sum)); return result; }
    private static List<Row> ownedRows() throws Exception {
        byte[] bytes; try (InputStream stream = CpuAffineMovementIndexingScatterRandomStructuralOracleTest.class.getResourceAsStream(BASE + "generated-coverage-inventory.tsv")) { assertTrue(stream != null, "checked inventory"); bytes = stream.readAllBytes(); }
        assertEquals(INVENTORY_SHA256, java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)), "checked inventory digest");
        Map<String, Row> unique = new LinkedHashMap<>();
        String text = new String(bytes, StandardCharsets.UTF_8); assertTrue(text.endsWith("\n") && !text.contains("\r"), "canonical LF inventory");
        for (String line : text.split("\n")) { String[] row = line.split("\t", -1); if (row.length != 27 || !row[21].equals("GENERATED")) continue; String family = family(row[0], row[2]); if (family != null) assertTrue(unique.put(row[0], new Row(row[0], family, row[2], family.equals("affine") || family.equals("indexing") ? dataType(row[4]) : "INT32", row[18], row[9], row[10], row[12])) == null, "duplicate owner: " + row[0]); }
        return List.copyOf(unique.values());
    }
    private static String family(String owner, String form) { if (owner.startsWith("affine-matrix:") && AFFINE.contains(form)) return "affine"; if (owner.startsWith("ordinary:") && MOVEMENT.contains(form)) return "movement"; if (owner.startsWith("ordinary:") && INDEXING.contains(form)) return "indexing"; if (owner.startsWith("ordinary:") && SCATTER.contains(form)) return "scatter"; if (owner.startsWith("ordinary:") && RANDOM.contains(form)) return "random"; return null; }
    private static String dataType(String lowered) { for (DataType type : DataType.values()) if (lowered.contains(":" + type.name())) return type.name(); throw new AssertionError("missing affine data type: " + lowered); }
    private record PreparedScatter(String owner, String form, DataType dataType, DataType indexType,
            String reduction, ScratchRequirement scratchRequirement, CpuKernelSpecialization specialization,
            CpuKernelIr kernelIr) { }
    private record ScratchRequirement(boolean required, long bytes, long offset, String descriptor) { }
    private record ScratchDataflow(boolean sizeGuard, boolean reset, boolean header,
            boolean limbRead, boolean limbWrite, boolean publication) { }
    private record ScatterCarrier(Object array, Object carrier, DataType type) { }
    private record ScatterFacts(String descriptor, int roles, int loads, int stores, int longLoads, int branches,
            boolean range, boolean orderedLoads, boolean baseCopy, boolean indexLoad, boolean updateLoad,
            boolean outputStore, List<String> invokes, List<String> opcodes) { }
    private record PreparedRandom(CpuOrdinaryNonPointwiseGeneratedMatrixTest.RandomOrOneHotCandidate candidate,
            long count, long[] geometry, CpuKernelSpecialization specialization, CpuKernelIr kernelIr) { }
    private record PreparedIndexing(String owner, String form,
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> context, long count, io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLowering.Geometry geometry,
            List<DataType> boundaryTypes, DataType dataType, DataType indexType,
            CpuKernelSpecialization specialization, CpuKernelIr kernelIr) { }
    private record IndexingFacts(String descriptor, int geometryLoads, int primitiveLoads, int primitiveStores,
            boolean indexWidth, boolean dataAccess, boolean familyMap, boolean boolStore,
            boolean forward, boolean backward, List<String> invokes) { }
    private record RandomAccess(int role, boolean write, boolean geometryAddress) { }
    private record RandomDataflowFacts(int roles, boolean prologue, boolean geometry,
            boolean stateWrites, boolean inputRead, boolean stateReads, boolean resultWrite,
            boolean maskWrite, boolean typedRoles, boolean range, boolean narrowOnceScale) { }
    private record Row(String owner, String family, String form, String dataType, String descriptor, String carriers, String access, String strategy) { }
}
