package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/** Semantic and retained Class-File evidence for CPU 0008L dense floating-mask boundaries. */
class CpuPointwiseMaskEvidenceTest {
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final List<CpuPointwiseOpcode> PREDICATES = List.of(
            CpuPointwiseOpcode.IS_FINITE, CpuPointwiseOpcode.IS_NAN, CpuPointwiseOpcode.IS_INF,
            CpuPointwiseOpcode.GREATER_THAN, CpuPointwiseOpcode.GREATER_OR_EQUAL,
            CpuPointwiseOpcode.LESS_THAN, CpuPointwiseOpcode.LESS_OR_EQUAL,
            CpuPointwiseOpcode.EQUAL, CpuPointwiseOpcode.NOT_EQUAL);

    @Test void exactInventoryAndFortyFourExecutionCases() throws Throwable {
        assertEquals(48, CpuPointwiseOpcode.values().length);
        assertEquals(13, Arrays.stream(CpuPointwiseOpcode.values())
                .filter(opcode -> opcode.vectorForm() == CpuPointwiseOpcode.VectorForm.MASK_PRODUCER
                        || opcode.vectorForm() == CpuPointwiseOpcode.VectorForm.VALUE_OR_MASK
                        || opcode.vectorForm() == CpuPointwiseOpcode.VectorForm.MASK_CONSUMER)
                .count());
        int cases = 0;
        for (DataType type : floatingTypes()) {
            for (CpuPointwiseOpcode opcode : PREDICATES) {
                execute(directPredicate(type, opcode), type, Scenario.ARRAY_EXACT);
                cases++;
            }
            execute(logical(type, CpuPointwiseOpcode.LOGICAL_NOT), type, Scenario.ARRAY_EXACT);
            execute(logical(type, CpuPointwiseOpcode.LOGICAL_AND), type, Scenario.ARRAY_EXACT);
            execute(logical(type, CpuPointwiseOpcode.LOGICAL_OR), type, Scenario.ARRAY_EXACT);
            execute(externalWhere(type), type, Scenario.ARRAY_EXACT);
            execute(fanout(type), type, Scenario.ARRAY_EXACT);
            cases += 5;
            for (Scenario scenario : Scenario.values()) {
                execute(directPredicate(type, CpuPointwiseOpcode.IS_NAN), type, scenario);
                execute(externalWhere(type), type, scenario);
                cases += 2;
            }
        }
        assertEquals(44, cases);
    }

    @Test void exactSeventyTwoDeterministicStructuralDossiers() throws Exception {
        List<Form> baselines = new ArrayList<>();
        for (DataType type : floatingTypes()) {
            for (CpuPointwiseOpcode opcode : PREDICATES)
                baselines.add(new Form(type + "-" + opcode, directPredicate(type, opcode), type));
            baselines.add(new Form(type + "-NOT", logical(type, CpuPointwiseOpcode.LOGICAL_NOT), type));
            baselines.add(new Form(type + "-AND", logical(type, CpuPointwiseOpcode.LOGICAL_AND), type));
            baselines.add(new Form(type + "-OR", logical(type, CpuPointwiseOpcode.LOGICAL_OR), type));
            baselines.add(new Form(type + "-WHERE", externalWhere(type), type));
            baselines.add(new Form(type + "-FANOUT", fanout(type), type));
        }
        assertEquals(28, baselines.size());
        List<Dossier> dossiers = new ArrayList<>();
        for (Form form : baselines) dossiers.add(dossier(form.name(), form.ir(), form.type(),
                arrayCarriers(form.ir())));
        for (DataType type : floatingTypes()) for (Form form : List.of(
                new Form("CLASS", directPredicate(type, CpuPointwiseOpcode.IS_FINITE), type),
                new Form("COMPARE", directPredicate(type, CpuPointwiseOpcode.GREATER_THAN), type),
                new Form("BINARY", logical(type, CpuPointwiseOpcode.LOGICAL_AND), type),
                new Form("WHERE", externalWhere(type), type), new Form("FANOUT", fanout(type), type))) {
            int boundaries = boundaryTypes(form.ir()).size();
            for (int boundary = 0; boundary < boundaries; boundary++) {
                var carriers = new ArrayList<>(arrayCarriers(form.ir()));
                carriers.set(boundary, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                dossiers.add(dossier(type + "-" + form.name() + "-S" + boundary,
                        form.ir(), type, carriers));
            }
            dossiers.add(dossier(type + "-" + form.name() + "-ALL-S", form.ir(), type,
                    java.util.Collections.nCopies(boundaries,
                            CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)));
        }
        assertEquals(72, dossiers.size());
        assertEquals(72, dossiers.stream().map(Dossier::name).distinct().count());
        retain(dossiers);
    }

    @Test void priorSchemaProjectionsRemainExplicitlySeparated() {
        assertEquals(61, CpuGeneratorSchema.CURRENT_VERSION);
        CpuKernelIr virtual = virtualWhere(DataType.FLOAT32);
        assertEquals(52, specialization(virtual, DataType.FLOAT32, arrayCarriers(virtual), 52)
                .classIdentitySchema());
        CpuKernelIr changed = directPredicate(DataType.FLOAT32, CpuPointwiseOpcode.IS_FINITE);
        assertEquals(61, specialization(changed, DataType.FLOAT32, arrayCarriers(changed), 61)
                .classIdentitySchema());
    }

    private static void execute(CpuKernelIr ir, DataType type, Scenario scenario) throws Throwable {
        int lanes = lanes(type), count = scenario.tail ? lanes + 3 : lanes * 2;
        int offset = scenario.offset ? 3 : 0, capacity = offset + count + lanes;
        List<DataType> types = boundaryTypes(ir);
        try (Arena arena = Arena.ofConfined()) {
            List<Object> arguments = new ArrayList<>();
            List<CpuKernelSpecialization.CarrierAccess> carriers = new ArrayList<>();
            int boundary = 0;
            for (CpuKernelIr.Value value : ir.values()) {
                if (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
                boolean segment = scenario.segment(value.dataType());
                Object carrier = carrier(value.dataType(), capacity, segment, arena);
                arguments.add(carrier);
                carriers.add(segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                        : arrayCarrier(value.dataType()));
                if (value.kind() == CpuKernelIr.Value.Kind.INPUT)
                    for (int lane = 0; lane < count; lane++) write(carrier, value.dataType(),
                            offset + lane, input(value.dataType(), boundary, lane));
                else fill(carrier, value.dataType(), capacity, 77d);
                boundary++;
            }
            MethodHandle entry = artifact(ir, type, carriers).entryPoint();
            if (scenario.parallel) {
                int split = lanes;
                invoke(entry, arguments, geometry(types.size(), count, offset), 0, split);
                invoke(entry, arguments, geometry(types.size(), count, offset + split), split, count);
            } else invoke(entry, arguments, geometry(types.size(), count, offset), 0, count);
            verify(ir, arguments, offset, count);
            for (int i = 0; i < offset; i++) for (int b = 0; b < types.size(); b++)
                if (boundaryValue(ir, b).kind() == CpuKernelIr.Value.Kind.OUTPUT)
                    assertEquals(77d, read(arguments.get(b), types.get(b), i), 0d, "prefix sentinel");
        }
    }

    private static void invoke(MethodHandle entry, List<Object> arguments, long[] geometry,
            long start, long end) throws Throwable {
        var call = new ArrayList<>(arguments);
        call.add(geometry); call.add(start); call.add(end);
        entry.invokeWithArguments(call);
    }

    private static void verify(CpuKernelIr ir, List<Object> arguments, int offset, int count) {
        for (int lane = 0; lane < count; lane++) {
            Object[] values = new Object[ir.values().size()];
            int boundary = 0;
            for (CpuKernelIr.Value value : ir.values()) {
                if (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
                if (value.kind() == CpuKernelIr.Value.Kind.INPUT)
                    values[value.ordinal()] = value.dataType() == DataType.BOOL
                            ? read(arguments.get(boundary), value.dataType(), offset + lane) == 1d
                            : read(arguments.get(boundary), value.dataType(), offset + lane);
                boundary++;
            }
            for (CpuKernelIr.Instruction instruction : ir.instructions())
                values[instruction.output()] = evaluate(instruction, values);
            for (CpuKernelIr.Store store : ir.stores()) {
                CpuKernelIr.Value value = ir.values().get(store.value());
                int outputBoundary = boundaryIndex(ir, value.ordinal());
                double expected = value.dataType() == DataType.BOOL
                        ? (Boolean.TRUE.equals(values[value.ordinal()]) ? 1d : 0d)
                        : ((Number) values[value.ordinal()]).doubleValue();
                assertEquals(expected, read(arguments.get(outputBoundary), value.dataType(),
                        offset + lane), 0d, "lane " + lane + " store " + store.value());
            }
        }
    }

    private static Object evaluate(CpuKernelIr.Instruction instruction, Object[] values) {
        CpuPointwiseOpcode op = instruction.opcode();
        double a = values[instruction.inputs().getFirst()] instanceof Number number
                ? number.doubleValue() : 0d;
        double b = instruction.inputs().size() > 1
                && values[instruction.inputs().get(1)] instanceof Number number
                ? number.doubleValue() : 0d;
        return switch (op) {
            case IS_FINITE -> Double.isFinite(a); case IS_NAN -> Double.isNaN(a);
            case IS_INF -> Double.isInfinite(a); case GREATER_THAN -> a > b;
            case GREATER_OR_EQUAL -> a >= b; case LESS_THAN -> a < b;
            case LESS_OR_EQUAL -> a <= b; case EQUAL -> a == b; case NOT_EQUAL -> a != b;
            case LOGICAL_NOT -> !(Boolean) values[instruction.inputs().getFirst()];
            case LOGICAL_AND -> (Boolean) values[instruction.inputs().get(0)]
                    && (Boolean) values[instruction.inputs().get(1)];
            case LOGICAL_OR -> (Boolean) values[instruction.inputs().get(0)]
                    || (Boolean) values[instruction.inputs().get(1)];
            case WHERE -> (Boolean) values[instruction.inputs().get(0)]
                    ? values[instruction.inputs().get(1)] : values[instruction.inputs().get(2)];
            default -> throw new AssertionError(op);
        };
    }

    private static Dossier dossier(String name, CpuKernelIr ir, DataType type,
            List<CpuKernelSpecialization.CarrierAccess> carriers) throws Exception {
        var generator = new CpuClassFileKernelGenerator();
        var specialization = specialization(ir, type, carriers, 61);
        byte[] first = generator.generateClassBytes(specialization, ir);
        byte[] second = generator.generateClassBytes(specialization, ir);
        assertArrayEquals(first, second, name);
        assertTrue(ClassFile.of().verify(first).isEmpty(), name);
        var model = ClassFile.of().parse(first);
        assertTrue(model.fields().isEmpty(), name); assertTrue(model.interfaces().isEmpty(), name);
        assertEquals(1, model.methods().size(), name);
        var owners = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .map(invoke -> invoke.owner().asInternalName()).collect(java.util.stream.Collectors.toSet());
        assertFalse(owners.stream().anyMatch(owner -> owner.startsWith(
                "io/github/pho001/synaptik")), name + " helper call " + owners);
        var invokes = model.methods().getFirst().code().orElseThrow().elementStream()
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .toList();
        assertFalse(invokes.stream().anyMatch(invoke -> invoke.name().stringValue()
                .equals("toArray")), name);
        assertTrue(invokes.stream().anyMatch(invoke -> invoke.name().stringValue()
                .equals("toLong")), name);
        return new Dossier(name, first, sha256(first));
    }

    private static void retain(List<Dossier> dossiers) throws Exception {
        String configured = System.getProperty("synaptik.cpu.pointwiseMask.structuralEvidenceRoot");
        if (configured == null || configured.isBlank()) return;
        Path root = Path.of(configured).toAbsolutePath();
        assertTrue(Files.isRegularFile(root.resolve("RUN-STRUCTURAL-EVIDENCE")));
        Path classes = root.resolve("cpu-pointwise-mask-structural/classes");
        Files.createDirectories(classes);
        StringBuilder manifest = new StringBuilder("name,sha256,bytes\n");
        for (Dossier dossier : dossiers) {
            Path classFile = classes.resolve(dossier.name() + ".class");
            Files.write(classFile, dossier.bytes());
            decompile(root, dossier.name(), classFile);
            manifest.append(dossier.name()).append(',').append(dossier.sha256()).append(',')
                    .append(dossier.bytes().length).append('\n');
        }
        Files.writeString(root.resolve("cpu-pointwise-mask-structural/manifest.csv"), manifest);
    }

    /**
     * Retains an independently readable JVM disassembly beside each generated Class-File. This is
     * deliberately enabled only by the retained-evidence sentinel, never by ordinary unit tests.
     *
     * @param root caller-owned evidence root
     * @param name unique dossier name
     * @param classFile generated Class-File to inspect
     * @throws Exception if the platform disassembler cannot inspect the generated class
     */
    private static void decompile(Path root, String name, Path classFile) throws Exception {
        Path output = root.resolve("cpu-pointwise-mask-structural/decompile/" + name + ".txt");
        Files.createDirectories(output.getParent());
        Path javap = Path.of(System.getProperty("java.home"), "bin", "javap");
        Process process = new ProcessBuilder(javap.toString(), "-c", "-p", classFile.toString())
                .redirectErrorStream(true).redirectOutput(output.toFile()).start();
        assertEquals(0, process.waitFor(), "javap " + name);
        assertTrue(Files.size(output) > 0L, "empty javap output " + name);
    }

    private static CpuGeneratedKernel artifact(CpuKernelIr ir, DataType type,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        var generator = new CpuClassFileKernelGenerator();
        var specialization = specialization(ir, type, carriers, 61);
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static CpuKernelSpecialization specialization(CpuKernelIr ir, DataType type,
            List<CpuKernelSpecialization.CarrierAccess> carriers, int schema) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, boundaryTypes(ir), carriers,
                bits(type), -1, List.of(), false, schema);
    }

    private static CpuKernelIr directPredicate(DataType type, CpuPointwiseOpcode opcode) {
        int arity = opcode.arity();
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < arity; i++) values.add(value(i, type, CpuKernelIr.Value.Kind.INPUT,
                CpuAccessPlan.AccessKind.READ));
        values.add(value(arity, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT,
                CpuAccessPlan.AccessKind.WRITE));
        return new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(opcode,
                java.util.stream.IntStream.range(0, arity).boxed().toList(), arity)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(arity, 0)));
    }

    private static CpuKernelIr logical(DataType type, CpuPointwiseOpcode opcode) {
        boolean unary = opcode == CpuPointwiseOpcode.LOGICAL_NOT;
        var values = new ArrayList<CpuKernelIr.Value>();
        values.add(value(0, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ));
        if (!unary) {
            values.add(value(1, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ));
            values.add(value(2, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ));
        }
        int first = values.size();
        values.add(value(first, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL,
                CpuAccessPlan.AccessKind.READ));
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        instructions.add(new CpuKernelIr.Instruction(unary ? CpuPointwiseOpcode.IS_FINITE
                : CpuPointwiseOpcode.GREATER_THAN, unary ? List.of(0) : List.of(0, 1), first));
        int second = first;
        if (!unary) {
            second = first + 1;
            values.add(value(second, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL,
                    CpuAccessPlan.AccessKind.READ));
            instructions.add(new CpuKernelIr.Instruction(opcode == CpuPointwiseOpcode.LOGICAL_AND
                    ? CpuPointwiseOpcode.IS_FINITE : CpuPointwiseOpcode.IS_NAN, List.of(2), second));
        }
        int output = values.size();
        values.add(value(output, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT,
                CpuAccessPlan.AccessKind.WRITE));
        instructions.add(new CpuKernelIr.Instruction(opcode,
                unary ? List.of(first) : List.of(first, second), output));
        return new CpuKernelIr(values, instructions, new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(output, 0)));
    }

    private static CpuKernelIr externalWhere(DataType type) {
        return new CpuKernelIr(List.of(value(0, DataType.BOOL, CpuKernelIr.Value.Kind.INPUT,
                        CpuAccessPlan.AccessKind.READ),
                value(1, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ),
                value(2, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ),
                value(3, type, CpuKernelIr.Value.Kind.OUTPUT, CpuAccessPlan.AccessKind.WRITE)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE, List.of(0, 1, 2), 3)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(3, 0)));
    }

    private static CpuKernelIr fanout(DataType type) {
        return new CpuKernelIr(List.of(value(0, type, CpuKernelIr.Value.Kind.INPUT,
                        CpuAccessPlan.AccessKind.READ),
                value(1, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ),
                value(2, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT,
                        CpuAccessPlan.AccessKind.WRITE),
                value(3, type, CpuKernelIr.Value.Kind.OUTPUT, CpuAccessPlan.AccessKind.WRITE)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.GREATER_THAN,
                                List.of(0, 1), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                                List.of(2, 0, 1), 3)),
                new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(2, 0), new CpuKernelIr.Store(3, 1)));
    }

    private static CpuKernelIr virtualWhere(DataType type) {
        return new CpuKernelIr(List.of(value(0, type, CpuKernelIr.Value.Kind.INPUT,
                        CpuAccessPlan.AccessKind.READ),
                value(1, type, CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.AccessKind.READ),
                value(2, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL,
                        CpuAccessPlan.AccessKind.READ),
                value(3, type, CpuKernelIr.Value.Kind.OUTPUT, CpuAccessPlan.AccessKind.WRITE)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.IS_FINITE, List.of(0), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                                List.of(2, 0, 1), 3)), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(3, 0)));
    }

    private static CpuKernelIr.Value value(int ordinal, DataType type,
            CpuKernelIr.Value.Kind kind, CpuAccessPlan.AccessKind access) {
        return new CpuKernelIr.Value(ordinal, type, kind, new CpuAccessPlan(access,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1));
    }

    private static List<DataType> boundaryTypes(CpuKernelIr ir) {
        return ir.values().stream().filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
    }

    private static CpuKernelIr.Value boundaryValue(CpuKernelIr ir, int boundary) {
        return ir.values().stream().filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .skip(boundary).findFirst().orElseThrow();
    }

    private static int boundaryIndex(CpuKernelIr ir, int ordinal) {
        int result = 0;
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
            if (value.ordinal() == ordinal) return result;
            result++;
        }
        throw new AssertionError(ordinal);
    }

    private static List<CpuKernelSpecialization.CarrierAccess> arrayCarriers(CpuKernelIr ir) {
        return boundaryTypes(ir).stream().map(CpuPointwiseMaskEvidenceTest::arrayCarrier).toList();
    }

    private static CpuKernelSpecialization.CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : type == DataType.FLOAT64 ? CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                : CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
    }

    private static long[] geometry(int boundaries, int count, int base) {
        long[] result = new long[2 + 4 * boundaries];
        result[0] = count;
        for (int i = 0; i < boundaries; i++) {
            result[2 + i] = base;
            result[2 + boundaries + i] = 1;
            result[2 + 3 * boundaries + i] = base + count;
        }
        return result;
    }

    private static Object carrier(DataType type, int count, boolean segment, Arena arena) {
        if (segment) return arena.allocate((long) count * type.byteWidth(), type.byteWidth());
        return type == DataType.FLOAT32 ? new float[count]
                : type == DataType.FLOAT64 ? new double[count] : new byte[count];
    }

    private static void fill(Object carrier, DataType type, int count, double value) {
        for (int i = 0; i < count; i++) write(carrier, type, i, value);
    }

    private static void write(Object carrier, DataType type, int index, double value) {
        if (carrier instanceof float[] array) array[index] = (float) value;
        else if (carrier instanceof double[] array) array[index] = value;
        else if (carrier instanceof byte[] array) array[index] = (byte) value;
        else if (type == DataType.FLOAT32) ((MemorySegment) carrier).setAtIndex(FLOAT, index,
                (float) value);
        else if (type == DataType.FLOAT64) ((MemorySegment) carrier).setAtIndex(DOUBLE, index, value);
        else ((MemorySegment) carrier).setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) value);
    }

    private static double read(Object carrier, DataType type, int index) {
        if (carrier instanceof float[] array) return array[index];
        if (carrier instanceof double[] array) return array[index];
        if (carrier instanceof byte[] array) return array[index];
        if (type == DataType.FLOAT32) return ((MemorySegment) carrier).getAtIndex(FLOAT, index);
        if (type == DataType.FLOAT64) return ((MemorySegment) carrier).getAtIndex(DOUBLE, index);
        return ((MemorySegment) carrier).getAtIndex(ValueLayout.JAVA_BYTE, index);
    }

    private static double input(DataType type, int boundary, int lane) {
        if (type == DataType.BOOL) return (lane & 1) == 0 ? 0d : 1d;
        double[] values = {Double.NaN, Double.POSITIVE_INFINITY, -3d, -0d, 0d, 2d, 7d};
        double value = values[(lane + boundary * 2) % values.length];
        return type == DataType.FLOAT32 ? (float) value : value;
    }

    private static int lanes(DataType type) {
        return type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.length()
                : DoubleVector.SPECIES_PREFERRED.length();
    }

    private static int bits(DataType type) {
        return type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : DoubleVector.SPECIES_PREFERRED.vectorBitSize();
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.FLOAT32, DataType.FLOAT64);
    }

    private static String sha256(byte[] value) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
    }

    private enum Scenario {
        ARRAY_EXACT(false, false, false, false), SEGMENT_OFFSET_TAIL(true, true, true, false),
        BOOL_SEGMENT_PARALLEL(false, false, false, true), NUMERIC_SEGMENT_BOOL_ARRAY(true, true, true, true);
        final boolean numericSegment, offset, tail, parallel;
        Scenario(boolean numericSegment, boolean offset, boolean tail, boolean parallel) {
            this.numericSegment = numericSegment; this.offset = offset;
            this.tail = tail; this.parallel = parallel;
        }
        boolean segment(DataType type) {
            return this == BOOL_SEGMENT_PARALLEL ? type == DataType.BOOL
                    : this == NUMERIC_SEGMENT_BOOL_ARRAY ? type != DataType.BOOL : numericSegment;
        }
    }

    private record Form(String name, CpuKernelIr ir, DataType type) { }
    private record Dossier(String name, byte[] bytes, String sha256) { }
}
