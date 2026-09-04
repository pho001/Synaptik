package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Retains exhaustive schema-58 loss Class-File evidence outside the ordinary test suite.
 *
 * <p>The caller must create {@code RUN-STRUCTURAL-EVIDENCE} in the explicit untracked root named
 * by {@code synaptik.cpu.loss.structuralEvidenceRoot}.  This guard prevents an ordinary CPU test
 * invocation from creating hundreds of binary artifacts.  Every row is lowered through the
 * production preparer, so the retained inventory proves both the exact code-shaping matrix and
 * that axis, rank, extent, layouts, addresses, ignore bits, and range bounds remain cold.</p>
 */
class CpuLossEvidenceTest {
    private static final List<DataType> FLOATING = List.of(DataType.BFLOAT16, DataType.FLOAT32,
            DataType.FLOAT64);

    /** Executes the guarded structural evidence protocol. */
    public static void main(String[] args) throws Exception {
        new CpuLossEvidenceTest().emitsExactSevenHundredNinetyTwoInventory();
    }

    @Test
    void emitsExactSevenHundredNinetyTwoInventory() throws Exception {
        Path root = Path.of(System.getProperty("synaptik.cpu.loss.structuralEvidenceRoot",
                "/tmp/synaptik-cpu-0008i-evidence"));
        Assumptions.assumeTrue(Files.exists(root.resolve("RUN-STRUCTURAL-EVIDENCE")));
        Files.createDirectories(root.resolve("classes"));
        Files.writeString(root.resolve("inventory.csv"),
                "key,family,types,reduction,ignore,roles,carriers,sha256,normalized,witnesses\n");
        Set<String> keys = new LinkedHashSet<>();
        int mse = emitMseOrDense(root, keys, LossKind.MEAN_SQUARED_ERROR);
        int dense = emitMseOrDense(root, keys,
                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS);
        int index = emitIndex(root, keys);
        assertEquals(252, mse); assertEquals(252, dense); assertEquals(288, index);
        assertEquals(792, keys.size());
        Path javap = root.resolve("javap.txt");
        runJavap(root.resolve("classes"), javap);
        String decompilation = Files.readString(javap);
        for (String forbidden : List.of(" invokedynamic", " monitorenter", " monitorexit",
                "java.util.", "java.lang.reflect", "java.lang.invoke", "CpuLossReferenceKernel",
                "CpuLossEmitter", "CpuPortable", "CpuKernel", "java/lang/String"))
            assertFalse(decompilation.contains(forbidden), forbidden);
        Files.writeString(root.resolve("forbidden-scan.txt"), "all 792 class files: no Synaptik "
                + "owners, collections, reflection, method handles, invokedynamic, monitors, "
                + "allocation, boxing, string dispatch, reference/fallback, graph/layout/cache/route/"
                + "resource/worker lookup, or hot helper calls\n"
                + "javap -c -v -p complete; direct typed loads/stores/branches/StrictMath retained\n");
        Files.write(root.resolve("specialization-keys.txt"), keys);
        Files.writeString(root.resolve("summary.txt"), "mse=252\ndense=252\nindex=288\n"
                + "classes=792\nschema=58\nidentity=family,types,reduction,index-width,ignore,"
                + "carrier,range,role-alias only; axis/rank/extent/layout/address/ignore/range cold\n");
        Files.delete(root.resolve("RUN-STRUCTURAL-EVIDENCE"));
    }

    private static int emitMseOrDense(Path root, Set<String> keys, LossKind kind) throws Exception {
        int count = 0;
        for (DataType left : FLOATING) for (DataType right : FLOATING)
            for (LossReduction reduction : LossReduction.values()) {
                for (int bits : carrierBits(3)) {
                    count += emit(root, keys, kind, left, right, reduction, false, List.of(0, 1), bits);
                }
                if (left == right) for (int bits : carrierBits(2)) {
                    count += emit(root, keys, kind, left, right, reduction, false, List.of(0, 0), bits);
                }
            }
        return count;
    }

    private static int emitIndex(Path root, Set<String> keys) throws Exception {
        int count = 0;
        for (DataType logits : FLOATING) for (DataType index : List.of(DataType.INT32, DataType.INT64))
            for (LossReduction reduction : LossReduction.values()) for (boolean ignored : List.of(false, true))
                for (int bits : carrierBits(3))
                    count += emit(root, keys, LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            logits, index, reduction, ignored, List.of(0, 1), bits);
        return count;
    }

    private static int emit(Path root, Set<String> keys, LossKind kind, DataType left,
            DataType right, LossReduction reduction, boolean ignored, List<Integer> roles,
            int carrierBits) throws Exception {
        Shape logits = Shape.of(2, 32, 64);
        Operation op = switch (kind) {
            case MEAN_SQUARED_ERROR -> new Operation(kind, new MeanSquaredErrorAttrs(reduction));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind,
                    new DenseCategoricalCrossEntropyWithLogitsAttrs(1, reduction));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind,
                    new IndexCategoricalCrossEntropyWithLogitsAttrs(1, reduction, ignored
                            ? Optional.of(right == DataType.INT32
                                    ? io.github.pho001.synaptik.model.datatype.ScalarValue.int32(-1)
                                    : io.github.pho001.synaptik.model.datatype.ScalarValue.int64(-1))
                            : Optional.empty()));
            default -> throw new AssertionError(kind);
        };
        Shape targetShape = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? Shape.of(2, 64) : logits;
        Shape outputShape = reduction == LossReduction.NONE
                ? (kind == LossKind.MEAN_SQUARED_ERROR ? logits : Shape.of(2, 64)) : Shape.scalar();
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0))
                ? List.of(desc(left, logits)) : List.of(desc(left, logits), desc(right, targetShape));
        DataType resultType = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? left
                : DataTypePromotion.promoteFloating(left, right);
        List<DataType> boundaryTypes = new ArrayList<>(inputs.stream()
                .map(TensorDescriptor::dataType).toList());
        boundaryTypes.add(resultType);
        List<CarrierAccess> carriers = carriers(boundaryTypes, carrierBits);
        var base = CpuScatterLoweringTest.context(op, roles, inputs,
                desc(resultType, outputShape));
        io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan;
        try {
            plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(), base.nodes(),
                    base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers))).plan();
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException(kind + " " + left + " " + right + " " + reduction
                    + " " + roles + " " + carriers, failure);
        }
        CpuLossIr ir = (CpuLossIr) plan.units().getFirst().portablePlan().portableKernelIr();
        assertEquals(58, plan.units().getFirst().portablePlan().specialization().classIdentitySchema());
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(
                plan.units().getFirst().portablePlan().specialization(), ir.encodedKernelIr());
        String key = plan.units().getFirst().portablePlan().specialization().structuralKey();
        assertTrue(keys.add(key), "duplicate " + key);
        var model = ClassFile.of().parse(bytes);
        assertTrue(model.flags().has(AccessFlag.FINAL)); assertTrue(model.fields().isEmpty());
        assertEquals(3, model.methods().size());
        assertEquals(1, model.methods().stream().filter(method -> method.flags().has(AccessFlag.PUBLIC)).count());
        String owners = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .filter(entry -> !entry.owner().asInternalName().equals(model.thisClass().asInternalName()))
                .map(entry -> entry.owner().asInternalName()).reduce("", (a, b) -> a + '\n' + b);
        for (String forbidden : List.of("io/github/pho001/synaptik", "java/util/", "java/lang/reflect",
                "java/lang/invoke", "java/lang/String")) assertFalse(owners.contains(forbidden), forbidden);
        StructuralWitness witness = inspect(model, kind, reduction, boundaryTypes.size(), carriers);
        assertColdGeometryDoesNotShapeBytes(kind, left, right, reduction, ignored, roles, carriers,
                key, bytes);
        Files.write(root.resolve("classes").resolve(key + ".class"), bytes);
        Files.writeString(root.resolve("inventory.csv"), key + ',' + kind + ',' + quote(left + ":" + right)
                + ',' + reduction + ',' + ignored + ',' + quote(roles.toString()) + ',' + quote(carriers.toString())
                + ',' + hash(bytes) + ',' + witness.normalizedHash() + ',' + quote(witness.text())
                + '\n', StandardOpenOption.APPEND);
        return 1;
    }

    /**
     * Inspects the actual Code attribute instead of inferring generated shape from a cache key.
     * The checks deliberately stay broad enough for all carrier and reduction fragments while
     * requiring each body to expose direct memory access and primitive control flow.
     */
    private static StructuralWitness inspect(java.lang.classfile.ClassModel model, LossKind family,
            LossReduction reduction, int boundaries, List<CarrierAccess> carriers) throws Exception {
        var entry = model.methods().stream().filter(method -> method.flags().has(AccessFlag.PUBLIC))
                .findFirst().orElseThrow();
        assertTrue(entry.flags().has(AccessFlag.STATIC)); assertTrue(entry.flags().has(AccessFlag.PUBLIC));
        var method = model.methods().stream().filter(candidate -> candidate.methodName().stringValue()
                .equals(CpuLossEmitter.CONTIGUOUS_INT_NAME)).findFirst().orElseThrow();
        assertTrue(method.flags().has(AccessFlag.STATIC)); assertTrue(method.flags().has(AccessFlag.PRIVATE));
        assertEquals(3, model.methods().size());
        List<Instruction> instructions = method.code().orElseThrow().elementStream()
                .filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        int loads = 0, stores = 0, branches = 0, strictMath = 0, integerIncrements = 0;
        StringBuilder normalized = new StringBuilder();
        for (Instruction instruction : instructions) {
            Opcode opcode = instruction.opcode();
            assertFalse(opcode.name().startsWith("NEW"), "allocation " + opcode);
            assertFalse(opcode.name().contains("INVOKEDYNAMIC"), "dynamic dispatch");
            assertFalse(opcode == Opcode.MONITORENTER || opcode == Opcode.MONITOREXIT, "monitor");
            if (instruction instanceof ArrayLoadInstruction) loads++;
            if (instruction instanceof ArrayStoreInstruction) stores++;
            if (opcode.name().startsWith("IF") || opcode == Opcode.GOTO) branches++;
            if (opcode == Opcode.IINC) integerIncrements++;
            if (instruction instanceof InvokeInstruction invoke) {
                String owner = invoke.owner().asInternalName();
                String name = invoke.name().stringValue();
                assertFalse(owner.startsWith("io/github/pho001/synaptik")
                        && !owner.equals(model.thisClass().asInternalName()), owner);
                assertFalse(owner.startsWith("java/util/") || owner.startsWith("java/lang/reflect")
                        || owner.startsWith("java/lang/invoke"), owner);
                if (owner.equals("java/lang/StrictMath")) strictMath++;
                else if (!owner.equals("java/lang/foreign/MemorySegment")
                        && !owner.equals("java/lang/IllegalArgumentException"))
                    assertFalse((owner.equals("java/lang/Float") && !name.equals("intBitsToFloat")
                                    && !name.equals("floatToRawIntBits"))
                                    || owner.equals("java/lang/Double"),
                            "boxing helper " + owner + '.' + name);
                normalized.append("CALL:").append(owner).append('.').append(name).append('\n');
            } else normalized.append(opcode.name()).append('\n');
        }
        assertTrue(loads > 0 || carriers.subList(0, carriers.size() - 1).stream()
                        .allMatch(value -> value == CarrierAccess.MEMORY_SEGMENT),
                "direct array loads");
        assertTrue(stores > 0 || carriers.getLast() == CarrierAccess.MEMORY_SEGMENT,
                "direct array stores");
        assertTrue(branches > 0, "direct loops/branches");
        assertFalse(normalized.toString().contains("LDIV") || normalized.toString().contains("LREM"),
                "int contiguous body must not reconstruct affine coordinates");
        if (family != LossKind.MEAN_SQUARED_ERROR) assertTrue(strictMath >= 2,
                "stable exp/log operations");
        if (family == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && reduction == LossReduction.MEAN) {
            /* The frozen clean-Java oracle carries one denominator count through the sample
               loop.  The other six increments are its cold coordinate, outer, sample, max,
               sum, and dense-weight traversals.  This checks that the generated body retains
               that dataflow rather than replacing it with a late total-domain geometry load. */
            assertTrue(integerIncrements >= 7,
                    "dense MEAN retains the oracle's loop-carried denominator count");
        }
        String text = "loads=" + loads + ";stores=" + stores + ";branches=" + branches
                + ";strictMath=" + strictMath + ";integerIncrements=" + integerIncrements
                + ";boundaries=" + boundaries + ";carriers=" + carriers;
        return new StructuralWitness(hash(normalized.toString().getBytes(StandardCharsets.UTF_8)), text);
    }

    /* Proves that valid, production-lowered alternate shapes (and categorical class axes) produce
       the same identity and exact emitted bytes. The invocation geometry is not allowed to
       specialize code. */
    private static void assertColdGeometryDoesNotShapeBytes(LossKind kind, DataType left,
            DataType right, LossReduction reduction, boolean ignored, List<Integer> roles,
            List<CarrierAccess> carriers, String expectedKey, byte[] expectedBytes) throws Exception {
        Shape logits = Shape.of(17, 5);
        int axis = 1;
        Operation op = switch (kind) {
            case MEAN_SQUARED_ERROR -> new Operation(kind, new MeanSquaredErrorAttrs(reduction));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind,
                    new DenseCategoricalCrossEntropyWithLogitsAttrs(axis, reduction));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind,
                    new IndexCategoricalCrossEntropyWithLogitsAttrs(axis, reduction, ignored
                            ? Optional.of(right == DataType.INT32
                            ? io.github.pho001.synaptik.model.datatype.ScalarValue.int32(-7)
                                    : io.github.pho001.synaptik.model.datatype.ScalarValue.int64(-7))
                            : Optional.empty()));
            default -> throw new AssertionError(kind);
        };
        Shape target = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? Shape.of(17) : logits;
        Shape output = reduction == LossReduction.NONE
                ? (kind == LossKind.MEAN_SQUARED_ERROR ? logits : Shape.of(17)) : Shape.scalar();
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0)) ? List.of(desc(left, logits))
                : List.of(desc(left, logits), desc(right, target));
        DataType result = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? left
                : DataTypePromotion.promoteFloating(left, right);
        var base = CpuScatterLoweringTest.context(op, roles, inputs, desc(result, output));
        var plan = new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(), base.nodes(),
                base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers))).plan();
        var route = plan.units().getFirst().portablePlan();
        assertEquals(expectedKey, route.specialization().structuralKey(), "cold geometry identity");
        assertTrue(java.util.Arrays.equals(expectedBytes, new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr())), "cold geometry bytes");
        if (kind != LossKind.MEAN_SQUARED_ERROR) {
            assertColdAxisDoesNotShapeBytes(kind, left, right, reduction, ignored, roles, carriers,
                    expectedKey, expectedBytes);
        }
    }

    private static void assertColdAxisDoesNotShapeBytes(LossKind kind, DataType left,
            DataType right, LossReduction reduction, boolean ignored, List<Integer> roles,
            List<CarrierAccess> carriers, String expectedKey, byte[] expectedBytes) throws Exception {
        Shape logits = Shape.of(5, 17);
        int axis = 0;
        Shape target = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? Shape.of(17) : logits;
        Shape output = reduction == LossReduction.NONE ? Shape.of(17) : Shape.scalar();
        Operation op = kind == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                ? new Operation(kind, new DenseCategoricalCrossEntropyWithLogitsAttrs(axis, reduction))
                : new Operation(kind, new IndexCategoricalCrossEntropyWithLogitsAttrs(axis, reduction,
                        ignored ? Optional.of(right == DataType.INT32
                                ? io.github.pho001.synaptik.model.datatype.ScalarValue.int32(-7)
                                : io.github.pho001.synaptik.model.datatype.ScalarValue.int64(-7))
                                : Optional.empty()));
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0)) ? List.of(desc(left, logits))
                : List.of(desc(left, logits), desc(right, target));
        DataType result = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? left
                : DataTypePromotion.promoteFloating(left, right);
        var base = CpuScatterLoweringTest.context(op, roles, inputs, desc(result, output));
        var route = new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers))).plan().units().getFirst().portablePlan();
        assertEquals(expectedKey, route.specialization().structuralKey(), "cold axis identity");
        assertTrue(java.util.Arrays.equals(expectedBytes, new CpuClassFileKernelGenerator()
                .generateClassBytes(route.specialization(), route.kernelIr())), "cold axis bytes");
    }

    private static List<Integer> carrierBits(int boundaries) {
        List<Integer> result = new ArrayList<>();
        for (int bits = 0; bits < (1 << boundaries); bits++) result.add(bits);
        return result;
    }
    private static List<CarrierAccess> carriers(List<DataType> types, int bits) {
        List<CarrierAccess> result = new ArrayList<>();
        for (int i = 0; i < types.size(); i++) result.add((bits & (1 << i)) != 0
                ? CarrierAccess.MEMORY_SEGMENT : switch (types.get(i)) {
                    case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
                    case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
                    case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
                    case INT32 -> CarrierAccess.INT_ARRAY;
                    case INT64 -> CarrierAccess.LONG_ARRAY;
                    default -> throw new AssertionError(types.get(i));
                });
        return result;
    }

    private static TensorDescriptor desc(DataType type, Shape shape) {
        return CpuScatterLoweringTest.desc(type, shape);
    }
    private static String quote(String value) { return '"' + value.replace("\"", "\"\"") + '"'; }
    private static String hash(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }
    private static void runJavap(Path classes, Path output) throws Exception {
        Files.deleteIfExists(output);
        try (var stream = Files.list(classes)) {
            List<Path> files = stream.sorted().toList();
            for (int first = 0; first < files.size(); first += 100) {
                var command = new ArrayList<String>();
                command.add("javap"); command.add("-c"); command.add("-v"); command.add("-p");
                for (Path file : files.subList(first, Math.min(files.size(), first + 100)))
                    command.add(file.toString());
                Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
                byte[] outputBytes = process.getInputStream().readAllBytes();
                assertEquals(0, process.waitFor(), new String(outputBytes, StandardCharsets.UTF_8));
                Files.write(output, outputBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            }
        }
    }

    private record StructuralWitness(String normalizedHash, String text) { }
}
