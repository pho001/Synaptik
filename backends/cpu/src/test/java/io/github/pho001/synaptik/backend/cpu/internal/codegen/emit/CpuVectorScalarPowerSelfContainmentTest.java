package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.StreamSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/** Semantic and complete-Class-File closure for CPU 0008Q1A vector scalar power. */
class CpuVectorScalarPowerSelfContainmentTest {
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ORDER);
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ORDER);
    private static final List<CpuKernelIr.PowerRealization> REALIZATIONS = List.of(
            CpuKernelIr.PowerRealization.POSITIVE_ONE,
            CpuKernelIr.PowerRealization.IDENTITY,
            CpuKernelIr.PowerRealization.SQUARE,
            CpuKernelIr.PowerRealization.RECIPROCAL);

    @Test void everyTypeRealizationAndOrderedCarrierPairIsSelfContained() {
        assertEquals(66, CpuGeneratorSchema.CURRENT_VERSION);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            CpuKernelSpecialization.CarrierAccess array = arrayCarrier(type);
            for (CpuKernelIr.PowerRealization realization : REALIZATIONS) {
                for (List<CpuKernelSpecialization.CarrierAccess> carriers : List.of(
                        List.of(array, array),
                        List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        List.of(array, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, array))) {
                    byte[] bytes = bytes(type, realization, carriers, false,
                            CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
                    assertStructure(type, realization, carriers, bytes);
                    byte[] parallel = bytes(type, realization, carriers, false,
                            CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR);
                    assertArrayEquals(bytes, parallel,
                            type + " " + realization + " orchestration must share generated body");
                }
            }
        }
    }

    @Test void arbitraryRangesVectorChunksScalarTailsAndEdgeValuesMatchCleanJava()
            throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int lanes = type == DataType.FLOAT32
                    ? FloatVector.SPECIES_PREFERRED.length()
                    : DoubleVector.SPECIES_PREFERRED.length();
            int count = lanes * 3 + 7;
            long start = 2;
            long end = count - 1L;
            for (CpuKernelIr.PowerRealization realization : REALIZATIONS) {
                for (List<CpuKernelSpecialization.CarrierAccess> carriers : carrierPairs(type)) {
                    try (Arena arena = Arena.ofConfined()) {
                        Object input = carrier(type, carriers.get(0), count, arena);
                        Object output = carrier(type, carriers.get(1), count, arena);
                        fill(type, input, count);
                        fillSentinel(type, output, count);
                        var generated = artifact(type, realization, carriers, false,
                                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
                        generated.entryPoint().invokeWithArguments(input, output,
                                geometry(2, count, start), start, end);
                        for (int index = 0; index < count; index++) {
                            double actual = read(type, output, index);
                            if (index < start || index >= end) {
                                assertBits(type, sentinel(type), actual,
                                        "outside range " + index);
                            } else {
                                assertBits(type, clean(type, realization,
                                        read(type, input, index)), actual,
                                        type + " " + realization + " lane " + index);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test void representativeFusedBodiesRetainDirectPowerDataflow() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int lanes = type == DataType.FLOAT32
                    ? FloatVector.SPECIES_PREFERRED.length()
                    : DoubleVector.SPECIES_PREFERRED.length();
            for (CpuKernelIr.PowerRealization realization : REALIZATIONS) {
                List<CpuKernelSpecialization.CarrierAccess> carriers = List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, arrayCarrier(type));
                byte[] bytes = bytes(type, realization, carriers, true,
                        CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
                assertStructure(type, realization, carriers, bytes);
                try (Arena arena = Arena.ofConfined()) {
                    int count = lanes + 1;
                    Object input = carrier(type, carriers.get(0), count, arena);
                    Object output = carrier(type, carriers.get(1), count, arena);
                    fill(type, input, count);
                    artifact(type, realization, carriers, true,
                            CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)
                            .entryPoint().invokeWithArguments(input, output, geometry(2, count),
                                    0L, (long) count);
                    for (int i = 0; i < count; i++) assertBits(type,
                            -clean(type, realization, read(type, input, i)), read(type, output, i),
                            "fused " + type + " " + realization + " lane " + i);
                }
            }
        }
    }

    @Test void fusedPowerDoesNotNarrowAnotherAccessedBoundary() throws Throwable {
        DataType type = DataType.FLOAT32;
        var unused = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
                List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.STRIDED), 0);
        var bias = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.LAST_AXIS_BIAS, 2,
                List.of(CpuAccessPlan.AxisRole.BROADCAST, CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var output = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 2,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS,
                        CpuAccessPlan.AxisRole.CONTIGUOUS), 2);
        CpuKernelIr ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, unused),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.INPUT, bias),
                new CpuKernelIr.Value(2, type, CpuKernelIr.Value.Kind.VIRTUAL, output),
                new CpuKernelIr.Value(3, type, CpuKernelIr.Value.Kind.OUTPUT, output)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW, List.of(0), 2,
                                new CpuKernelIr.ScalarImmediate(type, 0L),
                                CpuKernelIr.PowerRealization.POSITIVE_ONE),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD, List.of(2, 1), 3)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(3, 0)));
        var carriers = List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                List.of(type, type, type), carriers,
                FloatVector.SPECIES_PREFERRED.vectorBitSize(), -1,
                List.of(CpuKernelIr.PowerRealization.POSITIVE_ONE), false, 52);
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
        float[] ignored = {Float.NaN};
        float[] biasValues = {10.0f, 20.0f, 30.0f};
        float[] actual = {-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f};
        artifact.entryPoint().invokeWithArguments(ignored, biasValues, actual,
                fusedBiasGeometry(), 1L, 6L);
        assertArrayEquals(new float[] {-7.0f, 21.0f, 31.0f, 11.0f, 21.0f, 31.0f}, actual);
    }

    @Test void scalarSegmentBroadcastDoesNotPrepareUnusedVectorByteOrder() {
        DataType type = DataType.FLOAT32;
        CpuAccessPlan scalar = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.SCALAR_ALL_ZERO, 1,
                List.of(CpuAccessPlan.AxisRole.BROADCAST), 0);
        CpuAccessPlan denseRead = dense(CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan denseWrite = dense(CpuAccessPlan.AccessKind.WRITE);
        CpuKernelIr ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.INPUT, scalar),
                new CpuKernelIr.Value(2, type, CpuKernelIr.Value.Kind.VIRTUAL, denseWrite),
                new CpuKernelIr.Value(3, type, CpuKernelIr.Value.Kind.OUTPUT, denseWrite)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW, List.of(0), 2,
                                new CpuKernelIr.ScalarImmediate(type, 0x3f80_0000L),
                                CpuKernelIr.PowerRealization.IDENTITY),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD, List.of(2, 1), 3)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(3, 0)));
        var carriers = List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                List.of(type, type, type), carriers,
                FloatVector.SPECIES_PREFERRED.vectorBitSize(), -1,
                List.of(CpuKernelIr.PowerRealization.IDENTITY), false, 52);
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir);
        long nativeOrderCalls = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow()
                .elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).filter(call -> call.owner().asInternalName()
                        .equals("java/nio/ByteOrder")
                        && call.name().stringValue().equals("nativeOrder")).count();
        assertEquals(1L, nativeOrderCalls,
                "the scalar segment layout is prepared, but vector byte order is not needed");
    }

    @Test void directPowerRemainsVectorIneligible() {
        CpuKernelIr ir = ir(DataType.FLOAT32, CpuKernelIr.PowerRealization.DIRECT, false);
        var specialization = specialization(ir, DataType.FLOAT32,
                CpuKernelIr.PowerRealization.DIRECT,
                List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY),
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
        assertThrows(IllegalArgumentException.class,
                () -> new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir));
    }

    private static void assertStructure(DataType type, CpuKernelIr.PowerRealization realization,
            List<CpuKernelSpecialization.CarrierAccess> carriers, byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        assertEquals(1, model.methods().size());
        var method = model.methods().getFirst();
        assertEquals(CpuGeneratorSchema.ENTRY_NAME, method.methodName().stringValue());
        var code = method.code().orElseThrow();
        List<Instruction> instructions = code.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).toList();
        assertTrue(instructions.stream().anyMatch(i -> i.opcode().kind()
                == Opcode.Kind.BRANCH), "vector loop and scalar tail require branches");
        assertFalse(instructions.stream().anyMatch(i -> switch (i.opcode()) {
            case NEW, NEWARRAY, ANEWARRAY, MULTIANEWARRAY, INVOKEDYNAMIC -> true;
            default -> false;
        }));
        List<String> members = StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .map(entry -> entry.owner().name().stringValue() + "."
                        + entry.nameAndType().name().stringValue()).toList();
        assertTrue(members.stream().noneMatch(name -> name.startsWith(
                "io/github/pho001/synaptik")), members.toString());
        assertTrue(StreamSupport.stream(model.constantPool().spliterator(), false)
                .noneMatch(MethodHandleEntry.class::isInstance));
        assertTrue(StreamSupport.stream(model.constantPool().spliterator(), false)
                .noneMatch(DynamicConstantPoolEntry.class::isInstance));
        for (String forbidden : List.of("java/lang/reflect", "java/util/Map", "java/lang/String",
                "java/lang/invoke")) assertTrue(members.stream().noneMatch(name ->
                name.startsWith(forbidden)), forbidden + " " + members);
        String vector = type == DataType.FLOAT32 ? "jdk/incubator/vector/FloatVector"
                : "jdk/incubator/vector/DoubleVector";
        if (realization == CpuKernelIr.PowerRealization.POSITIVE_ONE
                || realization == CpuKernelIr.PowerRealization.RECIPROCAL) {
            assertTrue(members.contains(vector + ".broadcast"), members.toString());
        }
        if (realization == CpuKernelIr.PowerRealization.SQUARE)
            assertTrue(members.contains(vector + ".mul"), members.toString());
        if (realization == CpuKernelIr.PowerRealization.RECIPROCAL)
            assertTrue(members.contains(vector + ".div"), members.toString());
        if (realization == CpuKernelIr.PowerRealization.POSITIVE_ONE) {
            Opcode load = type == DataType.FLOAT32 ? Opcode.FALOAD : Opcode.DALOAD;
            assertFalse(instructions.stream().anyMatch(i -> i.opcode() == load));
            if (carriers.getFirst() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                assertTrue(members.stream().noneMatch(name -> name.equals(
                        "java/lang/foreign/MemorySegment.get")), members.toString());
        }
        String load = carriers.getFirst() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? vector + ".fromMemorySegment" : vector + ".fromArray";
        if (realization != CpuKernelIr.PowerRealization.POSITIVE_ONE)
            assertTrue(members.contains(load), members.toString());
        String store = carriers.get(1) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? vector + ".intoMemorySegment" : vector + ".intoArray";
        assertTrue(members.contains(store), members.toString());
        boolean readsSegment = realization != CpuKernelIr.PowerRealization.POSITIVE_ONE
                && carriers.getFirst() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        boolean writesSegment = carriers.get(1)
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        boolean accessesSegment = readsSegment || writesSegment;
        long nativeOrderCalls = instructions.stream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).filter(call -> call.owner().asInternalName()
                        .equals("java/nio/ByteOrder")
                        && call.name().stringValue().equals("nativeOrder")).count();
        assertEquals(accessesSegment ? 2L : 0L, nativeOrderCalls,
                "layout and vector byte order must be prepared once outside the loop");
        assertEquals(accessesSegment ? 1L : 0L,
                instructions.stream().filter(i -> i.opcode() == Opcode.LREM).count(),
                "dense long vector bound must be computed once");
        if (realization == CpuKernelIr.PowerRealization.POSITIVE_ONE && !writesSegment) {
            assertEquals(4L, instructions.stream()
                    .filter(i -> i.opcode() == Opcode.IINC).count(),
                    "positive-one array output must not advance its unused input address");
        }
    }

    private static byte[] bytes(DataType type, CpuKernelIr.PowerRealization realization,
            List<CpuKernelSpecialization.CarrierAccess> carriers, boolean fused,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy) {
        CpuKernelIr ir = ir(type, realization, fused);
        return new CpuClassFileKernelGenerator().generateClassBytes(
                specialization(ir, type, realization, carriers, strategy), ir);
    }

    private static CpuGeneratedKernel artifact(DataType type,
            CpuKernelIr.PowerRealization realization,
            List<CpuKernelSpecialization.CarrierAccess> carriers, boolean fused,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy) {
        CpuKernelIr ir = ir(type, realization, fused);
        var specialization = specialization(ir, type, realization, carriers, strategy);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static CpuKernelSpecialization specialization(CpuKernelIr ir, DataType type,
            CpuKernelIr.PowerRealization realization,
            List<CpuKernelSpecialization.CarrierAccess> carriers,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy) {
        int species = type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                : DoubleVector.SPECIES_PREFERRED.vectorBitSize();
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT, strategy,
                List.of(type, type), carriers, species, -1, List.of(realization), false, 52);
    }

    private static CpuKernelIr ir(DataType type, CpuKernelIr.PowerRealization realization,
            boolean fused) {
        long bits = switch (realization) {
            case POSITIVE_ONE -> 0L;
            case IDENTITY -> type == DataType.FLOAT32 ? 0x3f80_0000L : 0x3ff0_0000_0000_0000L;
            case SQUARE -> type == DataType.FLOAT32 ? 0x4000_0000L : 0x4000_0000_0000_0000L;
            case RECIPROCAL -> type == DataType.FLOAT32 ? 0xbf80_0000L : 0xbff0_0000_0000_0000L;
            case DIRECT -> type == DataType.FLOAT32 ? 0x4040_0000L : 0x4008_0000_0000_0000L;
        };
        CpuAccessPlan read = dense(CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan write = dense(CpuAccessPlan.AccessKind.WRITE);
        var values = new ArrayList<CpuKernelIr.Value>();
        values.add(new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, read));
        values.add(new CpuKernelIr.Value(1, type,
                fused ? CpuKernelIr.Value.Kind.VIRTUAL : CpuKernelIr.Value.Kind.OUTPUT,
                fused ? read : write));
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW, List.of(0), 1,
                new CpuKernelIr.ScalarImmediate(type, bits), realization));
        int output = 1;
        if (fused) {
            output = 2;
            values.add(new CpuKernelIr.Value(2, type, CpuKernelIr.Value.Kind.OUTPUT, write));
            instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(1), 2));
        }
        return new CpuKernelIr(values, instructions, new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(output, 0)));
    }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static List<List<CpuKernelSpecialization.CarrierAccess>> carrierPairs(DataType type) {
        CpuKernelSpecialization.CarrierAccess array = arrayCarrier(type);
        CpuKernelSpecialization.CarrierAccess segment =
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        return List.of(List.of(array, array), List.of(segment, segment),
                List.of(array, segment), List.of(segment, array));
    }

    private static CpuKernelSpecialization.CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
    }

    private static Object carrier(DataType type, CpuKernelSpecialization.CarrierAccess access,
            int count, Arena arena) {
        if (access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
            return arena.allocate((long) count * type.byteWidth(), type.byteWidth());
        return type == DataType.FLOAT32 ? new float[count] : new double[count];
    }

    private static void fill(DataType type, Object carrier, int count) {
        double[] edges = {+0.0d, -0.0d, Double.MIN_VALUE, Float.MIN_VALUE,
                Double.MIN_NORMAL, Float.MIN_NORMAL, 0.5d, -0.5d, 1.0d, -1.0d,
                Double.MAX_VALUE, Float.MAX_VALUE, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.NaN};
        for (int i = 0; i < count; i++) write(type, carrier, i, edges[i % edges.length]);
    }

    private static void fillSentinel(DataType type, Object carrier, int count) {
        for (int i = 0; i < count; i++) write(type, carrier, i, sentinel(type));
    }

    private static double sentinel(DataType type) {
        return type == DataType.FLOAT32 ? -123.25f : -123.25d;
    }

    private static double clean(DataType type, CpuKernelIr.PowerRealization realization,
            double value) {
        if (type == DataType.FLOAT32) {
            float x = (float) value;
            return switch (realization) {
                case POSITIVE_ONE -> 1.0f;
                case IDENTITY -> x;
                case SQUARE -> x * x;
                case RECIPROCAL -> 1.0f / x;
                case DIRECT -> throw new AssertionError();
            };
        }
        return switch (realization) {
            case POSITIVE_ONE -> 1.0d;
            case IDENTITY -> value;
            case SQUARE -> value * value;
            case RECIPROCAL -> 1.0d / value;
            case DIRECT -> throw new AssertionError();
        };
    }

    private static void write(DataType type, Object carrier, int index, double value) {
        if (carrier instanceof float[] array) array[index] = (float) value;
        else if (carrier instanceof double[] array) array[index] = value;
        else if (type == DataType.FLOAT32) ((MemorySegment) carrier).set(FLOAT,
                (long) index * Float.BYTES, (float) value);
        else ((MemorySegment) carrier).set(DOUBLE, (long) index * Double.BYTES, value);
    }

    private static double read(DataType type, Object carrier, int index) {
        if (carrier instanceof float[] array) return array[index];
        if (carrier instanceof double[] array) return array[index];
        return type == DataType.FLOAT32 ? ((MemorySegment) carrier).get(FLOAT,
                (long) index * Float.BYTES) : ((MemorySegment) carrier).get(DOUBLE,
                (long) index * Double.BYTES);
    }

    private static void assertBits(DataType type, double expected, double actual, String message) {
        if (type == DataType.FLOAT32) {
            float e = (float) expected;
            float a = (float) actual;
            if (Float.isNaN(e)) assertTrue(Float.isNaN(a), message);
            else assertEquals(Float.floatToRawIntBits(e), Float.floatToRawIntBits(a), message);
        } else if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual), message);
        else assertEquals(Double.doubleToRawLongBits(expected),
                Double.doubleToRawLongBits(actual), message);
    }

    private static long[] geometry(int count, long extent) {
        return geometry(count, extent, 0);
    }

    private static long[] geometry(int count, long extent, long base) {
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int i = 0; i < count; i++) {
            result[2 + i] = base;
            result[2 + count + i] = 1;
            result[2 + count + count + count + i] = extent;
        }
        return result;
    }

    private static long[] fusedBiasGeometry() {
        return new long[] {2, 3, 0, 1, 999, 1, 1,
                7, 11, 0, 1, 3, 1,
                0, 1, 0, 0, 0, 0};
    }
}
