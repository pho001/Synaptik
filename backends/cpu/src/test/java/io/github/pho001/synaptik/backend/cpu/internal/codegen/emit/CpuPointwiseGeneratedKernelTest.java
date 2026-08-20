package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;
import org.junit.jupiter.api.Test;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ByteVector;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;

class CpuPointwiseGeneratedKernelTest {
    private static final List<CpuPointwiseOpcode> SELF_CONTAINED_ACTIVATIONS = List.of(
            CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.SIGMOID, CpuPointwiseOpcode.GELU_EXACT,
            CpuPointwiseOpcode.GELU_TANH_APPROXIMATION, CpuPointwiseOpcode.SILU);

    @Test void coveredScalarActivationArtifactsAreSelfContainedTypedAndFreeOfDynamicConstructs() {
        var generator = new CpuClassFileKernelGenerator();
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
            for (CpuPointwiseOpcode opcode : SELF_CONTAINED_ACTIVATIONS) {
                CpuKernelIr ir = ir(new Case(opcode, type));
                List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types,
                        types.stream().map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList(),
                        0, -1);
                byte[] bytes = generator.generateClassBytes(specialization, ir);
                var model = ClassFile.of().parse(bytes);
                String primitive = type == DataType.FLOAT64 ? "[D" : "[F";
                assertAll(opcode + " " + type,
                        () -> assertEquals("(" + primitive + primitive + "[JJJ)V",
                                model.methods().getFirst().methodTypeSymbol().descriptorString()),
                        () -> assertGeneratedClassShape(model, false));
            }
        }
    }

    @Test void vectorActivationReferencesOnlyChunkLevelVectorMathAndKeepsScalarTailDirect() {
        var generator = new CpuClassFileKernelGenerator();
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
            for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.ERF,
                    CpuPointwiseOpcode.GELU_EXACT)) {
                CpuKernelIr ir = ir(new Case(opcode, type));
                List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types,
                        types.stream().map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList(),
                        vectorSpeciesBits(type), -1);
                var model = ClassFile.of().parse(generator.generateClassBytes(specialization, ir));
                List<MemberRefEntry> projectMembers = java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                        .filter(entry -> entry.owner().asInternalName()
                                .startsWith("io/github/pho001/synaptik")).toList();
                String vectorDescriptor = type == DataType.FLOAT64
                        ? "Ljdk/incubator/vector/DoubleVector;"
                        : "Ljdk/incubator/vector/FloatVector;";
                assertAll(opcode + " " + type,
                        () -> assertFalse(projectMembers.isEmpty()),
                        () -> assertTrue(projectMembers.stream().allMatch(entry -> entry.owner()
                                .asInternalName().equals("io/github/pho001/synaptik/backend/cpu/"
                                        + "internal/codegen/emit/CpuVectorMath"))),
                        () -> assertTrue(projectMembers.stream().allMatch(entry -> entry.type()
                                .stringValue().equals("(" + vectorDescriptor + ")"
                                        + vectorDescriptor))),
                        () -> assertGeneratedClassShape(model, true));
            }
        }
    }

    @Test void coveredActivationsPreserveRawBitsAcrossExceptionalAndFiniteInputs() throws Throwable {
        long[] doubleBits = {0L, Long.MIN_VALUE, 1L, Long.MIN_VALUE | 1L,
                0x7ff0000000000000L, 0xfff0000000000000L, 0x7ff0000000000042L,
                0xfff8000000000042L, 0x7fefffffffffffffL, 0xffefffffffffffffL,
                Double.doubleToRawLongBits(1.0), Double.doubleToRawLongBits(-1.0)};
        int[] floatBits = {0, Integer.MIN_VALUE, 1, Integer.MIN_VALUE | 1,
                0x7f800000, 0xff800000, 0x7f800042, 0xffc00042, 0x7f7fffff,
                0xff7fffff, Float.floatToRawIntBits(1.0f), Float.floatToRawIntBits(-1.0f)};
        for (CpuPointwiseOpcode opcode : SELF_CONTAINED_ACTIVATIONS) {
            double[] doubles = Arrays.stream(doubleBits).mapToDouble(Double::longBitsToDouble).toArray();
            double[] expectedDoubles = referenceUnary(opcode, doubles);
            double[] actualDoubles = (double[]) invokeUnary(opcode, DataType.FLOAT64, doubles);
            assertArrayEquals(Arrays.stream(expectedDoubles).mapToLong(Double::doubleToRawLongBits)
                    .toArray(), Arrays.stream(actualDoubles).mapToLong(Double::doubleToRawLongBits)
                    .toArray(), opcode + " FLOAT64 raw bits");
            float[] floats = new float[floatBits.length];
            for (int index = 0; index < floats.length; index++)
                floats[index] = Float.intBitsToFloat(floatBits[index]);
            float[] expectedFloats = referenceUnary(opcode, floats);
            float[] actualFloats = (float[]) invokeUnary(opcode, DataType.FLOAT32, floats);
            int[] expectedBits = new int[floats.length], actualBits = new int[floats.length];
            for (int index = 0; index < floats.length; index++) {
                expectedBits[index] = Float.floatToRawIntBits(expectedFloats[index]);
                actualBits[index] = Float.floatToRawIntBits(actualFloats[index]);
            }
            assertArrayEquals(expectedBits, actualBits, opcode + " FLOAT32 raw bits");
        }
    }

    @Test void denseHeapArrayScalarAndVectorBodiesUseOnlyEntryNarrowingAndOneSpeciesLoad() {
        CpuKernelIr ir = ir(new Case(CpuPointwiseOpcode.ADD, DataType.FLOAT64));
        List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
        var scalar = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
        var vector = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                DoubleVector.SPECIES_PREFERRED.vectorBitSize(), -1);
        var segments = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types,
                Collections.nCopies(types.size(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                0, -1);
        var generator = new CpuClassFileKernelGenerator();
        var scalarCode = ClassFile.of().parse(generator.generateClassBytes(scalar, ir))
                .methods().getFirst().code().orElseThrow();
        var vectorCode = ClassFile.of().parse(generator.generateClassBytes(vector, ir))
                .methods().getFirst().code().orElseThrow();
        long scalarNarrowing = scalarCode.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(i -> i.opcode() == Opcode.L2I).count();
        long vectorNarrowing = vectorCode.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(i -> i.opcode() == Opcode.L2I).count();
        long speciesLoads = vectorCode.elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(i -> i.opcode() == Opcode.GETSTATIC).count();
        assertAll(() -> assertEquals(types.size() + 2L, scalarNarrowing),
                () -> assertEquals(types.size() + 2L, vectorNarrowing),
                () -> assertEquals(CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT,
                        scalar.loopAddressing(ir)),
                () -> assertEquals(CpuKernelSpecialization.LoopAddressing.GENERAL_LONG,
                        segments.loopAddressing(ir)),
                () -> assertEquals(1, speciesLoads),
                () -> assertTrue(vectorCode.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).anyMatch(i -> i.opcode() == Opcode.IREM)));
    }

    @Test void generatedAndReferenceAgreeForEveryAdmittedOpcodeAndType() throws Throwable {
        var cases = new ArrayList<Case>();
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.ADD, CpuPointwiseOpcode.SUB,
                CpuPointwiseOpcode.MUL)) for (DataType type : numericTypes()) cases.add(new Case(opcode, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32))
            cases.add(new Case(CpuPointwiseOpcode.DIV, type));
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.MIN, CpuPointwiseOpcode.MAX))
            for (DataType type : numericTypes()) cases.add(new Case(opcode, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32))
            cases.add(new Case(CpuPointwiseOpcode.POW, type));
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SCALAR_ADD,
                CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL))
            for (DataType type : numericTypes()) cases.add(new Case(opcode, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32))
            cases.add(new Case(CpuPointwiseOpcode.SCALAR_DIV, type));
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SCALAR_MIN,
                CpuPointwiseOpcode.SCALAR_MAX)) for (DataType type : numericTypes())
            cases.add(new Case(opcode, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32))
            cases.add(new Case(CpuPointwiseOpcode.SCALAR_CLAMP, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
            for (CpuPointwiseOpcode opcode : CpuPointwiseOpcode.values())
                if (opcode.family() == CpuPointwiseOpcode.Family.UNARY) cases.add(new Case(opcode, type));
            for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.IS_FINITE,
                    CpuPointwiseOpcode.IS_NAN, CpuPointwiseOpcode.IS_INF)) cases.add(new Case(opcode, type));
        }
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.GREATER_THAN,
                CpuPointwiseOpcode.GREATER_OR_EQUAL, CpuPointwiseOpcode.LESS_THAN,
                CpuPointwiseOpcode.LESS_OR_EQUAL, CpuPointwiseOpcode.EQUAL,
                CpuPointwiseOpcode.NOT_EQUAL)) for (DataType type : numericTypes())
            cases.add(new Case(opcode, type));
        cases.add(new Case(CpuPointwiseOpcode.WHERE, DataType.FLOAT64));
        cases.add(new Case(CpuPointwiseOpcode.WHERE, DataType.FLOAT32));
        cases.add(new Case(CpuPointwiseOpcode.LOGICAL_AND, DataType.BOOL));
        cases.add(new Case(CpuPointwiseOpcode.LOGICAL_OR, DataType.BOOL));
        cases.add(new Case(CpuPointwiseOpcode.LOGICAL_NOT, DataType.BOOL));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                DataType.INT64, DataType.BOOL)) cases.add(new Case(CpuPointwiseOpcode.CAST, type));

        for (Case one : cases) assertCase(one);
        assertEquals(126, cases.size());
    }

    @Test void float64NumericSubsetUsesVectorBodiesWithScalarTails() throws Throwable {
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SUB,
                CpuPointwiseOpcode.SCALAR_ADD, CpuPointwiseOpcode.SCALAR_SUB,
                CpuPointwiseOpcode.SCALAR_MUL, CpuPointwiseOpcode.DIV,
                CpuPointwiseOpcode.SCALAR_DIV, CpuPointwiseOpcode.NEG,
                CpuPointwiseOpcode.MIN, CpuPointwiseOpcode.MAX,
                CpuPointwiseOpcode.SCALAR_MIN, CpuPointwiseOpcode.SCALAR_MAX,
                CpuPointwiseOpcode.SCALAR_CLAMP,
                CpuPointwiseOpcode.ABS, CpuPointwiseOpcode.RECIPROCAL,
                CpuPointwiseOpcode.LOG, CpuPointwiseOpcode.LOG1P, CpuPointwiseOpcode.EXP,
                CpuPointwiseOpcode.EXPM1, CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.SQRT,
                CpuPointwiseOpcode.RSQRT, CpuPointwiseOpcode.SIGN, CpuPointwiseOpcode.RELU,
                CpuPointwiseOpcode.TANH, CpuPointwiseOpcode.GELU_EXACT,
                CpuPointwiseOpcode.CAST)) {
            Case one = new Case(opcode, DataType.FLOAT64);
            CpuKernelIr ir = ir(one);
            List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
            List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                    .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
            var specialization = new CpuKernelSpecialization(
                    CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                    DoubleVector.SPECIES_PREFERRED.vectorBitSize(), -1);
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(specialization,
                    generator.generateClassBytes(specialization, ir));
            int count = DoubleVector.SPECIES_PREFERRED.length() + 1;
            var generated = new ArrayList<Object>();
            for (Object input : inputs(one)) generated.add(Arrays.copyOf((double[]) input, count));
            double[] output = new double[count]; generated.add(output);
            var call = new ArrayList<>(generated); call.add(geometry(generated.size(), count));
            call.add(0L); call.add((long) count);
            artifact.entryPoint().invokeWithArguments(call);
            double[] expected = new double[count];
            var reference = new ArrayList<Object>(generated.subList(0, generated.size() - 1));
            reference.add(expected);
            CpuScalarReferenceKernel.execute(ir, reference.stream().map(value -> argument(value,
                    DataType.FLOAT64)).toList(), bindings(generated.size(), count), 0, count);
            assertArrayEquals(expected, output, opcode.name());
        }
    }

    @Test void float32EligibleOpcodeSetUsesPreferredVectorsWithScalarTails() throws Throwable {
        for (CpuPointwiseOpcode opcode : Arrays.stream(CpuPointwiseOpcode.values())
                .filter(candidate -> candidate.vectorForm() == CpuPointwiseOpcode.VectorForm.VALUE)
                .filter(candidate -> candidate != CpuPointwiseOpcode.SCALAR_POW).toList()) {
            Case one = new Case(opcode, DataType.FLOAT32);
            CpuKernelIr ir = ir(one);
            List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
            List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                    .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
            var specialization = new CpuKernelSpecialization(
                    CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                    FloatVector.SPECIES_PREFERRED.vectorBitSize(), -1);
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(specialization,
                    generator.generateClassBytes(specialization, ir));
            int count = FloatVector.SPECIES_PREFERRED.length() + 1;
            var generated = new ArrayList<Object>();
            for (Object input : inputs(one)) generated.add(Arrays.copyOf((float[]) input, count));
            float[] output = new float[count]; generated.add(output);
            var call = new ArrayList<>(generated); call.add(geometry(generated.size(), count));
            call.add(0L); call.add((long) count);
            artifact.entryPoint().invokeWithArguments(call);
            float[] expected = new float[count];
            var reference = new ArrayList<Object>(generated.subList(0, generated.size() - 1));
            reference.add(expected);
            CpuScalarReferenceKernel.execute(ir, reference.stream().map(value -> argument(value,
                    DataType.FLOAT32)).toList(), bindings(generated.size(), count), 0, count);
            for (int lane = 0; lane < count; lane++) assertFloatVectorResult(
                    opcode, expected[lane], output[lane]);
        }
    }

    @Test void float32VectorFusionLengthsOneThroughEightKeepVirtualIntermediates() throws Throwable {
        int count = FloatVector.SPECIES_PREFERRED.length() + 1;
        for (int length = 1; length <= 8; length++) {
            var values = new ArrayList<CpuKernelIr.Value>();
            values.add(new CpuKernelIr.Value(0, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT,
                    dense(CpuAccessPlan.AccessKind.READ)));
            for (int ordinal = 1; ordinal <= length; ordinal++) values.add(new CpuKernelIr.Value(
                    ordinal, DataType.FLOAT32, ordinal == length ? CpuKernelIr.Value.Kind.OUTPUT
                            : CpuKernelIr.Value.Kind.VIRTUAL,
                    dense(ordinal == length ? CpuAccessPlan.AccessKind.WRITE
                            : CpuAccessPlan.AccessKind.READ)));
            var instructions = new ArrayList<CpuKernelIr.Instruction>();
            for (int ordinal = 1; ordinal <= length; ordinal++) instructions.add(
                    new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                            List.of(ordinal - 1), ordinal));
            CpuKernelIr ir = new CpuKernelIr(values, instructions,
                    new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(length, 0)));
            var specialization = new CpuKernelSpecialization(
                    CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                    List.of(DataType.FLOAT32, DataType.FLOAT32),
                    List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                            CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY),
                    FloatVector.SPECIES_PREFERRED.vectorBitSize(), -1);
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(specialization,
                    generator.generateClassBytes(specialization, ir));
            float[] input = new float[count], output = new float[count];
            for (int i = 0; i < count; i++) input[i] = i - 3.25f;
            artifact.entryPoint().invokeWithArguments(input, output, geometry(2, count),
                    0L, (long) count);
            for (int i = 0; i < count; i++) assertEquals(length % 2 == 0 ? input[i] : -input[i],
                    output[i], "fusion length " + length + " lane " + i);
        }
    }

    @Test void integralAndCanonicalBoolRowsUseExactPreferredVectorsAndScalarTails()
            throws Throwable {
        var cases = new ArrayList<Case>();
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.ADD,
                    CpuPointwiseOpcode.SUB, CpuPointwiseOpcode.MUL, CpuPointwiseOpcode.MIN,
                    CpuPointwiseOpcode.MAX, CpuPointwiseOpcode.SCALAR_ADD,
                    CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL,
                    CpuPointwiseOpcode.SCALAR_MIN, CpuPointwiseOpcode.SCALAR_MAX,
                    CpuPointwiseOpcode.CAST)) cases.add(new Case(opcode, type));
        }
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.LOGICAL_AND,
                CpuPointwiseOpcode.LOGICAL_OR, CpuPointwiseOpcode.LOGICAL_NOT,
                CpuPointwiseOpcode.CAST)) cases.add(new Case(opcode, DataType.BOOL));
        for (Case one : cases) assertVectorCase(one);
        assertEquals(26, cases.size());
    }

    @Test void virtualFloatingMasksComposeAndDriveWhereWithoutMaterialization() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int lanes = vectorLanes(type);
            for (boolean or : List.of(false, true)) {
                CpuKernelIr ir = maskIr(type, or);
                Object a = repeated(type, lanes + 1, -2, 3, Double.NaN, -0.0);
                Object b = repeated(type, lanes + 1, -1, 2, 0, +0.0);
                Object c = repeated(type, lanes + 1, 10, 20, 30, 40);
                Object d = repeated(type, lanes + 1, 11, 19, 31, 39);
                Object output = array(type, lanes + 1, false);
                invokeVector(ir, List.of(a, b, c, d, output), type, lanes + 1);
                for (int i = 0; i < lanes + 1; i++) {
                    double av = numberAt(a, i), bv = numberAt(b, i);
                    double cv = numberAt(c, i), dv = numberAt(d, i);
                    boolean condition = or ? av > bv || cv < dv : av > bv && cv < dv;
                    double expected = condition ? av : cv;
                    assertEquals(expected, numberAt(output, i),
                            type + " " + (or ? "OR" : "AND") + " mask lane " + i);
                }
            }
        }
    }

    @Test void virtualFloatingClassificationAndNotDriveWhere() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int count = vectorLanes(type) + 1;
            CpuKernelIr ir = classificationWhereIr(type);
            Object input = repeated(type, count, Double.NaN, Double.POSITIVE_INFINITY, -2, +0.0);
            Object fallback = repeated(type, count, 7, 8, 9, 10);
            Object output = array(type, count, false);
            invokeVector(ir, List.of(input, fallback, output), type, count);
            for (int i = 0; i < count; i++) {
                double value = numberAt(input, i);
                double expected = !Double.isNaN(value) ? value : numberAt(fallback, i);
                assertEquals(expected, numberAt(output, i), type + " classification lane " + i);
            }
        }
    }

    @Test void scalarCanonicalBoolBroadcastDrivesFloatingWhereMask() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int count = vectorLanes(type) + 1;
            for (byte condition : new byte[] {0, 1}) {
                CpuKernelIr ir = scalarWhereIr(type);
                Object whenTrue = repeated(type, count, 1, 2, 3, 4);
                Object whenFalse = repeated(type, count, -1, -2, -3, -4);
                for (boolean segmentCondition : List.of(false, true)) {
                    Object output = array(type, count, false);
                    Object conditionCarrier = segmentCondition
                            ? segmentOf(new byte[] {condition}) : new byte[] {condition};
                    List<CpuKernelSpecialization.CarrierAccess> carriers = List.of(
                            segmentCondition
                                    ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                                    : CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY,
                            heapCarrier(type), heapCarrier(type), heapCarrier(type));
                    invokeVector(ir, List.of(conditionCarrier, whenTrue, whenFalse, output),
                            type, count, carriers);
                    Object expected = condition == 1 ? whenTrue : whenFalse;
                    assertPrimitiveArrayEquals(expected, output,
                            type + " scalar condition segment=" + segmentCondition);
                }
            }
        }
    }

    @Test void generatedScalarPowerMatchesDirectOracleAndEveryProvedPlan() throws Throwable {
        double[] exponents = {+0.0d, -0.0d, 1.0d, 2.0d, -1.0d, 3.0d, -2.0d, 0.5d,
                Double.MIN_VALUE, Double.MAX_VALUE, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.longBitsToDouble(0x7ff8_0000_0000_0001L)};
        double[] bases = {+0.0d, -0.0d, Double.MIN_VALUE, 0x0.ffff_ffff_ffff_fp-1022,
                Double.MIN_NORMAL, -Double.MIN_NORMAL, 0.5d, -0.5d, 1.0d, -1.0d,
                Double.MAX_VALUE, -Double.MAX_VALUE, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.NaN,
                Double.longBitsToDouble(0x7ff8_0000_0000_0042L)};
        for (double exponent : exponents) {
            CpuKernelIr ir = powerIr(DataType.FLOAT64, Double.doubleToRawLongBits(exponent));
            var artifact = powerArtifact(ir, DataType.FLOAT64,
                    CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
            double[] output = new double[bases.length];
            artifact.entryPoint().invokeWithArguments(bases, output, geometry(2, bases.length),
                    0L, (long) bases.length);
            for (int index = 0; index < bases.length; index++) {
                double expected = StrictMath.pow(bases[index], exponent);
                if (Double.isNaN(expected)) assertTrue(Double.isNaN(output[index]));
                else assertEquals(Double.doubleToRawLongBits(expected),
                        Double.doubleToRawLongBits(output[index]),
                        "base=" + bases[index] + " exponent=" + exponent);
            }
        }

        float[] floatExponents = {+0.0f, -0.0f, 1.0f, 2.0f, -1.0f, 3.0f, -2.0f, 0.5f,
                Float.MIN_VALUE, Float.MAX_VALUE, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.intBitsToFloat(0x7fc0_0001)};
        float[] floatBases = {+0.0f, -0.0f, Float.MIN_VALUE, 0x0.fffffep-126f,
                Float.MIN_NORMAL, -Float.MIN_NORMAL, 0.5f, -0.5f, 1.0f, -1.0f,
                Float.MAX_VALUE, -Float.MAX_VALUE, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.NaN, Float.intBitsToFloat(0x7fc0_0042)};
        for (float exponent : floatExponents) {
            CpuKernelIr ir = powerIr(DataType.FLOAT32,
                    Float.floatToRawIntBits(exponent) & 0xffff_ffffL);
            var artifact = powerArtifact(ir, DataType.FLOAT32,
                    CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
            float[] output = new float[floatBases.length];
            artifact.entryPoint().invokeWithArguments(floatBases, output,
                    geometry(2, floatBases.length), 0L, (long) floatBases.length);
            for (int index = 0; index < floatBases.length; index++) {
                float expected = (float) StrictMath.pow((double) floatBases[index],
                        (double) exponent);
                if (Float.isNaN(expected)) assertTrue(Float.isNaN(output[index]));
                else assertEquals(Float.floatToRawIntBits(expected),
                        Float.floatToRawIntBits(output[index]),
                        "base=" + floatBases[index] + " exponent=" + exponent);
            }
        }
    }

    @Test void generatedBinaryAndScalarDivisionMatchPrimitiveOracleEdges() throws Throwable {
        double[] left = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                +0.0d, -0.0d, Double.MIN_VALUE, Double.MIN_NORMAL, Double.MAX_VALUE,
                -Double.MAX_VALUE, 1.0d, -1.0d};
        double[] right = {1.0d, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                +0.0d, -0.0d, 2.0d, Double.MAX_VALUE, Double.MIN_VALUE,
                -Double.MIN_VALUE, 3.0d, -3.0d};
        CpuKernelIr binary = divisionIr(DataType.FLOAT64, false,
                Double.doubleToRawLongBits(0.0d));
        var binaryArtifact = artifact(binary, List.of(DataType.FLOAT64, DataType.FLOAT64,
                DataType.FLOAT64), List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY));
        double[] binaryOutput = new double[left.length];
        binaryArtifact.entryPoint().invokeWithArguments(left, right, binaryOutput,
                geometry(3, left.length), 0L, (long) left.length);
        for (int index = 0; index < left.length; index++) assertFloatingResult(
                left[index] / right[index], binaryOutput[index]);

        float[] input = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                +0.0f, -0.0f, Float.MIN_VALUE, Float.MIN_NORMAL, Float.MAX_VALUE,
                -Float.MAX_VALUE, 1.0f, -1.0f};
        float denominator = -0.0f;
        CpuKernelIr scalar = divisionIr(DataType.FLOAT32, true,
                Float.floatToRawIntBits(denominator) & 0xffff_ffffL);
        var scalarArtifact = artifact(scalar, List.of(DataType.FLOAT32, DataType.FLOAT32),
                List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY));
        float[] scalarOutput = new float[input.length];
        scalarArtifact.entryPoint().invokeWithArguments(input, scalarOutput,
                geometry(2, input.length), 0L, (long) input.length);
        for (int index = 0; index < input.length; index++) assertFloatingResult(
                input[index] / denominator, scalarOutput[index]);
    }

    @Test void float32RsqrtNarrowsOnlyAfterDoublePrecisionSqrtAndReciprocal() throws Throwable {
        float input = Float.intBitsToFloat(0x0000_26f6);
        float intermediateRoot = (float) StrictMath.sqrt((double) input);
        float oldIntermediateNarrowing = 1.0f / intermediateRoot;
        float expected = (float) (1.0d / StrictMath.sqrt((double) input));
        assertAll(
                () -> assertEquals(0x6168_01ad, Float.floatToRawIntBits(oldIntermediateNarrowing)),
                () -> assertEquals(0x6168_01ae, Float.floatToRawIntBits(expected)),
                () -> assertNotEquals(Float.floatToRawIntBits(oldIntermediateNarrowing),
                        Float.floatToRawIntBits(expected)));

        CpuKernelIr ir = ir(new Case(CpuPointwiseOpcode.RSQRT, DataType.FLOAT32));
        var generated = artifact(ir, List.of(DataType.FLOAT32, DataType.FLOAT32),
                List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY));
        float[] output = new float[1];
        generated.entryPoint().invokeWithArguments(new float[] {input}, output,
                geometry(2, 1), 0L, 1L);
        assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(output[0]));
    }

    @Test void extremaClampTensorPowerAndLogicalRowsPreserveExactEdgeSemantics() throws Throwable {
        CpuKernelIr min = ir(new Case(CpuPointwiseOpcode.MIN, DataType.FLOAT64));
        var minArtifact = artifact(min, List.of(DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64),
                java.util.Collections.nCopies(3, CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY));
        double[] left = {-0.0d, +0.0d, Double.NaN, Double.POSITIVE_INFINITY};
        double[] right = {+0.0d, -0.0d, 1.0d, Double.NEGATIVE_INFINITY};
        double[] minimum = new double[4];
        minArtifact.entryPoint().invokeWithArguments(left, right, minimum, geometry(3, 4), 0L, 4L);
        assertAll(
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(minimum[0])),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(minimum[1])),
                () -> assertTrue(Double.isNaN(minimum[2])),
                () -> assertEquals(Double.NEGATIVE_INFINITY, minimum[3]));

        CpuKernelIr clamp = clampIr(DataType.FLOAT64, Double.doubleToRawLongBits(+0.0d),
                Double.doubleToRawLongBits(-0.0d));
        var clampArtifact = artifact(clamp, List.of(DataType.FLOAT64, DataType.FLOAT64),
                java.util.Collections.nCopies(2, CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY));
        double[] clamped = new double[4];
        clampArtifact.entryPoint().invokeWithArguments(
                new double[] {-1.0d, -0.0d, +0.0d, 1.0d}, clamped,
                geometry(2, 4), 0L, 4L);
        for (double value : clamped) assertEquals(Double.doubleToRawLongBits(-0.0d),
                Double.doubleToRawLongBits(value));

        CpuKernelIr power = ir(new Case(CpuPointwiseOpcode.POW, DataType.FLOAT32));
        var powerArtifact = artifact(power, List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                java.util.Collections.nCopies(3, CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY));
        float[] bases = {-2.0f, Float.MIN_VALUE, -0.0f, Float.NaN};
        float[] exponents = {3.0f, 0.5f, -1.0f, 2.0f};
        float[] powered = new float[4];
        powerArtifact.entryPoint().invokeWithArguments(bases, exponents, powered,
                geometry(3, 4), 0L, 4L);
        for (int index = 0; index < powered.length; index++) {
            float expected = (float) StrictMath.pow((double) bases[index], (double) exponents[index]);
            if (Float.isNaN(expected)) assertTrue(Float.isNaN(powered[index]));
            else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(powered[index]));
        }

        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.LOGICAL_AND,
                CpuPointwiseOpcode.LOGICAL_OR, CpuPointwiseOpcode.LOGICAL_NOT)) {
            CpuKernelIr logical = ir(new Case(opcode, DataType.BOOL));
            var logicalArtifact = artifact(logical,
                    java.util.Collections.nCopies(opcode.arity() + 1, DataType.BOOL),
                    java.util.Collections.nCopies(opcode.arity() + 1,
                            CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY));
            byte[] first = {0, 1, 1, 0};
            byte[] output = new byte[4];
            if (opcode.arity() == 2) logicalArtifact.entryPoint().invokeWithArguments(first,
                    new byte[] {0, 0, 1, 1}, output, geometry(3, 4), 0L, 4L);
            else logicalArtifact.entryPoint().invokeWithArguments(first, output,
                    geometry(2, 4), 0L, 4L);
            for (byte value : output) assertTrue(value == 0 || value == 1);
        }
    }

    @Test void zeroPowerDoesNotReadTheBaseAndSpecialPowerVectorizes() throws Throwable {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        CpuKernelIr zero = powerIr(DataType.FLOAT64, Double.doubleToRawLongBits(-0.0d));
        var scalar = powerArtifact(zero, DataType.FLOAT64,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
        double[] output = new double[3];
        scalar.entryPoint().invokeWithArguments(new double[0], output, geometry(2, 3), 0L, 3L);
        assertArrayEquals(new double[] {1.0d, 1.0d, 1.0d}, output);

        for (double exponent : List.of(+0.0d, 1.0d, 2.0d, -1.0d)) {
            CpuKernelIr ir = powerIr(DataType.FLOAT64, Double.doubleToRawLongBits(exponent));
            var vector = powerArtifact(ir, DataType.FLOAT64,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
            double[] input = new double[lanes + 1];
            for (int i = 0; i < input.length; i++) input[i] = i - 2.0d;
            double[] generated = new double[input.length];
            vector.entryPoint().invokeWithArguments(input, generated,
                    geometry(2, input.length), 0L, (long) input.length);
            for (int i = 0; i < input.length; i++) {
                double expected = StrictMath.pow(input[i], exponent);
                if (Double.isNaN(expected)) assertTrue(Double.isNaN(generated[i]));
                else assertEquals(Double.doubleToRawLongBits(expected),
                        Double.doubleToRawLongBits(generated[i]));
            }
        }
        int floatLanes = FloatVector.SPECIES_PREFERRED.length();
        for (float exponent : List.of(+0.0f, 1.0f, 2.0f, -1.0f)) {
            CpuKernelIr ir = powerIr(DataType.FLOAT32,
                    Float.floatToRawIntBits(exponent) & 0xffff_ffffL);
            var vector = powerArtifact(ir, DataType.FLOAT32,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR);
            float[] input = new float[floatLanes + 1];
            for (int i = 0; i < input.length; i++) input[i] = i - 2.0f;
            float[] generated = new float[input.length];
            vector.entryPoint().invokeWithArguments(input, generated,
                    geometry(2, input.length), 0L, (long) input.length);
            for (int i = 0; i < input.length; i++) assertFloatingResult(
                    (float) StrictMath.pow(input[i], exponent), generated[i]);
        }
    }

    @Test void exactSegmentsLoadAndStoreEveryAdmittedCarrierWidth() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                    DataType.INT64, DataType.BOOL)) {
                Case one = new Case(CpuPointwiseOpcode.CAST, type);
                CpuKernelIr ir = ir(one);
                List<DataType> types = List.of(type, type);
                List<CpuKernelSpecialization.CarrierAccess> carriers = List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
                var generator = new CpuClassFileKernelGenerator();
                var artifact = generator.defineClassBytes(specialization,
                        generator.generateClassBytes(specialization, ir));
                var input = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                var output = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                writeSegment(type, input);
                artifact.entryPoint().invokeWithArguments(input, output, geometry(2, 4), 0L, 4L);
                assertEquals(input.mismatch(output), -1L, type.name());

                Object heapInput = arrayValues(type, false);
                var mixedToSegment = artifact(ir, types, List.of(heapCarrier(type),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT));
                output.fill((byte) 0);
                mixedToSegment.entryPoint().invokeWithArguments(heapInput, output,
                        geometry(2, 4), 0L, 4L);
                assertEquals(-1L, segmentOf(heapInput).mismatch(output), type + " heap-to-segment");

                Object heapOutput = array(type, 4, false);
                var mixedToHeap = artifact(ir, types, List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, heapCarrier(type)));
                mixedToHeap.entryPoint().invokeWithArguments(input, heapOutput,
                        geometry(2, 4), 0L, 4L);
                assertEquals(-1L, input.mismatch(segmentOf(heapOutput)), type + " segment-to-heap");
            }
        }
    }

    @Test void preferredTypedVectorsUseSegmentAndMixedCarriers() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                    DataType.INT64, DataType.BOOL)) {
                int count = vectorLanes(type) + 1;
                Case one = new Case(CpuPointwiseOpcode.CAST, type);
                CpuKernelIr ir = ir(one);
                List<DataType> types = List.of(type, type);
                Object source = resize(arrayValues(type, false), count);
                var input = arena.allocate((long) count * type.byteWidth(), type.byteWidth());
                var output = arena.allocate((long) count * type.byteWidth(), type.byteWidth());
                input.copyFrom(segmentOf(source));

                var allSegment = vectorArtifact(ir, types, List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT), type);
                allSegment.entryPoint().invokeWithArguments(input, output,
                        geometry(2, count), 0L, (long) count);
                assertEquals(-1L, input.mismatch(output), type + " vector segments");

                output.fill((byte) 0);
                var mixed = vectorArtifact(ir, types, List.of(heapCarrier(type),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT), type);
                mixed.entryPoint().invokeWithArguments(source, output,
                        geometry(2, count), 0L, (long) count);
                assertEquals(-1L, segmentOf(source).mismatch(output), type + " vector mixed");
            }
        }
    }

    private static CpuGeneratedKernel artifact(CpuKernelIr ir, List<DataType> types,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static Object invokeUnary(CpuPointwiseOpcode opcode, DataType type, Object input)
            throws Throwable {
        CpuKernelIr ir = ir(new Case(opcode, type));
        int count = java.lang.reflect.Array.getLength(input);
        List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
        Object output = array(type, count, false);
        var call = new ArrayList<Object>(); call.add(input); call.add(output);
        call.add(geometry(types.size(), count)); call.add(0L); call.add((long) count);
        artifact(ir, types, types.stream().map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList())
                .entryPoint().invokeWithArguments(call);
        return output;
    }

    private static double[] referenceUnary(CpuPointwiseOpcode opcode, double[] input) {
        double[] result = new double[input.length];
        for (int index = 0; index < input.length; index++) result[index] = switch (opcode) {
            case ERF -> CpuScalarReferenceKernel.erf(input[index]);
            case SIGMOID -> CpuScalarReferenceKernel.sigmoid(input[index]);
            case GELU_EXACT -> CpuScalarReferenceKernel.gelu(input[index]);
            case GELU_TANH_APPROXIMATION -> CpuScalarReferenceKernel.geluTanhApproximation(input[index]);
            case SILU -> CpuScalarReferenceKernel.silu(input[index]);
            default -> throw new AssertionError(opcode);
        };
        return result;
    }

    private static float[] referenceUnary(CpuPointwiseOpcode opcode, float[] input) {
        float[] result = new float[input.length];
        for (int index = 0; index < input.length; index++) result[index] = (float) switch (opcode) {
            case ERF -> CpuScalarReferenceKernel.erf(input[index]);
            case SIGMOID -> CpuScalarReferenceKernel.sigmoid(input[index]);
            case GELU_EXACT -> CpuScalarReferenceKernel.gelu(input[index]);
            case GELU_TANH_APPROXIMATION -> CpuScalarReferenceKernel.geluTanhApproximation(input[index]);
            case SILU -> CpuScalarReferenceKernel.silu(input[index]);
            default -> throw new AssertionError(opcode);
        };
        return result;
    }

    private static void assertGeneratedClassShape(java.lang.classfile.ClassModel model,
            boolean allowVectorMath) {
        List<MemberRefEntry> members = java.util.stream.StreamSupport.stream(
                model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(members.stream().filter(entry -> entry.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik")).allMatch(entry -> allowVectorMath
                                && entry.owner().asInternalName().endsWith("/CpuVectorMath"))),
                () -> assertTrue(model.methods().stream().noneMatch(method -> method
                        .methodTypeSymbol().descriptorString().contains("Ljava/lang/Object;"))),
                () -> assertTrue(members.stream().noneMatch(entry -> entry.type().stringValue()
                        .contains("Ljava/lang/Object;") || entry.owner().asInternalName()
                                .startsWith("java/lang/reflect/") || entry.owner().asInternalName()
                                .startsWith("java/util/"))),
                () -> assertTrue(model.methods().stream().flatMap(method -> method.code().stream())
                        .flatMap(code -> code.elementStream()).noneMatch(
                                java.lang.classfile.instruction.NewObjectInstruction.class::isInstance)));
    }

    private static CpuGeneratedKernel vectorArtifact(CpuKernelIr ir, List<DataType> types,
            List<CpuKernelSpecialization.CarrierAccess> carriers, DataType laneType) {
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                vectorSpeciesBits(laneType), -1);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static void assertVectorCase(Case one) throws Throwable {
        CpuKernelIr ir = ir(one);
        int count = vectorLanes(one.type()) + 1;
        var arguments = new ArrayList<Object>();
        for (Object input : inputs(one)) arguments.add(resize(input, count));
        Object output = array(one.outputType(), count, false);
        arguments.add(output);
        invokeVector(ir, arguments, one.type(), count);
        Object expected = array(one.outputType(), count, false);
        var reference = new ArrayList<>(arguments.subList(0, arguments.size() - 1));
        reference.add(expected);
        List<DataType> types = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        CpuScalarReferenceKernel.execute(ir, reference.stream().map(value ->
                argument(value, types.get(reference.indexOf(value)))).toList(),
                bindings(types.size(), count), 0, count);
        assertPrimitiveArrayEquals(expected, output, one.opcode() + " " + one.type());
    }

    private static void invokeVector(CpuKernelIr ir, List<Object> arguments, DataType laneType,
            int count) throws Throwable {
        List<DataType> types = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
        invokeVector(ir, arguments, laneType, count, carriers);
    }

    private static void invokeVector(CpuKernelIr ir, List<Object> arguments, DataType laneType,
            int count, List<CpuKernelSpecialization.CarrierAccess> carriers) throws Throwable {
        List<DataType> types = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                vectorSpeciesBits(laneType), -1);
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
        var call = new ArrayList<Object>(arguments);
        call.add(geometry(arguments.size(), count)); call.add(0L); call.add((long) count);
        artifact.entryPoint().invokeWithArguments(call);
    }

    private static CpuKernelIr maskIr(DataType type, boolean or) {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < 4; i++) values.add(new CpuKernelIr.Value(i, type,
                CpuKernelIr.Value.Kind.INPUT, dense(CpuAccessPlan.AccessKind.READ)));
        for (int i = 4; i < 7; i++) values.add(new CpuKernelIr.Value(i, DataType.BOOL,
                CpuKernelIr.Value.Kind.VIRTUAL, dense(CpuAccessPlan.AccessKind.READ)));
        values.add(new CpuKernelIr.Value(7, type, CpuKernelIr.Value.Kind.OUTPUT,
                dense(CpuAccessPlan.AccessKind.WRITE)));
        return new CpuKernelIr(values, List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.GREATER_THAN, List.of(0, 1), 4),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.LESS_THAN, List.of(2, 3), 5),
                new CpuKernelIr.Instruction(or ? CpuPointwiseOpcode.LOGICAL_OR
                        : CpuPointwiseOpcode.LOGICAL_AND, List.of(4, 5), 6),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE, List.of(6, 0, 2), 7)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(7, 0)));
    }

    private static CpuKernelIr scalarWhereIr(DataType type) {
        CpuAccessPlan scalar = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.SCALAR_ALL_ZERO, 1,
                List.of(CpuAccessPlan.AxisRole.BROADCAST), 0);
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.BOOL, CpuKernelIr.Value.Kind.INPUT, scalar),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(2, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(3, type, CpuKernelIr.Value.Kind.OUTPUT,
                        dense(CpuAccessPlan.AccessKind.WRITE))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                        List.of(0, 1, 2), 3)), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(3, 0)));
    }

    private static CpuKernelIr classificationWhereIr(DataType type) {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(2, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(3, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(4, type, CpuKernelIr.Value.Kind.OUTPUT,
                        dense(CpuAccessPlan.AccessKind.WRITE))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.IS_NAN, List.of(0), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.LOGICAL_NOT,
                                List.of(2), 3),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                                List.of(3, 0, 1), 4)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(4, 0)));
    }

    private static int vectorLanes(DataType type) {
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case INT32 -> IntVector.SPECIES_PREFERRED.length();
            case INT64 -> LongVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            default -> throw new IllegalArgumentException("unsupported vector type");
        };
    }

    private static int vectorSpeciesBits(DataType type) {
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> LongVector.SPECIES_PREFERRED.vectorBitSize();
            case BOOL -> ByteVector.SPECIES_PREFERRED.vectorBitSize();
            default -> throw new IllegalArgumentException("unsupported vector type");
        };
    }

    private static Object resize(Object source, int count) {
        if (source instanceof double[] value) return Arrays.copyOf(value, count);
        if (source instanceof float[] value) return Arrays.copyOf(value, count);
        if (source instanceof int[] value) return Arrays.copyOf(value, count);
        if (source instanceof long[] value) return Arrays.copyOf(value, count);
        return Arrays.copyOf((byte[]) source, count);
    }

    private static Object repeated(DataType type, int count, double... values) {
        Object result = array(type, count, false);
        for (int i = 0; i < count; i++) {
            double value = values[i % values.length];
            if (result instanceof double[] array) array[i] = value;
            else ((float[]) result)[i] = (float) value;
        }
        return result;
    }

    private static double numberAt(Object array, int index) {
        return array instanceof double[] values ? values[index] : ((float[]) array)[index];
    }

    private static CpuGeneratedKernel powerArtifact(CpuKernelIr ir, DataType type,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy) {
        int species = strategy.compute()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                ? type == DataType.FLOAT32 ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
                        : DoubleVector.SPECIES_PREFERRED.vectorBitSize() : 0;
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT, strategy,
                List.of(type, type), List.of(heapCarrier(type), heapCarrier(type)), species, -1,
                List.of(ir.instructions().getFirst().powerRealization()));
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static CpuKernelIr powerIr(DataType type, long exponentBits) {
        CpuKernelIr.ScalarImmediate exponent = new CpuKernelIr.ScalarImmediate(type, exponentBits);
        CpuKernelIr.PowerRealization realization = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScalarPowerAnalysis()
                .analyze(exponent);
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.OUTPUT,
                        dense(CpuAccessPlan.AccessKind.WRITE))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_POW,
                        List.of(0), 1, exponent, realization)), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)));
    }

    private static CpuKernelIr divisionIr(DataType type, boolean scalar, long scalarBits) {
        int inputCount = scalar ? 1 : 2;
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int index = 0; index < inputCount; index++) values.add(new CpuKernelIr.Value(index,
                type, CpuKernelIr.Value.Kind.INPUT, dense(CpuAccessPlan.AccessKind.READ)));
        values.add(new CpuKernelIr.Value(inputCount, type, CpuKernelIr.Value.Kind.OUTPUT,
                dense(CpuAccessPlan.AccessKind.WRITE)));
        return new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(
                scalar ? CpuPointwiseOpcode.SCALAR_DIV : CpuPointwiseOpcode.DIV,
                java.util.stream.IntStream.range(0, inputCount).boxed().toList(), inputCount,
                scalar ? new CpuKernelIr.ScalarImmediate(type, scalarBits) : null)),
                new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(inputCount, 0)));
    }

    private static CpuKernelIr clampIr(DataType type, long lower, long upper) {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT,
                        dense(CpuAccessPlan.AccessKind.READ)),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.OUTPUT,
                        dense(CpuAccessPlan.AccessKind.WRITE))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.SCALAR_CLAMP,
                        List.of(0), 1, new CpuKernelIr.ClampImmediate(
                                new CpuKernelIr.ScalarImmediate(type, lower),
                                new CpuKernelIr.ScalarImmediate(type, upper)))),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
    }

    private static void assertFloatingResult(double expected, double actual) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
        else assertEquals(Double.doubleToRawLongBits(expected),
                Double.doubleToRawLongBits(actual));
    }

    private static void assertFloatingResult(float expected, float actual) {
        if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual));
        else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }

    private static void assertFloatVectorResult(CpuPointwiseOpcode opcode, float expected,
            float actual) {
        if (Float.isNaN(expected)) { assertTrue(Float.isNaN(actual), opcode.name()); return; }
        if (expected == 0.0f || Float.isInfinite(expected)) {
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual),
                    opcode.name());
            return;
        }
        if (opcode == CpuPointwiseOpcode.ERF || opcode == CpuPointwiseOpcode.GELU_EXACT) {
            assertEquals(expected, actual, Math.max(2e-5f, 2e-5f * Math.abs(expected)),
                    opcode.name());
            return;
        }
        int allowedUlps = switch (opcode) {
            case TANH -> 5;
            case LOG, LOG1P, EXP, EXPM1, RSQRT -> 2;
            case RECIPROCAL, SQRT -> 1;
            default -> 0;
        };
        assertTrue(floatUlpDistance(expected, actual) <= allowedUlps,
                () -> opcode + " expected=" + expected + " actual=" + actual);
    }

    private static long floatUlpDistance(float left, float right) {
        int a = Float.floatToRawIntBits(left);
        int b = Float.floatToRawIntBits(right);
        long orderedA = a < 0 ? 0x8000_0000L - a : 0x8000_0000L + a;
        long orderedB = b < 0 ? 0x8000_0000L - b : 0x8000_0000L + b;
        return Math.abs(orderedA - orderedB);
    }

    private static java.lang.foreign.MemorySegment segmentOf(Object array) {
        if (array instanceof double[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof float[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof int[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof long[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        return java.lang.foreign.MemorySegment.ofArray((byte[]) array);
    }

    private static void writeSegment(DataType type, java.lang.foreign.MemorySegment segment) {
        for (int i = 0; i < 4; i++) {
            long offset = (long) i * type.byteWidth();
            switch (type) {
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i - 1.5);
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i - 1.5f);
                case INT32 -> segment.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i * 17);
                case INT64 -> segment.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (long) i * Long.MAX_VALUE);
                case BOOL -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) (i & 1));
                default -> throw new IllegalArgumentException("unsupported segment type");
            }
        }
    }

    private static void assertCase(Case one) throws Throwable {
        CpuKernelIr ir = ir(one);
        List<CpuKernelIr.Value> boundaries = ir.values();
        List<DataType> types = boundaries.stream().map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(specialization,
                new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir));
        List<Object> inputArrays = inputs(one);
        Object generatedOutput = array(one.outputType(), 4, false);
        Object referenceOutput = array(one.outputType(), 4, false);
        List<Object> generated = new ArrayList<>(inputArrays); generated.add(generatedOutput);
        List<Object> reference = new ArrayList<>(inputArrays); reference.add(referenceOutput);
        long[] geometry = geometry(boundaries.size(), 4);
        var arguments = new ArrayList<Object>(generated);
        arguments.add(geometry); arguments.add(0L); arguments.add(4L);
        artifact.entryPoint().invokeWithArguments(arguments);
        CpuScalarReferenceKernel.execute(ir, reference.stream().map((Object value) -> argument(value,
                types.get(reference.indexOf(value)))).toList(), bindings(boundaries.size()), 0, 4);
        assertPrimitiveArrayEquals(referenceOutput, generatedOutput,
                one.opcode() + " " + one.type());
    }

    private static CpuKernelIr ir(Case one) {
        List<DataType> inputs = one.inputTypes();
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputs.size(); i++) values.add(new CpuKernelIr.Value(i, inputs.get(i),
                CpuKernelIr.Value.Kind.INPUT, dense(CpuAccessPlan.AccessKind.READ)));
        int output = inputs.size();
        values.add(new CpuKernelIr.Value(output, one.outputType(), CpuKernelIr.Value.Kind.OUTPUT,
                dense(CpuAccessPlan.AccessKind.WRITE)));
        CpuKernelIr.ScalarImmediate immediate = one.opcode().carriesScalarImmediate()
                ? immediate(one.type()) : null;
        CpuKernelIr.ClampImmediate clamp = one.opcode() == CpuPointwiseOpcode.SCALAR_CLAMP
                ? new CpuKernelIr.ClampImmediate(
                        new CpuKernelIr.ScalarImmediate(one.type(), negativeZero(one.type())),
                        new CpuKernelIr.ScalarImmediate(one.type(), positiveZero(one.type()))) : null;
        return new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(one.opcode(),
                java.util.stream.IntStream.range(0, inputs.size()).boxed().toList(), output,
                immediate, null, clamp)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(output, 0)));
    }

    private static List<Object> inputs(Case one) {
        if (one.opcode() == CpuPointwiseOpcode.WHERE) return List.of(
                new byte[] {1, 0, 1, 0}, arrayValues(one.type(), false), arrayValues(one.type(), true));
        var result = new ArrayList<Object>();
        for (int i = 0; i < one.opcode().arity(); i++) result.add(arrayValues(one.type(), i == 1));
        return result;
    }

    private static Object arrayValues(DataType type, boolean second) {
        return switch (type) {
            case FLOAT64 -> second ? new double[] {-0.0, Double.NaN, 3, -4}
                    : new double[] {0.0, Double.POSITIVE_INFINITY, -2, Double.MIN_VALUE};
            case FLOAT32 -> second ? new float[] {-0.0f, Float.NaN, 3, -4}
                    : new float[] {0.0f, Float.POSITIVE_INFINITY, -2, Float.MIN_VALUE};
            case INT32 -> second ? new int[] {1, -1, Integer.MAX_VALUE, 3}
                    : new int[] {Integer.MAX_VALUE, Integer.MIN_VALUE, -2, 7};
            case INT64 -> second ? new long[] {1, -1, Long.MAX_VALUE, 3}
                    : new long[] {Long.MAX_VALUE, Long.MIN_VALUE, -2, 7};
            case BOOL -> new byte[] {0, 1, 1, 0};
            default -> throw new IllegalArgumentException("unsupported test type");
        };
    }

    private static Object array(DataType type, int size, boolean ignored) {
        return switch (type) {
            case FLOAT64 -> new double[size]; case FLOAT32 -> new float[size];
            case INT32 -> new int[size]; case INT64 -> new long[size]; case BOOL -> new byte[size];
            default -> throw new IllegalArgumentException("unsupported test type");
        };
    }

    private static CpuKernelIr.ScalarImmediate immediate(DataType type) {
        return new CpuKernelIr.ScalarImmediate(type, switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(-0.0d);
            case FLOAT32 -> Float.floatToRawIntBits(-0.0f) & 0xffff_ffffL;
            case INT32 -> 0xffff_ffffL; case INT64 -> -1L;
            default -> throw new IllegalArgumentException("unsupported immediate");
        });
    }

    private static long negativeZero(DataType type) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(-0.0d)
                : Float.floatToRawIntBits(-0.0f) & 0xffff_ffffL;
    }

    private static long positiveZero(DataType type) { return 0L; }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static List<CpuAccessPlan.Binding> bindings(int count) {
        return bindings(count, 4);
    }

    private static List<CpuAccessPlan.Binding> bindings(int count, long extent) {
        var result = new ArrayList<CpuAccessPlan.Binding>();
        for (int i = 0; i < count; i++) result.add(CpuAccessPlan.Binding.create(
                dense(i == count - 1 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ),
                new long[] {extent}, 0, new long[] {1}, extent, 0, extent, extent));
        return result;
    }

    private static long[] geometry(int count, long extent) {
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int i = 0; i < count; i++) {
            result[2 + count + i] = 1;
            result[2 + count + count + count + i] = extent;
        }
        return result;
    }

    private static CpuBufferArgument argument(Object value, DataType type) {
        long bytes = java.lang.reflect.Array.getLength(value) * (long) type.byteWidth();
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) value, 0, bytes, false);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) value, 0, bytes, false);
            case INT32 -> new CpuBufferArgument.Ints((int[]) value, 0, bytes, false);
            case INT64 -> new CpuBufferArgument.Longs((long[]) value, 0, bytes, false);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) value, 0, bytes, false);
            default -> throw new IllegalArgumentException("unsupported argument type");
        };
    }

    private static CpuKernelSpecialization.CarrierAccess heapCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
            default -> throw new IllegalArgumentException("unsupported carrier");
        };
    }

    private static List<DataType> numericTypes() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32, DataType.INT64);
    }

    private static void assertPrimitiveArrayEquals(Object expected, Object actual, String message) {
        if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else if (expected instanceof long[] value) assertArrayEquals(value, (long[]) actual, message);
        else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }

    private record Case(CpuPointwiseOpcode opcode, DataType type) {
        List<DataType> inputTypes() {
            if (opcode == CpuPointwiseOpcode.WHERE) return List.of(DataType.BOOL, type, type);
            return java.util.Collections.nCopies(opcode.arity(), type);
        }
        DataType outputType() {
            return opcode.resultCategory() == CpuPointwiseOpcode.ResultCategory.BOOL
                    ? DataType.BOOL : type;
        }
    }
}
