package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Pins the current CPU generated providers' raw FLOAT32 stored-subnormal realization.
 * These backend observations do not narrow Model's ACCELERATOR DAZ freedom.
 */
class CpuStoredSubnormalProviderRegressionTest {
    private static final int MIN_SUBNORMAL = 0x0000_0001;
    private static final int POSITIVE_ZERO = 0x0000_0000;

    @Test
    void generatedScalarAndVectorDistinguishStoredSubnormalFromPositiveZero() throws Throwable {
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        assertTrue(lanes > 0);
        int count = lanes + 1; // One complete preferred-species chunk plus its scalar tail.
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (var strategy : List.of(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)) {
                float[] left = new float[count];
                float[] right = new float[count];
                for (int index = 0; index < count; index++) {
                    left[index] = Float.intBitsToFloat(index % 2 == 0
                            ? MIN_SUBNORMAL : POSITIVE_ZERO);
                    right[index] = Float.intBitsToFloat(index % 2 == 0
                            ? POSITIVE_ZERO : MIN_SUBNORMAL);
                }
                int[] leftBefore = bits(left);
                int[] rightBefore = bits(right);
                for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.GREATER_THAN,
                        CpuPointwiseOpcode.EQUAL, CpuPointwiseOpcode.MIN,
                        CpuPointwiseOpcode.MAX)) {
                    Object output = opcode == CpuPointwiseOpcode.GREATER_THAN
                            || opcode == CpuPointwiseOpcode.EQUAL
                            ? new byte[count] : new float[count];
                    invoke(opcode, profile, strategy, left, right, output, null);
                    String label = profile + " " + strategy + " " + opcode;
                    if (output instanceof byte[] predicates) {
                        byte[] expected = new byte[count];
                        if (opcode == CpuPointwiseOpcode.GREATER_THAN) {
                            for (int index = 0; index < count; index += 2) expected[index] = 1;
                        }
                        assertArrayEquals(expected, predicates, label);
                    } else {
                        int[] expected = new int[count];
                        Arrays.fill(expected, opcode == CpuPointwiseOpcode.MAX
                                ? MIN_SUBNORMAL : POSITIVE_ZERO);
                        assertArrayEquals(expected, bits((float[]) output), label);
                    }
                }
                for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SCALAR_MIN,
                        CpuPointwiseOpcode.SCALAR_MAX)) {
                    for (int scalarBits : List.of(POSITIVE_ZERO, MIN_SUBNORMAL)) {
                        float[] output = new float[count];
                        invoke(opcode, profile, strategy, left, null, output, scalarBits);
                        int[] expected = new int[count];
                        for (int index = 0; index < count; index++) {
                            int inputBits = leftBefore[index];
                            expected[index] = opcode == CpuPointwiseOpcode.SCALAR_MAX
                                    ? Math.max(inputBits, scalarBits)
                                    : Math.min(inputBits, scalarBits);
                        }
                        assertArrayEquals(expected, bits(output),
                                profile + " " + strategy + " " + opcode + " scalar=" + scalarBits);
                    }
                }
                assertArrayEquals(leftBefore, bits(left), profile + " left input bits");
                assertArrayEquals(rightBefore, bits(right), profile + " right input bits");
            }
        }
    }

    @Test
    void generatedScalarArgMaxSelectsStoredSubnormalUnderBothProfiles() throws Throwable {
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (ArgExtremaTiePolicy tie : ArgExtremaTiePolicy.values()) {
                float[] input = {Float.intBitsToFloat(POSITIVE_ZERO),
                        Float.intBitsToFloat(MIN_SUBNORMAL),
                        Float.intBitsToFloat(POSITIVE_ZERO)};
                var base = CpuArgExtremaLoweringTest.context(AggregateReductionKind.ARG_MAX,
                        DataType.FLOAT32, Shape.of(3), 0, false, tie);
                var context = new PrepareContext<>(profile, base.partition(), base.nodes(),
                        base.values(), base.memoryRequirements(), base.constants(),
                        new CpuPartitionAnalysisInputs(false, List.of(
                                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                                CpuKernelSpecialization.CarrierAccess.LONG_ARRAY)));
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                var route = plan.units().getFirst().portablePlan();
                var generator = new CpuClassFileKernelGenerator();
                var artifact = generator.defineClassBytes(route.specialization(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
                long[] output = {-1};
                artifact.entryPoint().invokeWithArguments(input, output,
                        plan.argExtremaGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
                assertArrayEquals(new long[] {1}, output, profile + " " + tie);
                assertArrayEquals(new int[] {POSITIVE_ZERO, MIN_SUBNORMAL, POSITIVE_ZERO},
                        bits(input), profile + " " + tie + " input bits");
            }
        }
    }

    private static void invoke(CpuPointwiseOpcode opcode, NumericalProfile profile,
            CpuPartitionPreparationPlan.ExecutionStrategy strategy, float[] left,
            float[] right, Object output, Integer scalarBits) throws Throwable {
        boolean binary = right != null;
        DataType outputType = output instanceof byte[] ? DataType.BOOL : DataType.FLOAT32;
        var values = new ArrayList<CpuKernelIr.Value>();
        values.add(new CpuKernelIr.Value(0, DataType.FLOAT32,
                CpuKernelIr.Value.Kind.INPUT, access(CpuAccessPlan.AccessKind.READ)));
        if (binary) values.add(new CpuKernelIr.Value(1, DataType.FLOAT32,
                CpuKernelIr.Value.Kind.INPUT, access(CpuAccessPlan.AccessKind.READ)));
        int result = values.size();
        values.add(new CpuKernelIr.Value(result, outputType,
                CpuKernelIr.Value.Kind.OUTPUT, access(CpuAccessPlan.AccessKind.WRITE)));
        var immediate = scalarBits == null ? null
                : new CpuKernelIr.ScalarImmediate(DataType.FLOAT32,
                        Integer.toUnsignedLong(scalarBits));
        var ir = new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(opcode,
                binary ? List.of(0, 1) : List.of(0), result, immediate, null)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(result, 0)));
        List<DataType> types = values.stream().map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream().map(type ->
                type == DataType.BOOL ? CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY).toList();
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()), profile, strategy,
                types, carriers, strategy == CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                        ? FloatVector.SPECIES_PREFERRED.vectorBitSize() : 0, -1,
                List.of(), false, strategy == CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                        && outputType == DataType.BOOL ? 61 : 52);
        var generator = new CpuClassFileKernelGenerator();
        byte[] generated = generator.generateClassBytes(specialization, ir);
        if (strategy == CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR) {
            var model = ClassFile.of().parse(generated);
            assertTrue(java.util.stream.StreamSupport.stream(model.constantPool().spliterator(),
                    false).filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                    .anyMatch(member -> member.owner().asInternalName()
                            .startsWith("jdk/incubator/vector/")),
                    opcode + " must exercise the generated Vector API body");
        }
        var artifact = generator.defineClassBytes(specialization, generated);
        int count = left.length;
        var arguments = new ArrayList<Object>();
        arguments.add(left);
        if (binary) arguments.add(right);
        arguments.add(output);
        arguments.add(geometry(arguments.size(), count));
        arguments.add(0L);
        arguments.add((long) count);
        artifact.entryPoint().invokeWithArguments(arguments);
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static long[] geometry(int boundaries, int count) {
        long[] result = new long[2 + 4 * boundaries];
        result[0] = count;
        for (int index = 0; index < boundaries; index++) {
            result[2 + boundaries + index] = 1;
            result[2 + 3 * boundaries + index] = count;
        }
        return result;
    }

    private static int[] bits(float[] values) {
        int[] result = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = Float.floatToRawIntBits(values[index]);
        }
        return result;
    }
}
