package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.invoke.MethodHandle;
import java.security.MessageDigest;
import java.util.List;
import java.util.Optional;

/** Test-only source-derived fixture builder and independent BFLOAT16 clean-Java oracle. */
final class CpuScalarImmediateClampEquivalenceOracle {
    record Fixture(String id, float immediate, String immediateLocation) { }
    record Artifact(Fixture fixture, CpuKernelSpecialization specialization, byte[] bytes,
            String hash, String binaryName, String structuralKey) { }

    /* The complete bounded mechanism here is one exact operation/type/carrier/layout/strategy/shape. */
    static List<Fixture> fixtures() {
        return List.of(new Fixture("bf16-mul-one", 1.0f, "entry:28"),
                new Fixture("bf16-mul-two", 2.0f, "entry:28"));
    }

    static Artifact artifact(Fixture fixture) {
        var shape = Shape.of(8);
        var descriptor = new TensorDescriptor(DataType.BFLOAT16, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var input = new ValueId(0); var output = new ValueId(1); var nodeId = new NodeId(0);
        var node = new CompiledNode(nodeId, new Operation(ScalarElementwiseKind.MUL,
                new ScalarValueAttrs(ScalarValue.bfloat16(fixture.immediate()))), List.of(input), List.of(output));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(nodeId));
        var context = new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(input, descriptor), new GraphValue(output, descriptor)),
                List.of(new LogicalMemoryRequirement(input, descriptor, Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(output, descriptor, Optional.of(partition), List.of(), true)),
                java.util.Map.of(), new CpuPartitionAnalysisInputs(true,
                        List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY,
                                CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        return new Artifact(fixture, route.specialization(), bytes, sha256(bytes),
                CpuGeneratorSchema.generatedBinaryName(route.specialization()), route.kernelIr().structuralKey());
    }

    static short[] invoke(Artifact artifact, short[] input) throws Throwable {
        short[] actual = new short[input.length]; long[] geometry = geometry(2, input.length);
        MethodHandle handle = new CpuClassFileKernelGenerator().defineClassBytes(artifact.specialization(), artifact.bytes()).entryPoint();
        handle.invokeExact(input, actual, geometry, 0L, (long) input.length);
        return actual;
    }

    /** Same represented-input widening, FLOAT32 multiplication, and ties-to-even narrowing as the generated loop. */
    static short[] cleanJava(Fixture fixture, short[] input) {
        short[] result = new short[input.length];
        for (int index = 0; index < input.length; index++) result[index] = bfloat(floatValue(input[index]) * fixture.immediate());
        return result;
    }
    static float floatValue(short bits) { return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16); }
    static short bfloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7fffffff) > 0x7f800000) return (short) 0x7fc0;
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
    }
    private static long[] geometry(int count, long extent) {
        long[] result = new long[2 + count + count + 2 * count]; result[0] = extent;
        for (int index = 0; index < count; index++) { result[2 + count + index] = 1; result[2 + 3 * count + index] = extent; }
        return result;
    }
    static String sha256(byte[] bytes) {
        try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
        catch (Exception failure) { throw new AssertionError(failure); }
    }
    private CpuScalarImmediateClampEquivalenceOracle() { }
}
