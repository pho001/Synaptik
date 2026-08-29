package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Frozen deterministic Class-File and compatibility evidence for CPU 0008. */
public final class CpuConv2dEvidenceTest {
    private static final Path EVIDENCE = Path.of(
            "/private/tmp/synaptik-cpu-0008-retained-evidence-20260826/generated");
    private record Target(String name, PrepareContext<CpuPartitionAnalysisInputs> context,
            int classes) { }

    @Test void retainsClosedRepresentativeGeneratedInventory() throws Exception {
        assertEquals(54, CpuGeneratorSchema.CURRENT_VERSION);
        List<Target> targets = targets();
        assertEquals(List.of("CONV-DENSE-F32", "CONV-GROUPED-F64", "CONV-DEPTHWISE-BF16",
                "CONV-GENERAL-MIXED", "CONV-FUSED-ADD", "CONV-FUSED-ADD-RELU", "CONV-PARALLEL-F32",
                "CONV-SPLIT-ADD-RELU"), targets.stream().map(Target::name).toList());
        Files.createDirectories(EVIDENCE);
        for (Target target : targets) inspect(target);
    }

    private static void inspect(Target target) throws Exception {
        var plan = new CpuPartitionPreparer().analyze(target.context).plan();
        Optional<CpuPartitionLowering.LoweredPartition> recognitionFree =
                target.name.startsWith("CONV-FUSED-")
                ? Optional.of(new CpuPartitionLowering().lower(target.context)) : Optional.empty();
        assertEquals(target.classes, plan.units().size(), target.name);
        if (recognitionFree.isPresent()) {
            assertEquals(recognitionFree.orElseThrow().portableKernelIr().structuralKey(),
                    plan.units().getFirst().portablePlan().portableKernelIr().structuralKey(),
                    target.name + " recognition-free lowering identity");
        }
        for (int index = 0; index < plan.units().size(); index++) {
            var route = plan.units().get(index).portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
            assertArrayEquals(bytes,
                    generator.generateClassBytes(route.specialization(), route.kernelIr()),
                    target.name + " unit " + index);
            if (recognitionFree.isPresent()) {
                assertArrayEquals(bytes, generator.generateClassBytes(route.specialization(),
                                recognitionFree.orElseThrow().kernelIr()),
                        target.name + " recognition-free class-byte identity");
            }
            var model = ClassFile.of().parse(bytes);
            assertAll(target.name,
                    () -> assertTrue(model.flags().has(java.lang.reflect.AccessFlag.FINAL)),
                    () -> assertTrue(model.fields().isEmpty()),
                    () -> assertEquals(1, model.methods().size()),
                    () -> assertEquals("invoke", model.methods().getFirst()
                            .methodName().stringValue()),
                    () -> assertTrue(model.methods().getFirst().flags()
                            .has(java.lang.reflect.AccessFlag.STATIC)));
            StringBuilder members = new StringBuilder();
            java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                    .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                    .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                            .append(member.name().stringValue())
                            .append(member.type().stringValue()).append('\n'));
            String references = members.toString();
            assertAll(target.name,
                    () -> assertFalse(references.contains("synaptik")),
                    () -> assertFalse(references.contains("java/lang/reflect")),
                    () -> assertFalse(references.contains("java/lang/invoke")),
                    () -> assertFalse(references.contains("java/util/")),
                    () -> assertFalse(references.contains(".valueOf")));
            retain(target.name + "-unit-" + index, bytes,
                    route.specialization().compatibilityBytes(), route.specialization().toString(),
                    model.methods().getFirst().methodTypeSymbol().displayDescriptor(), references,
                    plan.loweringManifest());
        }
    }

    private static List<Target> targets() {
        var result = new ArrayList<Target>();
        result.add(direct("CONV-DENSE-F32", List.of(DataType.FLOAT32, DataType.FLOAT32),
                Shape.of(4, 16, 32, 32), Shape.of(32, 16, 3, 3), Shape.of(4, 32, 30, 30),
                Conv2dAttrs.defaults(), null, Collections.nCopies(3, CarrierAccess.FLOAT_ARRAY), 1));
        result.add(direct("CONV-GROUPED-F64", List.of(DataType.FLOAT64, DataType.FLOAT64,
                        DataType.FLOAT64), Shape.of(2, 8, 35, 37), Shape.of(12, 2, 3, 2),
                Shape.of(2, 12, 17, 19), new Conv2dAttrs(2, 2, 1, 1, 2, 2, 4), null,
                Collections.nCopies(4, CarrierAccess.DOUBLE_ARRAY), 1));
        result.add(direct("CONV-DEPTHWISE-BF16", Collections.nCopies(3, DataType.BFLOAT16),
                Shape.of(2, 16, 32, 32), Shape.of(16, 1, 3, 3), Shape.of(2, 16, 32, 32),
                new Conv2dAttrs(1, 1, 1, 1, 1, 1, 16), null,
                Collections.nCopies(4, CarrierAccess.SHORT_ARRAY), 1));
        Shape x = Shape.of(1, 2, 4, 5), w = Shape.of(3, 2, 2, 2), y = Shape.of(1, 3, 3, 4);
        var layouts = List.of(LayoutDescriptor.of(x, new long[] {101, 43, 9, 1}, 7, true),
                LayoutDescriptor.of(w, new long[] {31, 13, 5, 2}, 3, true),
                LayoutDescriptor.of(y, new long[] {149, 47, 11, 2}, 5, true));
        result.add(direct("CONV-GENERAL-MIXED", List.of(DataType.BFLOAT16, DataType.FLOAT64),
                x, w, y, Conv2dAttrs.defaults(), layouts,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT), 1));
        var fused = CpuConv2dGeneratedKernelTest.fusedContext();
        result.add(new Target("CONV-FUSED-ADD", inputs(addOnly(fused),
                Collections.nCopies(5, CarrierAccess.FLOAT_ARRAY), 1), 1));
        result.add(new Target("CONV-FUSED-ADD-RELU", inputs(fused,
                Collections.nCopies(5, CarrierAccess.FLOAT_ARRAY), 1), 1));
        result.add(direct("CONV-PARALLEL-F32", List.of(DataType.FLOAT32, DataType.FLOAT32),
                Shape.of(8, 32, 48, 48), Shape.of(64, 32, 3, 3), Shape.of(8, 64, 46, 46),
                Conv2dAttrs.defaults(), null, Collections.nCopies(3, CarrierAccess.FLOAT_ARRAY), 4));
        result.add(new Target("CONV-SPLIT-ADD-RELU", inputs(publishConvOutput(fused),
                Collections.nCopies(6, CarrierAccess.FLOAT_ARRAY), 4), 2));
        return List.copyOf(result);
    }

    private static Target direct(String name, List<DataType> types, Shape input, Shape weight,
            Shape output, Conv2dAttrs attrs, List<LayoutDescriptor> layouts,
            List<CarrierAccess> carriers, int parallelism) {
        var base = CpuConv2dLoweringTest.context(types, input, weight, output, attrs, layouts);
        return new Target(name, inputs(base, carriers, parallelism), 1);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> inputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, List<CarrierAccess> carriers,
            int parallelism) {
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                parallelism, parallelism, 1);
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(true, carriers, config));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> publishConvOutput(
            PrepareContext<CpuPartitionAnalysisInputs> base) {
        var intermediate = base.nodes().getFirst().outputs().getFirst();
        var memory = base.memoryRequirements().stream().map(requirement ->
                requirement.valueId().equals(intermediate)
                    ? new LogicalMemoryRequirement(requirement.valueId(), requirement.descriptor(),
                        requirement.producerPartition(), requirement.consumerPartitions(), true)
                    : requirement).toList();
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(), memory,
                base.constants(), base.backendInputs());
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> addOnly(
            PrepareContext<CpuPartitionAnalysisInputs> base) {
        var nodes = base.nodes().subList(0, 2);
        var values = base.values().stream().filter(value -> value.id().value() <= 5).toList();
        var partition = new io.github.pho001.synaptik.planning.partition.PlannedPartition(
                CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(node -> node.id()).toList());
        var memory = base.memoryRequirements().stream()
                .filter(requirement -> requirement.valueId().value() <= 5)
                .map(requirement -> requirement.valueId().value() == 5
                        ? new LogicalMemoryRequirement(requirement.valueId(),
                            requirement.descriptor(), Optional.of(partition), List.of(), true)
                        : new LogicalMemoryRequirement(requirement.valueId(),
                            requirement.descriptor(), requirement.valueId().value() >= 4
                                ? Optional.of(partition) : Optional.empty(),
                            List.of(partition), false))
                .toList();
        return new PrepareContext<>(partition, nodes, values, memory, base.constants(),
                base.backendInputs());
    }

    private static void retain(String name, byte[] bytes, byte[] compatibility,
            String specialization, String descriptor, String members, String manifest)
            throws Exception {
        Files.write(EVIDENCE.resolve(name + ".class"), bytes);
        Files.write(EVIDENCE.resolve(name + ".compatibility"), compatibility);
        Files.writeString(EVIDENCE.resolve(name + ".sha256"), hex(bytes) + "\n");
        Files.writeString(EVIDENCE.resolve(name + ".specialization"), specialization + "\n");
        Files.writeString(EVIDENCE.resolve(name + ".descriptor"), descriptor + "\n");
        Files.writeString(EVIDENCE.resolve(name + ".members"), members);
        Files.writeString(EVIDENCE.resolve(name + ".manifest"), manifest + "\n");
    }

    private static String hex(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }
}
