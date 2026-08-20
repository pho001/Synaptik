package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Locks the cross-family structural inventory used by the generated/direct evidence closure. */
class CpuGeneratedDirectEvidenceClosureTest {
    private static final Set<String> NON_POINTWISE_FORMS = Set.of(
            "affine:copy", "affine:internal-view", "movement:pad", "movement:tile",
            "movement:concat", "movement:stack", "movement:unfold-axis",
            "movement:unfold2d", "movement:slice-update", "indexing:gather",
            "indexing:gather-elements", "indexing:gather-nd", "indexing:one-hot",
            "scatter:scatter-elements:none", "scatter:scatter-elements:add",
            "scatter:scatter-elements:min", "scatter:scatter-elements:max",
            "scatter:scatter-elements:mul", "scatter:scatter-add:add",
            "scatter:scatter-nd:none", "scatter:scatter-nd:add", "scatter:scatter-nd:min",
            "scatter:scatter-nd:max", "scatter:scatter-nd:mul", "fold:axis", "fold:2d",
            "ordering:sort", "ordering:argsort", "ordering:top-k", "random:initial-state",
            "random:dropout-f64", "random:dropout-f32", "scan:cum-sum:inclusive-forward",
            "scan:cum-sum:inclusive-reverse", "scan:cum-sum:exclusive-forward",
            "scan:cum-sum:exclusive-reverse", "scan:cum-prod:inclusive-forward",
            "scan:cum-prod:inclusive-reverse", "scan:cum-prod:exclusive-forward",
            "scan:cum-prod:exclusive-reverse", "aggregate:min:full", "aggregate:min:axis",
            "aggregate:min:multi-axis", "aggregate:max:full", "aggregate:max:axis",
            "aggregate:max:multi-axis", "aggregate:all:full", "aggregate:all:axis",
            "aggregate:all:multi-axis", "aggregate:any:full", "aggregate:any:axis",
            "aggregate:any:multi-axis", "aggregate:sum:full", "aggregate:sum:axis",
            "aggregate:sum:multi-axis", "aggregate:mean:full", "aggregate:mean:axis",
            "aggregate:mean:multi-axis", "aggregate:prod:full", "aggregate:prod:axis",
            "aggregate:prod:multi-axis");

    @Test void ledgerCoversEveryPointwiseOpcodeAndCurrentNonPointwiseForm() {
        assertAll(
                () -> assertEquals(48, CpuPointwiseOpcode.values().length),
                () -> assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class),
                        EnumSet.copyOf(List.of(CpuPointwiseOpcode.values()))),
                () -> assertEquals(61, NON_POINTWISE_FORMS.size()),
                () -> assertTrue(NON_POINTWISE_FORMS.stream().noneMatch(String::isBlank)));
    }

    @Test void representativesHaveOneTypedStaticEntryAndClosedMemberReferences() {
        var representatives = new ArrayList<Representative>();
        representatives.add(new Representative(pointwise(false, false), 0));
        representatives.add(new Representative(pointwise(true, false), 0));
        representatives.add(new Representative(pointwise(false, true), 1));
        representatives.add(new Representative(generated(CpuNonAffineMovementLoweringTest.context(
                new Operation(PadKind.PAD,
                        new PadAttrs(List.of(0L), List.of(0L), ScalarValue.int32(0))),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(8))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(8)))), 1));
        representatives.add(new Representative(generated(CpuIndexingLoweringTest.context(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, Shape.of(8)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(4))),
                CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, Shape.of(4)))), 2));
        representatives.add(new Representative(generated(CpuScatterLoweringTest.context(
                new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MIN)), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(4))),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)))), 2));
        representatives.add(new Representative(generated(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 8, 1)),
                DataType.FLOAT32, Shape.of(7, 2), Shape.of(8))), 0));
        representatives.add(new Representative(generated(CpuOrderingLoweringTest.context(
                new Operation(OrderingKind.ARGSORT, new SortAttrs(1, true)), DataType.INT64,
                Shape.of(4, 8), Shape.of(4, 8), false)), 0));
        representatives.add(new Representative(generated(CpuScanLoweringTest.context(
                CumulativeScanKind.CUM_PROD, DataType.INT64, Shape.of(4, 8), 1, true, true)), 1));
        representatives.add(new Representative(generated(CpuAggregateLoweringTest.context(
                AggregateReductionKind.MIN, DataType.BFLOAT16, Shape.of(4, 8),
                new AxisReductionAttrs(1, false), Shape.of(4))), 1));
        representatives.add(new Representative(generated(CpuAggregateLoweringTest.context(
                AggregateReductionKind.MEAN, DataType.FLOAT32, Shape.of(4, 8),
                new AxisReductionAttrs(1, false), Shape.of(4))), 1));
        representatives.forEach(representative -> {
            assertClosedClass(representative.bytes());
            assertSegmentLayoutsHoisted(representative);
        });
    }

    private static byte[] generated(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
        return new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
    }

    private static byte[] pointwise(boolean vector, boolean segment) {
        var denseRead = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var denseWrite = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(1, DataType.FLOAT32, CpuKernelIr.Value.Kind.OUTPUT, denseWrite)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.TANH, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
        var specialization = new CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint
                        .fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32, DataType.FLOAT32),
                segment ? List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                        : List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY),
                vector ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() : 0,
                -1);
        return new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir);
    }

    private static void assertClosedClass(byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("invoke", model.methods().getFirst().methodName().stringValue()),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.STATIC)),
                () -> assertTrue(model.methods().getFirst().methodTypeSymbol().descriptorString()
                        .endsWith("[JJJ)V")),
                () -> assertFalse(model.methods().getFirst().methodTypeSymbol().descriptorString()
                        .contains("Ljava/lang/Object;")),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(members.stream().noneMatch(member -> member.type().stringValue()
                        .contains("Ljava/lang/Object;") || member.owner().asInternalName()
                        .startsWith("java/lang/reflect/") || member.owner().asInternalName()
                        .startsWith("java/util/"))),
                () -> assertTrue(members.stream().filter(member -> member.owner().asInternalName()
                        .startsWith("io/github/pho001/synaptik")).allMatch(member -> member.owner()
                        .asInternalName().endsWith("/CpuVectorMath"))));
    }

    private static void assertSegmentLayoutsHoisted(Representative representative) {
        var invokes = ClassFile.of().parse(representative.bytes()).methods().getFirst().code()
                .orElseThrow().elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        long nativeOrder = invokes.stream().filter(instruction ->
                instruction.owner().asInternalName().equals("java/nio/ByteOrder")
                        && instruction.name().stringValue().equals("nativeOrder")).count();
        long withOrder = invokes.stream().filter(instruction ->
                instruction.owner().asInternalName().equals("java/lang/foreign/ValueLayout")
                        && instruction.name().stringValue().equals("withOrder")).count();
        int firstSegmentAccess = java.util.stream.IntStream.range(0, invokes.size())
                .filter(index -> invokes.get(index).owner().asInternalName()
                        .equals("java/lang/foreign/MemorySegment")
                        && (invokes.get(index).name().stringValue().equals("get")
                            || invokes.get(index).name().stringValue().equals("set")))
                .findFirst().orElse(invokes.size());
        int lastLayoutConstruction = java.util.stream.IntStream.range(0, invokes.size())
                .filter(index -> invokes.get(index).name().stringValue().equals("nativeOrder")
                        || invokes.get(index).name().stringValue().equals("withOrder"))
                .reduce((left, right) -> right).orElse(-1);
        assertAll(
                () -> assertEquals(representative.orderedLayoutCount(), nativeOrder),
                () -> assertEquals(representative.orderedLayoutCount(), withOrder),
                () -> assertTrue(lastLayoutConstruction < firstSegmentAccess));
    }

    private record Representative(byte[] bytes, int orderedLayoutCount) { }
}
