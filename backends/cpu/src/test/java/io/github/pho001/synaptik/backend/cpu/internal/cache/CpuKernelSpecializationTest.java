package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import java.util.List;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;

class CpuKernelSpecializationTest {
    @Test void foldIdentityExcludesColdGeometryAndSeparatesFamilyTypeCarrierAndRank() {
        var axisOne = fold(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var axisTwo = fold(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 6, 2)), DataType.FLOAT32, Shape.of(2, 3), Shape.of(6),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        Shape imageShape = Shape.of(1, 1, 3, 3);
        var image = fold(new Operation(WindowTransformKind.FOLD2D,
                new Fold2dAttrs(imageShape, CpuFoldLoweringTest.window(false))), DataType.FLOAT32,
                Shape.of(1, 4, 4), imageShape,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var integral = fold(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.INT32, Shape.of(3, 3), Shape.of(5),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY));
        var segment = fold(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5),
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY));
        assertAll(() -> assertEquals(axisOne.specialization(), axisTwo.specialization()),
                () -> assertArrayEquals(new CpuClassFileKernelGenerator().generateClassBytes(
                                axisOne.specialization(), axisOne.kernelIr()),
                        new CpuClassFileKernelGenerator().generateClassBytes(
                                axisTwo.specialization(), axisTwo.kernelIr())),
                () -> assertNotEquals(axisOne.specialization(), image.specialization()),
                () -> assertNotEquals(axisOne.specialization(), integral.specialization()),
                () -> assertNotEquals(axisOne.specialization(), segment.specialization()));
    }
    @Test void scatterCompatibilityExcludesGeometryAndSeparatesReductionAndScratchSignature() {
        var small=scatter(ScatterReduction.ADD,Shape.of(2),Shape.of(3),DataType.FLOAT32);
        var large=scatter(ScatterReduction.ADD,Shape.of(7),Shape.of(11),DataType.FLOAT32);
        var product=scatter(ScatterReduction.MUL,Shape.of(2),Shape.of(3),DataType.FLOAT32);
        var largeProduct=scatter(ScatterReduction.MUL,Shape.of(7),Shape.of(11),DataType.FLOAT32);
        assertAll(() -> assertEquals(small.specialization(),large.specialization()),
                () -> assertArrayEquals(new CpuClassFileKernelGenerator().generateClassBytes(
                                small.specialization(),small.kernelIr()),
                        new CpuClassFileKernelGenerator().generateClassBytes(
                                large.specialization(),large.kernelIr())),
                () -> assertNotEquals(small.specialization(),product.specialization()),
                () -> assertEquals(product.specialization(), largeProduct.specialization()),
                () -> assertArrayEquals(new CpuClassFileKernelGenerator().generateClassBytes(
                                product.specialization(), product.kernelIr()),
                        new CpuClassFileKernelGenerator().generateClassBytes(
                                largeProduct.specialization(), largeProduct.kernelIr())),
                () -> assertFalse(small.specialization().scratchParameter()),
                () -> assertTrue(product.specialization().scratchParameter()));
    }
    @Test void excludesCompatibleExtentsButIncludesNumericalAndStrategyFacts() {
        var one = CpuPartitionPreparerTest.analyze(Shape.of(1)).plan().units().getFirst()
                .portablePlan().specialization();
        var many = CpuPartitionPreparerTest.analyze(Shape.of(99)).plan().units().getFirst()
                .portablePlan().specialization();
        assertAll(
                () -> assertEquals(one, many),
                () -> assertEquals(one.structuralKey(), many.structuralKey()),
                () -> assertSame(CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        one.numericalMode()),
                () -> assertArrayEquals(one.compatibilityBytes(), many.compatibilityBytes()));
    }

    @Test void orderedCarrierPatternChangesKeyDescriptorAndGeneratedBytes() {
        var allSegment = specialization(List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.MEMORY_SEGMENT));
        var mixed = specialization(List.of(CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.MEMORY_SEGMENT));
        assertAll(
                () -> assertNotEquals(allSegment.structuralKey(), mixed.structuralKey()),
                () -> assertNotEquals(allSegment.entryType(), mixed.entryType()),
                () -> assertFalse(java.util.Arrays.equals(allSegment.compatibilityBytes(),
                        mixed.compatibilityBytes())));
    }

    @Test void vectorSpeciesChangesCompatibilityWhileParallelChunkFactsDoNot() {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        var old = CpuPartitionPreparerTest.context(Shape.of(lanes * 2));
        var scalarParallel = specialization(old, new PortableExecutionConfig(
                ComputePreference.SCALAR, 4, 2, 1));
        var scalarSingle = specialization(old, PortableExecutionConfig.DEFAULT);
        var vector = specialization(old, new PortableExecutionConfig(
                ComputePreference.VECTOR_IF_ELIGIBLE, 4, 2, 1));
        assertAll(
                () -> assertEquals(scalarSingle, scalarParallel),
                () -> assertEquals(0, scalarSingle.vectorSpeciesBitSize()),
                () -> assertEquals(DoubleVector.SPECIES_PREFERRED.vectorBitSize(),
                        vector.vectorSpeciesBitSize()),
                () -> assertNotEquals(scalarSingle.structuralKey(), vector.structuralKey()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuKernelSpecialization(vector.loweringFingerprint(),
                                vector.numericalMode(), vector.executionStrategy(),
                                vector.boundaryDataTypes(), vector.carrierPattern(),
                                vector.vectorSpeciesBitSize() / 2,
                                vector.materializedSourcePosition(),
                                vector.scalarPowerRealizations())));
    }

    @Test void typedPreferredSpeciesAndBoundaryTypeChangeVectorIdentity() {
        int count = Math.max(DoubleVector.SPECIES_PREFERRED.length(),
                FloatVector.SPECIES_PREFERRED.length()) * 2;
        Shape shape = Shape.of(count);
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1);
        var doubleSpecialization = specialization(CpuPartitionPreparerTest.context(shape), config);
        var floatDescriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var floatContext = CpuPartitionPreparerTest.context(floatDescriptor, floatDescriptor,
                floatDescriptor, floatDescriptor, new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), config));
        var floatSpecialization = specialization(floatContext, config);
        assertAll(
                () -> assertEquals(DoubleVector.SPECIES_PREFERRED.vectorBitSize(),
                        doubleSpecialization.vectorSpeciesBitSize()),
                () -> assertEquals(FloatVector.SPECIES_PREFERRED.vectorBitSize(),
                        floatSpecialization.vectorSpeciesBitSize()),
                () -> assertNotEquals(doubleSpecialization.structuralKey(),
                        floatSpecialization.structuralKey()),
                () -> assertFalse(java.util.Arrays.equals(doubleSpecialization.compatibilityBytes(),
                        floatSpecialization.compatibilityBytes())));
    }

    @Test void materializationStructureChangesIdentityWhileInstanceFactsAndVariantsStayBounded() {
        var shape = Shape.of(2, 3);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, 48, 1, 1);
        var materialized = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        PortableExecutionConfig.DEFAULT, policy)).plan();
        var direct = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan();
        assertAll(
                () -> assertEquals(0, materialized.units().getFirst().portablePlan()
                        .specialization().materializedSourcePosition()),
                () -> assertNotEquals(direct.units().getFirst().portablePlan().specialization()
                                .structuralKey(),
                        materialized.units().getFirst().portablePlan().specialization()
                                .structuralKey()),
                () -> assertEquals(new CpuSpecializationBudget(4, 1, 0, 0),
                        materialized.specializationBudget()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuSpecializationBudget(5, 1, 0, 0)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuSpecializationBudget(4, 2, 0, 0)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuSpecializationBudget(4, 1, 1, 0)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuSpecializationBudget(4, 1, 0, 1)));
    }

    @Test void scalarPowerRealizationIsExplicitCompatibilityMetadata() {
        var base = CpuPartitionPreparerTest.analyze(Shape.of(3)).plan().units().getFirst()
                .portablePlan().specialization();
        var direct = new CpuKernelSpecialization(base.loweringFingerprint(), base.numericalMode(),
                base.executionStrategy(), base.boundaryDataTypes(), base.carrierPattern(),
                base.vectorSpeciesBitSize(), base.materializedSourcePosition(),
                List.of(CpuKernelIr.PowerRealization.DIRECT));
        var square = new CpuKernelSpecialization(base.loweringFingerprint(), base.numericalMode(),
                base.executionStrategy(), base.boundaryDataTypes(), base.carrierPattern(),
                base.vectorSpeciesBitSize(), base.materializedSourcePosition(),
                List.of(CpuKernelIr.PowerRealization.SQUARE));
        assertAll(
                () -> assertNotEquals(direct.structuralKey(), square.structuralKey()),
                () -> assertFalse(java.util.Arrays.equals(direct.compatibilityBytes(),
                        square.compatibilityBytes())),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuKernelSpecialization(base.loweringFingerprint(),
                                base.numericalMode(), base.executionStrategy(),
                                base.boundaryDataTypes(), base.carrierPattern(),
                                base.vectorSpeciesBitSize(), base.materializedSourcePosition(), null)));
    }

    @Test void shortArrayIsTheSeventhCarrierAndOnlyMatchesBfloat16() {
        var fingerprint = CpuLoweringFingerprint.fromHex("0".repeat(64));
        var bfloat = new CpuKernelSpecialization(fingerprint,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                        .ExecutionStrategy.SCALAR,
                List.of(DataType.BFLOAT16, DataType.BFLOAT16),
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT), 0, -1, List.of());
        assertAll(
                () -> assertEquals(7, CarrierAccess.values().length),
                () -> assertEquals(short[].class, bfloat.entryType().parameterType(0)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuKernelSpecialization(fingerprint,
                                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                                io.github.pho001.synaptik.backend.cpu.internal.prepare
                                        .CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                                List.of(DataType.FLOAT32, DataType.FLOAT32),
                                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.FLOAT_ARRAY),
                                0, -1, List.of())));
    }

    @Test void indexingIdentityIncludesStructureAndExcludesCompatibleColdGeometry() {
        var axisZero = indexing(AxisGatherKind.GATHER, 0, Shape.of(2, 2), Shape.of(2),
                Shape.of(2, 2), DataType.FLOAT32, DataType.INT64,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        var axisOne = indexing(AxisGatherKind.GATHER, 1, Shape.of(2, 2), Shape.of(2),
                Shape.of(2, 2), DataType.FLOAT32, DataType.INT64,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        var otherExtents = indexing(AxisGatherKind.GATHER, 1, Shape.of(3, 4), Shape.of(5),
                Shape.of(3, 5), DataType.FLOAT32, DataType.INT64,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        var otherFamily = indexing(AxisGatherKind.GATHER_ELEMENTS, 1, Shape.of(2, 2),
                Shape.of(2, 2), Shape.of(2, 2), DataType.FLOAT32, DataType.INT64,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        var otherCarrier = indexing(AxisGatherKind.GATHER, 0, Shape.of(2, 2), Shape.of(2),
                Shape.of(2, 2), DataType.FLOAT32, DataType.INT64,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY));
        var generator = new CpuClassFileKernelGenerator();
        assertAll(
                () -> assertEquals(axisZero.specialization(), axisOne.specialization()),
                () -> assertEquals(axisOne.specialization(), otherExtents.specialization()),
                () -> assertArrayEquals(generator.generateClassBytes(axisZero.specialization(),
                                axisZero.kernelIr()),
                        generator.generateClassBytes(otherExtents.specialization(),
                                otherExtents.kernelIr())),
                () -> assertNotEquals(axisZero.specialization(), otherFamily.specialization()),
                () -> assertNotEquals(axisZero.specialization(), otherCarrier.specialization()),
                () -> assertEquals(34, CpuGeneratorSchema.CURRENT_VERSION),
                () -> assertEquals(-1, axisZero.specialization().materializedSourcePosition()));
    }

    private static CpuKernelSpecialization specialization(
            PrepareContext<CpuPartitionAnalysisInputs> old, PortableExecutionConfig config) {
        var context = new PrepareContext<>(old.partition(), old.nodes(), old.values(),
                old.memoryRequirements(), old.constants(), new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), config));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan().specialization();
    }

    private static CpuKernelSpecialization specialization(List<CarrierAccess> pattern) {
        var old = CpuPartitionPreparerTest.context(Shape.of(3));
        var context = new PrepareContext<>(old.partition(), old.nodes(), old.values(),
                old.memoryRequirements(), old.constants(),
                new CpuPartitionAnalysisInputs(false, pattern));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan().specialization();
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable
            .CpuPortableRoutePlan indexing(AxisGatherKind kind, int axis, Shape dataShape,
                    Shape indexShape, Shape outputShape, DataType dataType, DataType indexType,
                    List<CarrierAccess> carriers) {
        var base = CpuIndexingLoweringTest.context(new Operation(kind, new IndexAxisAttrs(axis)),
                List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(dataType, dataShape),
                        CpuIndexingLoweringTest.descriptor(indexType, indexShape)),
                CpuIndexingLoweringTest.descriptor(dataType, outputShape));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan
            scatter(ScatterReduction reduction,Shape dataShape,Shape updateShape,DataType type){
        var base=CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,reduction)),List.of(0,1,2),
                List.of(CpuScatterLoweringTest.desc(type,dataShape),
                        CpuScatterLoweringTest.desc(DataType.INT32,updateShape),
                        CpuScatterLoweringTest.desc(type,updateShape)),
                CpuScatterLoweringTest.desc(type,dataShape));
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),
                base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,
                        List.of(type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.INT_ARRAY,
                                CarrierAccess.INT_ARRAY,
                                type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.INT_ARRAY,
                                type==DataType.FLOAT32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.INT_ARRAY)));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan
            fold(Operation operation, DataType type, Shape input, Shape output,
                    List<CarrierAccess> carriers) {
        var base = CpuFoldLoweringTest.context(operation, type, input, output);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }
}
