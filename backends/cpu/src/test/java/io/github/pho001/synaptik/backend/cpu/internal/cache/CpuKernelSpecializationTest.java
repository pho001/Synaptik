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

class CpuKernelSpecializationTest {
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
}
