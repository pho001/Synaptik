package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuSpecializedSubgraphTest {
    @Test void factsSnapshotListsAndIdentityExcludesOccurrenceObjects() {
        var members = new ArrayList<>(List.of(0, 1));
        var units = new ArrayList<>(List.of(0, 1));
        var access = new AccessFact(DataType.FLOAT32, Shape.of(2, 3), 0,
                List.of(3L, 1L), CpuAccessPlan.Regime.DENSE_LINEAR, true);
        var epilogue = new Epilogue(AddInputOrder.PRECEDING_LEFT, Terminal.RELU,
                Optional.empty());
        var attributes = new ReductionAttributes(AggregateReductionKind.SUM,
                ReductionForm.SINGLE_AXIS, List.of(1), false, 0);
        var identity = new StructuralIdentity(Family.REDUCTION, Form.SUM,
                List.of(DataType.FLOAT32), List.of(DataType.FLOAT32),
                List.of(access, access), attributes, epilogue,
                List.of(unitFact(access), unitFact(access)));
        var fact = new ReductionEpilogue(Form.SUM, members, units,
                List.of(DataType.FLOAT32), List.of(DataType.FLOAT32),
                List.of(access, access), epilogue, identity);
        members.clear(); units.clear();
        assertAll(() -> assertEquals(List.of(0, 1), fact.memberNodeOrdinals()),
                () -> assertEquals(List.of(0, 1), fact.baselineUnitIndices()),
                () -> assertEquals(ExecutionDisposition.ORDINARY_SPLIT, fact.disposition()),
                () -> assertEquals(identity, fact.structuralIdentity()),
                () -> assertFalse(identity.toString().contains("ValueId")));
    }

    @Test void closedValidationRejectsDispositionBoundsAndMismatchedIdentity() {
        var access = new AccessFact(DataType.FLOAT64, Shape.scalar(), 0, List.of(),
                CpuAccessPlan.Regime.DENSE_LINEAR, true);
        var attrs = new MatmulAttributes(MatmulInputForm.ORDINARY);
        var identity = new StructuralIdentity(Family.MATMUL, Form.MATMUL,
                List.of(DataType.FLOAT64, DataType.FLOAT64), List.of(DataType.FLOAT64),
                List.of(access, access, access), attrs, Epilogue.none(), List.of());
        assertAll(
                () -> assertEquals(ExecutionDisposition.UNSUPPORTED_ANCHOR,
                        new MatmulEpilogue(List.of(0), List.of(),
                                List.of(DataType.FLOAT64, DataType.FLOAT64),
                                List.of(DataType.FLOAT64), List.of(access, access, access),
                                Epilogue.none(), identity).disposition()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Epilogue(AddInputOrder.NONE, Terminal.RELU,
                                Optional.of(new io.github.pho001.synaptik.model.operation
                                        .elementwise.scalar.ClampRangeAttrs(
                                                io.github.pho001.synaptik.model.datatype.ScalarValue.float64(0),
                                                io.github.pho001.synaptik.model.datatype.ScalarValue.float64(1))))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new MatmulEpilogue(List.of(0, 1, 2, 3, 4, 5, 6), List.of(),
                                List.of(DataType.FLOAT64, DataType.FLOAT64),
                                List.of(DataType.FLOAT64), List.of(access), Epilogue.none(), identity)));
    }

    @Test void closedFactsRejectImpossibleEleventhPositionAndThirdAssociatedUnit() {
        var access = new AccessFact(DataType.FLOAT32, Shape.scalar(), 0, List.of(),
                CpuAccessPlan.Regime.DENSE_LINEAR, true);
        var ten = java.util.Collections.nCopies(10, access);
        var attributes = new ConvolutionAttributes(1, List.of(1L), List.of(0L),
                List.of(1L), 1, false);
        var identity = new StructuralIdentity(Family.CONVOLUTION, Form.CONV1D_COMPOSITION,
                List.of(DataType.FLOAT32), List.of(DataType.FLOAT32), ten, attributes,
                Epilogue.none(), List.of(unitFact(access), unitFact(access)));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new StructuralIdentity(Family.CONVOLUTION,
                                Form.CONV1D_COMPOSITION, List.of(DataType.FLOAT32),
                                List.of(DataType.FLOAT32), java.util.Collections.nCopies(11, access),
                                attributes, Epilogue.none(), List.of())),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new ConvolutionEpilogue(Form.CONV1D_COMPOSITION,
                                List.of(0, 1, 2, 3, 4, 5), List.of(0, 1, 2),
                                List.of(DataType.FLOAT32), List.of(DataType.FLOAT32), ten,
                                Epilogue.none(), ExecutionDisposition.ORDINARY_SPLIT, identity)));
    }

    private static BaselineUnitFact unitFact(AccessFact access) {
        var boundary = new BoundaryResourceFact(access.dataType(), CpuKernelIr.Value.Kind.OUTPUT,
                new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                        CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0),
                List.of(), 0, List.of(), 1, 0, 1, 1, List.of(), 0, 0, 1,
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization
                        .CarrierAccess.FLOAT_ARRAY,
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization
                        .CarrierAccess.FLOAT_ARRAY);
        String key = "01".repeat(32);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(key),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32),
                List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY), 0, -1,
                List.of(), false);
        var execution = new BaselineExecutionFact(BaselineRoute.PORTABLE, specialization,
                BaselineCompute.SCALAR, BaselineOrchestration.SINGLE_THREAD, List.of(), 1, 1, 1,
                0, List.of(), Optional.empty(), RuntimeTopology.POINTWISE, List.of(), "test");
        return new BaselineUnitFact(key, execution, List.of(), List.of(boundary), 1,
                new WorkspaceResourceFact(WorkspaceRole.NONE, 0, 0));
    }
}
