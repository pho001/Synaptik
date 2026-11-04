package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Promotion is a scalar computation fact, never a rewrite of physical input carriers. */
class CpuPromotedPointwiseTest {
    @Test void everyNewModelValidTupleRetainsItsOriginalBoundaryTypesAndSelectsScalar() {
        for (var pair : List.of(List.of(DataType.FLOAT32, DataType.FLOAT64),
                List.of(DataType.FLOAT64, DataType.FLOAT32), List.of(DataType.INT32, DataType.INT64),
                List.of(DataType.INT64, DataType.INT32))) {
            var promoted = DataTypePromotion.promoteNumeric(pair.get(0), pair.get(1));
            for (var kind : BinaryArithmeticKind.values()) {
                if (!promoted.isFloating() && (kind == BinaryArithmeticKind.DIV || kind == BinaryArithmeticKind.POW)) continue;
                check(new Operation(kind, NoOperationAttrs.INSTANCE), pair, promoted,
                        CpuPointwiseOpcode.valueOf(kind.name()));
            }
            for (var kind : BinaryComparisonKind.values()) check(
                    new Operation(kind, NoOperationAttrs.INSTANCE), pair, DataType.BOOL,
                    CpuPointwiseOpcode.valueOf(kind.name()));
            if (promoted.isFloating()) check(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(DataType.BOOL, pair.get(0), pair.get(1)), promoted, CpuPointwiseOpcode.WHERE);
        }
    }

    @Test void lowOperandsPromoteBeforeArithmeticWhileCrossCategoryStillRejects() {
        var provider = new CpuCapabilityProvider();
        for (var low : List.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            var fixture = fixture(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                    List.of(low, DataType.FLOAT64), DataType.FLOAT64, CpuPointwiseOpcode.ADD);
            assertTrue(provider.supports(new OperationCapabilityQuery(fixture.operation(),
                    fixture.inputDescriptors(), List.of(fixture.outputDescriptor()))));
        }
        var invalid = fixture(new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                List.of(DataType.INT64, DataType.FLOAT64), DataType.BOOL, CpuPointwiseOpcode.EQUAL);
        assertFalse(provider.supports(new OperationCapabilityQuery(invalid.operation(), invalid.inputDescriptors(),
                List.of(invalid.outputDescriptor()))));
    }

    private static void check(Operation operation, List<DataType> inputs, DataType output, CpuPointwiseOpcode opcode) {
        var fixture = fixture(operation, inputs, output, opcode);
        var provider = new CpuCapabilityProvider();
        assertTrue(provider.supports(new OperationCapabilityQuery(operation, fixture.inputDescriptors(),
                List.of(fixture.outputDescriptor()))), fixture.id());
        var context = fixture.context(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(fixture.shape()),
                List.of(), new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                        1, 1, Long.MAX_VALUE),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        assertEquals(fixture.boundaryTypes(), route.specialization().boundaryDataTypes());
        assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, plan.executionStrategy());
        assertEquals(inputs, route.kernelIr().values().stream().limit(inputs.size()).map(v -> v.dataType()).toList());
    }

    private static CpuGeneratedDirectEvidenceClosureTest.PointwiseFixture fixture(Operation operation,
            List<DataType> inputs, DataType output, CpuPointwiseOpcode opcode) {
        return new CpuGeneratedDirectEvidenceClosureTest.PointwiseFixture(operation.kind() + inputs.toString(),
                operation, inputs, output, opcode, Shape.of(32));
    }
}
