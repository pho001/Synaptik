import graph.compile.planning.region.lowering.OperationSemanticClassifier;
import graph.compile.planning.region.lowering.OperationSemanticLevel;
import operations.Operation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class OperationTraitMetadataTest {
    @Test
    void everyOpTypeHasTraitMetadata() {
        for (Operation.OpType opType : Operation.OpType.values()) {
            assertNotNull(opType.category(), opType.name());
            assertNotNull(opType.semanticFamily(), opType.name());
            assertNotNull(opType.computationalCost(), opType.name());
            assertNotNull(opType.controlTrait(), opType.name());
            assertNotNull(opType.resultKind(), opType.name());
        }
    }

    @Test
    void representativeElementwiseTraitsMatchBackendNeutralClassification() {
        assertEquals(Operation.OpSemanticFamily.ARITHMETIC, Operation.OpType.ADD.semanticFamily());
        assertEquals(Operation.OpComputationalCost.CHEAP, Operation.OpType.ADD.computationalCost());
        assertEquals(Operation.OpResultKind.NUMERIC, Operation.OpType.ADD.resultKind());

        assertEquals(Operation.OpSemanticFamily.ARITHMETIC, Operation.OpType.SQRT.semanticFamily());
        assertEquals(Operation.OpComputationalCost.MEDIUM, Operation.OpType.SQRT.computationalCost());

        assertEquals(Operation.OpSemanticFamily.TRANSCENDENTAL, Operation.OpType.EXP.semanticFamily());
        assertEquals(Operation.OpComputationalCost.EXPENSIVE, Operation.OpType.EXP.computationalCost());

        assertEquals(Operation.OpSemanticFamily.COMPARISON, Operation.OpType.GT.semanticFamily());
        assertEquals(Operation.OpControlTrait.BRANCHLESS, Operation.OpType.GT.controlTrait());
        assertEquals(Operation.OpResultKind.BOOLEAN, Operation.OpType.GT.resultKind());

        assertEquals(Operation.OpSemanticFamily.LOGICAL, Operation.OpType.LOGICAL_AND.semanticFamily());
        assertEquals(Operation.OpControlTrait.BOOL_LOGIC, Operation.OpType.LOGICAL_AND.controlTrait());
        assertEquals(Operation.OpResultKind.BOOLEAN, Operation.OpType.LOGICAL_AND.resultKind());

        assertEquals(Operation.OpSemanticFamily.SELECTION, Operation.OpType.WHERE.semanticFamily());
        assertEquals(Operation.OpControlTrait.SELECT_MASK, Operation.OpType.WHERE.controlTrait());
        assertEquals(Operation.OpResultKind.NUMERIC, Operation.OpType.WHERE.resultKind());
    }

    @Test
    void representativeNonElementwiseTraitsMatchBackendNeutralClassification() {
        assertEquals(Operation.OpSemanticFamily.REDUCTION, Operation.OpType.SUM.semanticFamily());
        assertEquals(Operation.OpResultKind.NUMERIC, Operation.OpType.SUM.resultKind());

        assertEquals(Operation.OpSemanticFamily.REDUCTION, Operation.OpType.ARGMAX.semanticFamily());
        assertEquals(Operation.OpResultKind.INDEX, Operation.OpType.ARGMAX.resultKind());

        assertEquals(Operation.OpSemanticFamily.LAYOUT, Operation.OpType.RESHAPE.semanticFamily());
        assertEquals(Operation.OpResultKind.SHAPE_VIEW, Operation.OpType.RESHAPE.resultKind());

        assertEquals(Operation.OpSemanticFamily.LINEAR_ALGEBRA, Operation.OpType.MATMUL.semanticFamily());
        assertEquals(Operation.OpComputationalCost.EXPENSIVE, Operation.OpType.MATMUL.computationalCost());
    }

    @Test
    void semanticClassifierPreservesRepresentativeLevelsThroughTraits() {
        assertEquals(OperationSemanticLevel.PRIMITIVE,
                OperationSemanticClassifier.classify(Operation.OpType.ADD));
        assertEquals(OperationSemanticLevel.PRIMITIVE,
                OperationSemanticClassifier.classify(Operation.OpType.MATMUL));
        assertEquals(OperationSemanticLevel.LAYOUT,
                OperationSemanticClassifier.classify(Operation.OpType.RESHAPE));
        assertEquals(OperationSemanticLevel.LAYOUT,
                OperationSemanticClassifier.classify(Operation.OpType.NOOP));
        assertEquals(OperationSemanticLevel.CANONICAL_HIGH_LEVEL,
                OperationSemanticClassifier.classify(Operation.OpType.LINEAR));
        assertEquals(OperationSemanticLevel.UNKNOWN,
                OperationSemanticClassifier.classify(Operation.OpType.CONST_SCALAR));
    }
}
