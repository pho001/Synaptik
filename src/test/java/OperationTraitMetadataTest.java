import planning.region.lowering.OperationSemanticClassifier;
import planning.region.lowering.OperationSemanticLevel;
import operations.Operation;
import operations.elementwise.binary.add;
import operations.elementwise.compare.greaterThan;
import operations.elementwise.logical.logicalAnd;
import operations.elementwise.unary.exp;
import operations.elementwise.unary.sqrt;
import operations.elementwise.where.where;
import operations.layout.reshape;
import operations.linalg.matmul;
import operations.reduction.argMax;
import operations.reduction.sum;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OperationTraitMetadataTest {
    @Test
    void representativeOperationsDeclareObjectModelMetadata() {
        List<Operation> operations = List.of(
                new add(),
                new sqrt(),
                new exp(),
                new greaterThan(null),
                new logicalAnd(null),
                new where(),
                new sum(-1),
                new argMax(1, false),
                new reshape(new int[]{2, 3}),
                new matmul()
        );

        for (Operation operation : operations) {
            assertNotNull(operation.opType(), operation.getClass().getName());
            assertNotNull(operation.arityClass(), operation.getClass().getName());
            assertNotNull(operation.semanticFamily(), operation.getClass().getName());
            assertNotNull(operation.computationalCost(), operation.getClass().getName());
            assertNotNull(operation.controlTrait(), operation.getClass().getName());
            assertNotNull(operation.resultKind(), operation.getClass().getName());
        }
    }

    @Test
    void representativeElementwiseTraitsComeFromConcreteOperations() {
        Operation add = new add();
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, add.arityClass());
        assertTrue(add.isFusable());
        assertEquals(Operation.OpSemanticFamily.ARITHMETIC, add.semanticFamily());
        assertEquals(Operation.OpComputationalCost.CHEAP, add.computationalCost());
        assertEquals(Operation.OpResultKind.NUMERIC, add.resultKind());

        Operation sqrt = new sqrt();
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, sqrt.arityClass());
        assertTrue(sqrt.isFusable());
        assertEquals(Operation.OpSemanticFamily.ARITHMETIC, sqrt.semanticFamily());
        assertEquals(Operation.OpComputationalCost.MEDIUM, sqrt.computationalCost());

        Operation exp = new exp();
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, exp.arityClass());
        assertTrue(exp.isFusable());
        assertEquals(Operation.OpSemanticFamily.TRANSCENDENTAL, exp.semanticFamily());
        assertEquals(Operation.OpComputationalCost.EXPENSIVE, exp.computationalCost());

        Operation greaterThan = new greaterThan(null);
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, greaterThan.arityClass());
        assertTrue(greaterThan.isFusable());
        assertEquals(Operation.OpSemanticFamily.COMPARISON, greaterThan.semanticFamily());
        assertEquals(Operation.OpControlTrait.BRANCHLESS, greaterThan.controlTrait());
        assertEquals(Operation.OpResultKind.BOOLEAN, greaterThan.resultKind());

        Operation logicalAnd = new logicalAnd(null);
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, logicalAnd.arityClass());
        assertTrue(logicalAnd.isFusable());
        assertEquals(Operation.OpSemanticFamily.LOGICAL, logicalAnd.semanticFamily());
        assertEquals(Operation.OpControlTrait.BOOL_LOGIC, logicalAnd.controlTrait());
        assertEquals(Operation.OpResultKind.BOOLEAN, logicalAnd.resultKind());

        Operation where = new where();
        assertEquals(Operation.OpArityClass.ELEMENT_WISE, where.arityClass());
        assertTrue(where.isFusable());
        assertEquals(Operation.OpSemanticFamily.SELECTION, where.semanticFamily());
        assertEquals(Operation.OpControlTrait.SELECT_MASK, where.controlTrait());
        assertEquals(Operation.OpResultKind.NUMERIC, where.resultKind());
    }

    @Test
    void representativeNonElementwiseTraitsComeFromConcreteOperations() {
        Operation sum = new sum(-1);
        assertEquals(Operation.OpArityClass.REDUCTION, sum.arityClass());
        assertFalse(sum.isFusable());
        assertEquals(Operation.OpSemanticFamily.REDUCTION, sum.semanticFamily());
        assertEquals(Operation.OpResultKind.NUMERIC, sum.resultKind());

        Operation argMax = new argMax(1, false);
        assertEquals(Operation.OpArityClass.REDUCTION, argMax.arityClass());
        assertFalse(argMax.isFusable());
        assertEquals(Operation.OpSemanticFamily.REDUCTION, argMax.semanticFamily());
        assertEquals(Operation.OpResultKind.INDEX, argMax.resultKind());

        Operation reshape = new reshape(new int[]{2, 3});
        assertEquals(Operation.OpArityClass.LAYOUT, reshape.arityClass());
        assertFalse(reshape.isFusable());
        assertEquals(Operation.OpSemanticFamily.LAYOUT, reshape.semanticFamily());
        assertEquals(Operation.OpResultKind.SHAPE_VIEW, reshape.resultKind());

        Operation matmul = new matmul();
        assertEquals(Operation.OpArityClass.LINEAR_ALGEBRA, matmul.arityClass());
        assertFalse(matmul.isFusable());
        assertEquals(Operation.OpSemanticFamily.LINEAR_ALGEBRA, matmul.semanticFamily());
        assertEquals(Operation.OpComputationalCost.EXPENSIVE, matmul.computationalCost());
    }

    @Test
    void semanticClassifierUsesConcreteOperationMetadataWhenAvailable() {
        assertEquals(OperationSemanticLevel.PRIMITIVE,
                OperationSemanticClassifier.classify(new add()));
        assertEquals(OperationSemanticLevel.PRIMITIVE,
                OperationSemanticClassifier.classify(new matmul()));
        assertEquals(OperationSemanticLevel.LAYOUT,
                OperationSemanticClassifier.classify(new reshape(new int[]{2, 3})));
        assertEquals(OperationSemanticLevel.CANONICAL_HIGH_LEVEL,
                OperationSemanticClassifier.classify(Operation.OpType.LINEAR));
        assertEquals(OperationSemanticLevel.UNKNOWN,
                OperationSemanticClassifier.classify(Operation.OpType.CONST_SCALAR));
    }
}
