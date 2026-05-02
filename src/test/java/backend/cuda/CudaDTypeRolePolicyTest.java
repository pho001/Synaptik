package backend.cuda;

import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaDTypeRolePolicyTest {
    @Test
    void float32ComputeInputAndOutputAreSupported() {
        assertTrue(CudaDTypeRolePolicy.computeInput(DataType.FLOAT32).supported());
        assertTrue(CudaDTypeRolePolicy.computeOutput(DataType.FLOAT32).supported());
    }

    @Test
    void residencyOnlyDTypesDoNotBecomeComputeSupport() {
        for (DataType dataType : new DataType[]{DataType.BFLOAT16, DataType.BOOL, DataType.INT32}) {
            var input = CudaDTypeRolePolicy.computeInput(dataType);
            var output = CudaDTypeRolePolicy.computeOutput(dataType);

            assertFalse(input.supported());
            assertFalse(output.supported());
            assertTrue(input.detail().contains("RESIDENCY_ONLY_NOT_COMPUTE"));
            assertTrue(output.detail().contains("dtype residency is not native dtype compute"));
            assertTrue(CudaDTypeRolePolicy.residencyOnly(dataType).supported());
        }
    }

    @Test
    void int32IndexInputIsRoleScoped() {
        var index = CudaDTypeRolePolicy.indexInput(DataType.INT32);

        assertTrue(index.supported());
        assertTrue(index.detail().contains("INDEX_INPUT"));
        assertFalse(CudaDTypeRolePolicy.indexInput(DataType.FLOAT32).supported());
        assertFalse(CudaDTypeRolePolicy.computeOutput(DataType.INT32).supported());
    }

    @Test
    void boolPredicateInputIsRoleScoped() {
        var predicate = CudaDTypeRolePolicy.predicateInput(DataType.BOOL);

        assertTrue(predicate.supported());
        assertTrue(predicate.detail().contains("PREDICATE_INPUT"));
        assertFalse(CudaDTypeRolePolicy.predicateInput(DataType.FLOAT32).supported());
        assertFalse(CudaDTypeRolePolicy.computeOutput(DataType.BOOL).supported());
    }

    @Test
    void float64IsUnsupportedForCudaNativeRoles() {
        assertFalse(CudaDTypeRolePolicy.computeInput(DataType.FLOAT64).supported());
        assertFalse(CudaDTypeRolePolicy.computeOutput(DataType.FLOAT64).supported());
        assertFalse(CudaDTypeRolePolicy.residencyOnly(DataType.FLOAT64).supported());
    }
}
