package backend.cpu.fused.optimize;

import backend.cpu.fused.codegen.FusedDTypeOps;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class FusedPrecisionResolver {
    public static int resolve(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        DataType target = null;

        List<Tensor> all = new ArrayList<>();
        if (cluster != null) all.addAll(cluster);
        if (externalInputsInOrder != null) all.addAll(externalInputsInOrder);
        if (root != null) all.add(root);

        for (Tensor t : all) {
            if (t == null) continue;
            DataType dt = t.getDataType();
            if (dt == DataType.FLOAT64) {
                target = DataType.FLOAT64;
                break;
            }
            if (dt == DataType.FLOAT32) {
                target = DataType.FLOAT32;
                continue;
            }
            if (dt == DataType.BFLOAT16 && target == null) {
                target = DataType.BFLOAT16;
                continue;
            }
            if (dt == DataType.BOOL) {
                continue;
            }
            if (dt == DataType.INT32) {
                throw new UnsupportedOperationException("INT32 tensors are not supported in fused precision resolution.");
            }
        }

        return switch (target) {
            case null -> FusedDTypeOps.MODE_F32;
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case BFLOAT16 -> FusedDTypeOps.MODE_BF16;
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL tensors are not supported in fused precision resolution.");
        };
    }
}
