package graph.optimizer.fusion;

import graph.codegen.FusedDTypeOps;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class FusedPrecisionResolver {
    public static int resolve(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        DataType target = root != null ? root.getDataType() : DataType.FLOAT64;
        if (target == null) {
            target = DataType.FLOAT64;
        }

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
            if (dt == DataType.BOOL) {
                throw new UnsupportedOperationException("BOOL tensors are not supported in fused precision resolution.");
            }
            if (dt == DataType.FLOAT32 && target == DataType.FLOAT16) {
                target = DataType.FLOAT32;
            }
        }

        return switch (target) {
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case FLOAT16 -> FusedDTypeOps.MODE_F16;
            case BOOL -> throw new UnsupportedOperationException("BOOL tensors are not supported in fused precision resolution.");
        };
    }
}
