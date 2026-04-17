package tensor.ops.binary;

import operations.Operation;
import operations.add;
import operations.div;
import operations.max;
import operations.min;
import operations.mul;
import operations.sub;
import tensor.BroadcastPlan;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorDataTypeUtil;
import tensor.TensorPrimitiveBuilder;

public final class TensorBinaryOps {
    private TensorBinaryOps() {
    }

    public static Tensor add(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new add(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "+",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(outGrad, second.getShape()));
            }
        });
        return out;
    }

    public static Tensor sub(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new sub(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "-",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(outGrad.neg(), second.getShape()));
            }
        });
        return out;
    }

    public static Tensor mul(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new mul(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "*",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad.mul(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(outGrad.mul(first), second.getShape()));
            }
        });
        return out;
    }

    public static Tensor div(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new div(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "/",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(outGrad.div(second), first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor grad = outGrad.neg().mul(first).div(second.pow(2));
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(grad, second.getShape()));
            }
        });
        return out;
    }

    public static Tensor min(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new min(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "min",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                Tensor gradRaw = TensorPrimitiveBuilder.ternary(
                        first,
                        second,
                        outGrad,
                        plan.outShape(),
                        new operations.minGrad(plan, true),
                        "min_grad_a",
                        outGrad.getDataType()
                );
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(gradRaw, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = TensorPrimitiveBuilder.ternary(
                        first,
                        second,
                        outGrad,
                        plan.outShape(),
                        new operations.minGrad(plan, false),
                        "min_grad_b",
                        outGrad.getDataType()
                );
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(gradRaw, second.getShape()));
            }
        });
        return out;
    }

    public static Tensor max(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new max(plan);
        Tensor out = TensorPrimitiveBuilder.binary(first, second, plan.outShape(), op, "max",
                TensorDataTypeUtil.binary(first, second), null);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            if (first.getRequiresGrad()) {
                Tensor gradRaw = TensorPrimitiveBuilder.ternary(
                        first,
                        second,
                        outGrad,
                        plan.outShape(),
                        new operations.maxGrad(plan, true),
                        "max_grad_a",
                        outGrad.getDataType()
                );
                BinarySupport.accumulateGradient(first, TensorBroadcastOps.sumToShape(gradRaw, first.getShape()));
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = TensorPrimitiveBuilder.ternary(
                        first,
                        second,
                        outGrad,
                        plan.outShape(),
                        new operations.maxGrad(plan, false),
                        "max_grad_b",
                        outGrad.getDataType()
                );
                BinarySupport.accumulateGradient(second, TensorBroadcastOps.sumToShape(gradRaw, second.getShape()));
            }
        });
        return out;
    }
}
