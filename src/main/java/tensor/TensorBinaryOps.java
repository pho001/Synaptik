package tensor;

import operations.Operation;
import operations.add;
import operations.div;
import operations.max;
import operations.min;
import operations.mul;
import operations.sub;

import java.util.List;

final class TensorBinaryOps {
    private TensorBinaryOps() {}

    static Tensor add(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new add(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "+");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.sumToShape(outGrad, first.getShape());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.sumToShape(outGrad, second.getShape());
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    static Tensor sub(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new sub(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "-");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.sumToShape(outGrad, first.getShape());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.sumToShape(outGrad.neg(), second.getShape());
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    static Tensor mul(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new mul(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "*");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.sumToShape(outGrad.mul(second), first.getShape());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.sumToShape(outGrad.mul(first), second.getShape());
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    static Tensor div(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new div(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "/");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.sumToShape(outGrad.div(second), first.getShape());
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.sumToShape(
                        outGrad.neg().mul(first).div(second.pow(2)),
                        second.getShape()
                );
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    static Tensor min(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new min(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "min");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.minMaxGradForInput(first, second, outGrad, plan, true, false);
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.minMaxGradForInput(first, second, outGrad, plan, false, false);
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    static Tensor max(Tensor first, Tensor second) {
        BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
        Operation op = new max(plan);
        Tensor out = new Tensor(plan.outShape(), List.of(first, second), op, "max");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = TensorBroadcastOps.minMaxGradForInput(first, second, outGrad, plan, true, true);
                accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = TensorBroadcastOps.minMaxGradForInput(first, second, outGrad, plan, false, true);
                accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }

}
