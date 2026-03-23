package Tensor;

import Operations.Operation;
import Operations.add;
import Operations.div;
import Operations.max;
import Operations.min;
import Operations.mul;
import Operations.sub;

import java.util.List;

final class TensorBinaryOps {
    private TensorBinaryOps() {}

    static Tensor add(Tensor first, Tensor second) {
        Operation op = new add();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "+");
        out.setDataType(TensorDataTypeUtil.binary(first, second));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (first.getRequiresGrad()) {
                if (first.getGradient() == null) {
                    first.setGradient(outGrad);
                } else {
                    first.setGradient(first.getGradient().add(outGrad));
                }
            }

            if (second.getRequiresGrad()) {
                if (second.getGradient() == null) {
                    second.setGradient(outGrad);
                } else {
                    second.setGradient(second.getGradient().add(outGrad));
                }
            }
        });
        return out;
    }

    static Tensor sub(Tensor first, Tensor second) {
        Operation op = new sub();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "-");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (first.getRequiresGrad()) {
                if (first.getGradient() == null) {
                    first.setGradient(outGrad);
                } else {
                    first.setGradient(first.getGradient().add(outGrad));
                }
            }

            if (second.getRequiresGrad()) {
                Tensor gradForSecond = outGrad.neg();
                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });
        return out;
    }

    static Tensor mul(Tensor first, Tensor second) {
        Operation op = new mul();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "*");
        out.setDataType(TensorDataTypeUtil.binary(first, second));

        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (first.getRequiresGrad()) {
                Tensor gradForFirst = outGrad.mul(second);
                if (first.getGradient() == null) {
                    first.setGradient(gradForFirst);
                } else {
                    first.setGradient(first.getGradient().add(gradForFirst));
                }
            }

            if (second.getRequiresGrad()) {
                Tensor gradForSecond = outGrad.mul(first);
                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });

        return out;
    }

    static Tensor div(Tensor first, Tensor second) {
        Operation op = new div();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "/");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();

            if (first.getRequiresGrad()) {
                Tensor gradForFirst = outGrad.div(second);
                if (first.getGradient() == null) {
                    first.setGradient(gradForFirst);
                } else {
                    first.setGradient(first.getGradient().add(gradForFirst));
                }
            }

            if (second.getRequiresGrad()) {
                Tensor gradForSecond = outGrad.neg().mul(first).div(second.pow(2));
                if (second.getGradient() == null) {
                    second.setGradient(gradForSecond);
                } else {
                    second.setGradient(second.getGradient().add(gradForSecond));
                }
            }
        });
        return out;
    }

    static Tensor min(Tensor first, Tensor second) {
        Operation op = new min();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "min");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            double[] a = first.toDoubleArrayCopy();
            double[] b = second.toDoubleArrayCopy();
            double[] og = outGrad.toDoubleArrayCopy();
            double[] ga = new double[og.length];
            double[] gb = new double[og.length];

            for (int i = 0; i < og.length; i++) {
                if (a[i] < b[i]) {
                    ga[i] = og[i];
                } else if (a[i] > b[i]) {
                    gb[i] = og[i];
                } else {
                    double half = 0.5 * og[i];
                    ga[i] = half;
                    gb[i] = half;
                }
            }

            if (first.getRequiresGrad()) {
                Tensor gradForFirst = new Tensor(ga, first.getShape().clone(), null, "min_grad_a", first.getDataType());
                if (first.getGradient() == null) first.setGradient(gradForFirst);
                else first.setGradient(first.getGradient().add(gradForFirst));
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = new Tensor(gb, second.getShape().clone(), null, "min_grad_b", second.getDataType());
                if (second.getGradient() == null) second.setGradient(gradForSecond);
                else second.setGradient(second.getGradient().add(gradForSecond));
            }
        });
        return out;
    }

    static Tensor max(Tensor first, Tensor second) {
        Operation op = new max();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "max");
        out.setDataType(TensorDataTypeUtil.binary(first, second));
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            double[] a = first.toDoubleArrayCopy();
            double[] b = second.toDoubleArrayCopy();
            double[] og = outGrad.toDoubleArrayCopy();
            double[] ga = new double[og.length];
            double[] gb = new double[og.length];

            for (int i = 0; i < og.length; i++) {
                if (a[i] > b[i]) {
                    ga[i] = og[i];
                } else if (a[i] < b[i]) {
                    gb[i] = og[i];
                } else {
                    double half = 0.5 * og[i];
                    ga[i] = half;
                    gb[i] = half;
                }
            }

            if (first.getRequiresGrad()) {
                Tensor gradForFirst = new Tensor(ga, first.getShape().clone(), null, "max_grad_a", first.getDataType());
                if (first.getGradient() == null) first.setGradient(gradForFirst);
                else first.setGradient(first.getGradient().add(gradForFirst));
            }
            if (second.getRequiresGrad()) {
                Tensor gradForSecond = new Tensor(gb, second.getShape().clone(), null, "max_grad_b", second.getDataType());
                if (second.getGradient() == null) second.setGradient(gradForSecond);
                else second.setGradient(second.getGradient().add(gradForSecond));
            }
        });
        return out;
    }
}
