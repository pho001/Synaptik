package Tensor;

import Operations.Operation;
import Operations.add;
import Operations.div;
import Operations.mul;
import Operations.sub;

import java.util.List;

final class TensorBinaryOps {
    private TensorBinaryOps() {}

    static Tensor add(Tensor first, Tensor second) {
        Operation op = new add();
        Tensor out = new Tensor(first.getShape(), List.of(first, second), op, "+");

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
}
