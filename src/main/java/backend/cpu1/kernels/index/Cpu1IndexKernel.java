package backend.cpu1.kernels.index;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;

/**
 * Prepared index loop entry point.
 */
@FunctionalInterface
public interface Cpu1IndexKernel {
    void run(Cpu1PreparedIndexUnit unit, Cpu1TensorView input, Cpu1TensorView indices, Cpu1TensorView output);

    default void run(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    ) {
        if (updates != null) {
            throw new UnsupportedOperationException("cpu1 " + unit.opType() + " kernel does not accept updates input.");
        }
        run(unit, input, indices, output);
    }

    static Cpu1IndexKernel withUpdates(Cpu1IndexUpdateKernel kernel) {
        if (kernel == null) {
            throw new IllegalArgumentException("kernel cannot be null");
        }
        return new Cpu1IndexKernel() {
            @Override
            public void run(
                    Cpu1PreparedIndexUnit unit,
                    Cpu1TensorView input,
                    Cpu1TensorView indices,
                    Cpu1TensorView output
            ) {
                throw new UnsupportedOperationException("cpu1 " + unit.opType() + " kernel requires updates input.");
            }

            @Override
            public void run(
                    Cpu1PreparedIndexUnit unit,
                    Cpu1TensorView input,
                    Cpu1TensorView indices,
                    Cpu1TensorView updates,
                    Cpu1TensorView output
            ) {
                if (updates == null) {
                    throw new IllegalArgumentException("updates cannot be null for cpu1 " + unit.opType());
                }
                kernel.run(unit, input, indices, updates, output);
            }
        };
    }
}

@FunctionalInterface
interface Cpu1IndexUpdateKernel {
    void run(
            Cpu1PreparedIndexUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView indices,
            Cpu1TensorView updates,
            Cpu1TensorView output
    );
}
