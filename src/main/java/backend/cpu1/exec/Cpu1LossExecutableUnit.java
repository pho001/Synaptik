package backend.cpu1.exec;

import backend.cpu1.kernels.loss.crossentropy.Cpu1DenseCrossEntropyKernel;
import backend.cpu1.kernels.loss.crossentropy.Cpu1CrossEntropyKernel;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernel;
import backend.cpu1.prepare.Cpu1PreparedCrossEntropyLossUnit;
import backend.cpu1.prepare.Cpu1PreparedDenseCrossEntropyLossUnit;
import backend.cpu1.prepare.Cpu1PreparedNllLossUnit;
import runtime.execution.ExecutionContext;

public final class Cpu1LossExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit;
    private final Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit;
    private final Cpu1PreparedNllLossUnit preparedNllLossUnit;
    private final Cpu1CrossEntropyKernel crossEntropyKernel;
    private final Cpu1DenseCrossEntropyKernel denseCrossEntropyKernel;
    private final Cpu1NllLossKernel nllKernel;

    public Cpu1LossExecutableUnit(Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit) {
        if (preparedCrossEntropyLossUnit == null) {
            throw new IllegalArgumentException("preparedCrossEntropyLossUnit cannot be null");
        }
        this.preparedCrossEntropyLossUnit = preparedCrossEntropyLossUnit;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.crossEntropyKernel = preparedCrossEntropyLossUnit.kernel();
        this.denseCrossEntropyKernel = null;
        this.nllKernel = null;
    }

    public Cpu1LossExecutableUnit(Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit) {
        if (preparedDenseCrossEntropyLossUnit == null) {
            throw new IllegalArgumentException("preparedDenseCrossEntropyLossUnit cannot be null");
        }
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = preparedDenseCrossEntropyLossUnit;
        this.preparedNllLossUnit = null;
        this.crossEntropyKernel = null;
        this.denseCrossEntropyKernel = preparedDenseCrossEntropyLossUnit.kernel();
        this.nllKernel = null;
    }

    public Cpu1LossExecutableUnit(Cpu1PreparedNllLossUnit preparedNllLossUnit) {
        if (preparedNllLossUnit == null) {
            throw new IllegalArgumentException("preparedNllLossUnit cannot be null");
        }
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = preparedNllLossUnit;
        this.crossEntropyKernel = null;
        this.denseCrossEntropyKernel = null;
        this.nllKernel = preparedNllLossUnit.kernel();
    }

    public Cpu1PreparedCrossEntropyLossUnit preparedUnit() {
        if (preparedCrossEntropyLossUnit == null) {
            throw new IllegalStateException("This cpu1 loss executable does not expose cross entropy loss.");
        }
        return preparedCrossEntropyLossUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        if (preparedCrossEntropyLossUnit != null) {
            return preparedCrossEntropyLossUnit.scratchBufferSpec();
        }
        if (preparedDenseCrossEntropyLossUnit != null) {
            return preparedDenseCrossEntropyLossUnit.scratchBufferSpec();
        }
        return preparedNllLossUnit.scratchBufferSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        if (preparedCrossEntropyLossUnit != null) {
            crossEntropyKernel.run(preparedCrossEntropyLossUnit, context);
            return;
        }
        if (preparedDenseCrossEntropyLossUnit != null) {
            denseCrossEntropyKernel.run(preparedDenseCrossEntropyLossUnit, context);
            return;
        }
        nllKernel.run(preparedNllLossUnit, context);
    }
}
