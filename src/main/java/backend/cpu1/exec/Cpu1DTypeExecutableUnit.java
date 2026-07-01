package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;

/**
 * Base executable unit for prepared cpu1 dtype kernels.
 */
public abstract class Cpu1DTypeExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedDTypeUnit preparedUnit;

    protected Cpu1DTypeExecutableUnit(Cpu1PreparedDTypeUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedDTypeUnit preparedUnit() {
        return preparedUnit;
    }
}
