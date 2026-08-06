package io.github.pho001.synaptik.backend.cpu.internal.cache;

/** Current-only generated-kernel schema; no legacy reader or migration exists. */
public final class CpuGeneratorSchema {
    /** Current schema version. */ public static final int CURRENT_VERSION = 3;
    /** Generated entry name. */ public static final String ENTRY_NAME = "invoke";
    private CpuGeneratorSchema() { }

    /**
     * Returns a deterministic generated binary name.
     * @param specialization non-null exact structural specialization
     * @return a deterministic binary name in the CPU code-generation package; never {@code null}
     * @throws NullPointerException if {@code specialization} is {@code null}
     */
    public static String generatedBinaryName(CpuKernelSpecialization specialization) {
        return "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.Generated_"
                + specialization.structuralKey();
    }
}
