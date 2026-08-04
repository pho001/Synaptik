package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.classfile.ClassFile;

/**
 * Defines the schema-one compatibility constants shared by CPU specialization fingerprints,
 * generated class files, and the durable generated-artifact envelope. These constants version
 * CPU-internal encodings; they are neither public formats nor an architecture-level choice of
 * bytecode API.
 */
final class CpuGeneratorSchema {
    /** Canonical specialization encoding and generator-behavior version. */
    static final int CURRENT_VERSION = 1;
    /** Required major version for every class emitted by this generator foundation. */
    static final int CLASSFILE_MAJOR_VERSION = ClassFile.JAVA_26_VERSION;
    /** Required minor version for every generated class. */
    static final int CLASSFILE_MINOR_VERSION = 0;
    /** Exact Java feature used to emit, verify, and define stored class bytes. */
    static final int JAVA_FEATURE_VERSION = 26;
    /** Exact byte length of both lowering and specialization SHA-256 fingerprints. */
    static final int FINGERPRINT_BYTE_COUNT = 32;
    /** Durable envelope and namespace version. */
    static final int ARTIFACT_FORMAT_VERSION = 1;
    /** Canonical artifact-key metadata domain version. */
    static final int ARTIFACT_KEY_DOMAIN_VERSION = 1;
    /** Eight-byte durable-envelope discriminator. */
    static final String ARTIFACT_MAGIC = "SYNCPUK1";
    /** Domain separator used before canonical metadata in the path digest. */
    static final String ARTIFACT_KEY_DOMAIN =
            "synaptik.cpu.generated-kernel.artifact-key.v1";
    /** Fixed binary-name prefix for generated hidden-kernel classes. */
    static final String GENERATED_BINARY_NAME_PREFIX =
            "io.github.pho001.synaptik.backend.cpu.execution.CpuGenerated$";
    /** Exact generated static entry name. */
    static final String GENERATED_ENTRY_NAME = "invoke";

    /**
     * Returns the exact binary class name for one complete specialization.
     *
     * @param specialization non-null complete specialization
     * @return deterministic binary name derived from the complete specialization fingerprint
     * @throws NullPointerException if {@code specialization} is {@code null}
     */
    static String generatedBinaryName(CpuKernelSpecialization specialization) {
        return GENERATED_BINARY_NAME_PREFIX
                + java.util.Objects.requireNonNull(specialization, "specialization")
                        .specializationFingerprint();
    }

    private CpuGeneratorSchema() {}
}
