package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.classfile.ClassFile;

/**
 * Defines the schema-one compatibility constants shared by CPU specialization fingerprints and
 * generated class files. These constants version the current CPU-internal encoding; they are not
 * a persistent-cache format or an architecture-level choice of bytecode API.
 */
final class CpuGeneratorSchema {
    /** Canonical specialization encoding and generator-behavior version. */
    static final int CURRENT_VERSION = 1;
    /** Required major version for every class emitted by this generator foundation. */
    static final int CLASSFILE_MAJOR_VERSION = ClassFile.JAVA_26_VERSION;
    /** Exact byte length of both lowering and specialization SHA-256 fingerprints. */
    static final int FINGERPRINT_BYTE_COUNT = 32;

    private CpuGeneratorSchema() {}
}
