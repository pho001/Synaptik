package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.Objects;

/**
 * Strong owner of one verified hidden generated class, direct entry handle, and deterministic
 * class-byte snapshot. It owns no close lifecycle or persistent storage.
 */
public final class CpuGeneratedKernel {
    private final CpuKernelSpecialization specialization;
    private final MethodHandles.Lookup lookup;
    private final MethodHandle entryPoint;
    private final byte[] classBytes;

    /** Creates one artifact after the generator has verified and defined its exact class shape. */
    CpuGeneratedKernel(CpuKernelSpecialization specialization, MethodHandles.Lookup lookup,
            MethodHandle entryPoint, byte[] classBytes) {
        this.specialization = Objects.requireNonNull(specialization, "specialization");
        this.lookup = Objects.requireNonNull(lookup, "lookup");
        this.entryPoint = Objects.requireNonNull(entryPoint, "entryPoint");
        this.classBytes = Objects.requireNonNull(classBytes, "classBytes").clone();
        if (!entryPoint.type().equals(specialization.entryType())) throw new IllegalArgumentException(
                "entry point type must match specialization");
    }
    /** Returns compatibility facts.
     * @return the non-null immutable structural specialization */
    public CpuKernelSpecialization specialization() { return specialization; }
    /** Returns the defining lookup.
     * @return the non-null lookup retaining the hidden class */
    public MethodHandles.Lookup hiddenLookup() { return lookup; }
    /** Returns loaded identity.
     * @return the non-null hidden class compatibility identity */
    public Class<?> hiddenClass() { return lookup.lookupClass(); }
    /** Returns the invocation target.
     * @return the non-null exact direct static entry handle */
    public MethodHandle entryPoint() { return entryPoint; }
    /** Returns verified bytes.
     * @return a new defensive copy of deterministic class bytes */
    public byte[] classBytes() { return classBytes.clone(); }
}
