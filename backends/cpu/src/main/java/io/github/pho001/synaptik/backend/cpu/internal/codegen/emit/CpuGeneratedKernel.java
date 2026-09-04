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
    private static final String LOSS_CONTIGUOUS_INT = "lossContiguousInt";
    private static final String LOSS_GENERIC_AFFINE = "lossGenericAffine";
    private final CpuKernelSpecialization specialization;
    private final MethodHandles.Lookup lookup;
    private final MethodHandle entryPoint;
    private final byte[] classBytes;

    /**
     * Creates one artifact after the generator has verified and defined its exact class shape.
     *
     * @param specialization non-null exact structural specialization
     * @param lookup non-null defining lookup that strongly retains the hidden class
     * @param entryPoint non-null exact static entry handle matching the specialization
     * @param classBytes non-null verified deterministic bytes; copied defensively
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the handle type differs from the specialization
     */
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

    /**
     * Resolves one private loss helper through the retained defining lookup.
     *
     * <p>This is a cold binding operation for schema-58 loss artifacts only.  The caller proves
     * the selected geometry before requesting the direct helper; the generated public entry
     * remains available for all geometry and retains the generic affine fallback.</p>
     *
     * @param contiguousInt whether the caller proved the direct int-address contiguous form
     * @return the private helper with the same exact type as {@link #entryPoint()}
     * @throws IllegalArgumentException if this artifact has no matching private loss helper
     */
    public MethodHandle lossEntryPointFor(boolean contiguousInt) {
        try {
            return lookup.findStatic(lookup.lookupClass(), contiguousInt ? LOSS_CONTIGUOUS_INT
                    : LOSS_GENERIC_AFFINE, entryPoint.type());
        } catch (ReflectiveOperationException failure) {
            throw new IllegalArgumentException("generated loss helper is unavailable", failure);
        }
    }

    /** Returns verified bytes.
     * @return a new defensive copy of deterministic class bytes */
    public byte[] classBytes() { return classBytes.clone(); }
}
