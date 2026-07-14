package io.github.pho001.synaptik.backend.contract;

/**
 * Defines the coarse declarative category of a device: central processing unit or accelerator.
 *
 * <p>{@link #CPU} describes a device that executes through a general-purpose central processing
 * unit. Scalar, vector, assembly, and OpenBLAS implementations are routes within a CPU backend,
 * not additional device classes. {@link #ACCELERATOR} describes a non-CPU compute device intended
 * for offloaded computation. It may describe a graphics processing unit, neural processing unit,
 * or another accelerator without promising a subtype, capability, performance characteristic,
 * memory topology, vendor, or execution route.</p>
 *
 * <p>This type supplies category vocabulary only. A {@link BackendAvailabilitySnapshot} may
 * associate a category with a {@link BackendDeviceId}; the category is not stored in the device
 * identity itself. A category does not identify a backend or device, establish availability or
 * capability, express configuration preference, or select backend ownership or an implementation
 * route.</p>
 *
 * <p>The declaration order is stable for enum identity and diagnostics. Its ordinal order does
 * not express preference, score, priority, capability, or fallback order.</p>
 */
public enum DeviceClass {
    /** A device that executes through a general-purpose central processing unit. */
    CPU,

    /** A non-CPU compute device intended for offloaded computation. */
    ACCELERATOR
}
