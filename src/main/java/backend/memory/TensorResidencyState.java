package backend.memory;

import java.util.Objects;

/**
 * Mutable per-run residency state for one runtime tensor value.
 *
 * <p>The state is deliberately separate from semantic {@code Tensor} graph
 * construction. It belongs to a prepared execution run and records whether the
 * CPU representation and an optional device representation are current. Future
 * Metal zero-copy work can attach buffer handles next to this state without
 * changing the public tensor API.</p>
 */
public final class TensorResidencyState {
    private StorageResidency residency;
    private boolean cpuCurrent;
    private boolean nativeCurrent;
    private boolean deviceCurrent;
    private String deviceBackend;
    private String lastTransitionReason;

    private TensorResidencyState(
            StorageResidency residency,
            boolean cpuCurrent,
            boolean nativeCurrent,
            boolean deviceCurrent,
            String deviceBackend,
            String lastTransitionReason
    ) {
        this.residency = Objects.requireNonNull(residency, "residency cannot be null");
        this.cpuCurrent = cpuCurrent;
        this.nativeCurrent = nativeCurrent;
        this.deviceCurrent = deviceCurrent;
        this.deviceBackend = normalize(deviceBackend);
        this.lastTransitionReason = normalize(lastTransitionReason);
    }

    /**
     * Creates a state for a value that is current in CPU array storage.
     *
     * @param reason transition reason used in diagnostics
     * @return CPU-current residency state
     */
    public static TensorResidencyState cpuArrayCurrent(String reason) {
        return new TensorResidencyState(StorageResidency.CPU_ARRAY, true, false, false, "", reason);
    }

    /**
     * Creates a state for an allocated runtime tensor whose semantic value is not current yet.
     *
     * @param reason transition reason used in diagnostics
     * @return CPU-array residency with no current value
     */
    public static TensorResidencyState cpuArrayStale(String reason) {
        return new TensorResidencyState(StorageResidency.CPU_ARRAY, false, false, false, "", reason);
    }

    /**
     * Marks a CPU-side write as the newest representation.
     *
     * @param reason diagnostic reason
     */
    public void markCpuCurrent(String reason) {
        residency = StorageResidency.CPU_ARRAY;
        cpuCurrent = true;
        nativeCurrent = false;
        deviceCurrent = false;
        deviceBackend = "";
        lastTransitionReason = normalize(reason);
    }

    /**
     * Marks a native CPU storage write as the newest representation.
     *
     * @param reason diagnostic reason
     */
    public void markNativeCurrent(String reason) {
        residency = StorageResidency.CPU_NATIVE;
        cpuCurrent = false;
        nativeCurrent = true;
        deviceCurrent = false;
        deviceBackend = "";
        lastTransitionReason = normalize(reason);
    }

    /**
     * Marks a device-side write as the newest representation.
     *
     * @param residency new device residency; must not be {@link StorageResidency#CPU_ARRAY} or
     *                  {@link StorageResidency#CPU_NATIVE}
     * @param deviceBackend backend id such as {@code GPU_METAL}
     * @param reason diagnostic reason
     */
    public void markDeviceCurrent(StorageResidency residency, String deviceBackend, String reason) {
        if (residency == StorageResidency.CPU_ARRAY || residency == StorageResidency.CPU_NATIVE) {
            throw new IllegalArgumentException("device writes require a device residency.");
        }
        this.residency = Objects.requireNonNull(residency, "residency cannot be null");
        this.cpuCurrent = false;
        this.nativeCurrent = false;
        this.deviceCurrent = true;
        this.deviceBackend = normalize(deviceBackend);
        this.lastTransitionReason = normalize(reason);
    }

    /**
     * Marks a host-shared buffer as current from both CPU and device perspectives.
     *
     * <p>This state represents a true shared-buffer contract: CPU reads do not require a download,
     * and device reads do not require an upload, provided the owner obeys the synchronization rules
     * of the backing native buffer. It is only valid for {@link StorageResidency#HOST_SHARED_DEVICE_BUFFER}.</p>
     *
     * @param deviceBackend backend id such as {@code GPU_METAL}
     * @param reason diagnostic reason
     */
    public void markSharedBufferCurrent(String deviceBackend, String reason) {
        residency = StorageResidency.HOST_SHARED_DEVICE_BUFFER;
        cpuCurrent = true;
        nativeCurrent = false;
        deviceCurrent = true;
        this.deviceBackend = normalize(deviceBackend);
        lastTransitionReason = normalize(reason);
    }

    /**
     * Marks a successful device-to-CPU synchronization.
     *
     * @param reason diagnostic reason
     */
    public void markMaterializedToCpu(String reason) {
        residency = StorageResidency.CPU_ARRAY;
        cpuCurrent = true;
        nativeCurrent = false;
        deviceCurrent = false;
        deviceBackend = "";
        lastTransitionReason = normalize(reason);
    }

    /**
     * Returns whether CPU array data must be materialized before a CPU read.
     *
     * @return true when the CPU representation is stale and a device representation is current
     */
    public boolean requiresCpuMaterialization() {
        return !cpuCurrent && (nativeCurrent || deviceCurrent);
    }

    /**
     * Returns whether native CPU storage must be materialized before a native CPU read.
     *
     * @return true when native storage is stale but CPU array storage is current
     */
    public boolean requiresNativeMaterialization() {
        return !nativeCurrent && cpuCurrent;
    }

    public StorageResidency residency() {
        return residency;
    }

    public boolean cpuCurrent() {
        return cpuCurrent;
    }

    public boolean nativeCurrent() {
        return nativeCurrent;
    }

    public boolean deviceCurrent() {
        return deviceCurrent;
    }

    public String deviceBackend() {
        return deviceBackend;
    }

    public String lastTransitionReason() {
        return lastTransitionReason;
    }

    private static String normalize(String value) {
        return value == null ? "" : value;
    }
}
