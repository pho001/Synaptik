package runtime.memory.transfer;

/**
 * Static support class for one host/device transfer matrix entry.
 */
public enum DeviceTransferSupport {
    DIRECT,
    ARRAY_BRIDGE,
    UNSUPPORTED
}
