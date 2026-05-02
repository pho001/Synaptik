package backend.metal;

/**
 * Metal dtype capability role.
 */
public enum MetalDTypeRole {
    STORAGE("storage"),
    EXTERNAL_INPUT("externalInput"),
    EXTERNAL_INPUT_ROLE("externalInputRole"),
    COMPUTE("compute"),
    OUTPUT("output"),
    OPERATION("operation");

    private final String label;

    MetalDTypeRole(String label) {
        this.label = label;
    }

    public String label() {
        return label;
    }
}
