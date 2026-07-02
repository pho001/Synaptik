package io.github.pho001.synaptik.model;

/**
 * Classifies a {@link DataType} by its mathematical value domain.
 *
 * <p>The category is backend-independent and describes which semantic rules may apply to a data
 * type. It does not describe a physical storage format, a device capability, or an execution
 * route.</p>
 */
public enum DataTypeCategory {
    /** Data types that represent real-valued floating-point numbers. */
    FLOATING,

    /** Data types that represent signed whole numbers. */
    INTEGRAL,

    /** The data type category whose values are strictly logical false or true. */
    BOOLEAN
}
