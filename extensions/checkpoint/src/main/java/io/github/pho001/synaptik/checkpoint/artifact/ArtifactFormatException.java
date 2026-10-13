package io.github.pho001.synaptik.checkpoint.artifact;

import java.io.IOException;
import java.util.Objects;

/**
 * Reports an invalid, unsupported, corrupt, or configured-limit-exceeding artifact image.
 * No partially decoded image is returned when this exception is raised.
 */
public final class ArtifactFormatException extends IOException {
    /**
     * Constructs a format failure with a diagnostic description.
     *
     * @param message non-null description of the rejected artifact
     * @throws NullPointerException if {@code message} is null
     */
    public ArtifactFormatException(String message) {
        super(Objects.requireNonNull(message, "message"));
    }

    /**
     * Constructs a format failure caused by invalid encoded data or checked arithmetic.
     *
     * @param message non-null description of the rejected artifact
     * @param cause non-null underlying failure
     * @throws NullPointerException if {@code message} or {@code cause} is null
     */
    public ArtifactFormatException(String message, Throwable cause) {
        super(Objects.requireNonNull(message, "message"), Objects.requireNonNull(cause, "cause"));
    }
}
