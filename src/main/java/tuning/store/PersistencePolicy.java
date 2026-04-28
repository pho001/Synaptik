package tuning.store;

import java.nio.file.Path;

/**
 * Controls autotune persistence side effects.
 *
 * <p>The policy is declarative. Sessions write history entries only when
 * {@link #persistHistory()} is true and {@link #historyPath()} is non-null, and
 * write the selected best profile only when {@link #persistBestProfile()} is true
 * and {@link #bestProfilePath()} is non-null.</p>
 *
 * @param persistBestProfile whether to save the best profile
 * @param persistHistory whether to append per-candidate history entries
 * @param bestProfilePath optional best-profile destination
 * @param historyPath optional history destination
 */
public record PersistencePolicy(
        boolean persistBestProfile,
        boolean persistHistory,
        Path bestProfilePath,
        Path historyPath
) {
    /**
     * @return policy that performs no persistence
     */
    public static PersistencePolicy disabled() {
        return new PersistencePolicy(false, false, null, null);
    }
}
