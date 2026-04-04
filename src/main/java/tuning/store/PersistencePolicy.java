package tuning.store;

import java.nio.file.Path;

public record PersistencePolicy(
        boolean persistBestProfile,
        boolean persistHistory,
        Path bestProfilePath,
        Path historyPath
) {
    public static PersistencePolicy disabled() {
        return new PersistencePolicy(false, false, null, null);
    }
}
