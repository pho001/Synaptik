import Benchmark.autotune.UnsafeCandidateHistory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class UnsafeCandidateHistoryTest {

    @TempDir
    Path tempDir;

    @Test
    void saveAndLoadRoundTripsOnlyMatchingContextAndSanitizesReason() throws Exception {
        Path historyPath = tempDir.resolve("history.tsv");

        UnsafeCandidateHistory history = UnsafeCandidateHistory.empty("ctx-a");
        history.markUnsafe("fp-1", "AUTO_1", "BAD\tREASON\nDETAIL");
        history.markUnsafe("fp-1", "AUTO_1", "IGNORED_DUPLICATE");
        history.save(historyPath);

        String text = Files.readString(historyPath);
        assertTrue(text.contains("# fingerprint\tstatus\treason\ttimestamp\tcontext"));
        assertTrue(text.contains("fp-1\tUNSAFE\tBAD REASON DETAIL candidate=AUTO_1\t"));

        UnsafeCandidateHistory sameContext = UnsafeCandidateHistory.load(historyPath, "ctx-a");
        UnsafeCandidateHistory otherContext = UnsafeCandidateHistory.load(historyPath, "ctx-b");
        assertTrue(sameContext.isUnsafe("fp-1"));
        assertFalse(otherContext.isUnsafe("fp-1"));
    }

    @Test
    void saveSkipsWriteWhenNothingChanged() {
        Path historyPath = tempDir.resolve("history.tsv");

        UnsafeCandidateHistory.empty("ctx-a").save(historyPath);

        assertFalse(Files.exists(historyPath));
    }
}
