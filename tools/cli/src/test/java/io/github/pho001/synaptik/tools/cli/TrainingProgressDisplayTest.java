package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TrainingProgressDisplayTest {
    @Test
    void resizeAccountsForOwnedHardLineReflowWithoutRetainingWholeTerminalContents() {
        int[] widths = {0, 79, 120, 3};
        assertEquals(4, TrainingProgressDisplay.reflowedRows(widths, 4, 121, 121));
        assertEquals(7, TrainingProgressDisplay.reflowedRows(widths, 4, 121, 40));
        assertEquals(4, TrainingProgressDisplay.reflowedRows(widths, 4, 121, 240));
        assertEquals(0, TrainingProgressDisplay.reflowedRows(widths, 0, 121, 40));
    }

    @Test
    void finiteScalingHandlesZeroConstantsNegativesSubnormalsAndOpposingExtremes() {
        int rows = 7;
        for (double constant : new double[] {0, -0.0, -4, 5, Double.MIN_VALUE, Double.MAX_VALUE}) {
            assertEquals(3, TrainingProgressDisplay.plotRow(constant, constant, constant, rows));
        }
        for (double[] range : new double[][] {
                {-10, -1}, {-Double.MAX_VALUE, Double.MAX_VALUE},
                {Double.MIN_VALUE, Double.MIN_VALUE * 2}, {0, 1}, {-1, 0}}) {
            assertEquals(rows - 1, TrainingProgressDisplay.plotRow(range[0], range[0], range[1], rows));
            assertEquals(0, TrainingProgressDisplay.plotRow(range[1], range[0], range[1], rows));
        }
        assertEquals(3, TrainingProgressDisplay.plotRow(0, -Double.MAX_VALUE, Double.MAX_VALUE, rows));
    }

    @Test
    void plotKeepsActualFinitePointsAndExplicitNonfiniteGapColumns() {
        double[] values = {Double.NaN, -3, Double.POSITIVE_INFINITY, 0, Double.NEGATIVE_INFINITY};
        var plot = TrainingProgressDisplay.plot(values, 30, 5);
        // Only the five grid rows, excluding explanatory labels, participate in point counts.
        long points = plot.subList(2, 7).stream().flatMapToInt(String::chars).filter(ch -> ch == '*').count();
        long gaps = plot.subList(2, 7).stream().flatMapToInt(String::chars).filter(ch -> ch == '?').count();
        assertEquals(2, points);
        assertEquals(3, gaps);
        assertEquals('*', plot.get(6).charAt(1));
        assertEquals('*', plot.get(2).charAt(3));
        for (int col : new int[] {0, 2, 4}) {
            assertEquals('?', plot.get(6).charAt(col));
        }
        var allNonfinite = TrainingProgressDisplay.plot(new double[] {Double.NaN, Double.POSITIVE_INFINITY}, 30, 5);
        assertEquals(0, allNonfinite.subList(2, 7).stream()
                .flatMapToInt(String::chars).filter(ch -> ch == '*').count());
        assertEquals(2, allNonfinite.subList(2, 7).stream()
                .flatMapToInt(String::chars).filter(ch -> ch == '?').count());
    }

    @Test
    void frameFitsEveryAvailableSizeAndDoesNotEmitUserControlSequences() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        for (int epoch = 1; epoch <= 9; epoch++) {
            stats.beginEpoch(epoch);
            stats.step(epoch, 1, 12);
            clock.addAndGet(1_000_000_000);
            stats.endEpoch();
            stats.snapshot();
        }
        var snapshot = stats.snapshot();
        for (int width : new int[] {1, 2, 12, 31, 45, 79, 107, 108, 120, 240}) {
            for (int height : new int[] {1, 2, 5, 9, 16, 17, 24, 40}) {
                var frame = TrainingProgressDisplay.frame(snapshot, TrainingProgress.Outcome.FAILED,
                        "bad\u001b[2J\n\r\tmodel", width, height, 10, 1, 1,
                        "CPU\u001b[H", true);
                assertTrue(frame.size() <= height);
                assertFalse(frame.isEmpty());
                for (String line : frame) {
                    assertTrue(line.length() <= width);
                    assertTrue(line.chars().allMatch(ch -> ch >= 32 && ch <= 126));
                }
            }
        }
    }

    @Test
    void historyMovesBesideOrBelowMetricsInsteadOfDisappearingAtNormalSizes() {
        TrainingProgressStats stats = new TrainingProgressStats(() -> 0);
        stats.beginEpoch(1);
        stats.step(-2, 1, 8);
        var snapshot = stats.snapshot();
        var narrow = TrainingProgressDisplay.frame(snapshot, null, null, 79, 24, 1, 1, 1, "CPU", true);
        var wide = TrainingProgressDisplay.frame(snapshot, null, null, 139, 24, 1, 1, 1, "CPU", true);
        // Inspect the data point, not a logo/ANSI/text golden. Its horizontal location proves
        // the responsive layout while scalar/label wording remains unconstrained.
        int narrowPointColumn = pointColumn(narrow);
        int widePointColumn = pointColumn(wide);
        assertEquals(0, narrowPointColumn);
        assertTrue(widePointColumn >= 79);
        assertTrue(narrow.size() > wide.size());
    }

    @Test
    void realPlainAdapterIsAppendOnlyAndContainsNoTerminalControlsEvenOnFailureDetails() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream error = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        TrainingProgressStats stats = new TrainingProgressStats(() -> 0);
        stats.beginEpoch(1);
        stats.step(Double.NaN, 1, 12);
        try (TrainingProgressDisplay display = TrainingProgressDisplay.open(error, true, 1, 1, 1, "CPU")) {
            display.render(stats.snapshot(), null, null);
            String first = bytes.toString(StandardCharsets.UTF_8);
            display.render(stats.snapshot(), TrainingProgress.Outcome.FAILED, "bad\u001b[2J\nmodel");
            String all = bytes.toString(StandardCharsets.UTF_8);
            assertTrue(all.startsWith(first));
            assertTrue(all.length() > first.length());
            assertTrue(all.chars().allMatch(ch -> (ch >= 32 && ch <= 126) || ch == '\n' || ch == '\r'));
        }
    }

    private static int pointColumn(java.util.List<String> frame) {
        for (String line : frame) {
            int column = line.indexOf('*');
            if (column >= 0) {
                return column;
            }
        }
        throw new AssertionError("Actual finite data point was not rendered");
    }
}
