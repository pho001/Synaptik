package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TuiViewTest {
    @Test
    void everyViewportFitsUnicodeControlsAndKeepsFocusedInputInView() {
        TuiForm form = new TuiForm(TuiForm.Action.TRAIN);
        TuiFormTest.type(form, "界🙂".repeat(90));
        for (int columns : new int[] {0, 1, 2, 8, 20, 40, 72, 80, 140, 400}) {
            for (int rows : new int[] {0, 1, 2, 4, 8, 12, 24, 80}) {
                for (int selection = 0; selection < 4; selection++) {
                    fits(TuiView.menu(selection, columns, rows, "bad\033[2J\nnotice"), columns, rows);
                }
                TuiView.Frame frame = TuiView.form(form, columns, rows, "bad\033[2J\nnotice");
                fits(frame, columns, rows);
                if (columns >= 8 && rows > 0) {
                    assertNotNull(frame.cursor());
                    assertTrue(frame.lines().get(frame.cursor().row()).contains("🙂"));
                }
            }
        }
        // Vertical focus follows the real enabled fields rather than the top of a long form.
        for (int i = 0; i < 10; i++) TuiFormTest.key(form, TuiConsole.Kind.TAB);
        TuiView.Frame frame = TuiView.form(form, 30, 5, null);
        fits(frame, 30, 5);
        assertNotNull(frame.cursor());
    }

    @Test
    void horizontalWindowTracksActualCodePointCursorInBothDirections() {
        TuiForm form = new TuiForm(TuiForm.Action.INSPECT);
        TuiFormTest.type(form, "界🙂".repeat(30));
        TuiView.Frame end = TuiView.form(form, 12, 4, null);
        assertNotNull(end.cursor());
        assertTrue(end.cursor().column() < 11);
        TuiFormTest.key(form, TuiConsole.Kind.HOME);
        TuiView.Frame start = TuiView.form(form, 12, 4, null);
        assertNotNull(start.cursor());
        assertEquals(2, start.cursor().column());
        TuiFormTest.key(form, TuiConsole.Kind.RIGHT);
        TuiView.Frame one = TuiView.form(form, 12, 4, null);
        assertEquals(start.cursor().column() + 2, one.cursor().column());
        TuiFormTest.key(form, TuiConsole.Kind.RIGHT);
        TuiView.Frame two = TuiView.form(form, 12, 4, null);
        assertEquals(one.cursor().column() + 2, two.cursor().column());
    }

    @Test
    void jobUsesActualLossPointsAndRecentEventsWithinResizedViewport() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        stats.beginEpoch(1);
        stats.step(-9, 2, 24);
        stats.snapshot(); // History is sampled at refresh, not once for every recorded update.
        clock.addAndGet(1_000_000_000L);
        stats.step(3, 2, 24);
        TuiView.Progress progress = new TuiView.Progress(2, 10, 20, "CPU", stats.snapshot(), null, null);
        TuiView.Job job = new TuiView.Job("Train", true, false, 0, null, List.of(), progress, clock.get());
        List<String> events = List.of("old-event", "middle-event", "new-event\033[2J");
        TuiView.Frame large = TuiView.job(job, events, 140, 40);
        assertEquals(2, large.lines().stream().flatMapToInt(String::chars).filter(ch -> ch == '*').count());
        assertTrue(large.lines().stream().anyMatch(line -> line.contains("new-event")));
        for (int columns : new int[] {0, 1, 5, 31, 80, 140, 400}) {
            for (int rows : new int[] {0, 1, 2, 5, 12, 24, 40, 100}) {
                fits(TuiView.job(job, events, columns, rows), columns, rows);
            }
        }
        TuiView.Job inspection = new TuiView.Job("Inspect", false, false, 0, null,
                List.of("user model 界🙂", "unsafe\033[2J\nvalue"), null, 19);
        TuiView.Frame inspected = TuiView.job(inspection, List.of(), 80, 12);
        assertTrue(inspected.lines().stream().anyMatch(line -> line.contains("界🙂")));
        fits(inspected, 80, 12);
    }

    @Test
    void displayWidthNeverSplitsSupplementaryOrWideCodePoints() {
        assertEquals("界", SynaptikBrand.clip("界🙂x", 3));
        assertEquals("界🙂", SynaptikBrand.clip("界🙂x", 4));
        assertEquals(4, SynaptikBrand.width("界🙂"));
        assertEquals("e\u0301", SynaptikBrand.clip("e\u0301x", 1));
        assertFalse(SynaptikBrand.plain("bad\033[2J\r\n\u009b").codePoints().anyMatch(Character::isISOControl));
    }

    @Test
    void normalNarrowDashboardRetainsFinitePointsAndNonfiniteGapsAfterEpochHistoryFills() {
        AtomicLong clock = new AtomicLong();
        TrainingProgressStats stats = new TrainingProgressStats(clock::get);
        for (int epoch = 1; epoch <= 9; epoch++) {
            stats.beginEpoch(epoch);
            stats.step(epoch == 9 ? Double.NaN : epoch, 1, 8);
            clock.addAndGet(1_000_000);
            stats.endEpoch();
            stats.snapshot();
        }
        var progress = new TuiView.Progress(20, 1, 1, "CPU", stats.snapshot(), null, null);
        var job = new TuiView.Job("Train", true, false, 0, null, List.of(), progress, clock.get());
        var frame = TuiView.job(job, List.of("actual completion event"), 80, 24);
        fits(frame, 80, 24);
        assertEquals(8, frame.lines().stream().flatMapToInt(String::chars).filter(point -> point == '*').count());
        assertTrue(frame.lines().stream().anyMatch(line -> line.indexOf('?') >= 0
                && line.chars().allMatch(point -> point == ' ' || point == '*' || point == '?')));
    }

    @Test
    void editCursorFollowsRenderedReplacementWithoutChangingStoredFormatCharacters() {
        var form = new TuiForm(TuiForm.Action.INSPECT);
        int before = TuiView.form(form, 80, 24, null).cursor().column();
        TuiFormTest.type(form, "👩‍💻\uDB40\uDC01");
        var frame = TuiView.form(form, 80, 24, null);
        assertEquals(before + 6, frame.cursor().column()); // Two wide symbols and two safe replacement cells.
        fits(frame, 80, 24);
    }

    private static void fits(TuiView.Frame frame, int columns, int rows) {
        assertTrue(frame.lines().size() <= rows);
        for (String line : frame.lines()) {
            assertTrue(SynaptikBrand.width(line) <= Math.max(0, columns - 1));
            assertFalse(line.codePoints().anyMatch(Character::isISOControl));
            for (int i = 0; i < line.length(); i++) {
                if (Character.isHighSurrogate(line.charAt(i))) assertTrue(++i < line.length() && Character.isLowSurrogate(line.charAt(i)));
                else assertFalse(Character.isLowSurrogate(line.charAt(i)));
            }
        }
        if (frame.cursor() != null) {
            assertTrue(frame.cursor().row() >= 0 && frame.cursor().row() < frame.lines().size());
            assertTrue(frame.cursor().column() >= 0 && frame.cursor().column() < columns - 1);
        }
    }
}
