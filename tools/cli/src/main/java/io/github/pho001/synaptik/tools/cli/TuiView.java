package io.github.pho001.synaptik.tools.cli;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.jline.utils.WCWidth;

/** Pure viewport layout over detached UI data; no terminal, model or worker access. */
final class TuiView {
    record Progress(int epochs, long stepsPerEpoch, long examplesPerEpoch, String backend,
            TrainingProgressStats.Snapshot snapshot, TrainingProgress.Outcome outcome, String detail) {}
    record Job(String title, boolean running, boolean stopRequested, int status, String detail,
            List<String> inspection, Progress progress, long elapsedNanos) {
        Job { inspection = List.copyOf(inspection); }
    }
    record Frame(List<String> lines, TuiConsole.Cursor cursor) {
        Frame { lines = List.copyOf(lines); }
    }

    private TuiView() {}

    /**
     * Lays out the menu, keeping the selected action visible even in a tiny viewport.
     * @param selected action index: train, predict, inspect, quit
     * @param columns physical terminal columns; last column is reserved against wrapping
     * @param rows physical terminal rows
     * @param notice nullable diagnostic
     * @return plain bounded frame without an edit cursor
     */
    static Frame menu(int selected, int columns, int rows, String notice) {
        int width = Math.max(0, columns - 1);
        int height = Math.max(0, rows);
        List<String> result = new ArrayList<>();
        int index = Math.floorMod(selected, 4);
        if (height == 0) return new Frame(result, null);
        if (height >= 8) result.addAll(SynaptikBrand.lines(width, height));
        String[] actions = {"Train [t]", "Predict [p]", "Inspect [i]", "Quit [q]"};
        int footer = height >= 3 ? 1 : 0;
        int diagnostic = notice != null && !notice.isBlank() && height >= 6 ? 1 : 0;
        int available = height - result.size() - footer - diagnostic;
        int first = Math.max(0, Math.min(index - available / 2, 4 - available));
        for (int i = first; i < 4 && result.size() < height - footer - diagnostic; i++) {
            result.add((i == index ? "> " : "  ") + actions[i]);
        }
        if (diagnostic != 0) result.add(notice);
        if (footer != 0) result.add("Up/Down select  Enter open  t/p/i/q");
        return fit(result, null, width, height);
    }

    /**
     * Lays out enabled and disabled controls, scrolling vertically/horizontally to the edit cursor.
     * @param form UI-thread-owned form, read only during this call
     * @param columns physical terminal columns
     * @param rows physical terminal rows
     * @param notice nullable validation diagnostic
     * @return plain bounded frame with an in-viewport code-point-aware cursor when editable
     */
    static Frame form(TuiForm form, int columns, int rows, String notice) {
        int width = Math.max(0, columns - 1);
        int height = Math.max(0, rows);
        List<String> result = new ArrayList<>();
        if (height == 0) return new Frame(result, null);
        if (height >= 10) result.addAll(SynaptikBrand.lines(width, height));
        if (height >= 4) result.add(form.action() + " - Enter start / Esc back");
        int footer = height >= 3 ? 1 : 0;
        int diagnostic = notice != null && !notice.isBlank() && height >= 4 ? 1 : 0;
        int available = Math.max(1, height - result.size() - footer - diagnostic);
        List<String> body = new ArrayList<>();
        int editRow = 0;
        int editColumn = 0;
        boolean editVisible = false;
        List<TuiForm.Field> fields = form.fields();
        for (int i = 0; i < fields.size(); i++) {
            TuiForm.Field field = fields.get(i);
            boolean focus = i == form.focused();
            String mark = focus ? "> " : "  ";
            String prefix;
            if (width < 72) {
                body.add(mark + field.label() + (field.enabled() ? "" : " [disabled]"));
                prefix = focus ? "> " : "  ";
            } else {
                String label = SynaptikBrand.clip(field.label(), 36);
                prefix = mark + label + " ".repeat(Math.max(0, 36 - SynaptikBrand.width(label))) + ": ";
            }
            if (!field.enabled()) {
                body.add(prefix + "[disabled] " + field.value());
                continue;
            }
            int prefixWidth = Math.min(width, SynaptikBrand.width(prefix));
            int room = Math.max(0, width - prefixWidth);
            String value = SynaptikBrand.plain(field.value());
            int cursorOffset = focus ? value.offsetByCodePoints(0,
                    field.value().codePointCount(0, form.cursorOffset())) : 0;
            int start = 0;
            if (focus) {
                while (start < cursorOffset
                        && SynaptikBrand.width(value.substring(start, cursorOffset)) >= room) {
                    start = value.offsetByCodePoints(start, 1);
                }
                editRow = body.size();
                editColumn = prefixWidth + SynaptikBrand.width(value.substring(start, cursorOffset));
                editVisible = !field.choice() && editColumn < width;
            }
            String displayed = SynaptikBrand.clip(value.substring(start), room);
            body.add(prefix + displayed + (field.choice() ? "  <Left/Right>" : ""));
        }
        int start = Math.max(0, Math.min(editRow - available / 2, body.size() - available));
        int top = result.size();
        result.addAll(body.subList(start, Math.min(body.size(), start + available)));
        if (diagnostic != 0) result.add(notice);
        if (footer != 0) result.add(width < 19 ? "Enter/Esc" : width < 40 ? "Enter:start Esc:back"
                : "Enter start  Esc back | Tab/Shift-Tab or Up/Down fields; Left/Right edit/choose");
        TuiConsole.Cursor cursor = editVisible ? new TuiConsole.Cursor(top + editRow - start, editColumn) : null;
        return fit(result, cursor, width, height);
    }

    /**
     * Presents actual progress/inspection, completion state and the most recent bounded events.
     * @param job detached current job state
     * @param events bounded controller event list, oldest first; newest entries have priority
     * @param columns physical terminal columns
     * @param rows physical terminal rows
     * @return plain bounded frame without a cursor
     */
    static Frame job(Job job, List<String> events, int columns, int rows) {
        int width = Math.max(0, columns - 1);
        int height = Math.max(0, rows);
        List<String> result = new ArrayList<>();
        if (height == 0) return new Frame(result, null);
        String state = job.running() ? (job.stopRequested() ? "STOP REQUESTED / waiting for admitted work" : "RUNNING")
                : (job.status() == 0 ? "COMPLETED" : job.status() == 130 ? "CANCELLED" : "FAILED (status " + job.status() + ")");
        if (height >= 12) result.addAll(SynaptikBrand.lines(width, height));
        result.add(job.title() + " - " + state);
        if (height == 1) return fit(result, null, width, height);
        int footer = 1;
        String controls = job.running() ? "s / Ctrl-C stop training; q exit after work closes"
                : job.inspection().isEmpty() ? "Enter / Esc menu; q quit"
                : "Up/Down Home/End scroll; Enter/Esc menu; q quit";
        if (result.size() < height - footer && job.detail() != null && !job.detail().isBlank()) result.add(job.detail());
        if (result.size() < height - footer && (job.progress() == null || job.progress().snapshot() == null))
            result.add(String.format(Locale.ROOT, "Elapsed %.3fs", job.elapsedNanos() / 1_000_000_000.0));
        int eventRows = events.isEmpty() || height < 10 ? 0 : Math.min(4, Math.max(1, (height - result.size()) / 5));
        int bodyRows = Math.max(0, height - result.size() - footer - eventRows - (eventRows > 0 ? 1 : 0));
        if (job.progress() != null && bodyRows > 0) {
            Progress progress = job.progress();
            if (progress.snapshot() != null) {
                // Keep one headline/detail and one clock; strip the shared brand/status chrome.
                List<String> progressRows = TrainingProgressDisplay.frame(progress.snapshot(), progress.outcome(),
                        progress.detail(), Math.max(1, width), bodyRows + 2, progress.epochs(),
                        progress.stepsPerEpoch(), progress.examplesPerEpoch(), progress.backend(), false);
                int first = 2;
                for (int i = first; i < progressRows.size() && bodyRows > 0; i++, bodyRows--) result.add(progressRows.get(i));
            }
        } else {
            for (String line : job.inspection()) {
                if (bodyRows-- <= 0) break;
                // Public detached inspection metadata, never parsed CLI output.
                result.add(line);
            }
        }
        if (eventRows > 0) {
            result.add("Events (latest)");
            int first = Math.max(0, events.size() - eventRows);
            result.addAll(events.subList(first, events.size()));
        }
        result.add(controls);
        return fit(result, null, width, height);
    }

    /** Wraps detached metadata once per viewport width, without dropping any printable field. */
    static List<String> wrapInspection(List<String> metadata, int columns) {
        if (columns <= 0) return List.of();
        List<String> wrapped = new ArrayList<>();
        for (String text : metadata) {
            String safe = SynaptikBrand.plain(text);
            StringBuilder line = new StringBuilder(Math.min(safe.length(), columns));
            int used = 0;
            for (int offset = 0; offset < safe.length();) {
                int point = safe.codePointAt(offset);
                offset += Character.charCount(point);
                int cells = Math.max(0, WCWidth.wcwidth(point));
                if (cells > columns) { point = '?'; cells = 1; }
                if (used + cells > columns && !line.isEmpty()) {
                    wrapped.add(line.toString());
                    line.setLength(0);
                    used = 0;
                }
                line.appendCodePoint(point);
                used += cells;
            }
            wrapped.add(line.toString());
        }
        return List.copyOf(wrapped);
    }

    private static Frame fit(List<String> lines, TuiConsole.Cursor cursor, int width, int height) {
        List<String> result = new ArrayList<>(Math.min(lines.size(), height));
        for (int i = 0; i < lines.size() && i < height; i++) result.add(SynaptikBrand.clip(lines.get(i), width));
        if (cursor != null && (cursor.row() < 0 || cursor.row() >= result.size()
                || cursor.column() < 0 || cursor.column() >= width)) cursor = null;
        return new Frame(result, cursor);
    }
}
