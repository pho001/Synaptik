package io.github.pho001.synaptik.tools.cli;

import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.terminal.spi.SystemStream;
import org.jline.terminal.spi.TerminalProvider;
import org.jline.utils.InfoCmp.Capability;
import org.jline.utils.OSUtils;

/**
 * Stderr-only, non-raw terminal adapter. It never enters an alternate screen, consumes input,
 * runs a command, clears the screen, or chooses an implicit native provider. All text is bounded
 * ASCII, including user-derived labels and exception details.
 */
final class TrainingProgressDisplay implements TrainingProgress.Sink {
    private static final String[] WORDMARK = {
        " ___ __   __ _  _   _   ___ _____ ___ _  __",
        "/ __|\\ \\ / /| \\| | /_\\ | _ \\_   _|_ _| |/ /",
        "\\__ \\ \\ V / | .` |/ _ \\|  _/ | |  | || ' < ",
        "|___/  |_|  |_|\\_/_/ \\_\\_|   |_| |___|_|\\_\\"
    };
    private final PrintStream error;
    private final Terminal terminal;
    private final int epochs;
    private final long stepsPerEpoch;
    private final long examplesPerEpoch;
    private final String backend;
    private int ownedRows;
    private final int[] ownedLineWidths = new int[40];
    private int previousColumns;
    private boolean cursorHidden;
    private boolean closed;
    private boolean live;
    private boolean headerWritten;

    /**
     * Attempts only FFM, genuine stdin/stderr TTYs and useful capabilities/dimensions. An
     * unavailable provider or unsuitable terminal is an ordinary append-only selection.
     * @param error borrowed diagnostic stream; only the real System.err can receive terminal control
     * @param plain explicit append-only selection
     * @param epochs configured positive epoch count
     * @param stepsPerEpoch expected successful steps per complete epoch
     * @param examplesPerEpoch actual examples per complete epoch
     * @param backend detached descriptive label, not proof of native route selection
     * @return owned display, never null
     */
    static TrainingProgressDisplay open(PrintStream error, boolean plain, int epochs,
            long stepsPerEpoch, long examplesPerEpoch, String backend) {
        Terminal candidate = null;
        if (!plain && error == System.err && !OSUtils.IS_WINDOWS
                && !"dumb".equalsIgnoreCase(System.getenv("TERM"))) {
            try {
                TerminalProvider provider = TerminalProvider.load("ffm");
                String inputTerminal = provider.systemStreamName(SystemStream.Input);
                // Pinned Windows FFM writes stdout; POSIX dimensions come from stdin. Only
                // matching POSIX terminal identities qualify for stderr-only live rendering.
                if (inputTerminal != null && inputTerminal.equals(provider.systemStreamName(SystemStream.Error))
                        && provider.isSystemStream(SystemStream.Input) && provider.isSystemStream(SystemStream.Error)) {
                    candidate = TerminalBuilder.builder().name("Synaptik training").system(true)
                            .provider("ffm").systemOutput(TerminalBuilder.SystemOutput.SysErr)
                            .dumb(false).nativeSignals(true).build();
                    if (!usable(candidate)) {
                        candidate.close();
                        candidate = null;
                    }
                }
            } catch (IOException | RuntimeException | LinkageError unavailable) {
                if (candidate != null) {
                    try {
                        candidate.close();
                    } catch (IOException | RuntimeException | Error cleanup) {
                        if (cleanup != unavailable) unavailable.addSuppressed(cleanup);
                    }
                }
                candidate = null;
            }
        }
        return new TrainingProgressDisplay(error, candidate, epochs, stepsPerEpoch, examplesPerEpoch, backend);
    }

    private TrainingProgressDisplay(PrintStream error, Terminal terminal, int epochs,
            long stepsPerEpoch, long examplesPerEpoch, String backend) {
        this.error = error;
        this.terminal = terminal;
        this.live = terminal != null;
        this.epochs = epochs;
        this.stepsPerEpoch = stepsPerEpoch;
        this.examplesPerEpoch = examplesPerEpoch;
        this.backend = ascii(backend, 128);
    }

    /** @return selected native terminal, or null in append-only mode */
    Terminal terminal() {
        return terminal;
    }

    private static boolean usable(Terminal terminal) {
        Size size = terminal.getSize();
        return terminal.getType() != null && !terminal.getType().startsWith("dumb")
                && size.getColumns() >= 32 && size.getRows() >= 6
                && has(terminal, Capability.cursor_up) && has(terminal, Capability.carriage_return)
                && has(terminal, Capability.clr_eol) && has(terminal, Capability.cursor_invisible)
                && has(terminal, Capability.cursor_normal);
    }

    private static boolean has(Terminal terminal, Capability capability) {
        return terminal.getStringCapability(capability) != null;
    }

    @Override
    public synchronized void render(TrainingProgressStats.Snapshot snapshot,
            TrainingProgress.Outcome outcome, String detail) {
        if (closed) {
            return;
        }
        Size size = live ? terminal.getSize() : null;
        if (live && (size.getColumns() < 2 || size.getRows() < 2)) {
            restoreCursor();
            live = false;
            ownedRows = 0;
        }
        int width = live ? Math.min(size.getColumns() - 1, 240) : 110;
        int height = live ? Math.min(size.getRows() - 1, 40) : 24;
        List<String> lines = frame(snapshot, outcome, detail, width, height,
                epochs, stepsPerEpoch, examplesPerEpoch, backend, live || !headerWritten);
        if (live) {
            if (!headerWritten) {
                // Acquire a fresh row rather than erasing a caller's partial terminal line.
                terminal.writer().print("\r\n");
            }
            if (!cursorHidden) {
                terminal.puts(Capability.cursor_invisible);
                cursorHidden = true;
            }
            // The cursor rests below hard-newline-terminated owned rows. Account for their
            // visual reflow on width shrink, as JLine's partial-screen Display does. A height
            // shrink makes only the visible suffix reachable; never clear the entire screen.
            int visibleRows = reflowedRows(ownedLineWidths, ownedRows, previousColumns, size.getColumns());
            int eraseRows = Math.min(visibleRows, size.getRows() - 1);
            for (int i = 0; i < eraseRows; i++) {
                terminal.puts(Capability.cursor_up);
            }
            int rows = Math.max(eraseRows, lines.size());
            for (int i = 0; i < rows; i++) {
                terminal.puts(Capability.carriage_return);
                terminal.puts(Capability.clr_eol);
                if (i < lines.size()) {
                    terminal.writer().print(lines.get(i));
                }
                terminal.writer().print("\r\n");
            }
            // Erased surplus rows are blank, not part of the next frame. Leave the cursor
            // immediately below the new frame rather than retaining a growing blank region.
            for (int i = lines.size(); i < rows; i++) {
                terminal.puts(Capability.cursor_up);
            }
            ownedRows = lines.size();
            previousColumns = size.getColumns();
            for (int i = 0; i < ownedRows; i++) {
                ownedLineWidths[i] = lines.get(i).length();
            }
            terminal.flush();
            if (terminal.writer().checkError()) {
                throw new UncheckedIOException(new IOException("Training terminal write failed"));
            }
        } else {
            for (String line : lines) {
                error.println(line);
            }
            error.flush();
            if (error.checkError()) {
                throw new UncheckedIOException(new IOException("Training progress write failed"));
            }
        }
        headerWritten = true;
    }

    /**
     * Builds a bounded frame at the actual available dimensions. A one-row terminal prioritizes
     * the final status; wider terminals place loss history alongside scalar measurements.
     */
    static List<String> frame(TrainingProgressStats.Snapshot state, TrainingProgress.Outcome outcome,
            String detail, int width, int height, int epochs, long stepsPerEpoch,
            long examplesPerEpoch, String backend, boolean wordmark) {
        width = Math.max(1, Math.min(width, 240));
        height = Math.max(1, Math.min(height, 40));
        List<String> text = new ArrayList<>();
        String status = outcome == null ? "TRAINING" : outcome.name();
        text.add(status + (detail == null || detail.isBlank() ? "" : ": " + ascii(detail, 512)));
        text.add("Epoch " + state.epoch() + "/" + epochs + "  steps " + state.epochSteps() + "/"
                + stepsPerEpoch + "  total " + state.steps() + "  "
                + decimal(100.0 * state.epochSteps() / stepsPerEpoch) + "%");
        text.add("Examples " + state.examples() + "  epoch size " + examplesPerEpoch
                + "  backend " + ascii(backend, 128));
        text.add("Elapsed " + seconds(state.elapsedNanos()) + "s  current epoch " + seconds(state.epochNanos()) + "s");
        text.add("Step/s " + decimal(state.stepsPerSecond()) + "  input+target payload Mbit/s "
                + decimal(state.payloadMbitPerSecond()));
        text.add("Decoded payload bytes " + state.bytes());
        text.add("Pre-update loss " + (state.steps() == 0 ? "n/a" : number(state.loss()))
                + "  epoch mean " + (state.epochSteps() == 0 ? "n/a" : number(state.epochMean())));
        for (TrainingProgressStats.Epoch epoch : state.recentEpochs()) {
            text.add("Epoch " + epoch.number() + "  " + seconds(epoch.nanos()) + "s  "
                    + decimal(TrainingProgressStats.rate(epoch.steps(), epoch.nanos())) + " step/s  "
                    + decimal(TrainingProgressStats.mbitRate(epoch.bytes(), epoch.nanos()))
                    + " Mbit/s  mean " + number(epoch.meanLoss()));
        }
        List<String> result = new ArrayList<>();
        // Preserve useful scalar rows before spending the terminal's vertical space on art.
        if (wordmark && height >= 17 && width >= 45) {
            for (String line : WORDMARK) {
                result.add(line);
            }
        } else if (height > 2) {
            result.add(clip("SYNAPTIK", width));
        }
        boolean side = width >= 108;
        int left = side ? Math.min(76, width - 34) : width;
        List<String> plot = plot(state.losses(), side ? width - left - 3 : width, 6);
        int available = height - result.size();
        int body = side ? Math.min(Math.max(text.size(), plot.size()), available)
                : Math.min(text.size(), available >= 7 + plot.size() ? available - plot.size() : available);
        for (int i = 0; i < body; i++) {
            String row = i < text.size() ? clip(text.get(i), left) : "";
            if (side && i < plot.size()) {
                row += " ".repeat(left - row.length()) + " | " + plot.get(i);
            }
            result.add(clip(row, width));
        }
        if (!side) {
            for (String row : plot) {
                if (result.size() >= height) {
                    break;
                }
                result.add(clip(row, width));
            }
        }
        return result;
    }

    /**
     * Counts visual rows occupied by owned hard lines after terminal reflow, without allocating.
     * @param widths bounded ASCII widths of the previous frame
     * @param count number of valid hard lines
     * @param previousColumns previous physical terminal width, or zero before the first frame
     * @param columns current positive physical terminal width
     * @return owned visual row count, before clipping to the visible terminal height
     */
    static int reflowedRows(int[] widths, int count, int previousColumns, int columns) {
        if (previousColumns == 0 || previousColumns == columns) {
            return count;
        }
        int rows = 0;
        for (int i = 0; i < count; i++) {
            rows += Math.max(1, (widths[i] + columns - 1) / columns);
        }
        return rows;
    }

    /** Actual chronological finite points; nonfinite values occupy explicit '?' gap columns. */
    static List<String> plot(double[] samples, int width, int rows) {
        width = Math.max(1, Math.min(width, 120));
        rows = Math.max(1, Math.min(rows, 12));
        int count = Math.min(samples.length, width);
        int first = samples.length - count;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = first; i < samples.length; i++) {
            if (Double.isFinite(samples[i])) {
                min = Math.min(min, samples[i]);
                max = Math.max(max, samples[i]);
            }
        }
        List<String> result = new ArrayList<>();
        result.add(clip("Loss history (pre-update)", width));
        result.add(clip(min <= max ? number(min) + " .. " + number(max) : "No finite samples", width));
        char[][] cells = new char[rows][count];
        for (char[] row : cells) {
            java.util.Arrays.fill(row, ' ');
        }
        for (int i = 0; i < count; i++) {
            double value = samples[first + i];
            if (Double.isFinite(value)) {
                cells[plotRow(value, min, max, rows)][i] = '*';
            } else {
                cells[rows - 1][i] = '?';
            }
        }
        for (char[] row : cells) {
            result.add(new String(row));
        }
        result.add(clip("? = NaN/Infinity gap", width));
        return result;
    }

    /** Overflow-safe linear finite scaling, with constants (including signed zero) centered. */
    static int plotRow(double value, double min, double max, int rows) {
        if (min == max) {
            return (rows - 1) / 2;
        }
        double magnitude = Math.max(Math.abs(min), Math.abs(max));
        double low = min / magnitude;
        double high = max / magnitude;
        double fraction = (value / magnitude - low) / (high - low);
        return Math.max(0, Math.min(rows - 1, (int) Math.round((1 - fraction) * (rows - 1))));
    }

    private static String clip(String value, int width) {
        return value.length() <= width ? value : value.substring(0, width);
    }

    static String ascii(String value, int limit) {
        if (value == null) {
            return "";
        }
        StringBuilder result = new StringBuilder(Math.min(value.length(), limit));
        for (int i = 0; i < value.length() && i < limit; i++) {
            char ch = value.charAt(i);
            result.append(ch >= 32 && ch <= 126 ? ch : '?');
        }
        return result.toString();
    }

    private static String seconds(long nanos) {
        return decimal(nanos / 1_000_000_000.0);
    }

    private static String decimal(double value) {
        return Double.isNaN(value) ? "n/a" : String.format(Locale.ROOT, "%.3f", value);
    }

    private static String number(double value) {
        return String.format(Locale.ROOT, "%.6g", value);
    }

    private void restoreCursor() {
        if (cursorHidden) {
            terminal.puts(Capability.cursor_normal);
            terminal.flush();
            cursorHidden = false;
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        Throwable failure = null;
        if (terminal != null) {
            try {
                restoreCursor();
            } catch (RuntimeException | Error problem) {
                failure = problem;
            }
            try {
                terminal.close();
            } catch (IOException | RuntimeException | Error problem) {
                Throwable next = problem instanceof IOException io ? new UncheckedIOException(io) : problem;
                if (failure == null) {
                    failure = next;
                } else if (failure != next) {
                    failure.addSuppressed(next);
                }
            }
        }
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }
}
