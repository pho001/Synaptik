package io.github.pho001.synaptik.tools.cli;

import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.jline.terminal.Attributes;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.utils.Curses;
import org.jline.utils.InfoCmp.Capability;
import org.jline.utils.NonBlockingReader;

/**
 * UI-thread-owned raw terminal, input decoder and alternate screen. The signal callback only
 * sets a flag: it never interrupts a thread or touches numerical owners. System streams are borrowed.
 * Raw software-signal character mappings are disabled only in this console's owned attributes,
 * not through process-global properties, and restored with the original attributes on close.
 */
final class TuiConsole implements AutoCloseable {
    record Cursor(int row, int column) {}
    enum Kind {
        CHARACTER, ENTER, TAB, BACK_TAB, UP, DOWN, LEFT, RIGHT, HOME, END,
        DELETE, BACKSPACE, ESCAPE, INTERRUPT, EOF, TIMEOUT
    }
    record Key(Kind kind, int character) {}

    private final Terminal terminal;
    private final AtomicBoolean interrupted = new AtomicBoolean();
    private final Decoder decoder;
    private Attributes attributes;
    private Terminal.SignalHandler previousInt;
    private boolean signalOwned;
    private boolean rawOwned;
    private boolean screenOwned;
    private boolean cursorOwned;
    private boolean closed;

    /**
     * Opens a real qualified terminal and acquires raw/screen state transactionally.
     * @param error borrowed real System.err
     * @return owned console; close restores all acquired terminal state
     * @throws UncheckedIOException when full-screen operation is unavailable
     */
    static TuiConsole open(PrintStream error) {
        try {
            return acquire(CliTerminalFactory.openTui(error));
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    /**
     * Acquires state on an already owned terminal, closing it even if acquisition fails.
     * @param terminal qualified owned terminal, never a numerical owner
     * @return console owning the terminal
     */
    static TuiConsole acquire(Terminal terminal) {
        TuiConsole console = null;
        try {
            console = new TuiConsole(terminal);
            TuiConsole acquired = console;
            console.previousInt = terminal.handle(Terminal.Signal.INT, signal -> consoleInterrupt(acquired));
            console.signalOwned = true;
            // Save before entering raw mode: enterRawMode itself can fail after a partial change.
            console.attributes = new Attributes(terminal.getAttributes());
            console.rawOwned = true;
            terminal.enterRawMode();
            Attributes raw = new Attributes(terminal.getAttributes());
            raw.setLocalFlag(Attributes.LocalFlag.ISIG, false);
            // JLine 4 translates these bytes in software even with ISIG clear, then also
            // passes them to the decoder. Disable only this console's raw mappings to avoid
            // duplicate Ctrl-C handling; native INT still reaches the owned signal handler.
            raw.setControlChar(Attributes.ControlChar.VINTR, -1);
            raw.setControlChar(Attributes.ControlChar.VQUIT, -1);
            raw.setControlChar(Attributes.ControlChar.VSUSP, -1);
            raw.setControlChar(Attributes.ControlChar.VSTATUS, -1);
            terminal.setAttributes(raw);
            console.screenOwned = true;
            terminal.puts(Capability.enter_ca_mode);
            console.cursorOwned = true;
            terminal.puts(Capability.cursor_invisible);
            console.checkOutput();
            return console;
        } catch (RuntimeException | Error failure) {
            if (console != null) {
                console.cleanup(failure);
            } else {
                try {
                    terminal.close();
                } catch (IOException | RuntimeException | Error cleanup) {
                    suppress(failure, cleanup);
                }
            }
            throw failure;
        }
    }

    private static void consoleInterrupt(TuiConsole console) {
        console.interrupted.set(true);
    }

    private TuiConsole(Terminal terminal) {
        this.terminal = terminal;
        decoder = new Decoder(terminal.reader(), terminal);
    }

    /** @return a fresh actual viewport, never cached across frames */
    Size size() {
        return terminal.getSize();
    }

    /**
     * Reads one decoded Unicode/key event without blocking the UI indefinitely.
     * @param timeoutMillis maximum initial wait in milliseconds; zero performs a short poll
     * @return detached key; TIMEOUT and EOF are distinct
     * @throws UncheckedIOException on terminal input failure
     */
    Key readKey(long timeoutMillis) {
        try {
            return decoder.read(Math.max(1, timeoutMillis));
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    /**
     * Renders plain rows within the latest physical dimensions, reserving the wrap column.
     * @param lines detached plain frame; controls are sanitized before styling
     * @param cursor zero-based edit cursor, or null to hide it
     */
    void render(List<String> lines, Cursor cursor) {
        if (closed) throw new IllegalStateException("Console is closed");
        Size size = size();
        int rows = Math.max(0, size.getRows());
        int columns = Math.max(0, size.getColumns() - 1);
        terminal.puts(Capability.cursor_invisible);
        for (int row = 0; row < rows; row++) {
            terminal.puts(Capability.cursor_address, row, 0);
            terminal.puts(Capability.clr_eol);
            if (row < lines.size()) {
                String plain = SynaptikBrand.clip(lines.get(row), columns);
                terminal.writer().print(SynaptikBrand.style(plain, terminal));
            }
        }
        if (cursor != null && cursor.row() >= 0 && cursor.row() < rows
                && cursor.column() >= 0 && cursor.column() < columns) {
            terminal.puts(Capability.cursor_address, cursor.row(), cursor.column());
            terminal.puts(Capability.cursor_normal);
        }
        checkOutput();
    }

    /** @return and clears the pending native INT flag, allowing later jobs after a stop */
    boolean interruptionRequested() {
        return interrupted.getAndSet(false);
    }

    private void checkOutput() {
        terminal.flush();
        if (terminal.writer().checkError()) {
            throw new UncheckedIOException(new IOException("Interactive terminal write failed"));
        }
    }

    /** Restores cursor, screen, attributes and signal handler in reverse acquisition order. */
    @Override
    public void close() {
        Throwable failure = cleanup(null);
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
        if (failure instanceof IOException io) throw new UncheckedIOException(io);
    }

    private Throwable cleanup(Throwable failure) {
        if (closed) return failure;
        closed = true;
        if (cursorOwned) failure = attempt(failure, () -> {
            terminal.puts(Capability.cursor_normal);
            checkOutput();
        });
        if (screenOwned) failure = attempt(failure, () -> {
            terminal.puts(Capability.exit_ca_mode);
            checkOutput();
        });
        if (rawOwned) failure = attempt(failure, () -> terminal.setAttributes(attributes));
        if (signalOwned) failure = attempt(failure, () -> terminal.handle(Terminal.Signal.INT, previousInt));
        return attempt(failure, terminal::close);
    }

    private interface Cleanup { void run() throws IOException; }

    private static Throwable attempt(Throwable primary, Cleanup cleanup) {
        try {
            cleanup.run();
        } catch (IOException | RuntimeException | Error failure) {
            if (primary == null) return failure;
            suppress(primary, failure);
        }
        return primary;
    }

    private static void suppress(Throwable primary, Throwable cleanup) {
        if (primary != cleanup) primary.addSuppressed(cleanup);
    }

    /** Stateful UTF16/terminfo decoder; no surrogate or escape suffix becomes editable garbage. */
    static final class Decoder {
        private final NonBlockingReader reader;
        private final Map<String, Kind> sequences = new LinkedHashMap<>();
        private final ArrayDeque<Integer> pending = new ArrayDeque<>();
        private int highSurrogate = -1;

        Decoder(NonBlockingReader reader, Terminal terminal) {
            this.reader = reader;
            bind("\033[A", Kind.UP); bind("\033[B", Kind.DOWN);
            bind("\033[C", Kind.RIGHT); bind("\033[D", Kind.LEFT);
            bind("\033[H", Kind.HOME); bind("\033[F", Kind.END);
            bind("\033[1~", Kind.HOME); bind("\033[4~", Kind.END);
            bind("\033[7~", Kind.HOME); bind("\033[8~", Kind.END);
            bind("\033[3~", Kind.DELETE); bind("\033[Z", Kind.BACK_TAB);
            bind("\033OA", Kind.UP); bind("\033OB", Kind.DOWN);
            bind("\033OC", Kind.RIGHT); bind("\033OD", Kind.LEFT);
            bind("\033OH", Kind.HOME); bind("\033OF", Kind.END);
            if (terminal != null) {
                bind(terminal, Capability.key_up, Kind.UP); bind(terminal, Capability.key_down, Kind.DOWN);
                bind(terminal, Capability.key_left, Kind.LEFT); bind(terminal, Capability.key_right, Kind.RIGHT);
                bind(terminal, Capability.key_home, Kind.HOME); bind(terminal, Capability.key_end, Kind.END);
                bind(terminal, Capability.key_dc, Kind.DELETE); bind(terminal, Capability.key_btab, Kind.BACK_TAB);
                bind(terminal, Capability.key_enter, Kind.ENTER); bind(terminal, Capability.key_backspace, Kind.BACKSPACE);
            }
        }

        private void bind(Terminal terminal, Capability capability, Kind kind) {
            String value = terminal.getStringCapability(capability);
            if (value != null) bind(Curses.tputs(value), kind);
        }

        private void bind(String sequence, Kind kind) {
            if (!sequence.isEmpty()) sequences.put(sequence, kind);
        }

        Key read(long timeout) throws IOException {
            int ch = next(timeout);
            if (highSurrogate >= 0) {
                if (ch == NonBlockingReader.READ_EXPIRED) return key(Kind.TIMEOUT);
                int high = highSurrogate;
                highSurrogate = -1;
                if (ch >= 0 && Character.isLowSurrogate((char) ch)) {
                    return new Key(Kind.CHARACTER, Character.toCodePoint((char) high, (char) ch));
                }
                pending.addFirst(ch);
                return new Key(Kind.CHARACTER, 0xfffd);
            }
            if (ch == NonBlockingReader.READ_EXPIRED) return key(Kind.TIMEOUT);
            if (ch == NonBlockingReader.EOF || ch == 4) return key(Kind.EOF);
            if (ch == 3) return key(Kind.INTERRUPT);
            if (ch == '\r' || ch == '\n') return key(Kind.ENTER);
            if (ch == '\t') return key(Kind.TAB);
            if (ch == 127 || ch == 8) return key(Kind.BACKSPACE);
            if (ch == 27) return escape();
            Kind mapped = sequences.get(Character.toString((char) ch));
            if (mapped != null) return key(mapped);
            if (Character.isHighSurrogate((char) ch)) {
                highSurrogate = ch;
                return read(30);
            }
            if (Character.isLowSurrogate((char) ch)) return new Key(Kind.CHARACTER, 0xfffd);
            return Character.isISOControl(ch) ? key(Kind.TIMEOUT) : new Key(Kind.CHARACTER, ch);
        }

        private Key escape() throws IOException {
            StringBuilder sequence = new StringBuilder("\033");
            while (sequence.length() <= 32) {
                int next = next(30);
                if (next < 0) {
                    if (next == NonBlockingReader.EOF) pending.addFirst(next);
                    return sequence.length() == 1 ? key(Kind.ESCAPE) : key(Kind.TIMEOUT);
                }
                if (sequence.length() == 1 && next != '[' && next != 'O'
                        && sequences.keySet().stream().noneMatch(s -> s.startsWith("\033" + (char) next))) {
                    pending.addFirst(next);
                    return key(Kind.ESCAPE);
                }
                sequence.append((char) next);
                Kind exact = sequences.get(sequence.toString());
                if (exact != null) return key(exact);
                // Unknown CSI/function sequences are consumed, not replayed into a path field.
                if (sequence.length() >= 3 && next >= 0x40 && next <= 0x7e) return key(Kind.TIMEOUT);
            }
            return key(Kind.TIMEOUT);
        }

        private int next(long timeout) throws IOException {
            return pending.isEmpty() ? reader.read(Math.max(1, timeout)) : pending.removeFirst();
        }

        private static Key key(Kind kind) { return new Key(kind, 0); }
    }
}
