package io.github.pho001.synaptik.tools.cli;

import java.io.IOException;
import java.io.PrintStream;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.terminal.spi.SystemStream;
import org.jline.terminal.spi.TerminalProvider;
import org.jline.utils.InfoCmp.Capability;
import org.jline.utils.OSUtils;

/**
 * POSIX stdin/stderr qualification shared by both presentations; never tries another provider.
 * Construction disables automatic grapheme-mode probing so it neither consumes caller input
 * nor enables a terminal width policy that these codepoint-based presentations do not model.
 */
final class CliTerminalFactory {
    private CliTerminalFactory() {}

    /**
     * Acquires a non-raw progress terminal, or selects append-only output if unavailable.
     * @param error borrowed stream; only the actual System.err qualifies
     * @param plain explicit append-only selection
     * @return owned terminal, or null; closing it never closes the system streams
     */
    static Terminal openLiveProgress(PrintStream error, boolean plain) {
        if (plain) return null;
        try {
            return open(error, false);
        } catch (IOException | RuntimeException | LinkageError unavailable) {
            return null;
        }
    }

    /**
     * Acquires a fully qualified terminal without changing raw/screen state.
     * @param error borrowed diagnostic stream, required to be the real System.err
     * @return owned FFM terminal whose POSIX stdin and stderr name the same real TTY
     * @throws IOException if full-screen operation is unavailable; CLI commands remain usable
     */
    static Terminal openTui(PrintStream error) throws IOException {
        try {
            return open(error, true);
        } catch (RuntimeException | LinkageError unavailable) {
            throw new IOException("Interactive TUI requires the FFM provider and a capable POSIX stdin/stderr TTY; use CLI commands otherwise", unavailable);
        }
    }

    private static Terminal open(PrintStream error, boolean full) throws IOException {
        if (error != System.err || OSUtils.IS_WINDOWS || "dumb".equalsIgnoreCase(System.getenv("TERM"))) {
            throw unavailable();
        }
        TerminalProvider provider = TerminalProvider.load("ffm");
        String input = provider.systemStreamName(SystemStream.Input);
        if (input == null || !input.equals(provider.systemStreamName(SystemStream.Error))
                || !provider.isSystemStream(SystemStream.Input) || !provider.isSystemStream(SystemStream.Error)) {
            throw unavailable();
        }
        Terminal terminal = TerminalBuilder.builder().name("Synaptik").system(true)
                .provider("ffm").systemOutput(TerminalBuilder.SystemOutput.SysErr)
                // Keep progress input untouched and both frontends on their codepoint-width policy.
                .dumb(false).nativeSignals(true).graphemeCluster(false).build();
        boolean qualified;
        try {
            qualified = usable(terminal, full);
        } catch (RuntimeException | Error failure) {
            try {
                terminal.close();
            } catch (IOException | RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
        if (!qualified) {
            // A failing rejection-close is itself the primary failure, not a suppressed Error
            // hidden under an ordinary unavailable-terminal selection.
            terminal.close();
            throw unavailable();
        }
        return terminal;
    }

    /**
     * Checks real dimensions and all capabilities actually used by the selected presentation.
     * @param terminal candidate, not mutated
     * @param full whether alternate-screen/cursor-address capabilities are required
     * @return whether the candidate can render safely
     */
    static boolean usable(Terminal terminal, boolean full) {
        Size size = terminal.getSize();
        return terminal.getType() != null && !terminal.getType().startsWith("dumb")
                && size.getColumns() >= 32 && size.getRows() >= 6
                && has(terminal, Capability.cursor_up) && has(terminal, Capability.carriage_return)
                && has(terminal, Capability.clr_eol) && has(terminal, Capability.cursor_invisible)
                && has(terminal, Capability.cursor_normal)
                && (!full || (has(terminal, Capability.enter_ca_mode) && has(terminal, Capability.exit_ca_mode)
                    && has(terminal, Capability.cursor_address)));
    }

    private static boolean has(Terminal terminal, Capability capability) {
        return terminal.getStringCapability(capability) != null;
    }

    private static IOException unavailable() {
        return new IOException("Interactive terminal unavailable: requires matching real POSIX stdin/stderr TTYs, FFM and useful screen capabilities; use CLI/plain commands otherwise");
    }
}
