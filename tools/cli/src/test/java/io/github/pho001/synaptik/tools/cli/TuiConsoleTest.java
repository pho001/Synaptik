package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Proxy;
import java.util.ArrayDeque;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jline.terminal.Attributes;
import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.utils.InfoCmp.Capability;
import org.jline.utils.NonBlockingReader;
import org.junit.jupiter.api.Test;

class TuiConsoleTest {
    @Test
    void keyDecoderDistinguishesNavigationTimeoutEofAndSupplementaryInput() throws IOException {
        ScriptedReader input = new ScriptedReader();
        input.text("\033[A\033[B\033[C\033[D\033[H\033[F\033[3~\033[Z\t\r\177");
        TuiConsole.Decoder decoder = new TuiConsole.Decoder(input, null);
        for (TuiConsole.Kind expected : new TuiConsole.Kind[] {
                TuiConsole.Kind.UP, TuiConsole.Kind.DOWN, TuiConsole.Kind.RIGHT, TuiConsole.Kind.LEFT,
                TuiConsole.Kind.HOME, TuiConsole.Kind.END, TuiConsole.Kind.DELETE, TuiConsole.Kind.BACK_TAB,
                TuiConsole.Kind.TAB, TuiConsole.Kind.ENTER, TuiConsole.Kind.BACKSPACE}) {
            assertEquals(expected, decoder.read(1).kind());
        }
        input.values.add((int) '\ud83d');
        input.values.add(NonBlockingReader.READ_EXPIRED);
        assertEquals(TuiConsole.Kind.TIMEOUT, decoder.read(1).kind());
        input.values.add((int) '\ude42');
        assertEquals(new TuiConsole.Key(TuiConsole.Kind.CHARACTER, 0x1f642), decoder.read(1));
        input.text("\003\004");
        assertEquals(TuiConsole.Kind.INTERRUPT, decoder.read(1).kind());
        assertEquals(TuiConsole.Kind.EOF, decoder.read(1).kind());
        assertEquals(TuiConsole.Kind.TIMEOUT, decoder.read(1).kind());
        input.values.add(NonBlockingReader.EOF);
        assertEquals(TuiConsole.Kind.EOF, decoder.read(1).kind());
    }

    @Test
    void escapeAndUnknownSequencesCannotInjectSuffixesIntoAPath() throws IOException {
        ScriptedReader input = new ScriptedReader();
        TuiConsole.Decoder decoder = new TuiConsole.Decoder(input, null);
        input.text("\033");
        assertEquals(TuiConsole.Kind.ESCAPE, decoder.read(1).kind());
        input.text("\033x\033[999~y");
        assertEquals(TuiConsole.Kind.ESCAPE, decoder.read(1).kind());
        assertEquals('x', decoder.read(1).character());
        assertEquals(TuiConsole.Kind.TIMEOUT, decoder.read(1).kind());
        assertEquals('y', decoder.read(1).character());
        input.values.add((int) '\ud83d');
        input.text("a");
        assertEquals(0xfffd, decoder.read(1).character());
        assertEquals('a', decoder.read(1).character());
    }

    @Test
    void nativeAndRawInterruptsNeverInterruptThreadAndCloseRestoresOwnedStateOnce() {
        TerminalState state = new TerminalState();
        TuiConsole console = TuiConsole.acquire(state.terminal);
        assertTrue(state.alternate);
        assertFalse(state.visible);
        assertFalse(state.attributes.getLocalFlag(Attributes.LocalFlag.ECHO));
        assertFalse(state.attributes.getLocalFlag(Attributes.LocalFlag.ISIG));
        state.handler.handle(Terminal.Signal.INT);
        assertTrue(console.interruptionRequested());
        assertFalse(console.interruptionRequested());
        state.input.text("\003");
        assertEquals(TuiConsole.Kind.INTERRUPT, console.readKey(1).kind());
        assertFalse(Thread.currentThread().isInterrupted());
        console.close();
        assertRestored(state);
        console.close();
        assertEquals(1, state.closes);
    }

    @Test
    void rawSignalPolicyOwnsAndRestoresCallerCharacterMappingsWithoutGlobalChanges() {
        TerminalState state = new TerminalState();
        List<Attributes.ControlChar> controls = List.of(Attributes.ControlChar.VINTR,
                Attributes.ControlChar.VQUIT, Attributes.ControlChar.VSUSP, Attributes.ControlChar.VSTATUS);
        int mapping = 17;
        for (Attributes.ControlChar control : controls) state.attributes.setControlChar(control, mapping++);
        Map<Attributes.ControlChar, Integer> original = Map.copyOf(state.attributes.getControlChars());
        String softwareSignals = System.getProperty(TerminalBuilder.PROP_SOFTWARE_SIGNALS);
        try (TuiConsole console = TuiConsole.acquire(state.terminal)) {
            for (Attributes.ControlChar control : controls) assertEquals(-1, state.attributes.getControlChar(control));
            state.handler.handle(Terminal.Signal.INT);
            assertTrue(console.interruptionRequested());
            assertFalse(console.interruptionRequested());
            state.input.text("\003");
            assertEquals(TuiConsole.Kind.INTERRUPT, console.readKey(1).kind());
            assertEquals(softwareSignals, System.getProperty(TerminalBuilder.PROP_SOFTWARE_SIGNALS));
        }
        assertEquals(original, state.attributes.getControlChars());
        assertEquals(softwareSignals, System.getProperty(TerminalBuilder.PROP_SOFTWARE_SIGNALS));
        assertRestored(state);
    }

    @Test
    void partialAcquisitionFailureAlwaysReversesPreviouslyAcquiredResources() {
        for (String point : List.of("raw", "enter", "hide", "flush")) {
            TerminalState state = new TerminalState();
            AssertionError primary = new AssertionError(point);
            state.failures.put(point, primary);
            assertSame(primary, assertThrows(AssertionError.class, () -> TuiConsole.acquire(state.terminal)));
            assertRestored(state);
            assertEquals(1, state.closes);
        }
    }

    @Test
    void cleanupPreservesRuntimeAndErrorPrimariesAttemptsAllResourcesAndAvoidsSelfSuppression() {
        TerminalState state = new TerminalState();
        TuiConsole console = TuiConsole.acquire(state.terminal);
        AssertionError primary = new AssertionError("cursor restore");
        IOException secondary = new IOException("terminal close");
        state.failures.put("show", primary);
        state.failures.put("close", secondary);
        assertSame(primary, assertThrows(AssertionError.class, console::close));
        assertEquals(1, primary.getSuppressed().length);
        assertSame(secondary, primary.getSuppressed()[0]);
        assertRestored(state);

        TerminalState same = new TerminalState();
        RuntimeException shared = new IllegalStateException("same failure instance");
        same.failures.put("enter", shared);
        same.failures.put("close", shared);
        assertSame(shared, assertThrows(IllegalStateException.class, () -> TuiConsole.acquire(same.terminal)));
        assertEquals(0, shared.getSuppressed().length);
        assertRestored(same);
    }

    @Test
    void viewportIsRequeriedAndOutputNeverWritesBeyondCurrentGeometry() {
        TerminalState state = new TerminalState();
        try (TuiConsole console = TuiConsole.acquire(state.terminal)) {
            state.size = new Size(7, 3);
            console.render(List.of("界🙂ABCDE\033[2J", "second line", "third", "fourth"), new TuiConsole.Cursor(2, 5));
            assertTrue(state.maxAddressRow < 3);
            assertTrue(state.maxAddressColumn < 7);
            assertFalse(state.output.toString().contains("fourth"));
            assertFalse(state.output.toString().contains("\033"));
            assertTrue(state.visible);
            state.size = new Size(2, 1);
            state.maxAddressRow = 0;
            state.maxAddressColumn = 0;
            console.render(List.of("🙂too wide", "not visible"), new TuiConsole.Cursor(2, 5));
            assertEquals(0, state.maxAddressRow);
            assertEquals(0, state.maxAddressColumn);
            assertFalse(state.visible);
            assertFalse(state.output.toString().contains("not visible"));
        }
        assertRestored(state);
    }

    @Test
    void progressQualificationCannotAccidentallyAdmitAnIncapableFullScreen() {
        TerminalState state = new TerminalState();
        assertTrue(CliTerminalFactory.usable(state.terminal, false));
        assertTrue(CliTerminalFactory.usable(state.terminal, true));
        state.capabilities.remove(Capability.exit_ca_mode);
        assertTrue(CliTerminalFactory.usable(state.terminal, false));
        assertFalse(CliTerminalFactory.usable(state.terminal, true));
        state.size = new Size(0, 0);
        assertFalse(CliTerminalFactory.usable(state.terminal, false));
    }

    private static void assertRestored(TerminalState state) {
        assertFalse(state.alternate);
        assertTrue(state.visible);
        assertTrue(state.attributes.getLocalFlag(Attributes.LocalFlag.ECHO));
        assertTrue(state.attributes.getLocalFlag(Attributes.LocalFlag.ISIG));
        assertSame(Terminal.SignalHandler.SIG_DFL, state.handler);
        assertFalse(state.badRestoreOrder);
    }

    private static final class ScriptedReader extends NonBlockingReader {
        final ArrayDeque<Integer> values = new ArrayDeque<>();
        void text(String text) { text.chars().forEach(values::add); }
        @Override protected int read(long timeout, boolean peek) {
            if (values.isEmpty()) return READ_EXPIRED;
            return peek ? values.getFirst() : values.removeFirst();
        }
        @Override public int readBuffered(char[] buffer, int offset, int length, long timeout) {
            int next = read(timeout, false);
            if (next < 0) return next;
            buffer[offset] = (char) next;
            return 1;
        }
        @Override public void close() {}
    }

    /** Small terminal state machine: failures can occur after an externally visible partial change. */
    private static final class TerminalState {
        final ScriptedReader input = new ScriptedReader();
        final StringWriter output = new StringWriter();
        final PrintWriter writer = new PrintWriter(output);
        final Map<String, Throwable> failures = new HashMap<>();
        final Map<Capability, String> capabilities = new EnumMap<>(Capability.class);
        Attributes attributes = new Attributes();
        Terminal.SignalHandler handler = Terminal.SignalHandler.SIG_DFL;
        Size size = new Size(80, 24);
        boolean alternate;
        boolean visible = true;
        boolean badRestoreOrder;
        int closes;
        int maxAddressRow;
        int maxAddressColumn;
        final Terminal terminal;

        TerminalState() {
            for (Capability capability : List.of(Capability.cursor_up, Capability.carriage_return,
                    Capability.clr_eol, Capability.cursor_invisible, Capability.cursor_normal,
                    Capability.enter_ca_mode, Capability.exit_ca_mode, Capability.cursor_address)) {
                capabilities.put(capability, "");
            }
            attributes.setLocalFlag(Attributes.LocalFlag.ECHO, true);
            attributes.setLocalFlag(Attributes.LocalFlag.ISIG, true);
            attributes.setLocalFlag(Attributes.LocalFlag.ICANON, true);
            terminal = (Terminal) Proxy.newProxyInstance(Terminal.class.getClassLoader(), new Class<?>[] {Terminal.class},
                    (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "reader": return input;
                            case "writer": return writer;
                            case "getType": return "xterm";
                            case "getSize": return size;
                            case "getStringCapability": return capabilities.get((Capability) args[0]);
                            case "getNumericCapability": return null;
                            case "getAttributes": return new Attributes(attributes);
                            case "enterRawMode": {
                                Attributes before = new Attributes(attributes);
                                attributes.setLocalFlag(Attributes.LocalFlag.ECHO, false);
                                attributes.setLocalFlag(Attributes.LocalFlag.ICANON, false);
                                fail("raw");
                                return before;
                            }
                            case "setAttributes": {
                                Attributes next = new Attributes((Attributes) args[0]);
                                if (next.getLocalFlag(Attributes.LocalFlag.ECHO) && alternate) badRestoreOrder = true;
                                attributes = next;
                                return null;
                            }
                            case "handle": {
                                Terminal.SignalHandler previous = handler;
                                handler = (Terminal.SignalHandler) args[1];
                                if (handler == Terminal.SignalHandler.SIG_DFL && !attributes.getLocalFlag(Attributes.LocalFlag.ECHO)) badRestoreOrder = true;
                                return previous;
                            }
                            case "puts": {
                                Capability capability = (Capability) args[0];
                                if (capability == Capability.enter_ca_mode) { alternate = true; fail("enter"); }
                                if (capability == Capability.exit_ca_mode) {
                                    if (!visible) badRestoreOrder = true;
                                    alternate = false;
                                    fail("exit");
                                }
                                if (capability == Capability.cursor_invisible) { visible = false; fail("hide"); }
                                if (capability == Capability.cursor_normal) { visible = true; fail("show"); }
                                if (capability == Capability.cursor_address) {
                                    Object[] coordinates = (Object[]) args[1];
                                    maxAddressRow = Math.max(maxAddressRow, (int) coordinates[0]);
                                    maxAddressColumn = Math.max(maxAddressColumn, (int) coordinates[1]);
                                }
                                return true;
                            }
                            case "flush": fail("flush"); return null;
                            case "close": closes++; fail("close"); return null;
                            default: throw new UnsupportedOperationException(method.getName());
                        }
                    });
        }

        private void fail(String point) throws Throwable {
            Throwable failure = failures.remove(point);
            if (failure != null) throw failure;
        }
    }
}
