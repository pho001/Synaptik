package io.github.pho001.synaptik.tools.cli;

import java.util.List;
import org.jline.terminal.Terminal;
import org.jline.utils.Curses;
import org.jline.utils.InfoCmp.Capability;
import org.jline.utils.WCWidth;

/** Shared plain branding and display-column layout; styling happens only after layout. */
final class SynaptikBrand {
    private static final List<String> LOGO = List.of(
            "o---\\       /---o",
            "     +--o--+       SYNAPTIK",
            "o---/       \\---o");

    private SynaptikBrand() {}

    /**
     * Chooses art only when it leaves useful body space.
     * @param columns available display columns
     * @param rows available rows
     * @return plain, clipped branding rows
     */
    static List<String> lines(int columns, int rows) {
        if (rows >= 12 && columns >= 27) return LOGO;
        return rows > 0 ? List.of(clip("SYNAPTIK", columns)) : List.of();
    }

    /**
     * Removes terminal controls while retaining printable Unicode code points.
     * @param text nullable untrusted text
     * @return safe single-line text
     */
    static String plain(String text) {
        if (text == null) return "";
        StringBuilder result = new StringBuilder(text.length());
        text.codePoints().forEach(cp -> result.appendCodePoint(
                Character.isISOControl(cp) || Character.getType(cp) == Character.SURROGATE
                        || Character.getType(cp) == Character.FORMAT || WCWidth.wcwidth(cp) < 0 ? '?' : cp));
        return result.toString();
    }

    /**
     * Measures printable code points, not UTF16 units or escape bytes.
     * @param text safe plain text
     * @return occupied terminal columns
     */
    static int width(String text) {
        return text.codePoints().map(cp -> Math.max(0, WCWidth.wcwidth(cp))).sum();
    }

    /**
     * Clips without splitting a code point or exceeding the display width.
     * @param text untrusted text, sanitized before measuring
     * @param columns available columns, nonpositive means empty
     * @return plain bounded text
     */
    static String clip(String text, int columns) {
        String safe = plain(text);
        int end = 0;
        int used = 0;
        while (end < safe.length()) {
            int cp = safe.codePointAt(end);
            int next = Math.max(0, WCWidth.wcwidth(cp));
            if (used + next > Math.max(0, columns)) break;
            used += next;
            end += Character.charCount(cp);
        }
        return safe.substring(0, end);
    }

    /**
     * Applies terminal-native color to a complete, already fitted brand row only.
     * @param row plain fitted row; ordinary content is returned unchanged
     * @param terminal qualified real TTY, or null for append-only output
     * @return styled row with unchanged display width and no background color
     */
    static String style(String row, Terminal terminal) {
        if (terminal == null || System.getenv().containsKey("NO_COLOR")) return row;
        Integer colors = terminal.getNumericCapability(Capability.max_colors);
        String cyan = terminal.getStringCapability(Capability.set_a_foreground);
        String dim = terminal.getStringCapability(Capability.enter_dim_mode);
        String bold = terminal.getStringCapability(Capability.enter_bold_mode);
        String reset = terminal.getStringCapability(Capability.exit_attribute_mode);
        if (colors == null || colors < 8 || cyan == null || dim == null || bold == null || reset == null) return row;
        if (row.equals("SYNAPTIK")) return Curses.tputs(bold) + row + Curses.tputs(reset);
        if (!LOGO.contains(row)) return row;
        StringBuilder result = new StringBuilder(row.length() * 8);
        for (int i = 0; i < row.length(); i++) {
            char ch = row.charAt(i);
            if (ch == 'o' || ch == '+') result.append(Curses.tputs(cyan, 6));
            else if (ch == '-' || ch == '/' || ch == '\\') result.append(Curses.tputs(dim));
            else if (Character.isLetter(ch)) result.append(Curses.tputs(bold));
            result.append(ch).append(Curses.tputs(reset));
        }
        return result.toString();
    }
}
