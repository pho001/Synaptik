package io.github.pho001.synaptik.tools.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TuiFormTest {
    @Test
    void editingPreservesCodePointBoundariesAndExactPathValues() {
        TuiForm form = new TuiForm(TuiForm.Action.INSPECT);
        type(form, "--data/界🙂e\u0301.model");
        String original = form.fields().getFirst().value();
        key(form, TuiConsole.Kind.HOME);
        for (int i = 0; i < "--data/界".codePointCount(0, "--data/界".length()); i++) key(form, TuiConsole.Kind.RIGHT);
        assertEquals(original.indexOf("🙂"), form.cursorOffset());
        key(form, TuiConsole.Kind.DELETE);
        assertEquals(original.replace("🙂", ""), form.fields().getFirst().value());
        type(form, "🧠");
        assertEquals(original.replace("🙂", "🧠"), form.fields().getFirst().value());
        key(form, TuiConsole.Kind.BACKSPACE);
        assertEquals(original.replace("🙂", ""), form.fields().getFirst().value());
        key(form, TuiConsole.Kind.LEFT);
        assertEquals(original.indexOf("界"), form.cursorOffset());
        key(form, TuiConsole.Kind.END);
        assertEquals(form.fields().getFirst().value().length(), form.cursorOffset());
        CliArguments args = CliArguments.parse(form.arguments());
        assertEquals(form.fields().getFirst().value(), args.value("model", null));
    }

    @Test
    void loadedModelDisablesExcludesAndSkipsNewModelGeometryWithoutLosingEdits() {
        TuiForm form = new TuiForm(TuiForm.Action.TRAIN);
        type(form, "input.csv");
        key(form, TuiConsole.Kind.TAB);
        type(form, "output.model");
        key(form, TuiConsole.Kind.TAB);
        type(form, "loaded.model");
        CliArguments loaded = CliArguments.parse(form.arguments());
        for (String option : new String[] {"batch-size", "outputs", "seed"}) assertFalse(loaded.has(option));
        assertTrue(form.fields().subList(4, 7).stream().noneMatch(TuiForm.Field::enabled));
        key(form, TuiConsole.Kind.TAB);
        int epochs = form.focused();
        key(form, TuiConsole.Kind.TAB);
        assertTrue(form.focused() > epochs + 1);
        assertTrue(form.fields().get(form.focused()).enabled());
        key(form, TuiConsole.Kind.BACK_TAB);
        key(form, TuiConsole.Kind.BACK_TAB);
        assertEquals(2, form.focused());
        clear(form);
        CliArguments fresh = CliArguments.parse(form.arguments());
        assertTrue(fresh.has("batch-size"));
        assertTrue(fresh.has("outputs"));
        assertTrue(fresh.has("seed"));
    }

    @Test
    void backendChoiceControlsOnlyApplicableLibraryAndApiOptions() {
        TuiForm form = new TuiForm(TuiForm.Action.PREDICT);
        type(form, "data.csv");
        key(form, TuiConsole.Kind.TAB);
        type(form, "predictions.csv");
        key(form, TuiConsole.Kind.TAB);
        type(form, "model.bin");
        key(form, TuiConsole.Kind.TAB);
        key(form, TuiConsole.Kind.HOME);
        key(form, TuiConsole.Kind.RIGHT);
        key(form, TuiConsole.Kind.TAB);
        type(form, "/native/metal.dylib");
        CliArguments metal = CliArguments.parse(form.arguments());
        assertEquals("metal", metal.value("backend", null));
        assertEquals("/native/metal.dylib", metal.value("metal-library", null));
        assertFalse(metal.has("onednn-api"));
        key(form, TuiConsole.Kind.BACK_TAB);
        key(form, TuiConsole.Kind.RIGHT);
        key(form, TuiConsole.Kind.TAB);
        type(form, "/native/dnn.dylib");
        key(form, TuiConsole.Kind.TAB);
        key(form, TuiConsole.Kind.END);
        CliArguments dnn = CliArguments.parse(form.arguments());
        assertEquals("onednn", dnn.value("backend", null));
        assertEquals("/native/dnn.dylib", dnn.value("onednn-library", null));
        assertEquals("graph", dnn.value("onednn-api", null));
        assertFalse(dnn.has("metal-library"));
        key(form, TuiConsole.Kind.TAB);
        assertEquals(0, form.focused());
    }

    @Test
    void predictionCannotFallThroughToCsvOnScreenAndNavigationRemainsDiscoverable() {
        TuiForm form = new TuiForm(TuiForm.Action.PREDICT);
        assertThrows(IllegalArgumentException.class, form::arguments);
        key(form, TuiConsole.Kind.TAB);
        type(form, " ");
        assertThrows(IllegalArgumentException.class, form::arguments);
        type(form, "results.csv");
        assertEquals(" results.csv", CliArguments.parse(form.arguments()).value("output", null));
        assertEquals(TuiForm.Result.SUBMIT, form.handle(new TuiConsole.Key(TuiConsole.Kind.ENTER, 0)));
        assertEquals(TuiForm.Result.BACK, form.handle(new TuiConsole.Key(TuiConsole.Kind.ESCAPE, 0)));
        String before = form.fields().get(form.focused()).value();
        assertEquals(TuiForm.Result.UNCHANGED, form.handle(new TuiConsole.Key(TuiConsole.Kind.CHARACTER, 0xd800)));
        assertEquals(before, form.fields().get(form.focused()).value());
    }

    static void type(TuiForm form, String value) {
        value.codePoints().forEach(cp -> form.handle(new TuiConsole.Key(TuiConsole.Kind.CHARACTER, cp)));
    }

    static void key(TuiForm form, TuiConsole.Kind key) {
        form.handle(new TuiConsole.Key(key, 0));
    }

    static void clear(TuiForm form) {
        key(form, TuiConsole.Kind.END);
        while (form.cursorOffset() > 0) key(form, TuiConsole.Kind.BACKSPACE);
    }
}
