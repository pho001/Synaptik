package io.github.pho001.synaptik.text;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Exact frozen IDs and code-point whitespace semantics, without execution dependencies. */
final class WhitespaceTokenizerTest {
    @Test
    void exposesOnlyTheSelectedConcreteApi() {
        assertTrue(Modifier.isFinal(WhitespaceTokenizer.class.getModifiers()));
        assertEquals(0, WhitespaceTokenizer.class.getConstructors().length);
        assertEquals(Set.of("of", "vocabularySize", "paddingId", "unknownId", "encode"),
                Arrays.stream(WhitespaceTokenizer.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(method -> method.getName()).collect(Collectors.toSet()));
    }

    @Test
    void copiesOrderedVocabularyWithoutSpecialSpellings() {
        var words = new ArrayList<>(List.of("world", "Hello", "PAD", "UNK"));
        var tokenizer = WhitespaceTokenizer.of(words);
        words.clear();
        assertEquals(6, tokenizer.vocabularySize());
        assertEquals(0, tokenizer.paddingId());
        assertEquals(1, tokenizer.unknownId());
        assertArrayEquals(new long[] {3, 2, 4, 5, 1}, tokenizer.encode("Hello world PAD UNK missing"));
        assertArrayEquals(new long[] {2, 3}, WhitespaceTokenizer.of(List.of("Hello", "world"))
                .encode("Hello world"));
    }

    @Test
    void splitsMaximalRunsUsingJavaWhitespaceNotSpaceCharacters() {
        var tokenizer = WhitespaceTokenizer.of(List.of("a", "b", "a\u00a0b", "a\u2007b", "a\u202fb", "a\u200bb"));
        assertArrayEquals(new long[] {2, 3, 2, 3, 2},
                tokenizer.encode(" \ta\n\r b\u2003a\u2028b\u3000a  "));
        assertArrayEquals(new long[] {4, 5, 6, 7}, tokenizer.encode("a\u00a0b a\u2007b a\u202fb a\u200bb"));
        assertArrayEquals(new long[] {1, 1}, tokenizer.encode("missing-run another-run"));
    }

    @Test
    void preservesCasePunctuationNormalizationAndExactUtf16() {
        var tokenizer = WhitespaceTokenizer.of(List.of("Hello", "Hello,", "é", "e\u0301", "😀", "\ud800", "\udc00"));
        assertArrayEquals(new long[] {2, 1, 3, 4, 5, 6, 7, 8, 1},
                tokenizer.encode("Hello hello Hello, é e\u0301 😀 \ud800 \udc00 😀!"));
    }

    @Test
    void everyEncodingOwnsAFreshArrayEvenWhenEmpty() {
        var tokenizer = WhitespaceTokenizer.of(List.of());
        assertEquals(2, tokenizer.vocabularySize());
        assertArrayEquals(new long[] {1}, tokenizer.encode("anything"));
        for (String input : List.of("", " \t\u2003", "anything")) {
            long[] first = tokenizer.encode(input);
            long[] second = tokenizer.encode(input);
            assertNotSame(first, second);
            assertArrayEquals(first, second);
            if (first.length != 0) {
                first[0] = 99;
                assertArrayEquals(new long[] {1}, second);
            }
        }
    }

    @Test
    void rejectsInvalidVocabularyAndNamedNulls() {
        assertEquals("ordinaryTokens", assertThrows(NullPointerException.class,
                () -> WhitespaceTokenizer.of(null)).getMessage());
        assertTrue(assertThrows(NullPointerException.class,
                () -> WhitespaceTokenizer.of(Arrays.asList("a", null))).getMessage().contains("row 1"));
        for (var words : List.of(List.of(""), List.of("a", "a"), List.of("a b"), List.of("a\tb"), List.of("a\u2003b"))) {
            assertThrows(IllegalArgumentException.class, () -> WhitespaceTokenizer.of(words));
        }
        assertEquals("text", assertThrows(NullPointerException.class,
                () -> WhitespaceTokenizer.of(List.of()).encode(null)).getMessage());
    }
}
