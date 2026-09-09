package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves the checked disposition ledger against live, execution-produced coverage facts.
 *
 * <p>The canonical inventory is a frozen accounting snapshot: it identifies the exact generated
 * or rejected cases that must be accounted for, but it does not itself promote an oracle or
 * performance result. This registry resolves an immutable TSV selector against the live
 * {@link CpuGeneratedCoverageEvidenceRegistry.Record} produced by the current fixture execution.
 * A selector must match a generated record and exactly one most-specific applicability relation.
 * Missing, ambiguous, malformed, or stale relations fail rather than inheriting a nearby
 * operation-family result.</p>
 */
final class CpuGeneratedCoverageDispositionRegistry {
    private static final String RESOURCE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/"
            + "generated-coverage-disposition-registry.tsv";
    private static final String HEADER = "evidence-id\tkind\tdisposition\tapplicability\tevidence-reference";

    /** Independent evidence dimensions; an oracle result never supplies performance evidence. */
    enum Kind { ORACLE, PERFORMANCE }

    /**
     * The canonical ledger relation selected for one live owner and evidence dimension.
     *
     * @param evidenceId immutable TSV identifier for the selected relation; never {@code null}
     * @param disposition exact resolved state for that relation; never {@code null}
     */
    record Resolution(String evidenceId, String disposition) { }
    private CpuGeneratedCoverageDispositionRegistry() { }

    /**
     * Resolves one live generated record using the checked canonical ledger.
     *
     * @param kind independent oracle or performance dimension to resolve; never {@code null}
     * @param record live execution-produced facts for one owner; must describe a generated row
     * @return the unique canonical relation and its exact disposition; never {@code null}
     * @throws AssertionError if the record is not generated or no unique relation applies
     */
    static Resolution resolve(Kind kind, CpuGeneratedCoverageEvidenceRegistry.Record record) {
        Entry entry = resolve(entries(), kind, record);
        return new Resolution(entry.id, entry.disposition);
    }

    /**
     * Resolves one kind using the most-specific matching canonical selector.
     *
     * <p>Specificity is the number of exact applicability terms.  This makes a narrow
     * owner/form promotion independent of TSV order while an equal-specificity overlap fails
     * closed instead of accidentally choosing whichever row was parsed first.</p>
     *
     * @param entries canonical entries to resolve
     * @param kind evidence kind being resolved
     * @param record execution-produced owner facts
     * @return the unique most-specific matching entry
     */
    static Entry resolve(List<Entry> entries, Kind kind, CpuGeneratedCoverageEvidenceRegistry.Record record) {
        if (!record.outcome().equals("GENERATED")) {
            throw new AssertionError("non-generated owner cannot inherit a disposition: " + record.occurrenceId());
        }
        List<Entry> matches = entries.stream().filter(entry -> entry.kind == kind && entry.applies(record)).toList();
        if (matches.isEmpty()) throw new AssertionError("no " + kind + " evidence record for " + record.occurrenceId());
        int specificity = matches.stream().mapToInt(Entry::specificity).max().orElseThrow();
        List<Entry> mostSpecific = matches.stream().filter(entry -> entry.specificity() == specificity).toList();
        if (mostSpecific.size() != 1) throw new AssertionError("ambiguous " + kind + " evidence records for "
                + record.occurrenceId() + ": " + mostSpecific.stream().map(Entry::id).sorted().toList());
        return mostSpecific.getFirst();
    }

    /**
     * Verifies that the frozen ledger has complete, non-ambiguous relations for live fixture data.
     *
     * @param records live execution-produced records keyed by owner; never {@code null}
     * @throws AssertionError if a schema, selector, cardinality, or resolution invariant fails
     */
    static void verifyCanonicalRegistry(Map<String, CpuGeneratedCoverageEvidenceRegistry.Record> records) {
        List<Entry> canonical = entries();
        assertEquals(25, canonical.size());
        assertEquals(List.of("oracle-0009c-affine-v1", "oracle-0009c-concat-v1", "oracle-0009c-expand-v1",
                "oracle-0009c-expand-dims-v1", "oracle-0009c-gather-elements-v1", "oracle-0009c-gather-nd-v1",
                "oracle-0009c-gather-v1", "oracle-0009c-initial-state-v1", "oracle-0009c-one-hot-v1",
                "oracle-0009c-pad-v1", "oracle-0009c-permute-v1", "oracle-0009c-random-dropout-v1",
                "oracle-0009c-reshape-v1", "oracle-0009c-scatter-elements-v1", "oracle-0009c-scatter-nd-v1",
                "oracle-0009c-select-v1", "oracle-0009c-slice-update-v1", "oracle-0009c-slice-v1",
                "oracle-0009c-squeeze-v1", "oracle-0009c-stack-v1", "oracle-0009c-tile-v1", "oracle-0009c-unfold-axis-v1",
                "oracle-0009c-unfold2d-v1", "oracle-exact-fixture-classfile-v1",
                "performance-exact-fixture-unmeasured-v1").stream().sorted().toList(),
                canonical.stream().map(Entry::id).sorted().toList());
        verify(canonical, records);
    }

    /**
     * Verifies an arbitrary parsed ledger against supplied live records for mutation controls.
     *
     * @param entries parsed candidate relations; never {@code null}
     * @param records live execution-produced records keyed by owner; never {@code null}
     * @throws AssertionError if a generated record has no unique valid relation
     */
    static void verify(List<Entry> entries, Map<String, CpuGeneratedCoverageEvidenceRegistry.Record> records) {
        assertTrue(entries.stream().noneMatch(entry -> entry.disposition.equals("PENDING")));
        assertEquals(entries.size(), entries.stream().map(Entry::id).distinct().count(), "duplicate evidence id");
        entries.forEach(entry -> {
            entry.verifyReference();
            assertTrue(records.values().stream().anyMatch(entry::applies), "orphan selector: " + entry.id);
        });
    }

    /**
     * Parses the checked registry resource without converting it into execution evidence.
     *
     * @return immutable canonical relations in resource order; never {@code null}
     * @throws AssertionError if the resource is missing or violates its strict TSV schema
     */
    static List<Entry> entries() {
        try (InputStream input = CpuGeneratedCoverageDispositionRegistry.class.getResourceAsStream(RESOURCE)) {
            assertNotNull(input, "canonical disposition registry");
            String text = new String(input.readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(text.endsWith("\n") && !text.contains("\r"), "canonical LF registry");
            String[] lines = text.split("\n", -1);
            assertEquals(HEADER, lines[0]);
            Map<String, Entry> unique = new LinkedHashMap<>();
            for (int line = 1; line < lines.length - 1; line++) {
                String[] fields = lines[line].split("\t", -1);
                assertEquals(5, fields.length, "registry line " + (line + 1));
                for (String field : fields) assertFalse(field.isBlank(), "blank registry field " + (line + 1));
                Entry entry = new Entry(fields[0], Kind.valueOf(fields[1]), fields[2], fields[3], fields[4]);
                assertFalse(entry.disposition.equals("PENDING"), entry.id);
                assertNull(unique.put(entry.id, entry), "duplicate evidence id: " + entry.id);
            }
            return List.copyOf(unique.values());
        } catch (java.io.IOException failure) { throw new AssertionError("cannot read canonical disposition registry", failure); }
    }

    /**
     * One immutable selector and the evidence document it cites.
     *
     * @param id unique ledger identifier; never blank
     * @param kind independent dimension governed by this entry; never {@code null}
     * @param disposition fail-closed state selected when the relation applies; never blank
     * @param applicability semicolon-separated exact predicates over a live record; never blank
     * @param reference immutable evidence reference supporting the disposition; never blank
     */
    record Entry(String id, Kind kind, String disposition, String applicability, String reference) {
        boolean applies(CpuGeneratedCoverageEvidenceRegistry.Record record) {
            for (String term : applicability.split(";")) {
                String[] pair = term.split("=", -1);
                if (pair.length != 2) throw new AssertionError("malformed applicability for " + id);
                String actual = switch (pair[0]) {
                    case "outcome" -> record.outcome();
                    case "owner-prefix" -> record.occurrenceId();
                    case "operation-form" -> record.operationForm();
                    case "evidence-key" -> record.evidenceKey();
                    case "provider-supported" -> Boolean.toString(record.providerSupported());
                    case "preparer-admitted" -> Boolean.toString(record.preparerAdmitted());
                    case "prepared-ir-structural-key" -> record.preparedIrStructuralKey();
                    case "class-sha256" -> record.classHash();
                    case "normalized-body-key" -> record.normalizedBodyKey();
                    default -> throw new AssertionError("unknown applicability key " + pair[0] + " for " + id);
                };
                if (pair[1].equals("present")) {
                    if (actual.isBlank() || actual.startsWith("N/A_") || actual.equals("NO_GENERATED_UNIT")) return false;
                } else if (pair[0].equals("owner-prefix")) {
                    if (!actual.startsWith(pair[1])) return false;
                } else if (!actual.equals(pair[1])) return false;
            }
            return true;
        }
        /** Counts exact predicates; a presence check is a fallback guard, not a selector. */
        int specificity() {
            return (int) java.util.Arrays.stream(applicability.split(";"))
                    .filter(term -> !term.endsWith("=present")).count();
        }
        void verifyReference() {
            if (!reference.equals("CpuGeneratedCoverageCheckpointTest#exactCombinationInventoryReproducesEveryCanonicalFixtureExecution")
                    && !reference.equals("CpuAffineMovementIndexingScatterRandomStructuralOracleTest#exactOwnedRowsRetainGeneratedAbiAndScopedHygieneWhileOnlyOtherFamiliesRemainPartial")) {
                throw new AssertionError("unknown evidence reference for " + id + ": " + reference);
            }
        }
    }
}
