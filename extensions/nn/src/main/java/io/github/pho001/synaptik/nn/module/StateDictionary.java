package io.github.pho001.synaptik.nn.module;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable ordered in-memory snapshot of a complete module tree's state bindings.
 *
 * <p>The entry list is defensively copied and preserves encounter order. Entry paths are unique.
 * {@link #entries()} returns the retained unmodifiable copy. Structural immutability does not copy,
 * evaluate, materialize, or transfer ownership of the Tensor referenced by each entry.</p>
 *
 * <p>This value is an in-memory object boundary, not serialized checkpoint data. Record equality,
 * hashing, and diagnostic text describe its Java components and are not a persistent schema or
 * wire format.</p>
 *
 * @param entries non-null encounter-ordered entries with no null element or duplicate exact path;
 *     the generated accessor returns the unmodifiable defensive copy, never this caller-owned list
 */
public record StateDictionary(List<StateEntry> entries) {
    /**
     * Creates an ordered structural snapshot of the supplied entries.
     *
     * @param entries non-null caller-owned list read and defensively copied in encounter order;
     *     every entry must be non-null and every exact path unique
     * @throws NullPointerException if {@code entries} is null or an entry is null; an entry failure
     *     identifies its zero-based index
     * @throws IllegalArgumentException if the first repeated exact path is encountered
     */
    public StateDictionary {
        Objects.requireNonNull(entries, "entries");
        Set<String> paths = new HashSet<>();
        for (int index = 0; index < entries.size(); index++) {
            StateEntry entry = Objects.requireNonNull(entries.get(index), "entries[" + index + "]");
            if (!paths.add(entry.path())) {
                throw new IllegalArgumentException("duplicate state path: " + entry.path());
            }
        }
        entries = List.copyOf(entries);
    }
}
