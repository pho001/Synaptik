package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/** Mutable generic live-owner fixture with no public or NN seam. */
class TestParameterSource implements TrainingParameters {
    private final List<Binding> bindings = new ArrayList<>();
    private List<Entry> snapshot;

    /** Creates an empty test owner; subclasses declare bindings before admission. */
    TestParameterSource() { }

    /**
     * Adds a binding without imposing NN or Training descriptor restrictions.
     * @param path diagnostic path
     * @param tensor exact Tensor
     * @return mutable test binding
     */
    protected final Binding parameter(String path, Tensor tensor) {
        Binding binding = new Binding(path, tensor);
        bindings.add(binding);
        return binding;
    }

    /** @return immutable membership captured on the first admission */
    @Override
    public final List<Entry> entries() {
        if (snapshot == null) {
            snapshot = bindings.stream().map(b -> new Entry(b.path, b.value)).toList();
        }
        return snapshot;
    }

    /** @throws IllegalStateException if this owner's live bindings changed */
    @Override
    public void validateUnchanged() {
        List<Entry> expected = entries();
        if (expected.size() != bindings.size()) throw new IllegalStateException("membership changed");
        for (int index = 0; index < expected.size(); index++) {
            Binding actual = bindings.get(index);
            Entry entry = expected.get(index);
            if (!entry.path().equals(actual.path) || entry.tensor() != actual.value) {
                throw new IllegalStateException("live binding changed");
            }
        }
    }

    /** Mutable live Tensor slot, deliberately independent of NN Parameter. */
    static final class Binding {
        private final String path;
        private Tensor value;

        private Binding(String path, Tensor value) {
            this.path = path;
            this.value = value;
        }

        /** @return current exact Tensor binding */
        Tensor value() { return value; }

        /** @param replacement exact Tensor to install in the live test owner */
        void replace(Tensor replacement) { value = replacement; }
    }
}
