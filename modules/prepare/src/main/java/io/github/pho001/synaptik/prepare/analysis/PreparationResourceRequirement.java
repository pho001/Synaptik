package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.Objects;

/**
 * Declares one exact backend-neutral shared resource need discovered during partition analysis.
 *
 * <p>A buffer requirement associates one projected graph value with exact byte size and
 * alignment. A workspace requirement uses an analysis-local non-negative identity for backend
 * scratch that has no graph-value identity. Both variants are immutable declarations for later
 * shared slot assignment. Neither variant is a Runtime slot, address, allocation, storage object,
 * resource handle, lifetime interval, executable, or per-run binding.</p>
 */
public sealed interface PreparationResourceRequirement
        permits PreparationResourceRequirement.Buffer, PreparationResourceRequirement.Workspace {
    /**
     * Declares the exact shared buffer representation required for one projected graph value.
     *
     * @param valueId non-null graph-local value identity; retained exactly for compile/prepare
     *     association and never interpreted as a Runtime slot
     * @param byteSize exact non-negative number of bytes required; zero is valid
     * @param byteAlignment positive power-of-two byte alignment
     */
    record Buffer(ValueId valueId, long byteSize, long byteAlignment)
            implements PreparationResourceRequirement {
        /**
         * Creates one exact graph-value buffer declaration.
         *
         * @param valueId non-null projected value identity to retain exactly
         * @param byteSize exact non-negative byte size; zero is valid
         * @param byteAlignment positive power-of-two alignment measured in bytes
         * @throws NullPointerException if {@code valueId} is {@code null}; the message is
         *     {@code valueId}
         * @throws IllegalArgumentException if {@code byteSize} is negative; the message is
         *     {@code byteSize must be non-negative}
         * @throws IllegalArgumentException if {@code byteAlignment} is not a positive power of
         *     two; the message is {@code byteAlignment must be a positive power of two}
         */
        public Buffer {
            Objects.requireNonNull(valueId, "valueId");
            if (byteSize < 0) {
                throw new IllegalArgumentException("byteSize must be non-negative");
            }
            if (byteAlignment <= 0
                    || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
        }

        /**
         * Returns the projected logical value associated with this declaration.
         *
         * @return exact non-null immutable graph-value identity supplied at construction
         */
        @Override
        public ValueId valueId() {
            return valueId;
        }

        /**
         * Returns the exact requested byte count.
         *
         * @return non-negative byte size; zero is valid
         */
        @Override
        public long byteSize() {
            return byteSize;
        }

        /**
         * Returns the exact requested byte alignment.
         *
         * @return positive power-of-two alignment measured in bytes
         */
        @Override
        public long byteAlignment() {
            return byteAlignment;
        }
    }

    /**
     * Declares one exact backend workspace need local to a partition analysis result.
     *
     * @param requirementId non-negative identity unique among workspaces in one analysis result;
     *     the value is not a Runtime slot or global identity
     * @param byteSize exact non-negative number of bytes required; zero is valid
     * @param byteAlignment positive power-of-two byte alignment
     */
    record Workspace(long requirementId, long byteSize, long byteAlignment)
            implements PreparationResourceRequirement {
        /**
         * Creates one exact analysis-local workspace declaration.
         *
         * @param requirementId non-negative analysis-local workspace identity
         * @param byteSize exact non-negative byte size; zero is valid
         * @param byteAlignment positive power-of-two alignment measured in bytes
         * @throws IllegalArgumentException if {@code requirementId} is negative; the message is
         *     {@code requirementId must be non-negative}
         * @throws IllegalArgumentException if {@code byteSize} is negative; the message is
         *     {@code byteSize must be non-negative}
         * @throws IllegalArgumentException if {@code byteAlignment} is not a positive power of
         *     two; the message is {@code byteAlignment must be a positive power of two}
         */
        public Workspace {
            if (requirementId < 0) {
                throw new IllegalArgumentException("requirementId must be non-negative");
            }
            if (byteSize < 0) {
                throw new IllegalArgumentException("byteSize must be non-negative");
            }
            if (byteAlignment <= 0
                    || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "byteAlignment must be a positive power of two");
            }
        }

        /**
         * Returns this declaration's analysis-local workspace identity.
         *
         * @return non-negative identity scoped only to one analysis result
         */
        @Override
        public long requirementId() {
            return requirementId;
        }

        /**
         * Returns the exact requested byte count.
         *
         * @return non-negative byte size; zero is valid
         */
        @Override
        public long byteSize() {
            return byteSize;
        }

        /**
         * Returns the exact requested byte alignment.
         *
         * @return positive power-of-two alignment measured in bytes
         */
        @Override
        public long byteAlignment() {
            return byteAlignment;
        }
    }
}
