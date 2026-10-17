package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable diagnostics for one completed CPU-owned OpenBLAS discovery call.
 *
 * <p>The result records only normalized platform text, exact typed selections, attempt outcomes,
 * and failure strings. It owns no provider, native object, throwable, route decision, cost,
 * qualification evidence, or thread state.</p>
 *
 * @param requestedMode the mode that produced this result
 * @param platformSnapshot the normalized platform used only for automatic discovery
 * @param attempts the exact ordered load attempts, copied defensively
 * @param selected the exact successfully loaded selection, present only when loaded
 * @param status the closed discovery outcome
 */
record CpuOpenBlasDiscoveryResult(CpuOpenBlasDiscoveryRequest.Mode requestedMode,
        Optional<PlatformSnapshot> platformSnapshot, List<Attempt> attempts,
        Optional<Selection> selected, Status status) {
    /** Closed discovery outcome independent of route qualification. */
    enum Status {
        /** Discovery was explicitly disabled. */
        DISABLED,
        /** Every requested candidate failed to load or bind. */
        UNAVAILABLE,
        /** One candidate loaded and bound the provider's required symbols. */
        LOADED
    }

    /** Normalized operating-system classification for the bounded table. */
    enum OperatingSystemFamily {
        /** macOS or Darwin. */
        MACOS,
        /** Linux. */
        LINUX,
        /** Windows. */
        WINDOWS,
        /** Any unrecognized or absent operating-system token. */
        OTHER
    }

    /** Outcome of one exact loader delegation. */
    enum AttemptStatus {
        /** The provider loaded and completely bound the exact selection. */
        LOADED,
        /** The provider reported its typed loading or binding failure. */
        LOAD_OR_BIND_FAILURE
    }

    /**
     * Immutable normalized platform facts used to choose automatic candidates.
     *
     * @param operatingSystemFamily the classified operating-system family
     * @param normalizedOperatingSystem the trimmed lower-case operating-system token, possibly
     *     empty
     * @param normalizedArchitecture the trimmed lower-case architecture token, possibly empty
     */
    record PlatformSnapshot(OperatingSystemFamily operatingSystemFamily,
            String normalizedOperatingSystem, String normalizedArchitecture) {
        /**
         * Validates already-normalized diagnostic platform facts.
         *
         * @param operatingSystemFamily the required operating-system family
         * @param normalizedOperatingSystem the normalized operating-system token
         * @param normalizedArchitecture the normalized architecture token
         * @throws NullPointerException if a component is {@code null}
         * @throws IllegalArgumentException if a token is not trimmed and lower-case or the
         *     operating-system family disagrees with the normalized token
         */
        PlatformSnapshot {
            Objects.requireNonNull(operatingSystemFamily, "operatingSystemFamily");
            Objects.requireNonNull(normalizedOperatingSystem, "normalizedOperatingSystem");
            Objects.requireNonNull(normalizedArchitecture, "normalizedArchitecture");
            if (!normalizedOperatingSystem.equals(normalize(normalizedOperatingSystem))
                    || !normalizedArchitecture.equals(normalize(normalizedArchitecture))) {
                throw new IllegalArgumentException("platform tokens must already be normalized");
            }
            if (operatingSystemFamily != familyOf(normalizedOperatingSystem)) {
                throw new IllegalArgumentException(
                        "operating-system family and normalized token disagree");
            }
        }

        /**
         * Normalizes and classifies one snapshotted pair of platform-property values.
         * A {@code null} value represents an absent property and becomes the empty token.
         *
         * @param operatingSystem the raw operating-system property value, or {@code null}
         * @param architecture the raw architecture property value, or {@code null}
         * @return immutable normalized and classified platform facts
         */
        static PlatformSnapshot from(String operatingSystem, String architecture) {
            String normalizedOperatingSystem = normalize(operatingSystem);
            String normalizedArchitecture = normalize(architecture);
            return new PlatformSnapshot(familyOf(normalizedOperatingSystem),
                    normalizedOperatingSystem,
                    normalizedArchitecture);
        }

        /**
         * Produces the stable diagnostic token used by classification and metadata.
         *
         * @param value the raw platform value, or {@code null} when absent
         * @return the trimmed lower-case token, or the empty string for absence
         */
        private static String normalize(String value) {
            return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        }

        /**
         * Classifies one already-normalized operating-system token for the fixed table.
         *
         * @param normalizedOperatingSystem the non-null trimmed lower-case token
         * @return the matching closed family, or {@link OperatingSystemFamily#OTHER}
         */
        private static OperatingSystemFamily familyOf(String normalizedOperatingSystem) {
            if (normalizedOperatingSystem.startsWith("mac")
                    || normalizedOperatingSystem.startsWith("darwin")) {
                return OperatingSystemFamily.MACOS;
            }
            if (normalizedOperatingSystem.startsWith("linux")) {
                return OperatingSystemFamily.LINUX;
            }
            if (normalizedOperatingSystem.startsWith("windows")) {
                return OperatingSystemFamily.WINDOWS;
            }
            return OperatingSystemFamily.OTHER;
        }
    }

    /** Typed exact provider selection retained without resolving a loader path. */
    sealed interface Selection permits LibraryName, AbsoluteLibraryPath { }

    /**
     * Exact operating-system loader-name selection.
     *
     * @param value the nonblank name passed unchanged to the provider
     */
    record LibraryName(String value) implements Selection {
        /**
         * Validates an exact loader name without trimming or rewriting it.
         *
         * @param value the nonblank exact name
         * @throws NullPointerException if {@code value} is {@code null}
         * @throws IllegalArgumentException if {@code value} is blank
         */
        LibraryName {
            Objects.requireNonNull(value, "value");
            if (value.isBlank()) throw new IllegalArgumentException("library name must not be blank");
        }
    }

    /**
     * Exact absolute provider path selection.
     *
     * @param value the absolute path passed unchanged to the provider
     */
    record AbsoluteLibraryPath(Path value) implements Selection {
        /**
         * Validates an exact path without resolving, normalizing, or probing it.
         *
         * @param value the exact absolute path
         * @throws NullPointerException if {@code value} is {@code null}
         * @throws IllegalArgumentException if {@code value} is relative
         */
        AbsoluteLibraryPath {
            Objects.requireNonNull(value, "value");
            if (!value.isAbsolute()) throw new IllegalArgumentException("library path must be absolute");
        }
    }

    /**
     * Immutable record of one exact candidate delegation.
     *
     * @param position the zero-based position in the attempted candidate sequence
     * @param selection the exact typed selection passed to the loader
     * @param status the attempt outcome
     * @param failureType the thrown provider failure type name, present only for failure
     * @param failureMessage the nullable provider failure message represented as an optional,
     *     present only when the failed exception supplied a message
     */
    record Attempt(int position, Selection selection, AttemptStatus status,
            Optional<String> failureType, Optional<String> failureMessage) {
        /**
         * Validates one attempt and its success/failure diagnostics.
         *
         * @param position the non-negative zero-based attempt position
         * @param selection the exact selection attempted
         * @param status the attempt outcome
         * @param failureType the failure type, required only for failure
         * @param failureMessage the optional failure message, empty when absent or successful
         * @throws NullPointerException if a required reference is {@code null}
         * @throws IllegalArgumentException if position or diagnostic relationships are invalid
         */
        Attempt {
            Objects.requireNonNull(selection, "selection");
            Objects.requireNonNull(status, "status");
            failureType = Objects.requireNonNull(failureType, "failureType");
            failureMessage = Objects.requireNonNull(failureMessage, "failureMessage");
            if (position < 0) throw new IllegalArgumentException("attempt position must be non-negative");
            if (status == AttemptStatus.LOADED
                    && (failureType.isPresent() || failureMessage.isPresent())) {
                throw new IllegalArgumentException("loaded attempt must not contain failure diagnostics");
            }
            if (status == AttemptStatus.LOAD_OR_BIND_FAILURE && failureType.isEmpty()) {
                throw new IllegalArgumentException("failed attempt must contain a failure type");
            }
            if (failureType.isPresent() && failureType.orElseThrow().isBlank()) {
                throw new IllegalArgumentException("failure type must not be blank");
            }
        }

        /**
         * Creates a successful attempt with no failure diagnostics.
         *
         * @param position the zero-based position
         * @param selection the exact loaded selection
         * @return an immutable successful attempt
         */
        static Attempt loaded(int position, Selection selection) {
            return new Attempt(position, selection, AttemptStatus.LOADED, Optional.empty(),
                    Optional.empty());
        }

        /**
         * Creates a failed attempt from immutable diagnostic strings.
         *
         * @param position the zero-based position
         * @param selection the exact failed selection
         * @param failureType the nonblank thrown type name
         * @param failureMessage the nullable thrown message
         * @return an immutable failed attempt
         */
        static Attempt failed(int position, Selection selection, String failureType,
                String failureMessage) {
            return new Attempt(position, selection, AttemptStatus.LOAD_OR_BIND_FAILURE,
                    Optional.of(Objects.requireNonNull(failureType, "failureType")),
                    Optional.ofNullable(failureMessage));
        }
    }

    /**
     * Defensively copies metadata and validates mode, platform, attempts, selection, and status.
     *
     * @param requestedMode the required originating request mode
     * @param platformSnapshot the platform present exactly for automatic mode
     * @param attempts the ordered attempt list
     * @param selected the successful selection present exactly for loaded status
     * @param status the required result status
     * @throws NullPointerException if a required reference or collection member is {@code null}
     * @throws IllegalArgumentException if any status or ordering relationship is inconsistent
     */
    CpuOpenBlasDiscoveryResult {
        Objects.requireNonNull(requestedMode, "requestedMode");
        platformSnapshot = Objects.requireNonNull(platformSnapshot, "platformSnapshot");
        attempts = List.copyOf(attempts);
        selected = Objects.requireNonNull(selected, "selected");
        Objects.requireNonNull(status, "status");
        if ((requestedMode == CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC)
                != platformSnapshot.isPresent()) {
            throw new IllegalArgumentException(
                    "platform snapshot must be present exactly for automatic discovery");
        }
        for (int index = 0; index < attempts.size(); index++) {
            if (attempts.get(index).position() != index) {
                throw new IllegalArgumentException("attempt positions must be contiguous and ordered");
            }
        }
        Set<Selection> unique = new HashSet<>();
        if (!attempts.stream().allMatch(attempt -> unique.add(attempt.selection()))) {
            throw new IllegalArgumentException("attempt selections must be unique");
        }
        validateModeAttempts(requestedMode, platformSnapshot, attempts);
        switch (status) {
            case DISABLED -> {
                if (requestedMode != CpuOpenBlasDiscoveryRequest.Mode.DISABLED
                        || !attempts.isEmpty() || selected.isPresent()) {
                    throw new IllegalArgumentException("disabled result relationships disagree");
                }
            }
            case UNAVAILABLE -> {
                if (requestedMode == CpuOpenBlasDiscoveryRequest.Mode.DISABLED
                        || attempts.isEmpty() || selected.isPresent()
                        || attempts.stream().anyMatch(attempt ->
                                attempt.status() != AttemptStatus.LOAD_OR_BIND_FAILURE)
                        || requestedMode == CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC
                                && attempts.size() != CpuOpenBlasDiscovery.automaticCandidates(
                                        platformSnapshot.orElseThrow()).size()) {
                    throw new IllegalArgumentException("unavailable result relationships disagree");
                }
            }
            case LOADED -> {
                if (requestedMode == CpuOpenBlasDiscoveryRequest.Mode.DISABLED
                        || attempts.isEmpty() || selected.isEmpty()
                        || attempts.getLast().status() != AttemptStatus.LOADED
                        || !attempts.getLast().selection().equals(selected.orElseThrow())
                        || attempts.stream().limit(attempts.size() - 1L).anyMatch(attempt ->
                                attempt.status() != AttemptStatus.LOAD_OR_BIND_FAILURE)) {
                    throw new IllegalArgumentException("loaded result relationships disagree");
                }
            }
        }
    }

    /**
     * Validates that attempt count, selection kind, and automatic prefix match the request mode.
     *
     * @param mode the non-null originating request mode
     * @param platformSnapshot the platform present exactly for automatic mode
     * @param attempts the non-null immutable ordered attempts
     * @throws IllegalArgumentException if the attempts disagree with the mode or fixed table
     */
    private static void validateModeAttempts(CpuOpenBlasDiscoveryRequest.Mode mode,
            Optional<PlatformSnapshot> platformSnapshot, List<Attempt> attempts) {
        if (mode == CpuOpenBlasDiscoveryRequest.Mode.DISABLED && !attempts.isEmpty()) {
            throw new IllegalArgumentException("disabled discovery must not contain attempts");
        }
        if ((mode == CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME
                || mode == CpuOpenBlasDiscoveryRequest.Mode.EXACT_ABSOLUTE_PATH)
                && attempts.size() != 1) {
            throw new IllegalArgumentException("exact discovery must contain one attempt");
        }
        if (mode == CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME
                && !(attempts.getFirst().selection() instanceof LibraryName)
                || mode == CpuOpenBlasDiscoveryRequest.Mode.EXACT_ABSOLUTE_PATH
                && !(attempts.getFirst().selection() instanceof AbsoluteLibraryPath)) {
            throw new IllegalArgumentException("attempt selection type disagrees with exact mode");
        }
        if (mode == CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC) {
            List<Selection> expected = CpuOpenBlasDiscovery.automaticCandidates(
                    platformSnapshot.orElseThrow());
            if (attempts.size() > expected.size()) {
                throw new IllegalArgumentException("automatic attempts exceed the candidate table");
            }
            for (int index = 0; index < attempts.size(); index++) {
                if (!attempts.get(index).selection().equals(expected.get(index))) {
                    throw new IllegalArgumentException(
                            "automatic attempts must be an exact candidate-table prefix");
                }
            }
        }
    }
}
