import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class SourceTreeHygieneTest {

    @Test
    void sourceTreeDoesNotContainCompiledOrTempArtifacts() throws IOException {
        List<Path> roots = List.of(Path.of("src"), Path.of("test"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .map(Path::normalize)
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.endsWith(".class")
                                || name.equals(".DS_Store")
                                || name.startsWith(".tmp")
                                || name.contains(".tmp");
                    })
                    .map(Path::toString)
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Source tree contains generated artifacts: " + offenders);
        }
    }
}
