import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test

val trainingApiConsumer = extensions.getByType<SourceSetContainer>().create("trainingApiConsumer")
val nnApiConsumer = extensions.getByType<SourceSetContainer>().create("nnApiConsumer")

dependencies {
    add(trainingApiConsumer.implementationConfigurationName, project(":extensions:training"))
    add(nnApiConsumer.implementationConfigurationName, project(":extensions:nn"))
}

tasks.named<Test>("test") {
    dependsOn(tasks.named(trainingApiConsumer.compileJavaTaskName))
    dependsOn(tasks.named(nnApiConsumer.compileJavaTaskName))

    // Tests read repository files outside their classpath; declare them to avoid false UP-TO-DATE results.
    inputs.files(rootProject.fileTree(rootProject.projectDir) {
        include("settings.gradle.kts", "**/build.gradle.kts")
        include("modules/runtime/src/main/java/**/*.java", "backends/cpu/src/main/**")
        exclude("**/build/**", "**/.gradle/**", "**/.git/**")
    }).withPropertyName("architectureSourceFiles")
        .withPathSensitivity(PathSensitivity.RELATIVE)
}
