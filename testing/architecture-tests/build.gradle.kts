import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test

val trainingApiConsumer = extensions.getByType<SourceSetContainer>().create("trainingApiConsumer")

dependencies {
    add(trainingApiConsumer.implementationConfigurationName, project(":extensions:training"))
}

tasks.named<Test>("test") {
    dependsOn(tasks.named(trainingApiConsumer.compileJavaTaskName))
}
