# Kafka

Shared Kafka cluster is used for non-prod environments, so isolation is achieved with proper naming conventions and
permissions management.

---

## Naming rules

1. **Business purpose clarity**: Topic names must make the business purpose clear. Examples: `antifraud-requests`,
   `contact-prepared`, `contract-signed-by-bank`, `email-responses`.

    * Naming pattern `${domain_resource}-events` is reserved for REST API conventions.

2. **Environment suffix**: To run the same topic in multiple environments, attach the environment as a suffix:

    * Examples: `antifraud-requests-dev`, `antifraud-requests-uat`.

3. **Topic versioning**: If backward-incompatible changes occur in event structure, append version:

    * Example: `antifraud-requests-1-dev`.

4. **Consumer group uniqueness**: Each consumer group must have a unique name to avoid overlap.

    * For services, use the service name as group ID.
    * For local access, each developer must define a unique group ID.

### Proposed naming rules (DRAFT)

Based on [this article](https://devshawn.com/blog/apache-kafka-topic-naming-conventions/), topic names should include *
*environment**, **domain**, **classification**, and **description**:

* Format: `${env}.${domain}.${classification}.${description}`
* Rules:

    * `env`: `dev`, `uat`, or `prod`
    * `domain`: official domain name (e.g., `cards`, `clients`, `payments`)
    * `classification`: type/purpose (e.g., `events`, `cdc`, `requests`, `responses`, `internal`)
    * `description`: logical name of the topic

**Examples:**

* `antifraud-requests-dev` → `dev.clients.requests.antifraud`
* `contact-prepared-uat` → `uat.clients.events.contract-prepared`
* `email-responses-prod` → `prod.clients.internal-responses.email`

**Open questions:**

* Do we need to add a zone prefix (DMZ, LAN)?
* Does AVALaunch resources provisioning support this schema?

---

## Usage patterns

1. All Kafka cluster access must be secured with SSL/TLS. Each service has a dedicated certificate for managing *
   *READ/WRITE** permissions.
2. Topics are divided into **EXTERNAL** (part of public API) and **INTERNAL** (for inter-service communication).
   Services may configure two Kafka clusters for logical
   separation. [Spring Boot Kafka starter library](https://gitlab.avalaunch.aval/applications/cards/kafka-starter-lib)
   simplifies configuration.
3. Each topic must define a **retention period** — avoid relying on defaults.
4. Each event must specify a **partition key** to ensure correct processing order. Typically, a unique business ID is
   used.
5. Define a **retry policy** per consumer group to handle failures (retry forever, dead-letter queue, DB persistence,
   logging, etc.). Default policies risk lost events.
6. Each event must include **metadata** either in the body or headers:

    * Unique event ID (idempotent processing)
    * Event creation time (tracing)

---

## Schema management

1. Each topic must aggregate events of the same structure to avoid consumer-side complexity.
2. Event structure should be defined in **Avro schema format** and stored in a **Kafka schema registry**. Producer-side
   validation should verify structure before sending.
3. DTO classes can be generated automatically from Avro schemas and distributed as versioned JARs across services.

References: [Avro spec](https://avro.apache.org/docs/current/spec.html)

---

## Frameworks and tools

1. Use higher-level libraries instead of raw Kafka Java client:

    * [Spring Kafka](https://spring.io/projects/spring-kafka) – simple communication
    * [Spring Cloud Stream Kafka Binder](https://cloud.spring.io/spring-cloud-stream-binder-kafka/) – event streaming
      and data processing
2. Enable distributed tracing with [Spring Cloud Sleuth](https://spring.io/projects/spring-cloud-sleuth). Kafka is
   supported out-of-the-box.
3. For local non-prod access, [KafkaTool](https://www.kafkatool.com/) can be used with dedicated credentials.