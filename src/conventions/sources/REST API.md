# REST API

The REST architectural style has been chosen for the API platform. At the same time, many business processes are
asynchronous, which makes it impossible to fully implement them on the basis of HTTP as a synchronous protocol.
Therefore, the HTTP API is extended with events in Kafka to notify about changes in resource state.

---

## Key principles

It is very important to follow consistent principles for REST API:

1. All changes in the system are represented by standard CRUD operations on resources.
2. Each resource is uniquely identifiable (it has a unique ID). For example, in the Marqeta API, the resource identifier
   is called *token*.
3. For CRUD operations, the full capabilities of the HTTP protocol are used (methods, status codes, content type
   management, etc.).
4. The server implementation remains stateless (no client state is stored, only resource state).
5. The state of resources can be cached on intermediate hosts if the resource has been marked as cacheable by HTTP
   metadata.
6. The request and response must contain complete information for their processing (each request is considered
   independent).
7. Idempotency of all operations on resources (clients can make the same call repeatedly with the same result on the
   server).

---

## Design rules

1. Use nouns but not verbs. HTTP methods are used to represent operations.
2. GET method and query parameters must not alter the state of resources.
3. Use HTTP headers to specify data serialization format. **Content-Type** and **Accept** headers define
   request/response formats.
4. Provide filtering, sorting, field selection, and paging for collections.

**API examples with filtering and sorting**

```http
GET /cars?color=red           # Returns a list of red cars
GET /cars?seats<=2            # Returns a list of cars with a maximum of 2 seats
GET /cars?sort=-manufactorer,+model
GET /cars?fields=manufacturer,model,id,color
GET /cars?offset=10&limit=5
```

5. Version API according to the common versioning approach.
6. Handle errors with HTTP status codes.
7. Use the error payload to specify the root of the error.

**Error payload example**

```json
{
  "errors": [
    {
      "userMessage": "Sorry, the requested resource does not exist",
      "internalMessage": "No car found in the database",
      "code": 34,
      "more info": "http://dev.mwaysolutions.com/blog/api/v1/errors/12345"
    }
  ]
}
```

---

## Kafka events usage to extend REST API

For asynchronous scenarios, event streams are used in addition to the HTTP protocol. Key principles for using this
approach:

1. Each resource type corresponds to a single channel of events streaming.
2. All events for a resource type have the same structure and carry information about changes in the state of a
   particular resource.
3. Each event has its own identifier to provide tracking and implement client-side idempotent processing.
4. Each event is linked to a specific resource through its identifier.
5. All events for a specific resource are strictly time-ordered and time-stamped.

---

## Naming rules

All resources are named according to the entities of the domain model in order to best match business processes, as well
as distribute the implementation on the server between microservices. Examples: *Card*, *Card Transition*, *Limit
Request*.

---

## HTTP endpoints

* For each resource type, a root endpoint is created to create and retrieve a list of resources of this type. Its name
  is the same as the plural name of the resource type.

    * Example: For resource type *Card* the root endpoint will be `/cards`, allowing to get a list of resources via *
      *HTTP GET** and create new resources via **HTTP POST**.

* To access a specific resource, a child endpoint with a resource identifier is used.

    * Example: `/cards/${card_id}`, allowing to get a concrete resource via **HTTP GET** and update it via **HTTP PUT**.

* To provide operations with only a part of a specific resource, deeper child endpoints can be created.

    * Example: `/cards/${card_id}/pin`, allowing to get or update the PIN of a card.

---

## Kafka topics

* Kafka topics are used to organize event streams for all types of resources.
* A separate topic is used for each type of resource, named by the resource type.

    * Example: For *Card*, topic name is `card-events`.
* Environment suffixes are added to topic names.

    * Example: `card-events-dev` (DEV), `card-events-uat` (UAT).

Each event includes:

* Unique identifier (`event_id`)
* Resource identifier (`token` or `id`)
* Creation time (`event_creation_date`)
* Changes in the resource state (`data` or `state`)

Events for the same resource must use the same Kafka partition key (e.g., resource identifier) for sequential
processing.

---

## Spring Web MVC usage

1. All REST controllers must use dedicated DTOs for inputs and outputs. Domain and persistent entities are not allowed.
2. All inputs are validated according to the API contract.
3. REST controllers contain no business logic—they handle HTTP adaptation and invoke business services.
4. If the API-first approach is used, REST controllers must inherit from interfaces generated from OpenAPI specs. DTOs
   are also taken from generated JARs.
5. Common error handlers must be implemented in a separate `@ControllerAdvice` class. Specific error handlers may be
   placed inside the corresponding controller.
6. Each REST controller must restrict content types for inputs and outputs to supported ones (JSON by default) to avoid
   serialization issues.
