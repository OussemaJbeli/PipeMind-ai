# PipeMind AI

**PipeMind AI** is the intelligence layer of PipeMind, an intelligent CI/CD failure analysis platform.

While the PipeMind backend is responsible for orchestration, integrations, application state, queues, permissions, and execution, this repository is responsible for **understanding CI/CD data and turning it into useful engineering intelligence**.

The purpose of this service is not simply to send logs to an LLM and display its response.

PipeMind AI should progressively combine:

```text id="f5x0p3"
Log Processing
      +
Failure Classification
      +
Anomaly Detection
      +
Similarity Search
      +
Project Context
      +
RAG
      +
LLMs
      +
Historical Knowledge
      ↓
Intelligent CI/CD Analysis
```

The architecture should remain open to new AI/ML approaches as the project evolves.

---

# The Role of PipeMind AI

A CI/CD platform can tell us:

```text id="q9yqhv"
Pipeline failed.
Job failed.
Exit code: 1.
```

PipeMind AI should try to answer:

```text id="7n8j9m"
What failed?

Why did it fail?

What changed?

Is this failure related to the current commit?

Have we seen something similar before?

How confident are we?

What is the safest solution?

Can the problem potentially be fixed automatically?
```

The AI service therefore acts as the **reasoning and intelligence layer** of PipeMind.

---

# Position in the Architecture

```text id="b6a6qt"
Developer
    ↓
Git Push
    ↓
GitLab / GitHub / Jenkins
    ↓
PipeMind Backend
    ↓
Build Analysis Context
    ↓
┌─────────────────────────────┐
│       PipeMind AI           │
│                             │
│ Log Processing              │
│ Failure Classification      │
│ Anomaly Detection           │
│ Similarity Engine           │
│ RAG                         │
│ LLM Integration             │
│ Recommendation Engine       │
└──────────────┬──────────────┘
               │
        Structured Analysis
               ↓
       PipeMind Backend
               ↓
          PostgreSQL
               ↓
       PipeMind Frontend
```

The AI service does not own the main application state.

The backend remains responsible for storing and orchestrating the system.

---

# Main Intelligence Pipeline

A typical failure analysis can follow a pipeline similar to:

```text id="hkjz0g"
Raw CI/CD Context
        ↓
Log Processor
        ↓
Failure Classification
        ↓
Anomaly Detection
        ↓
Similarity Search
        ↓
Context Retrieval / RAG
        ↓
LLM Reasoning
        ↓
Recommendation Engine
        ↓
Structured Analysis
```

This pipeline is a conceptual architecture rather than a requirement that every failure must pass through every component.

Some analyses may need only part of the pipeline.

---

# 1. Log Processor

CI/CD logs are often noisy and contain large amounts of information that are irrelevant to the actual failure.

The Log Processor should transform raw logs into useful structured information.

For example:

```text id="a1s2d3"
2026-08-29 18:21:02 npm test
2026-08-29 18:21:04 PASS home.test.ts
2026-08-29 18:21:05 FAIL login.test.ts
2026-08-29 18:21:05 Expected status 200
2026-08-29 18:21:05 Received status 401
2026-08-29 18:21:06 Process exited with code 1
```

could become:

```json id="j2x4b7"
{
  "failed_file": "login.test.ts",
  "error_type": "assertion",
  "expected": 200,
  "received": 401,
  "exit_code": 1
}
```

Possible techniques include:

```text id="w4h8zq"
Regex
Parsing
Stack-trace extraction
Error signature extraction
Noise removal
Tokenization
Log normalization
```

The processor should reduce unnecessary information before expensive AI processing.

---

# 2. Failure Classifier

The classifier attempts to determine what type of failure occurred.

Possible categories could include:

```text id="9m3q2r"
BUILD
TEST
DEPENDENCY
DATABASE
NETWORK
DOCKER
DEPLOYMENT
CONFIGURATION
AUTHENTICATION
PERMISSION
INFRASTRUCTURE
RESOURCE
UNKNOWN
```

For example:

```text id="3f8n2k"
Expected 200
Received 401
Login test failed
```

could produce:

```text id="e4t6p1"
Category:
AUTHENTICATION

Type:
TEST_FAILURE

Confidence:
96%
```

The implementation may evolve from simple approaches to more advanced ML.

Potential approaches include:

```text id="c6y9v2"
Rule-based classification
TF-IDF + traditional ML
Embeddings
Transformer models
LLM classification
Hybrid approaches
```

The project should favor measurable evaluation over choosing a model simply because it is more sophisticated.

---

# 3. Anomaly Detector

Not every problem is an explicit failure.

A pipeline can succeed while behaving abnormally.

PipeMind AI should therefore be capable of identifying unusual behavior.

Examples:

```text id="q8p3f1"
Build duration increased by 700%

Memory consumption is significantly higher

Deployment time suddenly increased

Failure frequency increased

A normally stable job starts retrying

Test count unexpectedly decreases
```

Possible approaches include:

```text id="k3s7w1"
Statistical baselines
Moving averages
Threshold detection
Isolation Forest
Clustering
Time-series analysis
Other anomaly detection methods
```

The system should consider project history when determining what is normal.

A 10-minute build may be normal for one project and abnormal for another.

---

# 4. Similarity Engine

One of PipeMind's important capabilities is learning from historical failures.

The Similarity Engine compares a new failure against previously observed incidents.

For example:

```text id="v5m2x9"
Current failure:

HTTP 401
login.test.ts
Expected 200
Received 401
```

Historical failures might produce:

```text id="d7k4p8"
Failure #921 → 94% similarity
Failure #874 → 87% similarity
Failure #743 → 51% similarity
```

This can be implemented using embeddings and vector similarity.

A possible flow:

```text id="u1b6r0"
Failure
   ↓
Embedding Model
   ↓
Vector Representation
   ↓
Vector Database
   ↓
Nearest Historical Failures
```

PostgreSQL with a vector extension can potentially serve this purpose, depending on the final architecture.

---

# 5. Project Context

Logs alone are often insufficient to understand a failure.

PipeMind AI should be able to reason using additional project context supplied by the backend.

Context can include:

```text id="f4z1q7"
Project technology
Repository information
Branch
Commit
Changed files
Pipeline configuration
Failed job
Relevant logs
Previous pipeline
Previous failures
Project documentation
Environment information
Dependency information
```

For example:

```text id="s8p2m4"
Project:
Vue 3 + TypeScript

Changed:
auth.ts

Failure:
HTTP 401

Previous successful pipeline:
#1841

Similar historical failure:
#921
```

This context can dramatically improve the quality of analysis.

---

# 6. RAG Engine

The RAG layer allows PipeMind AI to retrieve relevant knowledge before asking an LLM to reason about a problem.

The retrieved information could include:

```text id="n2k6v5"
Previous failures
Resolved incidents
Project documentation
CI/CD documentation
Architecture information
Known errors
Repository-specific knowledge
```

A conceptual flow:

```text id="e9r1w3"
Current Failure
      ↓
Create Query / Embedding
      ↓
Retrieve Relevant Knowledge
      ↓
Build Context
      ↓
LLM
      ↓
Grounded Analysis
```

The goal is to prevent the LLM from reasoning only from generic knowledge.

PipeMind should ideally be able to say:

> "This resembles a failure that happened in this project three months ago."

rather than:

> "HTTP 401 usually means authentication failed."

---

# 7. LLM Integration

LLMs provide the higher-level reasoning capability of PipeMind.

The first external provider may be:

```text id="r7x3m2"
Gemini API
```

However, the architecture should not be tightly coupled to Gemini.

The service should be able to support multiple providers.

For example:

```text id="a3k8q1"
                    LLM Provider
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Gemini      Ollama      Other
             API       Local LLM   Provider
```

A provider abstraction can allow the rest of PipeMind AI to work without caring which model is being used.

---

# External and Local AI

PipeMind should support two general modes.

## External AI

Example:

```text id="h7d2m8"
PipeMind AI
     ↓
Gemini API
     ↓
Analysis
```

Advantages may include:

```text id="p5c8n1"
Strong general reasoning
No local GPU requirement
Easy experimentation
Access to powerful models
```

Potential concerns include:

```text id="q6m4z9"
Privacy
API costs
Network dependency
Rate limits
External data processing
```

---

## Local AI

A local model can potentially be integrated through tools such as Ollama or another compatible inference layer.

```text id="b2k9x4"
PipeMind AI
     ↓
Local Model Runtime
     ↓
LLM
```

Potential advantages:

```text id="w8n3r6"
Local processing
Reduced external dependency
Better control over sensitive logs
No per-request external API cost
Offline experimentation
```

Potential limitations include:

```text id="e1p7q5"
Hardware requirements
Model quality
Inference speed
Memory requirements
Operational complexity
```

PipeMind should not assume that local AI is always better or that external AI is always better.

The architecture should make the choice configurable.

---

# 8. Root Cause Analysis

The main objective of the intelligence layer is not merely to summarize logs.

It should attempt to identify the **most likely root cause**.

For example:

```text id="t4y8k2"
Failure:

Expected 200
Received 401

Changed file:
auth.ts

Previous successful pipeline:
#1841

Historical similar failure:
#921
```

PipeMind AI could produce:

```text id="m6v3p8"
Root Cause:
The authentication token is not being attached
to subsequent API requests.

Confidence:
91%

Evidence:
- Login test returned HTTP 401.
- auth.ts changed in the current commit.
- Similar failure #921 had the same root cause.
- Previous pipeline succeeded.
```

The system should distinguish between:

```text id="s1q4d7"
Observed fact
Inference
Prediction
Recommendation
```

This is important for user trust.

---

# 9. Recommendation Engine

After identifying a likely cause, PipeMind should determine useful next actions.

For example:

```text id="k9w2e6"
Recommendation:

Check the Axios authentication interceptor
in src/services/auth.ts.

Verify that the JWT token is stored after login
and attached to subsequent requests.
```

Recommendations may eventually include:

```text id="p4r8n2"
Retry pipeline
Inspect configuration
Modify a file
Generate a patch
Create an issue
Create a merge request
Rollback deployment
```

The AI service should provide the recommendation and relevant metadata.

Execution remains controlled by the backend.

---

# 10. Remediation Intelligence

PipeMind may eventually move from:

```text id="z7v4x1"
Detect
 ↓
Explain
 ↓
Recommend
```

toward:

```text id="c5m8q2"
Detect
 ↓
Explain
 ↓
Recommend
 ↓
Generate Fix
 ↓
Validate
 ↓
Request Approval
 ↓
Execute
```

The AI service can help determine:

```text id="n6b2p9"
Possible fix
Expected impact
Confidence
Risk
Affected files
Validation strategy
```

It should not independently bypass backend authorization or execute arbitrary infrastructure operations.

The backend remains the execution boundary.

---

# AI Provider Abstraction

The internal architecture should avoid spreading provider-specific code throughout the project.

A conceptual design:

```text id="y8c3m6"
LLMProvider
    │
    ├── GeminiProvider
    │
    ├── OllamaProvider
    │
    └── FutureProvider
```

This allows the same analysis pipeline to work with different models.

The exact abstraction can evolve as the requirements become clearer.

---

# Suggested Architecture

The initial service can be implemented as a Python API service.

A possible structure:

```text id="r3k7w2"
PipeMind-ai/

├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── analyze.py
│   │   ├── classify.py
│   │   ├── similarity.py
│   │   └── health.py
│   │
│   ├── services/
│   │   ├── log_processor.py
│   │   ├── classifier.py
│   │   ├── anomaly_detector.py
│   │   ├── similarity.py
│   │   ├── rag.py
│   │   └── recommendation.py
│   │
│   ├── providers/
│   │   ├── base.py
│   │   ├── gemini.py
│   │   └── ollama.py
│   │
│   ├── models/
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   └── config.py
│
├── models/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── dataset.py
│
├── tests/
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

This is a starting point, not a mandatory final structure.

---

# API Communication

The backend should communicate with PipeMind AI through an internal API.

For example:

```text id="j4v9p3"
POST /analyze
```

with a structured analysis context.

The AI service can return structured results such as:

```json id="x5m2n8"
{
  "category": "authentication",
  "severity": "high",
  "confidence": 0.91,
  "root_cause": "Missing authorization header",
  "evidence": [],
  "similar_failures": [],
  "recommendations": [],
  "remediation": {
    "available": true,
    "requires_approval": true
  }
}
```

The contract should be versionable so that improvements to the AI service do not unexpectedly break the backend.

---

# AI Should Be Explainable

PipeMind is intended for engineering environments where developers need to trust the result.

The AI should therefore try to provide:

```text id="q7r3k1"
Conclusion
Evidence
Confidence
Reasoning context
Historical references
Recommendation
Risk
```

Rather than simply:

```text id="u8n4m6"
"The problem is auth.ts."
```

A good analysis should allow the developer to ask:

> "Why does PipeMind believe this?"

and receive evidence from the actual project and pipeline.

---

# Historical Learning

PipeMind should become more useful as it observes more pipelines.

A successful resolution can become future knowledge.

Example:

```text id="c4v8z2"
Failure #1842

Problem:
Missing Authorization header

Solution:
Updated Axios interceptor

Result:
Pipeline succeeded
```

Later:

```text id="n7x3p5"
Failure #2097

Similarity:
96%

Previous resolution:
Update Axios interceptor
```

This creates a feedback loop:

```text id="w2m6q9"
Observe
  ↓
Analyze
  ↓
Recommend
  ↓
Developer resolves
  ↓
Observe outcome
  ↓
Store knowledge
  ↓
Improve future analysis
```

This historical intelligence is one of the important ideas behind PipeMind.

---

# Evaluation

AI quality should be measurable.

The project should eventually evaluate questions such as:

```text id="p8k4v2"
Was the failure classified correctly?

Was the predicted root cause correct?

Was the retrieved historical failure relevant?

Was the recommendation useful?

Did the proposed remediation work?

How confident was the system?

How often was the AI wrong?
```

Possible metrics can include:

```text id="x3m7q1"
Classification accuracy
Precision / Recall / F1
Similarity relevance
Retrieval quality
Root-cause accuracy
Recommendation success rate
False-positive rate
Latency
Token usage
Cost
```

The project should avoid evaluating the AI only through subjective impressions.

---

# Privacy and Security

CI/CD logs can contain sensitive information.

PipeMind AI should therefore assume that logs may contain:

```text id="b6n2r8"
Tokens
URLs
Environment variables
Credentials
Internal hostnames
Repository information
Infrastructure information
```

The service should support appropriate sanitization/redaction before sending information to external LLM providers.

For example:

```text id="m9q4t1"
Raw Log
   ↓
Secret Detection / Redaction
   ↓
Safe Context
   ↓
External LLM
```

Local AI may provide another option for environments where logs should remain inside the infrastructure.

Security should be treated as part of the AI architecture, not as an afterthought.

---

# What Belongs in PipeMind AI

### Belongs here

```text id="a7d3k9"
AI/ML algorithms
Log intelligence
NLP
Failure classification
Anomaly detection
Embeddings
Similarity search logic
RAG
LLM integrations
Prompt/context construction
AI evaluation
Model training
Inference
Recommendation generation
AI-related experiments
```

### Does not primarily belong here

```text id="v2p8m4"
User authentication
CI/CD credentials management
Main application database logic
Frontend UI
Project CRUD
External webhook management
Queue orchestration
Permission management
Direct remediation execution
```

Those responsibilities belong primarily to PipeMind Backend or PipeMind Front.

---

# Relationship With Other Repositories

```text id="k5n8q2"
                         PipeMind
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
   PipeMind-front    PipeMind-back      PipeMind-ai
   Vue + TypeScript  Laravel 12         Python
          │                 │                 │
          │                 │      Intelligence│
          │                 └──────────►       │
          │                            │       │
          │                            ├── ML  │
          │                            ├── RAG │
          │                            ├── LLM │
          │                            └── NLP │
          │                                    │
          └────────────── API ◄────────────────┘
                            │
                            ▼
                       PipeMind-data
```

`PipeMind-data` provides shared project knowledge, datasets, schemas, research material, experiments, and documentation.

---

# The BO-12 Example

A simplified real-world analysis could look like:

```text id="z3x7m1"
Developer pushes:

BO-12-login
      ↓
CI/CD Pipeline
      ↓
Unit Tests FAIL
      ↓
Backend collects:
- Logs
- Changed files
- Commit
- Project information
- Pipeline history
      ↓
PipeMind AI
      ↓
Log Processor
      ↓
Authentication Failure
      ↓
Historical Similarity: 94%
      ↓
RAG retrieves previous resolution
      ↓
LLM analyzes the complete context
      ↓
Recommendation Engine
      ↓
Structured result
```

Result:

```text id="c8p4y2"
Category:
Authentication

Severity:
High

Confidence:
91%

Likely Root Cause:
Authorization token is not attached to
API requests.

Evidence:
- HTTP 401
- auth.ts changed
- Similar previous failure

Recommendation:
Inspect the Axios authentication interceptor.

Possible Remediation:
Generate a patch and run the test suite.
```

The backend then stores and exposes this result to the frontend.

---

# Technology Direction

The initial AI service is expected to use Python because the ecosystem provides strong support for:

```text id="n6v2x8"
Machine Learning
NLP
Embeddings
Data Processing
LLMs
Experimentation
Scientific Computing
```

A Python web framework such as FastAPI can expose the intelligence layer as an internal service.

The project may use libraries and technologies appropriate to each intelligence problem rather than forcing every feature through an LLM.

For example:

```text id="r1k5m7"
Traditional algorithms
        +
ML models
        +
Embedding models
        +
Vector search
        +
LLMs
```

The goal is to use the **right intelligence technique for the right problem**.

---

# Guiding Philosophy

PipeMind AI should not be:

> "Send everything to Gemini and hope for a good answer."

It should become:

> **A layered intelligence system that combines deterministic processing, machine learning, historical knowledge, project context, retrieval, and generative AI to understand CI/CD failures.**

The LLM is an important component, but it is not the entire intelligence layer.

The strongest version of PipeMind should combine:

```text id="y6m2q8"
What happened?
       ↓
What type of problem is it?
       ↓
Is it abnormal?
       ↓
Have we seen it before?
       ↓
What project context matters?
       ↓
What is the most likely explanation?
       ↓
What should the developer do?
       ↓
Can the solution be safely automated?
```

The architecture described here is a foundation rather than a fixed implementation.

New models, algorithms, retrieval strategies, providers, and intelligence techniques can be introduced when experimentation demonstrates that they improve PipeMind's ability to understand, explain, and resolve CI/CD problems.
