# Evidence retrieval

`CustomerServiceAssistant` accepts a caller-supplied `EvidenceRetriever`. When
configured, it calls `search(query, task)` before each generation and includes up
to five `Evidence(source, text)` results in the system context. Each source is
limited to 256 characters and each text excerpt to 2,000 characters. With no
retriever configured, no search is performed.

Use an authorized local index or an external provider only after disclosing that
the current user query may be sent to it and obtaining any required consent.
Minimize or redact personal details before external search. Evidence excerpts are
treated as untrusted data rather than instructions; generated answers are asked
to name sources and surface uncertainty, but applications should verify citations
and important claims.

For `task="opinion"`, the assistant is instructed to distinguish verifiable facts
from its reasoned assessment. An opinion is not a feeling, a professional
determination, or a substitute for source review.
