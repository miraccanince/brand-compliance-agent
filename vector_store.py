import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from langchain_ollama import OllamaEmbeddings

from config import QDRANT_URL as DEFAULT_QDRANT_URL
from models import BrandRule

COLLECTION  = "brand_rules"
VECTOR_SIZE = 768  # nomic-embed-text


class RuleVectorStore:
    def __init__(self, url: str = DEFAULT_QDRANT_URL):
        # Use in-memory mode when no Qdrant server is available
        try:
            self.client = QdrantClient(url=url, timeout=3)
            self.client.get_collections()  # connection check
        except Exception:
            self.client = QdrantClient(":memory:")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION in existing:
            try:
                info = self.client.get_collection(COLLECTION)
                vectors = info.config.params.vectors
                current_size = vectors.size if hasattr(vectors, "size") else VECTOR_SIZE
                if current_size != VECTOR_SIZE:
                    self.client.delete_collection(COLLECTION)
                    existing = []
            except Exception:
                self.client.delete_collection(COLLECTION)
                existing = []
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def store_rules(self, rules: list[BrandRule], session_id: str) -> None:
        points = []
        for rule in rules:
            text = f"{rule.category}: {rule.rule}" + (f" ({rule.value})" if rule.value else "")
            vector = self.embeddings.embed_query(text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "category": rule.category,
                        "rule": rule.rule,
                        "value": rule.value,
                    },
                )
            )
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search_relevant_rules(
        self, query: str, session_id: str, limit: int = 15
    ) -> list[BrandRule]:
        vector = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=limit,
        )
        return [
            BrandRule(
                category=r.payload["category"],
                rule=r.payload["rule"],
                value=r.payload.get("value"),
            )
            for r in results.points
        ]
