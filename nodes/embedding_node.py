
from state.state import ResearchPaperState

from services.embedding_service import EmbeddingService
from services.document_management_service import (
    DocumentManagementService
)


def embedding_node(
    state: ResearchPaperState,
) -> ResearchPaperState:

    print("=" * 60)
    print("EMBEDDING NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})

    document_id = artifacts["document_id"]
    owner_id = artifacts["owner_id"]
    version = artifacts["version"]
    ingestion_id = artifacts["ingestion_id"]

    chunk_file = artifacts["chunk_file"]

    collection_name = state.get(
        "collection_name",
        "research_papers",
    )

    document_service = DocumentManagementService()

    try:

        # -----------------------------------------
        # 1. Index new version in Qdrant
        # -----------------------------------------

        result = EmbeddingService.index_chunks(
            chunk_file=chunk_file,
            document_id=document_id,
            owner_id=owner_id,
            version=version,
            ingestion_id=ingestion_id,
            collection_name=collection_name,
        )

        print(
            f"Successfully indexed "
            f"{result['indexed_chunks']} chunks."
        )

        # -----------------------------------------
        # 2. Mark version as indexed
        # -----------------------------------------

        document_service.update_version_status(
            document_id=document_id,
            owner_id=owner_id,
            version=version,
            status="indexed",
            extra_fields={
                "chunk_count": result["indexed_chunks"],
                "collection_name": collection_name,
            },
        )

        # -----------------------------------------
        # 3. Activate new version
        # -----------------------------------------

        activation = document_service.activate_version(
            document_id=document_id,
            owner_id=owner_id,
            version=version,
        )

        print(
            f"Active version: {activation['active_version']}"
        )

        # -----------------------------------------
        # 4. Delete old embeddings
        # -----------------------------------------

        old_version = activation["previous_version"]

        deletion_result = None

        if old_version is not None:

            deletion_result = EmbeddingService.delete_version(
                document_id=document_id,
                owner_id=owner_id,
                version=old_version,
                collection_name=collection_name,
            )

            print(
                f"Deleted old version: {old_version}"
            )

        return {
            **state,

            "artifacts": {
                **artifacts,

                "collection_name": collection_name,

                "embedding": {
                    "dense_model": result["dense_model"],
                    "sparse_model": result["sparse_model"],
                    "vector_dimension": result["vector_dimension"],
                    "indexed_chunks": result["indexed_chunks"],
                    "verified_chunks": result["verified_chunks"],
                },

                "active_version": activation["active_version"],
                "previous_version": old_version,
                "old_version_deletion": deletion_result,
            },

            "error": "",
        }

    
    except Exception as exc:

        print(f"Embedding/version activation failed: {exc}")

        try:

            current_document = document_service.get_document(
                document_id=document_id,
                owner_id=owner_id,
            )

            active_version = (
                current_document.get("active_version")
                if current_document else None
            )

            if active_version != version:

                print(
                    f"Version {version} is not active. "
                    "Cleaning up its Qdrant points."
                )

                # 1. Remove partially or fully indexed points
                try:

                    EmbeddingService.cleanup_failed_version(
                        document_id=document_id,
                        owner_id=owner_id,
                        version=version,
                        collection_name=collection_name,
                    )

                except Exception as cleanup_error:

                    print(
                        f"Qdrant cleanup failed: {cleanup_error}"
                    )

                # 2. Mark MongoDB version as failed
                document_service.update_version_status(
                    document_id=document_id,
                    owner_id=owner_id,
                    version=version,
                    status="failed",
                    error=str(exc),
                )

            else:

                print(
                    f"Version {version} is already active. "
                    "Preserving its Qdrant points."
                )

        except Exception as status_error:

            print(
                f"Failure recovery encountered an error: "
                f"{status_error}"
            )

        return {
            **state,
            "error": f"Embedding/version management failed: {exc}",
        }

    finally:
        document_service.close()
