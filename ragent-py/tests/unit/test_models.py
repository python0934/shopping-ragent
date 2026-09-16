"""Unit tests for ORM models — table names, columns, defaults, constraints."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import RelationshipProperty

# Import all models to register them on Base.metadata before tests run
import app.models  # noqa: F401
from app.core.database import Base


class TestAllModelsRegistered:
    """Verify all 28 tables are registered on the Base metadata."""

    def test_total_table_count(self):
        assert len(Base.metadata.tables) == 28

    def test_all_expected_table_names(self):
        expected = {
            "t_user", "t_biz_change_log", "t_sample_question",
            "t_conversation", "t_conversation_summary", "t_message", "t_message_feedback",
            "t_knowledge_base", "t_knowledge_document", "t_knowledge_chunk",
            "t_knowledge_document_chunk_log", "t_knowledge_document_schedule",
            "t_knowledge_document_schedule_exec",
            "t_intent_node", "t_query_term_mapping", "t_rag_trace_run", "t_rag_trace_node",
            "t_agent_profile", "t_agent_prompt", "t_agent_conversation",
            "t_agent_message", "t_agent_state", "t_agent_context_compaction",
            "t_ingestion_pipeline", "t_ingestion_pipeline_node",
            "t_ingestion_task", "t_ingestion_task_node",
            "t_knowledge_vector",
        }
        actual = set(Base.metadata.tables.keys())
        assert actual == expected


class TestSystemModels:
    """Test system domain model definitions."""

    def test_user_table_name(self):
        from app.models.system import UserDO
        mapper = inspect(UserDO)
        assert mapper.local_table.name == "t_user"

    def test_user_columns(self):
        from app.models.system import UserDO
        mapper = inspect(UserDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert col_names == {"id", "username", "password", "role", "avatar", "create_time", "update_time", "deleted"}

    def test_user_username_unique(self):
        from app.models.system import UserDO
        mapper = inspect(UserDO)
        col = mapper.columns["username"]
        assert col.unique is True

    def test_biz_change_log_has_jsonb(self):
        from app.models.system import BizChangeLogDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(BizChangeLogDO)
        for col_name in ["before_snapshot", "after_snapshot", "change_diff"]:
            assert isinstance(mapper.columns[col_name].type, JSONB)

    def test_biz_change_log_success_default(self):
        from app.models.system import BizChangeLogDO
        mapper = inspect(BizChangeLogDO)
        col = mapper.columns["success"]
        assert col.default is not None

    def test_sample_question_columns(self):
        from app.models.system import SampleQuestionDO
        mapper = inspect(SampleQuestionDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "question" in col_names
        assert "title" in col_names


class TestChatModels:
    """Test chat domain model definitions."""

    def test_message_jsonb_columns(self):
        from app.models.chat import MessageDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(MessageDO)
        for col_name in ["sources", "recommended_questions", "retrieved_chunks"]:
            assert isinstance(mapper.columns[col_name].type, JSONB)

    def test_message_status_default(self):
        from app.models.chat import MessageDO
        mapper = inspect(MessageDO)
        col = mapper.columns["message_status"]
        assert col.default is not None
        assert col.default.arg == "NORMAL"

    def test_conversation_unique_fields(self):
        from app.models.chat import ConversationDO
        mapper = inspect(ConversationDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "conversation_id" in col_names
        assert "user_id" in col_names

    def test_feedback_vote_not_null(self):
        from app.models.chat import MessageFeedbackDO
        mapper = inspect(MessageFeedbackDO)
        assert mapper.columns["vote"].nullable is False


class TestKnowledgeModels:
    """Test knowledge domain model definitions."""

    def test_knowledge_base_unique_collection(self):
        from app.models.knowledge import KnowledgeBaseDO
        mapper = inspect(KnowledgeBaseDO)
        assert mapper.columns["collection_name"].unique is True

    def test_knowledge_document_jsonb(self):
        from app.models.knowledge import KnowledgeDocumentDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(KnowledgeDocumentDO)
        assert isinstance(mapper.columns["ingestion_spec"].type, JSONB)

    def test_knowledge_document_defaults(self):
        from app.models.knowledge import KnowledgeDocumentDO
        mapper = inspect(KnowledgeDocumentDO)
        assert mapper.columns["status"].default.arg == "pending"
        assert mapper.columns["enabled"].default.arg == 1

    def test_chunk_log_no_deleted(self):
        """Chunk log table has no soft-delete column (audit log)."""
        from app.models.knowledge import KnowledgeDocumentChunkLogDO
        mapper = inspect(KnowledgeDocumentChunkLogDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "deleted" not in col_names

    def test_schedule_unique_doc_id(self):
        from app.models.knowledge import KnowledgeDocumentScheduleDO
        mapper = inspect(KnowledgeDocumentScheduleDO)
        assert mapper.columns["doc_id"].unique is True


class TestRagModels:
    """Test RAG intent & query model definitions."""

    def test_intent_node_jsonb(self):
        from app.models.rag import IntentNodeDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(IntentNodeDO)
        assert isinstance(mapper.columns["collection_names"].type, JSONB)

    def test_intent_node_defaults(self):
        from app.models.rag import IntentNodeDO
        mapper = inspect(IntentNodeDO)
        assert mapper.columns["kind"].default.arg == 0
        assert mapper.columns["sort_order"].default.arg == 0
        assert mapper.columns["enabled"].default.arg == 1

    def test_trace_run_unique_trace_id(self):
        from app.models.rag import RagTraceRunDO
        mapper = inspect(RagTraceRunDO)
        assert mapper.columns["trace_id"].unique is True

    def test_query_term_mapping_columns(self):
        from app.models.rag import QueryTermMappingDO
        mapper = inspect(QueryTermMappingDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "source_term" in col_names
        assert "target_term" in col_names
        assert "match_type" in col_names


class TestAgentModels:
    """Test agent domain model definitions."""

    def test_agent_profile_unique_name(self):
        from app.models.agent import AgentProfileDO
        mapper = inspect(AgentProfileDO)
        assert mapper.columns["name"].unique is True

    def test_agent_prompt_composite_unique(self):
        from app.models.agent import AgentPromptDO
        mapper = inspect(AgentPromptDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "agent_id" in col_names
        assert "slot_key" in col_names

    def test_agent_state_composite_pk(self):
        """AgentState uses composite primary key (user_id, session_id, state_key)."""
        from app.models.agent import AgentStateDO
        mapper = inspect(AgentStateDO)
        pk_cols = {c.key for c in mapper.primary_key}
        assert pk_cols == {"user_id", "session_id", "state_key"}

    def test_agent_state_no_deleted(self):
        """AgentState has no soft-delete column."""
        from app.models.agent import AgentStateDO
        mapper = inspect(AgentStateDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "deleted" not in col_names

    def test_agent_message_jsonb(self):
        from app.models.agent import AgentMessageDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(AgentMessageDO)
        assert isinstance(mapper.columns["blocks"].type, JSONB)

    def test_agent_context_compaction_no_deleted(self):
        """Compaction is append-only audit log — no deleted column."""
        from app.models.agent import AgentContextCompactionDO
        mapper = inspect(AgentContextCompactionDO)
        col_names = {c.key for c in mapper.column_attrs}
        assert "deleted" not in col_names
        assert "update_time" not in col_names


class TestIngestionModels:
    """Test ingestion pipeline model definitions."""

    def test_pipeline_node_jsonb(self):
        from app.models.ingestion import IngestionPipelineNodeDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(IngestionPipelineNodeDO)
        assert isinstance(mapper.columns["settings_json"].type, JSONB)
        assert isinstance(mapper.columns["condition_json"].type, JSONB)

    def test_task_jsonb(self):
        from app.models.ingestion import IngestionTaskDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(IngestionTaskDO)
        assert isinstance(mapper.columns["logs_json"].type, JSONB)
        assert isinstance(mapper.columns["metadata_json"].type, JSONB)

    def test_task_node_output_is_text(self):
        """output_json is stored as TEXT (full output), not JSONB."""
        from app.models.ingestion import IngestionTaskNodeDO
        from sqlalchemy import Text
        mapper = inspect(IngestionTaskNodeDO)
        assert isinstance(mapper.columns["output_json"].type, Text)


class TestVectorModel:
    """Test vector storage model."""

    def test_vector_table_name(self):
        from app.models.vector import KnowledgeVectorDO
        mapper = inspect(KnowledgeVectorDO)
        assert mapper.local_table.name == "t_knowledge_vector"

    def test_vector_metadata_column(self):
        """metadata column is renamed to metadata_ in Python to avoid clash."""
        from app.models.vector import KnowledgeVectorDO
        from sqlalchemy.dialects.postgresql import JSONB
        mapper = inspect(KnowledgeVectorDO)
        # Python attribute name is metadata_, mapped to DB column 'metadata'
        assert isinstance(mapper.columns["metadata_"].type, JSONB)
        # Verify the actual DB column name
        assert mapper.columns["metadata_"].name == "metadata"


class TestSoftDeletePattern:
    """Verify soft-delete pattern is consistent across models."""

    # Models that SHOULD have deleted column
    SOFT_DELETE_MODELS = [
        "UserDO", "SampleQuestionDO",
        "ConversationDO", "ConversationSummaryDO", "MessageDO", "MessageFeedbackDO",
        "KnowledgeBaseDO", "KnowledgeDocumentDO", "KnowledgeChunkDO",
        "IntentNodeDO", "QueryTermMappingDO",
        "RagTraceRunDO", "RagTraceNodeDO",
        "AgentProfileDO", "AgentPromptDO", "AgentConversationDO", "AgentMessageDO",
        "IngestionPipelineDO", "IngestionPipelineNodeDO", "IngestionTaskDO", "IngestionTaskNodeDO",
    ]

    # Models that should NOT have deleted column
    NO_SOFT_DELETE_MODELS = [
        "BizChangeLogDO",  # audit log
        "KnowledgeDocumentChunkLogDO",  # audit log
        "KnowledgeDocumentScheduleDO",  # unique constraint on doc_id
        "KnowledgeDocumentScheduleExecDO",  # execution record
        "AgentStateDO",  # composite PK state store
        "AgentContextCompactionDO",  # append-only audit
        "KnowledgeVectorDO",  # vector storage
    ]

    def test_soft_delete_models_have_deleted(self):
        import app.models as models_pkg
        for model_name in self.SOFT_DELETE_MODELS:
            model_cls = getattr(models_pkg, model_name)
            mapper = inspect(model_cls)
            col_names = {c.key for c in mapper.column_attrs}
            assert "deleted" in col_names, f"{model_name} should have 'deleted' column"

    def test_no_soft_delete_models(self):
        import app.models as models_pkg
        for model_name in self.NO_SOFT_DELETE_MODELS:
            model_cls = getattr(models_pkg, model_name)
            mapper = inspect(model_cls)
            col_names = {c.key for c in mapper.column_attrs}
            assert "deleted" not in col_names, f"{model_name} should NOT have 'deleted' column"
