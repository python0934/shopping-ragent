"""
ORM models package — re-exports all domain models for convenient imports.

Usage:
    from app.models import UserDO, ConversationDO, KnowledgeBaseDO, ...
"""

# System domain
from app.models.system import BizChangeLogDO, SampleQuestionDO, UserDO

# Chat domain
from app.models.chat import (
    ConversationDO,
    ConversationSummaryDO,
    MessageDO,
    MessageFeedbackDO,
)

# Knowledge domain
from app.models.knowledge import (
    KnowledgeBaseDO,
    KnowledgeChunkDO,
    KnowledgeDocumentChunkLogDO,
    KnowledgeDocumentDO,
    KnowledgeDocumentScheduleDO,
    KnowledgeDocumentScheduleExecDO,
)

# RAG intent & query domain
from app.models.rag import (
    IntentNodeDO,
    QueryTermMappingDO,
    RagTraceNodeDO,
    RagTraceRunDO,
)

# Agent domain
from app.models.agent import (
    AgentContextCompactionDO,
    AgentConversationDO,
    AgentMessageDO,
    AgentProfileDO,
    AgentPromptDO,
    AgentStateDO,
)

# Ingestion pipeline domain
from app.models.ingestion import (
    IngestionPipelineDO,
    IngestionPipelineNodeDO,
    IngestionTaskDO,
    IngestionTaskNodeDO,
)

# Vector storage
from app.models.vector import KnowledgeVectorDO

__all__ = [
    # System
    "UserDO",
    "BizChangeLogDO",
    "SampleQuestionDO",
    # Chat
    "ConversationDO",
    "ConversationSummaryDO",
    "MessageDO",
    "MessageFeedbackDO",
    # Knowledge
    "KnowledgeBaseDO",
    "KnowledgeDocumentDO",
    "KnowledgeChunkDO",
    "KnowledgeDocumentChunkLogDO",
    "KnowledgeDocumentScheduleDO",
    "KnowledgeDocumentScheduleExecDO",
    # RAG
    "IntentNodeDO",
    "QueryTermMappingDO",
    "RagTraceRunDO",
    "RagTraceNodeDO",
    # Agent
    "AgentProfileDO",
    "AgentPromptDO",
    "AgentConversationDO",
    "AgentMessageDO",
    "AgentStateDO",
    "AgentContextCompactionDO",
    # Ingestion
    "IngestionPipelineDO",
    "IngestionPipelineNodeDO",
    "IngestionTaskDO",
    "IngestionTaskNodeDO",
    # Vector
    "KnowledgeVectorDO",
]
