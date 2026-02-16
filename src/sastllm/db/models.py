from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .db import Base


class RepositoryModel(Base):
    __tablename__ = "repositories"

    # Attributes
    repository_id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    label = Column(String(255), nullable=True)
    processed = Column(Boolean, default=False, nullable=False)
    split = Column(String(50), nullable=True)  # e.g., 'train', 'val', 'test'
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    files = relationship("FileModel", back_populates="repository")


class FileModel(Base):
    __tablename__ = "files"

    # Attributes
    file_id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.repository_id"), nullable=False)
    language = Column(String(50), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1024), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    repository = relationship("RepositoryModel", back_populates="files")
    snippets = relationship("SnippetModel", back_populates="file")


class SnippetModel(Base):
    __tablename__ = "snippets"

    # Attributes
    snippet_id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    code = Column(Text, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    file = relationship("FileModel", back_populates="snippets")
    functionalities = relationship("FunctionalityModel", back_populates="snippet")


class FunctionalityModel(Base):
    __tablename__ = "functionalities"

    # Attributes
    functionality_id = Column(Integer, primary_key=True)
    snippet_id = Column(Integer, ForeignKey("snippets.snippet_id"), nullable=False)
    description = Column(Text, nullable=False)
    tag = Column(Text, nullable=False)
    cluster_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    snippet = relationship("SnippetModel", back_populates="functionalities")


class CSNCodeSnippetModel(Base):
    __tablename__ = "csn_snippets"

    # Attributes
    csn_snippet_id = Column(Integer, primary_key=True)
    repository = Column(String(255), nullable=False)
    filepath = Column(String(1024), nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    code = Column(Text, nullable=False)
    functionality = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RepositoryPredictionModel(Base):
    __tablename__ = "repository_predictions"

    # Attributes
    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey("repositories.repository_id"), nullable=False)
    method = Column(String(32), nullable=False)  # e.g., 'llm' | 'ml'
    classification = Column(String(64), nullable=True)  # 'malware' | 'benignware'
    flags_json = Column(JSONB, nullable=True)  # ["F1", "F6", ...]
    justification = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    repository = relationship("RepositoryModel")
