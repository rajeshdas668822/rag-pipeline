"""Multi-format document loader for the RAG reranking lab.

Supports extraction from PDF (pypdf), Markdown/Text (UTF-8), and JSON/JSONL
(structured records with id + text). Provides detailed error reporting for
unsupported formats, empty files, and duplicate document IDs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TextIO

from pypdf import PdfReader

from .config import Config
from .rag_types import Document

__all__ = ["DocumentLoader", "UnsupportedFileTypeError", "ExtractionError", "DuplicateDocumentIdError"]

SUPPORTED_EXT: set[str] = {".pdf", ".md", ".txt", ".json", ".jsonl"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file has an unsupported extension."""

    def __init__(self, path: str):
        self.path = path
        ext = Path(path).suffix.lower()
        super().__init__(f"Unsupported file type: {ext}. Supported extensions: {SUPPORTED_EXT}")


class ExtractionError(ValueError):
    """Raised when text extraction from a file fails."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"Extraction failed for {path}: {message}")


class DuplicateDocumentIdError(ValueError):
    """Raised when a duplicate document ID is encountered."""

    def __init__(self, doc_id: str, path: str, existing_path: str):
        self.doc_id = doc_id
        self.path = path
        self.existing_path = existing_path
        super().__init__(
            f"Duplicate document ID '{doc_id}' found in '{path}' "
            f"(already present in '{existing_path}')"
        )


class DocumentLoader:
    """Load documents from files in various formats.

    Supports PDF, Markdown/Text, and JSON/JSONL files. Each document is
    assigned a unique ID and returned with its extracted text and source path.

    Args:
        config: Config instance (used for potential future configuration).
    """

    def __init__(self, config: Config):
        self._config = config
        self._seen_ids: dict[str, str] = {}

    def load_file(self, path: str) -> list[Document]:
        """Load a single document file.

        Args:
            path: Path to the document file.

        Returns:
            List of Document objects extracted from the file.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported.
            ExtractionError: If text extraction fails.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ExtractionError(path, f"File not found: {path}")

        return self._load_file_internal(path, file_path)

    def load_directory(self, data_dir: str) -> list[Document]:
        """Load all documents from a directory.

        Args:
            data_dir: Path to the directory containing documents.

        Returns:
            List of all Document objects found in the directory.

        Raises:
            UnsupportedFileTypeError: If any file has an unsupported extension.
            ExtractionError: If any file fails to extract.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        documents: list[Document] = []
        for file_path in sorted(data_path.iterdir()):
            if file_path.is_file():
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                except UnsupportedFileTypeError as e:
                    print(f"Skipping unsupported file: {e.path}")
                except ExtractionError as e:
                    print(f"Skipping file due to extraction error: {e.path}: {e.message}")

        return documents

    def _load_file_internal(self, path: str, file_path: Path) -> list[Document]:
        """Internal loader that dispatches to format-specific handlers.

        Args:
            path: Path string for error reporting.
            file_path: Path object for file operations.

        Returns:
            List of Document objects.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported.
            ExtractionError: If text extraction fails.
        """
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path, file_path)
        elif ext in (".md", ".txt"):
            return self._load_text(path, file_path)
        elif ext in (".json", ".jsonl"):
            return self._load_json(path, file_path)
        else:
            raise UnsupportedFileTypeError(path)

    def _load_pdf(self, path: str, file_path: Path) -> list[Document]:
        """Load documents from a PDF file.

        Args:
            path: Path string for error reporting.
            file_path: Path object for file operations.

        Returns:
            List of Document objects.

        Raises:
            ExtractionError: If PDF extraction fails.
        """
        try:
            reader = PdfReader(file_path)
            text_parts: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)
            if not text.strip():
                raise ExtractionError(path, "PDF contains no extractable text")
            doc_id = self._assign_id(path, 0)
            self._check_duplicate(doc_id, path)
            return [Document(
                doc_id=doc_id,
                text=text,
                source_path=path,
                metadata={"page_count": len(reader.pages)}
            )]
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(path, f"PDF extraction failed: {e}") from e

    def _load_text(self, path: str, file_path: Path) -> list[Document]:
        """Load documents from a text or markdown file.

        Args:
            path: Path string for error reporting.
            file_path: Path object for file operations.

        Returns:
            List of Document objects.

        Raises:
            ExtractionError: If text reading fails.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            if not text.strip():
                raise ExtractionError(path, "File is empty or contains only whitespace")
            doc_id = self._assign_id(path, 0)
            self._check_duplicate(doc_id, path)
            return [Document(
                doc_id=doc_id,
                text=text,
                source_path=path,
                metadata={"format": "text" if path.endswith(".txt") else "markdown"}
            )]
        except ExtractionError:
            raise
        except UnicodeDecodeError as e:
            raise ExtractionError(path, f"UTF-8 decoding failed: {e}") from e
        except Exception as e:
            raise ExtractionError(path, f"Text reading failed: {e}") from e

    def _load_json(self, path: str, file_path: Path) -> list[Document]:
        """Load documents from a JSON or JSONL file.

        Args:
            path: Path string for error reporting.
            file_path: Path object for file operations.

        Returns:
            List of Document objects.

        Raises:
            ExtractionError: If JSON parsing fails or records are malformed.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            if path.endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    return self._extract_documents_from_jsonl(data, path)
                elif isinstance(data, dict):
                    return self._extract_documents_from_jsonl([data], path)
                else:
                    raise ExtractionError(path, "JSON root must be an object or array")
            else:  # .jsonl
                records = self._parse_jsonl(content, path)
                return self._extract_documents_from_jsonl(records, path)
        except ExtractionError:
            raise
        except json.JSONDecodeError as e:
            raise ExtractionError(path, f"JSON parsing failed: {e}") from e
        except Exception as e:
            raise ExtractionError(path, f"JSON processing failed: {e}") from e

    def _parse_jsonl(self, content: str, path: str) -> list[dict]:
        """Parse JSONL content into a list of records.

        Args:
            content: Raw JSONL string.
            path: Path string for error reporting.

        Returns:
            List of parsed records (dictionaries).

        Raises:
            ExtractionError: If any line is invalid JSON.
        """
        records: list[dict] = []
        for i, line in enumerate(content.strip().split("\n"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ExtractionError(path, f"Invalid JSON on line {i}: {e}") from e
        return records

    def _extract_documents_from_jsonl(self, records: list[dict], path: str) -> list[Document]:
        """Extract documents from a list of JSON records.

        Args:
            records: List of dictionaries with id and text fields.
            path: Path string for error reporting.

        Returns:
            List of Document objects.

        Raises:
            ExtractionError: If any record is missing id or text fields.
        """
        documents: list[Document] = []
        for i, record in enumerate(records, start=0):
            if not isinstance(record, dict):
                raise ExtractionError(path, f"Record {i} is not a dictionary")
            if "id" not in record:
                raise ExtractionError(path, f"Record {i} missing 'id' field")
            if "text" not in record:
                raise ExtractionError(path, f"Record {i} missing 'text' field")
            
            doc_id = str(record["id"])
            self._check_duplicate(doc_id, path)
            
            metadata = {k: v for k, v in record.items() if k not in ("id", "text")}
            documents.append(Document(
                doc_id=doc_id,
                text=str(record["text"]),
                source_path=path,
                metadata=metadata
            ))
        return documents

    def _assign_id(self, path: str, seq: int) -> str:
        """Generate a unique document ID from path and sequence.

        Args:
            path: Source file path.
            seq: Sequence number for multiple documents per file.

        Returns:
            Formatted document ID.
        """
        stem = Path(path).stem
        return f"{stem}_{seq}"

    def _check_duplicate(self, doc_id: str, path: str) -> None:
        """Check for duplicate document IDs.

        Args:
            doc_id: Document ID to check.
            path: Current file path for error reporting.

        Raises:
            DuplicateDocumentIdError: If doc_id was already seen.
        """
        if doc_id in self._seen_ids:
            raise DuplicateDocumentIdError(doc_id, path, self._seen_ids[doc_id])
        self._seen_ids[doc_id] = path
