import pytest
from unittest.mock import patch

from chatbot.flow import rewrite_query, retrieve
from chatbot.vectordb import VectorDB


# -------------------------------------------------
# Mock LLM response for query rewriting
# -------------------------------------------------
def mock_generate_response(llm, prompt):
    """
    Simulates LLM behavior for query rewriting
    """
    prompt_lower = prompt.lower()

    if "tell me more" in prompt_lower:
        return "AWS services mentioned in documents"

    return "AWS services"


# -------------------------------------------------
# Test: rewrite_query node works correctly
# -------------------------------------------------
@patch("chatbot.flow.get_llm")
@patch("chatbot.flow.generate_response", side_effect=mock_generate_response)
def test_rewrite_query(mock_generate, mock_get_llm):
    # Prevent real Azure client creation
    mock_get_llm.return_value = object()

    state = {
        "question": "tell me more",
        "memory": [
            {"role": "user", "content": "What AWS services are mentioned?"},
            {"role": "assistant", "content": "EC2 and S3"}
        ]
    }

    result = rewrite_query(state)

    assert "rewritten_query" in result
    assert result["rewritten_query"] == "AWS services mentioned in documents"


# -------------------------------------------------
# Test: retrieve uses rewritten query if present
# -------------------------------------------------
@patch.object(VectorDB, "similarity_search")
def test_retrieve_uses_rewritten_query(mock_search):
    mock_search.return_value = []

    state = {
        "question": "tell me more",
        "rewritten_query": "AWS services mentioned in documents",
        "context": [],
        "sources": []
    }

    retrieve(state)

    mock_search.assert_called_once()
    args, _ = mock_search.call_args

    assert args[0] == "AWS services mentioned in documents"


# -------------------------------------------------
# Test: retrieve falls back to original question
# -------------------------------------------------
@patch.object(VectorDB, "similarity_search")
def test_retrieve_fallback_to_question(mock_search):
    mock_search.return_value = []

    state = {
        "question": "What AWS services are mentioned?",
        "context": [],
        "sources": []
    }

    retrieve(state)

    mock_search.assert_called_once()
    args, _ = mock_search.call_args

    assert args[0] == "What AWS services are mentioned?"


# -------------------------------------------------
# Integration-style test: rewrite → retrieve flow
# -------------------------------------------------
@patch("chatbot.flow.get_llm")
@patch("chatbot.flow.generate_response", side_effect=mock_generate_response)
@patch.object(VectorDB, "similarity_search")
def test_rewrite_and_retrieve_flow(mock_search, mock_generate, mock_get_llm):
    mock_get_llm.return_value = object()
    mock_search.return_value = []

    state = {
        "question": "tell me more",
        "context": [],
        "sources": [],
        "answer": "",
        "memory": [
            {"role": "user", "content": "What AWS services are mentioned?"},
            {"role": "assistant", "content": "EC2 and S3"}
        ]
    }

    # Step 1: rewrite
    rewritten = rewrite_query(state)
    state.update(rewritten)

    # Step 2: retrieve
    retrieve(state)

    args, _ = mock_search.call_args
    assert args[0] == "AWS services mentioned in documents"
