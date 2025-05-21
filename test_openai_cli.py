"""
Tests for the OpenAI CLI application.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import openai_cli
from openai_cli import AiConversation


@pytest.fixture
def mock_env_api_key():
    """Fixture to mock environment variable for API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def ai_conversation(mock_env_api_key):
    """Fixture to create an AiConversation instance with mocked API key."""
    return AiConversation()


@pytest.fixture
def mock_session_post():
    """Fixture to mock session.post."""
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        yield mock_post


def test_load_api_key_from_env(ai_conversation):
    """Test loading API key from environment variables."""
    assert ai_conversation.api_key == "test_api_key"


def test_create_message(ai_conversation):
    """Test creating a message dictionary."""
    message = ai_conversation._create_message("user", "Hello")
    assert message == {"role": "user", "content": "Hello"}


def test_send_request_success(ai_conversation, mock_session_post):
    """Test successful API request."""
    messages = [{"role": "user", "content": "Hello"}]
    response = ai_conversation._send_request(messages)
    
    # Verify the response
    assert response == "Test response"
    
    # Verify the request was made correctly
    mock_session_post.assert_called_once()
    args, kwargs = mock_session_post.call_args
    assert args[0] == AiConversation.API_URL
    assert kwargs["headers"]["Authorization"] == f"Bearer {ai_conversation.api_key}"
    assert kwargs["json"]["model"] == AiConversation.MODEL
    assert kwargs["json"]["messages"] == messages


def test_process_command_exit(ai_conversation):
    """Test processing exit command."""
    result = ai_conversation._process_command("/exit")
    assert result is False


def test_process_command_clear(ai_conversation):
    """Test processing clear command."""
    # Add a message to conversation history
    ai_conversation.messages.append({"role": "user", "content": "test"})
    
    # Process clear command
    result = ai_conversation._process_command("/clear")
    
    # Verify result and that history was cleared except system message
    assert result is True
    assert len(ai_conversation.messages) == 1
    assert ai_conversation.messages[0]["role"] == "system"


def test_send_request_error(ai_conversation):
    """Test API request with error."""
    with patch.object(ai_conversation.session, 'post') as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        messages = [{"role": "user", "content": "Hello"}]
        response = ai_conversation._send_request(messages)
        
        assert response is None
