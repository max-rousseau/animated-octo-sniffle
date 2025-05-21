"""
Tests for the OpenAI CLI application.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import openai_cli


@pytest.fixture
def mock_env_api_key():
    """Fixture to mock environment variable for API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def mock_requests_post():
    """Fixture to mock requests.post."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        yield mock_post


def test_load_api_key_from_env(mock_env_api_key):
    """Test loading API key from environment variables."""
    api_key = openai_cli.load_api_key()
    assert api_key == "test_api_key"


def test_create_message():
    """Test creating a message dictionary."""
    message = openai_cli.create_message("user", "Hello")
    assert message == {"role": "user", "content": "Hello"}


def test_send_request_success(mock_env_api_key, mock_requests_post):
    """Test successful API request."""
    messages = [{"role": "user", "content": "Hello"}]
    response = openai_cli.send_request("test_api_key", messages)
    
    # Verify the response
    assert response == "Test response"
    
    # Verify the request was made correctly
    mock_requests_post.assert_called_once()
    args, kwargs = mock_requests_post.call_args
    assert args[0] == openai_cli.API_URL
    assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"
    assert kwargs["json"]["model"] == openai_cli.MODEL
    assert kwargs["json"]["messages"] == messages


@patch("requests.post")
def test_send_request_error(mock_post):
    """Test API request with error."""
    mock_post.side_effect = requests.exceptions.RequestException("API Error")
    
    messages = [{"role": "user", "content": "Hello"}]
    response = openai_cli.send_request("test_api_key", messages)
    
    assert response is None
