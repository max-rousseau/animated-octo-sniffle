#!/usr/bin/env python
"""
Command line interface for OpenAI GPT-4o model.

This program allows users to have a conversation with OpenAI's GPT-4o model
through a simple command line interface. It maintains conversation history
and sends it with each new request to provide context.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Constants
MODEL = "gpt-4o"
API_URL = "https://api.openai.com/v1/chat/completions"


def load_api_key() -> str:
    """
    Load the OpenAI API key from environment variables or .env file.

    Returns:
        str: The OpenAI API key.

    Raises:
        SystemExit: If the API key is not found.
    """
    # Try to load from .env file first
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please set the OPENAI_API_KEY environment variable or add it to a .env file.")
        sys.exit(1)
    
    return api_key


def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Create a message dictionary for the OpenAI API.

    Args:
        role (str): The role of the message sender (system, user, or assistant).
        content (str): The content of the message.

    Returns:
        Dict[str, str]: A dictionary containing the role and content.
    """
    return {"role": role, "content": content}


def send_request(api_key: str, messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Send a request to the OpenAI API.

    Args:
        api_key (str): The OpenAI API key.
        messages (List[Dict[str, str]]): The list of messages to send.

    Returns:
        Optional[str]: The response from the API, or None if there was an error.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error making request to OpenAI API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None


def main() -> None:
    """
    Main function to run the OpenAI CLI.
    
    This function initializes the conversation, loads the API key,
    and handles the conversation loop.
    """
    print("Welcome to the OpenAI GPT-4o CLI!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print()
    
    api_key = load_api_key()
    
    # Initialize conversation with a system message
    messages = [
        create_message(
            "system", 
            "You are a helpful assistant. Provide concise and accurate responses."
        )
    ]
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Check for clear command
        if user_input.lower() == "clear":
            messages = [messages[0]]  # Keep only the system message
            print("Conversation cleared.")
            continue
        
        # Add user message to conversation
        messages.append(create_message("user", user_input))
        
        # Send request to OpenAI API
        response = send_request(api_key, messages)
        
        if response:
            print("\nAssistant:", response)
            print()
            
            # Add assistant response to conversation history
            messages.append(create_message("assistant", response))
        else:
            print("Failed to get a response. Please try again.")
            # Remove the last user message since we didn't get a response
            messages.pop()


if __name__ == "__main__":
    main()
