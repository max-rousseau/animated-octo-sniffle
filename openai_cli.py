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


class AiConversation:
    """
    A class to manage conversations with the OpenAI API.
    
    This class handles loading API keys, maintaining conversation history,
    and sending requests to the OpenAI API.
    """
    
    # Constants
    MODEL = "gpt-4o"
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self) -> None:
        """Initialize the AiConversation with API key and session."""
        self.api_key = self._load_api_key()
        self.session = requests.Session()
        
        # Initialize conversation with a system message
        self.messages = [
            self._create_message(
                "system", 
                "You are a helpful assistant. Provide concise and accurate responses."
            )
        ]
    
    def _load_api_key(self) -> str:
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
    
    def _create_message(self, role: str, content: str) -> Dict[str, str]:
        """
        Create a message dictionary for the OpenAI API.

        Args:
            role (str): The role of the message sender (system, user, or assistant).
            content (str): The content of the message.

        Returns:
            Dict[str, str]: A dictionary containing the role and content.
        """
        return {"role": role, "content": content}
    
    def _send_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Send a request to the OpenAI API using the session.

        Args:
            messages (List[Dict[str, str]]): The list of messages to send.

        Returns:
            Optional[str]: The response from the API, or None if there was an error.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        try:
            response = self.session.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error making request to OpenAI API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
    
    def _process_command(self, command: str) -> bool:
        """
        Process a command starting with a slash.
        
        Args:
            command (str): The command to process.
            
        Returns:
            bool: True if the conversation should continue, False if it should end.
        """
        cmd = command[1:].lower()  # Remove the slash and convert to lowercase
        
        if cmd in ["exit", "quit"]:
            print("Goodbye!")
            return False
        
        if cmd == "clear":
            self.messages = [self.messages[0]]  # Keep only the system message
            print("Conversation cleared.")
            return True
        
        print(f"Unknown command: {command}")
        return True
    
    def chat(self) -> None:
        """
        Start the conversation loop with the OpenAI API.
        
        This method handles user input, sends requests to the API,
        and displays responses.
        """
        print("Welcome to the OpenAI GPT-4o CLI!")
        print("Commands:")
        print("  /exit or /quit - End the conversation")
        print("  /clear - Start a new conversation")
        print("  (Press Ctrl+C at any time to exit)")
        print()
        
        try:
            while True:
                # Get user input
                user_input = input("You: ")
                
                # Check if it's a command (starts with /)
                if user_input.startswith('/'):
                    if not self._process_command(user_input):
                        break
                    continue
                
                # Add user message to conversation
                self.messages.append(self._create_message("user", user_input))
                
                # Send request to OpenAI API
                response = self._send_request(self.messages)
                
                if response:
                    print("\nAssistant:", response)
                    print()
                    
                    # Add assistant response to conversation history
                    self.messages.append(self._create_message("assistant", response))
                else:
                    print("Failed to get a response. Please try again.")
                    # Remove the last user message since we didn't get a response
                    self.messages.pop()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")


def main() -> None:
    """
    Main function to run the OpenAI CLI.
    
    This function creates an AiConversation instance and starts the chat.
    """
    conversation = AiConversation()
    conversation.chat()


if __name__ == "__main__":
    main()
