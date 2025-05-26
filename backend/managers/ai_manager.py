import dspy
import os
from typing import Dict, Any, Optional
import logging
from utils.logger import Logger

logger = Logger("ai_manager", see_time=True, console_log=False)

class AIManager:
    """
    Manages AI model configuration and initialization for the auto-analyst system.
    Handles different providers (OpenAI, Anthropic, etc.) and model settings.
    """
    
    def __init__(self):
        self.current_provider = None
        self.current_model = None
        self.lm = None
        self.anthropic_client = None
        self.initialized = False
        
    def configure_model(self, provider: str, model: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Configure the AI model for the system.
        
        Args:
            provider: AI provider (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-sonnet-4-20250514')
            api_key: Optional API key (if not set in environment)
            
        Returns:
            Configuration result
        """
        try:
            # Set API key if provided
            if api_key:
                if provider.lower() == 'openai':
                    os.environ['OPENAI_API_KEY'] = api_key
                elif provider.lower() == 'anthropic':
                    os.environ['ANTHROPIC_API_KEY'] = api_key
                # Add more providers as needed
            
            # Configure model based on provider
            if provider.lower() == 'openai':
                self.lm = dspy.OpenAI(model=model)
                dspy.configure(lm=self.lm)
            elif provider.lower() == 'anthropic':
                # Use Anthropic directly with modern Messages API
                try:
                    import anthropic
                    api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
                    if not api_key:
                        raise ValueError("Anthropic API key not found")
                    
                    self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                    # Test the connection with the modern Messages API
                    test_response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    logger.log_message("Anthropic client configured successfully with modern API", level=logging.INFO)
                except Exception as e:
                    raise ValueError(f"Failed to configure Anthropic: {str(e)}")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            self.current_provider = provider
            self.current_model = model
            self.initialized = True
            
            logger.log_message(f"Successfully configured {provider} {model}", level=logging.INFO)
            
            return {
                "status": "success",
                "provider": provider,
                "model": model,
                "message": f"Successfully configured {provider} {model}"
            }
            
        except Exception as e:
            logger.log_message(f"Error configuring model: {str(e)}", level=logging.ERROR)
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to configure {provider} {model}"
            }
    
    def generate_response(self, messages: list, **kwargs) -> str:
        """
        Generate a response using the configured model.
        
        Args:
            messages: List of messages in OpenAI format
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            # Add system prompt for balanced conversation and code generation
            system_prompt = """You are Francisco's friendly AI assistant specialized in data science and Python programming. You are warm, personable, and remember details about your conversations.

PERSONALITY TRAITS:
- Always greet users warmly and remember their names when they introduce themselves
- Be enthusiastic about data science and genuinely interested in helping
- Remember previous conversations and reference them when appropriate
- Use a conversational, friendly tone while being professional
- Show genuine interest in the user's projects and goals
- Acknowledge when users share personal information (name, interests, projects)

CONVERSATION GUIDELINES:
1. ALWAYS remember and use the user's name when they've introduced themselves
2. Be conversational and friendly for ALL interactions
3. Remember context from the entire conversation history
4. For greetings and personal questions, respond warmly and personally
5. Ask follow-up questions to show genuine interest
6. Reference previous parts of the conversation when relevant

WHEN TO GENERATE CODE:
- User asks for graphs, charts, or visualizations
- User requests data analysis or statistics  
- User mentions matplotlib, pandas, numpy, seaborn, plotly
- User asks to "create", "generate", "show", "plot", "analyze" data
- User uploads data files and asks for analysis
- User asks for EDA (Exploratory Data Analysis)

WHEN NOT TO GENERATE CODE:
- Simple greetings ("Hello", "Hi", "How are you?") - respond warmly instead
- Personal introductions ("My name is...", "I am...") - acknowledge and remember
- General questions about topics - provide helpful explanations
- Asking for explanations or definitions - give clear, friendly explanations
- Casual conversation - engage naturally

CRITICAL CODE FORMATTING RULES (when generating code):
1. ALWAYS write Python code in properly formatted markdown code blocks with ```python
2. Each statement MUST be on a separate line with proper indentation
3. NEVER concatenate multiple statements on the same line
4. Use proper line breaks between logical sections
5. Add clear comments to explain what each section does
6. Ensure proper spacing around operators (=, +, -, etc.)
7. Use consistent indentation (4 spaces)
8. Each print() statement should be on its own line
9. Each variable assignment should be on its own line
10. Each function call should be on its own line
11. NEVER break strings or file paths across lines
12. Use proper quotes for strings (no broken quotes)

INSIGHTS AND ANALYSIS REQUIREMENTS:
When performing data analysis, always provide:
1. Clear, actionable insights about the data
2. Specific observations about patterns, trends, and anomalies
3. Recommendations for next steps
4. Interpretation of statistical findings
5. Business or practical implications of the findings
6. Suggestions for further analysis or data collection

EXAMPLE OF CORRECT FORMATTING:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
df = pd.read_csv('uploads/filename.csv')

# Display basic information
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Check for missing values
print("\\n=== MISSING VALUES ===")
missing_values = df.isnull().sum()
print(missing_values)

# Create visualization
plt.figure(figsize=(10, 6))
plt.hist(df['column'], bins=30, alpha=0.7)
plt.title('Distribution of Column')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
```

EXAMPLE OF INCORRECT FORMATTING (DO NOT DO THIS):
```python
import pandas as pd;import matplotlib.pyplot as plt;print("Hello")df=pd.read_csv('uploads/file
name.csv');print(df.shape)plt.hist(df['col'])plt.show()
```

AVAILABLE LIBRARIES (when coding):
- numpy (as np) - Numerical computing
- pandas (as pd) - Data manipulation and analysis  
- matplotlib.pyplot (as plt) - Static plotting
- seaborn (as sns) - Statistical visualization

IMPORTANT MEMORY RULES:
- If a user says "My name is [Name]", always remember and use their name in future responses
- If a user asks "Do you remember my name?", always confirm and use their name
- Reference previous topics discussed in the conversation
- Show continuity and memory across the conversation

EXAMPLE RESPONSES:
User: "Hi! My name is Francisco, and I love data science. How are you?"
Good Response: "Hi Francisco! It's wonderful to meet you! I'm doing great, thank you for asking. It's exciting to hear that you love data science - that's fantastic! What aspects of data science do you enjoy the most? Are you working on any interesting projects or datasets right now? I'd love to hear more about your data science journey!"

User: "Do you remember my name?"
Good Response: "Of course, Francisco! I absolutely remember you. How could I forget someone who loves data science as much as you do? Is there anything specific you'd like to work on today?"

IMPORTANT: 
- Always be warm, friendly, and remember personal details
- When generating code, ensure perfect formatting with no concatenated lines
- Provide rich, detailed insights when analyzing data
- The code will be automatically executed, so make sure it's syntactically perfect"""

            # Prepare messages with system prompt
            formatted_messages = []
            
            if self.current_provider == 'anthropic':
                # For Anthropic, add system prompt as first user message if no system message exists
                has_system = any(msg.get('role') == 'system' for msg in messages)
                if not has_system:
                    # Include conversation history for context
                    conversation_context = ""
                    if len(messages) > 1:
                        conversation_context = "\n\nCONVERSATION HISTORY:\n"
                        for i, msg in enumerate(messages[:-1]):
                            if msg['role'] in ['user', 'assistant']:
                                conversation_context += f"{msg['role'].title()}: {msg['content'][:200]}...\n"
                    
                    formatted_messages.append({
                        "role": "user", 
                        "content": f"SYSTEM: {system_prompt}{conversation_context}\n\nCurrent user message: {messages[-1]['content']}"
                    })
                else:
                    formatted_messages = messages
            else:
                # For other providers, add system message normally and include all conversation history
                formatted_messages = [{"role": "system", "content": system_prompt}] + messages
            
            if self.current_provider == 'openai' and self.lm:
                # Use DSPy for OpenAI
                if len(formatted_messages) == 1:
                    response = self.lm(formatted_messages[0]['content'])
                    return response[0] if isinstance(response, list) else str(response)
                else:
                    # Handle conversation format with full history
                    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in formatted_messages])
                    response = self.lm(conversation_text)
                    return response[0] if isinstance(response, list) else str(response)
                    
            elif self.current_provider == 'anthropic' and self.anthropic_client:
                # Use Anthropic directly with modern Messages API
                response = self.anthropic_client.messages.create(
                    model=self.current_model,
                    max_tokens=kwargs.get('max_tokens', 4000),
                    messages=formatted_messages
                )
                return response.content[0].text
            else:
                raise ValueError("No model configured")
                
        except Exception as e:
            logger.log_message(f"Error generating response: {str(e)}", level=logging.ERROR)
            raise
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Current configuration details
        """
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "initialized": self.initialized
        }
    
    def is_configured(self) -> bool:
        """
        Check if AI model is properly configured.
        
        Returns:
            True if configured, False otherwise
        """
        return self.initialized and (self.lm is not None or self.anthropic_client is not None)
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get list of available models for each provider.
        
        Returns:
            Dictionary with available models by provider
        """
        return {
            "openai": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            "anthropic": [
                # Claude 4 models (latest)
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                # Claude 3.7 models
                "claude-3-7-sonnet-20250219",
                # Claude 3.5 models
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                # Claude 3 models (legacy)
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        } 