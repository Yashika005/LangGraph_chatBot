import os
from openai import AzureOpenAI

def get_llm():
    """Initialize and return Azure OpenAI client"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not all([endpoint, api_key, api_version]):
        raise ValueError("Azure OpenAI credentials not found in environment variables")
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    return client

def generate_response(llm_client, prompt):
    """Helper function to generate response using Azure OpenAI"""
    try: 
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        
        # Check if prompt contains conversation history markers
        system_message = "You are a helpful assistant that answers questions based on provided context. Please answer all greetings politely and professionally."
        
        # If prompt has memory context, parse it for structured messages
        if "Previous conversation:" in prompt:
            # Extract the memory part and the rest
            parts = prompt.split("Context:", 1)
            if len(parts) == 2:
                memory_part, rest = parts
                memory_part = memory_part.replace("Previous conversation:\n", "").strip()
                
                # Parse memory messages
                messages = [{"role": "system", "content": system_message}]
                
                # Add memory messages
                lines = memory_part.split("\n")
                for line in lines:
                    if ":" in line:
                        role_content = line.split(":", 1)
                        if len(role_content) == 2:
                            role, content = role_content
                            role = role.strip().lower()
                            if role in ["user", "assistant"]:
                                messages.append({"role": role, "content": content.strip()})
                
                # Add current context and question
                context_question_parts = rest.split("Question:", 1)
                if len(context_question_parts) == 2:
                    context, question = context_question_parts
                    messages.append({"role": "user", "content": f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}"})
            else:
                # Fallback to original method
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        
        response = llm_client.chat.completions.create(
            messages=messages,
            model=deployment,
            max_completion_tokens=13107,
            temperature=0.7,  # Lower temperature for more focused answers
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        
        return response.choices[0].message.content
        
    except Exception as e: 
        error_msg = str(e)
        raise Exception(f"Error generating response from Azure OpenAI: {error_msg}")