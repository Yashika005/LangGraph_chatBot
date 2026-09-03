import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment}")
print(f"API Version: {api_version}")
print(f"API Key: {api_key[: 10]}..." if api_key else "API Key:  None")

try:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    
    print("\n Client created successfully")
    print("Sending test request...")
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Say hello in one word",
            }
        ],
        model=deployment,
        max_tokens=10,
    )
    
    print(f"SUCCESS: {response.choices[0]. message.content}")
    
except Exception as e:
    print(f" ERROR: {str(e)}")
    import traceback
    traceback.print_exc()