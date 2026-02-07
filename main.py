"""
Build Agent - Main entry point
"""

import os
import ssl

# Use system trust store (macOS Keychain) for SSL certificates
import truststore
truststore.inject_into_ssl()

import certifi
from dotenv import load_dotenv

# Fallback: set certificate bundle paths
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

def main():
   # Load environment variables
   load_dotenv()

   # Create LLM using HuggingFace
   endpoint = HuggingFaceEndpoint(
       repo_id="HuggingFaceH4/zephyr-7b-beta",
       task="text-generation",
       max_new_tokens=1024,
       temperature=0.7,
   )
   llm = ChatHuggingFace(llm=endpoint)

   # Run the query
   query = "Find the best places to visit in Doddaballapura"
   print(f"Query: {query}")
  
   result = llm.invoke([HumanMessage(content=query)])
   print(f"\nResult: {result.content}")

if __name__ == "__main__":
   main()