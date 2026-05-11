# AWS AgentCore + LangGraph Project

This project demonstrates how to build and deploy AI agents using AWS Bedrock AgentCore, LangGraph, and Groq LLMs.

## Features
- AI Agent workflows using LangGraph
- Memory-enabled agents
- FAQ Retrieval System
- CSV-based knowledge retrieval
- AWS Bedrock AgentCore integration
- Groq LLM support
- Vector Search using FAISS
- HuggingFace Embeddings

## Tech Stack
- Python
- LangGraph
- AWS Bedrock AgentCore
- Groq API
- FAISS
- HuggingFace Embeddings
- LangChain

## Project Structure
- `00_langgraph_agent.py` → Basic LangGraph agent
- `01_agentcore_runtime.py` → AgentCore runtime integration
- `02_agentcore_memory.py` → Memory-enabled AI agent

## How to Run

```bash
uv sync
python 00_langgraph_agent.py
