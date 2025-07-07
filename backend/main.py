#!/usr/bin/env python3
"""
FastAPI Backend for SQL Agent Chat
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys

# Add the parent directory to the path to import agent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Change to parent directory to access .env and parquet file
os.chdir(parent_dir)

from agent import Agent

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Asset Management Assistant API",
    description="Backend API for the Real Estate Asset Management Assistant",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
try:
    print("🔄 Initializing Real Estate Asset Management Assistant...")
    agent = Agent()
    print("✅ Real Estate Asset Management Assistant initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    agent = None

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    query_classification: Optional[str] = None
    sql_query: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent_ready=agent is not None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint for real estate asset management queries"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Initialize state for the agent
        from agent import AgentState
        
        initial_state = AgentState(
            user_query=chat_message.message,
            iteration_count=0,
            max_iterations=agent.max_iterations
        )
        
        # Run the agent graph
        config = {"configurable": {"thread_id": chat_message.user_id}}
        final_state = agent.graph.invoke(initial_state, config)
        
        # Extract information from the final state
        response_text = final_state.get("final_answer", "No response generated")
        query_classification = final_state.get("query_classification", None)
        sql_query = final_state.get("sql_query", None)
        
        return ChatResponse(
            response=response_text,
            query_classification=query_classification,
            sql_query=sql_query,
            error=None
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            error=str(e)
        )

@app.get("/agent/info")
async def get_agent_info():
    """Get information about the Real Estate Asset Management Assistant and available property data"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Get database info
        with agent.db.engine.connect() as conn:
            result = conn.execute(agent.db.engine.execute("SELECT COUNT(*) FROM cortex"))
            row_count = result.fetchone()[0]
        
        return {
            "database_info": {
                "table_name": "cortex",
                "row_count": row_count,
                "columns": agent.CORTEX_COLUMNS_DESCRIPTION
            },
            "agent_config": {
                "model": agent.llm_model,
                "temperature": agent.llm_temperature,
                "max_iterations": agent.max_iterations
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 