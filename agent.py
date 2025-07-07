#!/usr/bin/env python3
"""
Real Estate Asset Management Assistant

This script implements an LLM-powered assistant for real estate asset managers using LangGraph that can:
- Analyze property portfolio data and financial performance
- Convert natural language queries to SQL for real estate databases
- Execute SQL queries on property management databases
- Handle errors and regenerate queries for complex real estate analytics
- Provide human-readable insights for property managers and investors
- Classify queries as relevant (property data), irrelevant (missing data), or off-topic
"""

import os
import duckdb
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

# State management
@dataclass
class AgentState:
    """State object for the agent"""
    user_query: str = ""
    query_classification: str = ""  # 'relevant_query', 'irrelevant_query', or 'other_subject'
    sql_query: str = ""
    sql_results: Optional[str] = None
    error_message: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 3
    final_answer: str = ""
    messages: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []

class NodeType(Enum):
    """Enum for different node types"""
    CHECK_RELEVANCE = "check_relevance"
    CONVERT_TO_SQL = "convert_to_sql"
    EXECUTE_SQL = "execute_sql"
    REGENERATE_QUERY = "regenerate_query"
    GENERATE_HUMAN_READABLE = "generate_human_readable_answer"
    GENERATE_FUNNY_RESPONSE = "generate_funny_response"
    HANDLE_IRRELEVANT_QUERY = "handle_irrelevant_query"
    END_MAX_ITERATIONS = "end_max_iterations"

class Agent:
    """LangGraph-powered Real Estate Asset Management Assistant"""
    
    # Column descriptions for the cortex table
    CORTEX_COLUMNS_DESCRIPTION = """
    - entity_name: Name of the entity responsible for managing properties
    - property_name: Name of the property (can be None)
    - tenant_name: Name of the tenant (can be None)
    - ledger_type: Type of ledger entry (e.g., expenses, revenue)
    - ledger_group: Group classification
    - ledger_category: Category classification
    - ledger_code: Code for the ledger entry
    - ledger_description: Description of the ledger entry
    - month: Month in format YYYY-M## (e.g., 2025-M01)
    - quarter: Quarter in format YYYY-Q# (e.g., 2025-Q1)
    - year: Year (e.g., 2025)
    - profit: Numeric profit/loss value
    """
    
    def __init__(self, parquet_file: str = None, openai_api_key: str = None):
        """Initialize the Real Estate Asset Management Assistant"""
        # Load environment variables
        load_dotenv()
        # Get configuration from environment variables with fallbacks
        self.parquet_file = parquet_file or os.getenv("PARQUET_FILE", "cortex.parquet")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass it directly.")
        
        # Initialize LLMs with different temperatures for different tasks
        self.llm_creative = ChatOpenAI(
            model=self.llm_model,
            temperature=float(os.getenv("LLM_TEMPERATURE_CREATIVE", "0.7")),  # More creative for funny responses
            api_key=self.openai_api_key
        )
        
        self.llm_analytical = ChatOpenAI(
            model=self.llm_model,
            temperature=float(os.getenv("LLM_TEMPERATURE_ANALYTICAL", "0.1")),  # More precise for SQL generation
            api_key=self.openai_api_key
        )
        
        self.llm_classifier = ChatOpenAI(
            model=self.llm_model,
            temperature=float(os.getenv("LLM_TEMPERATURE_CLASSIFIER", "0.0")),  # Most precise for classification
            api_key=self.openai_api_key
        )
        
        self.llm_summarizer = ChatOpenAI(
            model=self.llm_model,
            temperature=float(os.getenv("LLM_TEMPERATURE_SUMMARIZER", "0.3")),  # Balanced for human-readable summaries
            api_key=self.openai_api_key
        )
        
        # Default LLM (for backward compatibility)
        self.llm = self.llm_analytical
        
        self.db = self._setup_database()
        self.sql_tool = self._setup_sql_tool()
        self.graph = self._build_graph()
    
    def _setup_database(self) -> SQLDatabase:
        """Set up DuckDB database with parquet file"""
        try:
            # Load parquet file
            df = pd.read_parquet(self.parquet_file)
            print(f"✅ Loaded parquet file: {self.parquet_file}")
            print(f"📊 DataFrame shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Create SQLAlchemy engine with DuckDB
            engine = create_engine("duckdb:///:memory:")
            
            # Load data into DuckDB using pandas
            df.to_sql("cortex", engine, if_exists="replace", index=False)
            
            # Create SQLDatabase instance
            db = SQLDatabase(engine=engine)
            
            # Test the connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM cortex"))
                count = result.fetchone()[0]
                print(f"✅ Database connection test: {count} rows in cortex table")
            
            return db
            
        except Exception as e:
            print(f"❌ Error setting up database: {e}")
            raise
    
    def _setup_sql_tool(self) -> QuerySQLDataBaseTool:
        """Set up SQL execution tool"""
        return QuerySQLDataBaseTool(db=self.db)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node(NodeType.CHECK_RELEVANCE.value, self._check_relevance_node)
        graph.add_node(NodeType.CONVERT_TO_SQL.value, self._convert_to_sql_node)
        graph.add_node(NodeType.EXECUTE_SQL.value, self._execute_sql_node)
        graph.add_node(NodeType.REGENERATE_QUERY.value, self._regenerate_query_node)
        graph.add_node(NodeType.GENERATE_HUMAN_READABLE.value, self._generate_human_readable_node)
        graph.add_node(NodeType.GENERATE_FUNNY_RESPONSE.value, self._generate_funny_response_node)
        graph.add_node(NodeType.HANDLE_IRRELEVANT_QUERY.value, self._handle_irrelevant_query_node)
        graph.add_node(NodeType.END_MAX_ITERATIONS.value, self._end_max_iterations_node)
        
        # Add edges
        graph.add_edge(START, NodeType.CHECK_RELEVANCE.value)
        
        # Conditional edges from check_relevance
        graph.add_conditional_edges(
            NodeType.CHECK_RELEVANCE.value,
            self._route_after_relevance_check,
            {
                "relevant_query": NodeType.CONVERT_TO_SQL.value,
                "irrelevant_query": NodeType.HANDLE_IRRELEVANT_QUERY.value,
                "other_subject": NodeType.GENERATE_FUNNY_RESPONSE.value
            }
        )
        
        # From convert_to_sql to execute_sql
        graph.add_edge(NodeType.CONVERT_TO_SQL.value, NodeType.EXECUTE_SQL.value)
        
        # Conditional edges from execute_sql
        graph.add_conditional_edges(
            NodeType.EXECUTE_SQL.value,
            self._route_after_sql_execution,
            {
                "success": NodeType.GENERATE_HUMAN_READABLE.value,
                "error_retry": NodeType.REGENERATE_QUERY.value,
                "error_max_attempts": NodeType.END_MAX_ITERATIONS.value
            }
        )
        
        # From regenerate_query back to convert_to_sql (for retry loop)
        graph.add_edge(NodeType.REGENERATE_QUERY.value, NodeType.CONVERT_TO_SQL.value)
        
        # End nodes
        graph.add_edge(NodeType.GENERATE_HUMAN_READABLE.value, END)
        graph.add_edge(NodeType.GENERATE_FUNNY_RESPONSE.value, END)
        graph.add_edge(NodeType.HANDLE_IRRELEVANT_QUERY.value, END)
        graph.add_edge(NodeType.END_MAX_ITERATIONS.value, END)
        
        return graph.compile(checkpointer=MemorySaver())
    
    def _check_relevance_node(self, state: AgentState) -> AgentState:
        """Check if the user query is relevant to our database and subject"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an assistant to a real-estate asset-management team.
            Your task is to decide whether an incoming user question can be answered solely with the data that exists in our SQL database (schema below).
            Return exactly one label — relevant_query, irrelevant_query, or other_subject — and nothing else.

            Database schema (table "cortex"):
            {self.CORTEX_COLUMNS_DESCRIPTION}

            How to choose the label:
                1.	relevant_query
            • Use when the question can be fully answered with the columns above.
            • Typical themes: entity / property / tenant info, ledger details (type, group, category, code, description, profit), time filters (month, quarter, year), aggregations or trends.
            • Example: "What was the total profit for each property in 2024?"
            • Reasoning: Only requires property_name, year, and profit, all present in the schema, so it is directly answerable.
                2.	irrelevant_query
            • Use when the question is data-related but cannot be answered with the columns above.
            • Typical themes: attributes we do not store (addresses, square footage, valuations, occupancy %), data from other sources or databases.
            • Example: "What is the street address of each property managed by PropCo 1?"
            • Reasoning: The schema has no "address" column; the data are missing, so the query cannot be fulfilled.
                3.	other_subject
            • Use when the input is not a database or SQL question.
            • Typical themes: general conversation, weather, sports, jokes, personal advice, programming tasks unrelated to our data.
            • Example: "Will it rain tomorrow in Tel Aviv?"
            • Reasoning: This is a weather question, unrelated to the database or any SQL query.

            Output format:
            Return only one label — relevant_query, irrelevant_query, or other_subject.
            """),
            HumanMessage(content=f"User query: {state.user_query}")
        ])
        
        chain = prompt | self.llm_classifier | StrOutputParser()
        result = chain.invoke({})
        
        # Store the classification result
        state.query_classification = result.lower().strip()
        return state
    
    def _convert_to_sql_node(self, state: AgentState) -> AgentState:
        """Convert natural language to SQL"""
        # Get database schema
        schema_info = self.db.get_table_info()
        
        # Build the prompt content conditionally
        prompt_content = f"""You are an expert SQL query generator. Convert the user's natural language request into a valid SQL query.
            
            Database Schema:
            {schema_info}
            
            {self.CORTEX_COLUMNS_DESCRIPTION}
            
            Rules:
            1. Only use tables and columns that exist in the schema
            2. Generate syntactically correct SQL for DuckDB
            3. Use appropriate WHERE clauses, JOINs, and aggregations
            4. Return only the SQL query without explanations
            5. Use LIMIT clauses when appropriate to avoid overwhelming results
            6. For "top" queries, use ORDER BY with LIMIT
            7. Handle NULL values appropriately"""
        
        # Add previous attempt context only if this is not the first attempt
        if state.sql_query:
            prompt_content += f"""
            
            Previous attempt context:
            Previous SQL: {state.sql_query}
            {f"Previous error: {state.error_message}" if state.error_message else ""}"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=prompt_content),
            HumanMessage(content=f"Convert this request to SQL: {state.user_query}")
        ])
        
        chain = prompt | self.llm_analytical | StrOutputParser()
        state.sql_query = chain.invoke({}).strip()
        
        return state
    
    def _execute_sql_node(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        try:
            # Clean the SQL query - remove markdown formatting
            sql_query = state.sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]  # Remove ```sql
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]   # Remove ``` if no language specified
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]  # Remove closing ```
            
            sql_query = sql_query.strip()
            print(f"🔍 Executing SQL: {sql_query}")
            
            # Execute the SQL query
            result = self.sql_tool.run(sql_query)
            
            print(f"📊 SQL Result: {result}")
            
            if result and result.strip():
                state.sql_results = result
                state.error_message = None
            else:
                state.sql_results = None
                state.error_message = "No rows returned"
            
        except Exception as e:
            print(f"❌ SQL Execution Error: {e}")
            state.sql_results = None
            state.error_message = str(e)
        
        return state
    
    def _regenerate_query_node(self, state: AgentState) -> AgentState:
        """Regenerate SQL query based on error"""
        state.iteration_count += 1
        
        schema_info = self.db.get_table_info()
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert SQL debugger. The previous SQL query failed or returned no results.
            
            Database Schema:
            {schema_info}
            
            Original user request: {state.user_query}
            Previous SQL query: {state.sql_query}
            Error/Issue: {state.error_message}
            
            Please generate a corrected SQL query that addresses the error or issue.
            Consider:
            1. Syntax errors and fix them
            2. Table/column name mismatches
            3. Logic errors in WHERE clauses
            4. Missing JOINs or incorrect JOIN conditions
            5. If no rows returned, try broader search criteria
            
            Return only the corrected SQL query."""),
            HumanMessage(content="Please provide the corrected SQL query.")
        ])
        
        chain = prompt | self.llm_analytical | StrOutputParser()
        state.sql_query = chain.invoke({}).strip()
        
        return state
    
    def _generate_human_readable_node(self, state: AgentState) -> AgentState:
        """Generate human-readable answer from SQL results"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a real estate asset manager assistant. Convert the raw SQL results into a clear, concise, and human-readable summary.
            
            Guidelines:
            1. Summarize the key findings
            2. Use natural language
            3. Include relevant numbers and insights
            4. Make it easy to understand for non-technical users
            5. Be concise but informative"""),
            HumanMessage(content=f"""
            Original question: {state.user_query}
            SQL query used: {state.sql_query}
            SQL results: {state.sql_results}
            
            Please provide a human-readable summary of these results.
            """)
        ])
        
        chain = prompt | self.llm_summarizer | StrOutputParser()
        state.final_answer = chain.invoke({})
        
        return state
    
    def _generate_funny_response_node(self, state: AgentState) -> AgentState:
        """Generate funny response for non-real estate queries"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a witty real estate asset manager assistant. The user asked something completely unrelated to property analytics.
            
            Respond with a short, humorous message that:
            1. Acknowledges their off-topic question
            2. Explains you only do property analytics
            3. Includes real estate humor
            4. Suggests they ask about properties instead
            
            Keep it brief and entertaining!"""),
            HumanMessage(content=f"User asked: {state.user_query}")
        ])
        
        chain = prompt | self.llm_creative | StrOutputParser()
        state.final_answer = chain.invoke({})
        
        return state
    
    def _handle_irrelevant_query_node(self, state: AgentState) -> AgentState:
        """Handle queries that are data-related but cannot be answered with our database"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a real estate asset manager assistant that needs to explain why a property-related query cannot be answered with the available data.
            
            Available data in our database:
            {self.CORTEX_COLUMNS_DESCRIPTION}
            
            The user asked a property-related question that cannot be answered with our database.
            Provide a clear, helpful response that:
            1. Acknowledges their question is property-related
            2. Explains why it cannot be answered with our current property database
            3. Lists what property data we do have available
            4. Suggests alternative questions they could ask about our property portfolio
            5. Be polite and helpful, not dismissive
            
            Focus on explaining the limitations of our property data while being encouraging about what we can help with."""),
            HumanMessage(content=f"""
            User's question: {state.user_query}
            
            Please explain why this cannot be answered with our database and suggest alternatives.
            """)
        ])
        
        chain = prompt | self.llm_analytical | StrOutputParser()
        state.final_answer = chain.invoke({})
        
        return state
    
    def _end_max_iterations_node(self, state: AgentState) -> AgentState:
        """Handle max iterations reached"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a real estate asset manager assistant that needs to apologize for not being able to complete a property analytics task.
            
            After multiple attempts, you couldn't generate a working SQL query for the user's property-related request.
            Provide a polite, apologetic response that:
            1. Acknowledges the difficulty with the property analytics
            2. Explains you tried multiple times to analyze the property data
            3. Suggests the user might rephrase their property question
            4. Offers to help with a different property analytics approach
            
            Be understanding and helpful."""),
            HumanMessage(content=f"""
            Original question: {state.user_query}
            Number of attempts: {state.iteration_count}
            Last error: {state.error_message}
            
            Please provide an apologetic response.
            """)
        ])
        
        chain = prompt | self.llm_analytical | StrOutputParser()
        state.final_answer = chain.invoke({})
        
        return state
    
    def _route_after_relevance_check(self, state: AgentState) -> str:
        """Route based on relevance check"""
        return state.query_classification
    
    def _route_after_sql_execution(self, state: AgentState) -> str:
        """Route based on SQL execution results"""
        if state.sql_results is not None and state.error_message is None:
            return "success"
        elif state.iteration_count >= state.max_iterations:
            return "error_max_attempts"
        else:
            return "error_retry"
    
    def run(self, user_input: str) -> str:
        """Run the Real Estate Asset Management Assistant with user input"""
        # Initialize state
        initial_state = AgentState(
            user_query=user_input,
            iteration_count=0,
            max_iterations=self.max_iterations
        )
        
        # Run the graph
        config = {"configurable": {"thread_id": "1"}}
        final_state = self.graph.invoke(initial_state, config)
        
        # LangGraph returns a dictionary, not an AgentState object
        if isinstance(final_state, dict):
            return final_state.get("final_answer", "No response generated")
        else:
            return final_state.final_answer

    def _get_llm_with_temperature(self, base_llm: ChatOpenAI, temperature: float) -> ChatOpenAI:
        """Create a new LLM instance with a specific temperature"""
        return ChatOpenAI(
            model=base_llm.model,
            temperature=temperature,
            api_key=base_llm.api_key
        )
    
    def update_node_temperature(self, node_name: str, temperature: float):
        """Update the temperature for a specific node type"""
        if node_name == "classifier":
            self.llm_classifier = self._get_llm_with_temperature(self.llm_classifier, temperature)
        elif node_name == "sql_generator":
            self.llm_analytical = self._get_llm_with_temperature(self.llm_analytical, temperature)
        elif node_name == "summarizer":
            self.llm_summarizer = self._get_llm_with_temperature(self.llm_summarizer, temperature)
        elif node_name == "creative":
            self.llm_creative = self._get_llm_with_temperature(self.llm_creative, temperature)
        else:
            raise ValueError(f"Unknown node type: {node_name}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the agent
    try:
        agent = Agent()
        
        # Test cases
        test_queries = [
            "How many records are in the database?",
            "What's the weather like today?",  # Non-SQL query
            "Show me the top 5 entries by some numeric column",
            "What are the column names in the data?",
        ]
        
        print("🏢 Real Estate Asset Management Assistant initialized successfully!")
        """print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            print("-" * 30)
            
            try:
                response = agent.run(query)
                print(f"🎯 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 30)"""
        
        # Interactive mode
        print("\n🎮 Interactive Mode - Type 'quit' to exit")
        print("💡 Ask questions about your property portfolio!")
        print("=" * 50)
        
        while True:
            user_input = input("\n👤 Your property query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = agent.run(user_input)
                print(f"🏢 Assistant: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("💡 Make sure you have set your OPENAI_API_KEY in the .env file")
        print("💡 Install required packages: pip install -r requirements.txt")
        print("💡 Check that your .env file exists and contains the required variables") 