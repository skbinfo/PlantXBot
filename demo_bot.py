from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import os
import math
import sqlite3
import pandas as pd
import uuid
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from retry import retry
from cachetools import TTLCache
import time
import yaml
import requests
import structlog
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from typing import List, Dict, Optional, Tuple, Set
import re
import random
# Vector DB imports (Optional if you don't use them, but kept for structure)
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize Flask application
app = Flask(__name__)
CORS(app)


# 1. Basic Settings
# Ensure you set 'GROQ_API_KEY' in your environment variables or hardcode here (not recommended for production)
API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_GROQ_API_KEY_HERE')
DB_PATH = os.getenv('DB_PATH', './database') # Folder containing your sqlite file
DB_FILE = 'my_database.db' # Your SQLite file name
DOWNLOAD_DIR = os.path.join(DB_PATH, 'public/downloads')
BASE_URL = 'http://localhost:5000/public/downloads' # URL where CSVs can be downloaded

# 2. Vector Database Settings (Optional)
# If you have RAG/Embeddings, point to them here.
CHROMA_PERSIST_DIR = os.path.join(DB_PATH, 'Chroma_db')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 3. Domain Context
# Describe what your database is about generally. This helps the AI understand the context.
DOMAIN_DESCRIPTION = """
You are an AI assistant for a [INSERT DOMAIN HERE, e.g., Sales Inventory, Hospital Records] system.
This database contains information about [INSERT ENTITIES, e.g., products, customers, transactions].
"""

# 4. Database Schema Definition
# REPLACE this structure with your actual database schematic.
# The 'semantic_type' helps the AI understand what the column represents (e.g., 'price', 'category', 'identifier').
USER_DEFINED_SCHEMA = {
    'my_database.db': {
        'tables': {
            'example_table': { 
                'columns': [
                    {'name': 'id', 'sqlite_type': 'INTEGER', 'semantic_type': 'primary_identifier', 'description': 'Unique ID.'},
                    {'name': 'name', 'sqlite_type': 'TEXT', 'semantic_type': 'name', 'description': 'Name of the item.'},
                    {'name': 'category', 'sqlite_type': 'TEXT', 'semantic_type': 'category', 'description': 'Category or classification.'},
                    {'name': 'value_column', 'sqlite_type': 'TEXT', 'semantic_type': 'measurement_value', 'description': 'A numerical value stored as text (needs CAST).'},
                    {'name': 'description', 'sqlite_type': 'TEXT', 'semantic_type': 'description', 'description': 'Detailed text description.'},
                    {'name': 'created_at', 'sqlite_type': 'TEXT', 'semantic_type': 'date', 'description': 'Record creation date.'}
                ],
                'description': 'A generic example table containing item records.',
                'primary_keys': ['id'],
                'common_joins': {}, 
                'notes': ["Numerical fields like value_column are stored as TEXT and need CASTing."]
            },
            # Add more tables here...
        }
    }
}

# 5. Database to Conceptual Name Mapping
# Map internal table names to "Nice Names" for the user.
DATABASE_MAPPING = {
    "example_table": "Main Inventory",
    # "users_table": "User Directory"
}

# Structured Logging Setup
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

CACHE = TTLCache(maxsize=1000, ttl=3600)
CONTEXT_CACHE = TTLCache(maxsize=100, ttl=3600)
DISPLAY_ROW_LIMIT = 10
MAX_ROWS_FOR_PANDAS_STATS = 50000
MAX_ROWS_FOR_LLM_SUMMARY = 1000

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "20 per minute"]
)

class SchemaManager:
    def __init__(self, db_path: str, user_schema: Dict):
        self.db_path = db_path
        self.schema = user_schema
        self.update_schema_from_db()

    def _enrich_schema_with_dynamic_info(self):
        """Checks actual DB types and updates schema if defined types match columns."""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for db_file, db_data in self.schema.items():
                for table_name, table_data in db_data['tables'].items():
                    try:
                        cursor.execute(f"PRAGMA table_info('{table_name}');")
                        pragma_info = {row[1]: row[2] if row[2] else 'TEXT' for row in cursor.fetchall()}
                        
                        for col_dict in table_data['columns']:
                            col_name = col_dict['name']
                            if col_name in pragma_info:
                                db_type = pragma_info[col_name].upper()
                                # Update type if it differs
                                if db_type != col_dict.get('sqlite_type', '').upper():
                                    col_dict['sqlite_type'] = db_type
                    except Exception as e:
                        logger.warning(f"Could not verify schema for table {table_name}: {e}")
            conn.close()
        except Exception as e:
            logger.error(f"Failed to enrich schema: {e}", exc_info=True)

    def update_schema_from_db(self):
        """
        Placeholder for logic that scans the DB for dynamic categories (like unique species).
        In this dummy code, we just trigger the type enrichment.
        """
        self._enrich_schema_with_dynamic_info()

    def get_schema_for_prompt(self) -> Dict:
        """Returns a simplified schema dictionary for the LLM prompt."""
        simplified = {}
        for db_name, db_data in self.schema.items():
            simplified[db_name] = {'tables': {}}
            for table_name, table_details in db_data['tables'].items():
                simplified[db_name]['tables'][table_name] = {
                    'columns': [{'name': c['name'], 'type': c.get('sqlite_type', 'TEXT'), 'desc': c.get('description', '')} for c in table_details['columns']],
                    'description': table_details['description'],
                    'notes': table_details.get('notes', [])
                }
        return simplified

    def get_tables(self) -> List[str]:
        # Assuming single DB file structure in user schema
        return list(self.schema[DB_FILE]['tables'].keys())

# Initialize Schema Manager
schema_manager = SchemaManager(os.path.join(DB_PATH, DB_FILE), USER_DEFINED_SCHEMA)

# Initialize Vector DB (Chroma) - Optional
logger.info("Initializing ChromaDB vector store...")
vector_db = None
try:
    if os.path.isdir(CHROMA_PERSIST_DIR):
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )
        logger.info(f"ChromaDB loaded. Collection count: {vector_db._collection.count()}")
    else:
        logger.warning(f"ChromaDB directory not found at {CHROMA_PERSIST_DIR}. RAG features disabled.")
except Exception as e:
    logger.error(f"Failed to load ChromaDB: {e}", exc_info=True)

def get_relevant_context_from_vectordb(query: str, k: int = 4) -> str:
    if vector_db is None:
        return "No knowledge base context available."
    try:
        relevant_docs = vector_db.similarity_search(query, k=k)
        if not relevant_docs:
            return "No specific context found in knowledge base."
        return "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return "Error retrieving context."

# --- Helper Functions (Sanitization, Stats, etc.) ---

def sanitize_query_input(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string")
    return query

def sanitize_sql(sql_query: str) -> str:
    sql_query = sql_query.strip()
    forbidden = ['DROP', 'DELETE FROM', 'TRUNCATE', 'INSERT INTO', 'UPDATE ', 'ALTER']
    if any(token in sql_query.upper() for token in forbidden) and not sql_query.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Potentially malicious SQL: {sql_query}")
    if sql_query.endswith(';'):
        sql_query = sql_query[:-1]
    return sql_query

def clean_nan_and_inf(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_and_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_and_inf(elem) for elem in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def execute_sql_query(sql_query: str, table_context: str) -> Tuple[List[Dict] | str, str]:
    try:
        sql_to_execute = sanitize_sql(sql_query)
        db_path_full = os.path.join(DB_PATH, DB_FILE)
        if not os.path.exists(db_path_full):
            return f"Error: Database file not found.", table_context
        
        conn = sqlite3.connect(f"file:{db_path_full}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_to_execute)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return [], table_context
        return [dict(row) for row in results], table_context
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return f"Error executing SQL: {str(e)}", table_context

def generate_csv(results: List[Dict], query_context: str) -> str:
    if not results: return ""
    try:
        df = pd.DataFrame(results)
        filename = f"{re.sub(r'[^a-zA-Z0-9]', '_', query_context[:30])}_{uuid.uuid4()}.csv"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        df.to_csv(filepath, index=False)
        return f"{BASE_URL.rstrip('/')}/{filename}"
    except Exception as e:
        logger.error(f"CSV Gen Error: {e}")
        return ""

def generate_statistics_from_results(results: List[Dict], query_context: str) -> Dict:
    if not results:
        return {"count": 0, "column_stats": {}, "preview": [], "stats_based_on_sample": False}
    
    count = len(results)
    stats_based_on_sample = False
    data_for_stats = results

    if count > MAX_ROWS_FOR_PANDAS_STATS:
        data_for_stats = random.sample(results, MAX_ROWS_FOR_PANDAS_STATS)
        stats_based_on_sample = True
    
    df = pd.DataFrame(data_for_stats)
    column_stats = {}
    
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if pd.api.types.is_numeric_dtype(non_null) and not non_null.empty:
            column_stats[col] = {
                "min": float(non_null.min()), "max": float(non_null.max()),
                "mean": float(non_null.mean()), "non_null_count": int(len(non_null))
            }
        elif not non_null.empty:
            column_stats[col] = {
                "distinct_count": int(non_null.nunique()),
                "top_values": {str(k): int(v) for k, v in non_null.astype(str).value_counts().head(5).items()}
            }
            
    return {
        "count": count,
        "column_stats": column_stats,
        "preview": results[:DISPLAY_ROW_LIMIT],
        "stats_based_on_sample": stats_based_on_sample
    }

def summarize_conversation(conversation_id: Optional[str] = None, max_turns: int = 3) -> str:
    if not conversation_id or conversation_id not in CONTEXT_CACHE:
        return "No prior context."
    history = CONTEXT_CACHE.get(conversation_id, {}).get('history', [])
    return "\n".join([f"User: {t['query']} -> Bot: {t['summary_preview']}" for t in history[-max_turns:]])

# --- LLM Invocation & Parsing ---

@retry(tries=3, delay=2, backoff=2, logger=logger)
def invoke_groq_model(chain, input_data: Dict) -> str:
    try:
        response_object = chain.invoke(input_data)
        raw_text = response_object.content.strip()
        # Attempt to extract JSON if wrapped in markdown or tags
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_text) or \
                     re.search(r"<JSON_START>([\s\S]*?)<JSON_END>", raw_text) or \
                     re.search(r"(\{[\s\S]*?\})", raw_text)
        if json_match:
            return json_match.group(1)
        return raw_text
    except Exception as e:
        logger.error(f"LLM Invoke Error: {e}")
        raise

def parse_json_response(response_text: str) -> Optional[Dict]:
    try:
        # Basic cleanup
        cleaned = response_text.strip()
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned) # Remove trailing commas
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(cleaned) # Fallback to YAML
        except:
            return None

def generate_textual_summary_from_stats(executed_queries_stats, user_query, plan, common_items=None):
    # Simplified version of the original summary generator
    summary = ["System-Generated Data Report:"]
    for i, stat in enumerate(executed_queries_stats):
        summary.append(f"\nQuery {i+1}: {stat.get('description_from_llm')}")
        if stat.get('error_if_any'):
            summary.append(f"  Status: Error - {stat['error_if_any']}")
        else:
            rows = stat.get('total_rows_found', 0)
            summary.append(f"  Rows Found: {rows}")
            if rows > 0:
                summary.append(f"  Key Stats: {json.dumps(stat.get('key_column_statistics', {}), default=str)}")
    return "\n".join(summary)


# --- PROMPT TEMPLATES (GENERIC) ---

INTENT_CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "db_summary", "domain_description", "history"],
    template="""
You are an AI assistant for a specific database system.
Domain Description: {domain_description}
Available Tables: {db_summary}

User Query: {user_query}
Conversation History: {history}

Classify the query intent into ONE of these categories:
1. "METADATA_DIRECT_ANSWER_PREFERRED": General questions about the domain or what data is available (e.g., "What does this DB contain?").
2. "DATA_RETRIEVAL": Requests for specific records, counts, comparisons, or analysis (e.g., "Find items with value > 10", "List all categories").
3. "OUT_OF_SCOPE": Queries unrelated to the database content.
4. "GENERAL_CONVERSATION": Greetings or non-data chitchat.

Output ONLY the category string.
Classification:
"""
)

SQL_PLAN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "schemas_json", "db_file", "history"],
    template="""
You are a SQL Expert. Generate a valid JSON plan to answer the user's request using SQLite.
User Query: {user_query}
Database File: {db_file}
Schemas: {schemas_json}
History: {history}

Rules:
1. Output ONLY a valid JSON object wrapped in <JSON_START> and <JSON_END>.
2. Use "query_type" to describe the action (data_retrieval, metadata_lookup).
3. "analysis_plan": A text description of your strategy.
4. "queries": A list of objects, each containing:
   - "sql": The valid SQLite query. **ALWAYS Quote identifiers** (e.g., SELECT "col" FROM "table").
   - "target_table": The table name.
   - "database_conceptual_name": A user-friendly name for the table.
   - "purpose_type": e.g., "data_preview", "aggregation", "entity_lookup".
5. Handle numeric columns stored as TEXT by using `CAST("col" AS REAL)`.

JSON Output:
<JSON_START>
{{
  "query_type": "data_retrieval",
  "analysis_plan": "...",
  "queries": [
    {{
      "sql": "SELECT ...",
      "target_table": "...",
      "database_conceptual_name": "...",
      "description": "...",
      "purpose_type": "..."
    }}
  ]
}}
<JSON_END>
"""
)

SUMMARY_INTERPRET_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "plan", "stats_summary", "history"],
    template="""
You are a helpful assistant. Summarize the database results for the user.
User Query: {user_query}
Plan: {plan}
Data Stats/Results: 
{stats_summary}

History: {history}

Output a JSON object with:
1. "summary": A conversational paragraph explaining the findings. Do not mention SQL or technical errors unless necessary. Be accurate to the stats provided.
2. "databases_conceptually_involved": A list of database/table names used.

Output JSON:
{{
  "summary": "...",
  "databases_conceptually_involved": ["..."]
}}
"""
)

# --- Main Query Processing Logic ---

def process_query(query: str, conversation_id: Optional[str] = None, user_api_key: Optional[str] = None) -> Dict:
    start_time = time.time()
    
    # Use global key if user specific not provided
    active_api_key = user_api_key if user_api_key else API_KEY
    if not active_api_key or "YOUR_GROQ_API_KEY" in active_api_key:
        return {"summary": "Error: API Key not configured.", "executed_queries_details": []}

    conversation_id = conversation_id or str(uuid.uuid4())
    sanitized_query = sanitize_query_input(query)
    
    # Initialize LLM
    llm = ChatGroq(api_key=active_api_key, model_name='llama-3.1-70b-versatile', temperature=0.1)
    
    # 0. Context Prep
    convo_history = summarize_conversation(conversation_id)
    table_list = ", ".join(DATABASE_MAPPING.values())
    
    # 1. Intent Classification
    try:
        intent_resp = invoke_groq_model(INTENT_CLASSIFICATION_PROMPT_TEMPLATE | llm, {
            "user_query": sanitized_query,
            "db_summary": table_list,
            "domain_description": DOMAIN_DESCRIPTION,
            "history": convo_history
        })
        intent = intent_resp.strip().replace('"', '')
    except Exception as e:
        logger.error(f"Intent Error: {e}")
        intent = "DATA_RETRIEVAL"

    # Handle Non-Data Intents
    if intent == "GENERAL_CONVERSATION":
        return {"summary": "Hello! I can help you query the database. What do you need?", "metadata": {"intent": intent}}
    if intent == "OUT_OF_SCOPE":
        return {"summary": "I can only answer questions related to the configured database domain.", "metadata": {"intent": intent}}

    # 2. SQL Planning
    schemas_str = json.dumps(schema_manager.get_schema_for_prompt(), indent=2)
    
    try:
        plan_raw = invoke_groq_model(SQL_PLAN_PROMPT_TEMPLATE | llm, {
            "user_query": sanitized_query,
            "schemas_json": schemas_str,
            "db_file": DB_FILE,
            "history": convo_history
        })
        plan_json = parse_json_response(plan_raw)
    except Exception as e:
        return {"summary": "I had trouble planning the database query.", "error": str(e)}

    if not plan_json or 'queries' not in plan_json:
        return {"summary": "I couldn't generate a valid plan to answer your question."}

    # 3. Execution
    results_for_display = []
    stats_for_summary = []
    
    for q in plan_json.get('queries', []):
        sql = q.get('sql')
        target = q.get('target_table')
        desc = q.get('description')
        
        # Execute
        rows, _ = execute_sql_query(sql, target)
        
        # Stats & processing
        is_error = isinstance(rows, str)
        stats = {}
        if not is_error:
            stats = generate_statistics_from_results(rows, target)
            csv_url = generate_csv(rows, target)
        
        # Store for Frontend
        results_for_display.append({
            "sql": sql,
            "description": desc,
            "results_preview": stats.get('preview', []) if not is_error else [],
            "row_count": stats.get('count', 0) if not is_error else 0,
            "error": rows if is_error else None,
            "download_url": csv_url if not is_error else ""
        })
        
        # Store for Stage 2 LLM
        stats_for_summary.append({
            "description_from_llm": desc,
            "error_if_any": rows if is_error else None,
            "total_rows_found": stats.get('count', 0) if not is_error else 0,
            "key_column_statistics": stats.get('column_stats', {}) if not is_error else {}
        })

    # 4. Summarization
    text_stats = generate_textual_summary_from_stats(stats_for_summary, sanitized_query, plan_json.get('analysis_plan'))
    
    try:
        summary_raw = invoke_groq_model(SUMMARY_INTERPRET_PROMPT_TEMPLATE | llm, {
            "user_query": sanitized_query,
            "plan": plan_json.get('analysis_plan'),
            "stats_summary": text_stats,
            "history": convo_history
        })
        summary_json = parse_json_response(summary_raw)
        final_summary = summary_json.get('summary', "Here are the results.")
    except Exception as e:
        final_summary = "I have the data but couldn't generate a natural language summary. Please check the result tables."

    # Update History Cache
    context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
    context_data['history'].append({"query": sanitized_query, "summary_preview": final_summary[:100] + "..."})
    CONTEXT_CACHE[conversation_id] = context_data

    return {
        "summary": final_summary,
        "executed_queries_details": results_for_display,
        "metadata": {
            "execution_time": time.time() - start_time,
            "intent": intent,
            "conversation_id": conversation_id
        }
    }

# --- Flask Endpoints ---

@app.route('/query', methods=['POST'])
@limiter.limit("100 per hour")
def query_endpoint():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query'"}), 400
        
        # Optional: Allow user to pass their own key in header or body
        user_key = request.headers.get('X-Groq-API-Key') or data.get('api_key')
        
        response = process_query(data['query'], data.get('conversation_id'), user_key)
        return jsonify(clean_nan_and_inf(response))
    except Exception as e:
        logger.error(f"Endpoint Error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    # Ensure DB file exists for dummy run
    if not os.path.exists(os.path.join(DB_PATH, DB_FILE)):
        print(f"WARNING: {DB_FILE} not found in {DB_PATH}. Create a dummy sqlite DB to test.")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
