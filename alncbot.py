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
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize Flask application
app = Flask(__name__)
CORS(app)

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

API_KEY = os.getenv('GROQ_API_KEY', '')
DB_PATH = os.getenv('DB_PATH', '/var/www/html/PlantXBot')
DB_FILE = 'all_databases.db'
DOWNLOAD_DIR = os.path.join(DB_PATH, 'public/downloads')
BASE_URL = 'http://14.139.61.8/PlantXBot/public/downloads'
CACHE = TTLCache(maxsize=1000, ttl=3600)
CONTEXT_CACHE = TTLCache(maxsize=100, ttl=3600)
DISPLAY_ROW_LIMIT = 10
MAX_ROWS_FOR_PANDAS_STATS = 50000
MAX_ROWS_FOR_LLM_SUMMARY = 1000

# --- ChromaDB Configuration ---
CHROMA_PERSIST_DIR = os.path.join(DB_PATH, 'Chroma_db')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- MODIFIED: Database to Table Mapping ---
DATABASE_MAPPING = {
    "anninter2": "AnnInter2 Database",
}
# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "20 per minute"]
)

class SchemaManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # --- MODIFIED: Added PtRNAdb to the schema definition ---
        self.schema = {
            'all_databases.db': {
                'tables': {
                    'linc_data': {
                        'columns': [
                            {'name': 'AlnC_ID', 'sqlite_type': 'TEXT', 'semantic_type': 'primary_identifier', 'description': 'Unique identifier for lncRNA.'},
                            {'name': 'Sample_code', 'sqlite_type': 'TEXT', 'semantic_type': 'sample_id', 'description': 'Sample code identifier.'},
                            {'name': 'Species', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'Species name.'},
                            {'name': 'Tissue', 'sqlite_type': 'TEXT', 'semantic_type': 'tissue_type', 'description': 'Tissue type.'},
                            {'name': 'Source_transcript_ID', 'sqlite_type': 'TEXT', 'semantic_type': 'identifier', 'description': 'Source transcript identifier.'},
                            {'name': 'Coding_probability', 'sqlite_type': 'REAL', 'semantic_type': 'probability', 'description': 'Probability of being coding.'},
                            {'name': 'Non_coding_probability', 'sqlite_type': 'REAL', 'semantic_type': 'probability', 'description': 'Probability of being non-coding.'},
                            {'name': 'Length', 'sqlite_type': 'INT', 'semantic_type': 'length_value', 'description': 'Sequence length.'},
                            {'name': 'GC_content', 'sqlite_type': 'REAL', 'semantic_type': 'sequence_feature', 'description': 'GC content percentage.'},
                            {'name': 'NCBI_link', 'sqlite_type': 'TEXT', 'semantic_type': 'url', 'description': 'Link to NCBI record.'},
                            {'name': 'Sequence', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_sequence', 'description': 'lncRNA sequence.'}
                        ],
                        'description': 'Long non-coding RNA (lncRNA) data with coding probability and expression.',
                        'primary_keys': ['AlnC_ID'],
                        'common_joins': {
                            'athisomir': "On `Sequence` or genomic coordinates.",
                            'anninter2': "On `Sequence` with `interactor1_seq` or `interactor2_seq`."
                        },
                        'notes': ["Focuses on lncRNAs across multiple species. Use `Non_coding_probability` for lncRNA classification."]
                  }
                }
            }
        }
        # Initialize species and tissues for all tables
        self.species = {table_name: set() for table_name in self.schema['all_databases.db']['tables']}
        self.tissues = {table_name: set() for table_name in self.schema['all_databases.db']['tables']}
        self.update_schema_from_db()

    def _enrich_schema_with_dynamic_info(self):
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for table_name, table_data in self.schema['all_databases.db']['tables'].items():
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                pragma_info = {row[1]: row[2] if row[2] else 'TEXT' for row in cursor.fetchall()}
                for col_dict in table_data['columns']:
                    col_name = col_dict['name']
                    if col_name in pragma_info:
                        db_type = pragma_info[col_name].upper()
                        if db_type != col_dict.get('sqlite_type', '').upper():
                            logger.debug(f"Updating SQLite type for {table_name}.{col_name} to {db_type}")
                            col_dict['sqlite_type'] = db_type
                    else:
                        logger.warning(f"Column '{col_name}' in schema for {table_name} not found in DB")
            conn.close()
        except Exception as e:
            logger.error(f"Failed to enrich schema: {e}", exc_info=True)

    def update_schema_from_db(self):
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for table_name in self.schema['all_databases.db']['tables']:
                self.species.setdefault(table_name, set())
                self.tissues.setdefault(table_name, set())
                table_columns = {col['name'] for col in self.schema['all_databases.db']['tables'][table_name]['columns']}

                # --- MODIFIED: Determine species column based on table ---
                species_col = None
                if table_name == 'linc_data':
                    species_col = 'Species' if 'Species' in table_columns else None
                elif table_name == 'PtRNAdb': # <-- NEW: Handle PtRNAdb
                    species_col = 'Plant Source' if 'Plant Source' in table_columns else None
                elif table_name in ['athisomir', 'anninter2']:
                    # Implicitly Arabidopsis thaliana for athisomir and anninter2
                    self.species[table_name].add('Arabidopsis thaliana')

                if species_col:
                    cursor.execute(f"SELECT DISTINCT \"{species_col}\" FROM \"{table_name}\" WHERE \"{species_col}\" IS NOT NULL AND \"{species_col}\" != ''")
                    self.species[table_name].update(row[0] for row in cursor.fetchall())

                # Tissue column determination (PtRNAdb has no tissue column)
                tissue_col = 'Tissue' if 'Tissue' in table_columns else None
                if tissue_col:
                    cursor.execute(f"SELECT DISTINCT \"{tissue_col}\" FROM \"{table_name}\" WHERE \"{tissue_col}\" IS NOT NULL AND \"{tissue_col}\" != ''")
                    self.tissues[table_name].update(row[0] for row in cursor.fetchall())

            conn.close()
            logger.info("Schema updated with species and tissues")
            self._enrich_schema_with_dynamic_info()
        except Exception as e:
            logger.error(f"Failed to update schema: {str(e)}", exc_info=True)

    def get_schema_for_prompt(self) -> Dict:
        return {
            db_name: {
                'tables': {
                    table_name: {
                        'columns': [{'name': c['name'], 'type': c.get('sqlite_type', 'TEXT'), 'semantic': c.get('semantic_type', 'generic'), 'desc': c['description'][:100]} for c in table_details['columns']],
                        'description': table_details['description'],
                        'notes': table_details.get('notes', []),
                        'common_joins': table_details.get('common_joins', {}),
                        'primary_keys': table_details.get('primary_keys', [])
                    } for table_name, table_details in db_data['tables'].items()
                }
            } for db_name, db_data in self.schema.items()
        }

    def get_tables(self) -> List[str]:
        return list(self.schema['all_databases.db']['tables'].keys())

    def get_species(self, table: str) -> Set[str]:
        return self.species.get(table, set())

    def get_tissues(self, table: str) -> Set[str]:
        return self.tissues.get(table, set())

schema_manager = SchemaManager(os.path.join(DB_PATH, DB_FILE))

# ... (rest of the file from line 410 to 658 is unchanged) ...
logger.info("Initializing ChromaDB vector store...")
vector_db = None
try:
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(f"ChromaDB persistence directory not found at: {CHROMA_PERSIST_DIR}")
    
    # Define the embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Load the persisted database
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )
    logger.info(f"ChromaDB loaded successfully from {CHROMA_PERSIST_DIR}. Collection has {vector_db._collection.count()} documents.")
except Exception as e:
    logger.error(f"FATAL: Failed to load ChromaDB. The app may not be able to answer informational queries. Error: {e}", exc_info=True)
    # The app will continue to run, but vector search will be disabled.

def get_relevant_context_from_vectordb(query: str, k: int = 4) -> str:
    """
    Queries the ChromaDB to find relevant document chunks for a given query.
    """
    if vector_db is None:
        logger.warning("Vector DB is not available, returning empty context.")
        return "No vector context available due to a system error."

    logger.info(f"Querying vector DB for: '{query}'")
    try:
        # Perform similarity search
        relevant_docs = vector_db.similarity_search(query, k=k)

        if not relevant_docs:
            logger.info("No relevant documents found in vector DB for the query.")
            return "No specific context found in the knowledge base for this query."

        # Combine the content of the relevant documents into a single string
        context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"Retrieved {len(relevant_docs)} relevant context chunks from vector DB.")
        return context_str
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        return "An error occurred while retrieving context from the knowledge base."

def sanitize_query_input(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string")
    return query

def sanitize_sql(sql_query: str) -> str:
    """
    Sanitizes the SQL query to prevent malicious commands and invalid identifiers.
    This version correctly distinguishes between table names and column names.
    """
    sql_query = sql_query.strip()
    # Prevent dangerous DML/DDL statements unless the query is a common table expression (WITH)
    if any(token in sql_query.upper() for token in ['DROP', 'DELETE FROM', 'TRUNCATE', 'INSERT INTO', 'UPDATE ']) and not sql_query.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Potentially malicious SQL command blocked: {sql_query}")

    # Create sets of valid table and column names from the schema
    valid_columns = set()
    for table_details in schema_manager.schema['all_databases.db']['tables'].values():
        valid_columns.update(col['name'] for col in table_details['columns'])
    
    valid_tables = set(schema_manager.get_tables())

    # Check all quoted identifiers in the query
    for match in re.finditer(r'"([^"]+)"', sql_query):
        identifier = match.group(1)
        
        # An identifier is valid if it's a known table, a known column, or a function like COUNT
        is_valid_table = identifier in valid_tables
        is_valid_column = identifier in valid_columns
        is_aggregate_function = re.match(r'COUNT\(.+\)', identifier, re.IGNORECASE)

        if not (is_valid_table or is_valid_column or is_aggregate_function):
            error_msg = (
                f"Invalid table or column name in SQL: '{identifier}'. "
                f"Please ensure all identifiers are from the provided schema."
            )
            logger.error(error_msg, valid_tables=sorted(list(valid_tables)), valid_columns=sorted(list(valid_columns)))
            raise ValueError(error_msg)

    # Remove trailing semicolon if present
    if sql_query.endswith(';'):
        sql_query = sql_query[:-1]
        
    return sql_query

def clean_nan_and_inf(obj):
    """
    Recursively walk a data structure and replace NaN/Infinity with None.
    This is essential for robust JSON serialization from pandas data.
    """
    if isinstance(obj, dict):
        return {k: clean_nan_and_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_and_inf(elem) for elem in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None  # Replace NaN/inf with None (becomes null in JSON)
    return obj

def generate_textual_summary_from_stats(
    executed_queries_stats_list: List[Dict],
    original_user_query: str,
    analysis_plan: str,
    common_items_info_list: Optional[List[Dict]] = None
) -> str:
    if not executed_queries_stats_list and not common_items_info_list:
        return (
            f"For your query: '{original_user_query}', the AI's plan was: '{analysis_plan}'. "
            "However, no data summaries from SQL queries or common item checks are available. "
            "Please try rephrasing to focus on miRNAs, lncRNAs, or RNA interactions."
        )

    summary_parts = ["System-Generated Data Summary:"]

    if common_items_info_list:
        for common_items_info in common_items_info_list:
            if not common_items_info:
                continue

            item_type_name = common_items_info.get('item_type', 'unknown items')
            summary_parts.append(f"\n--- Common '{item_type_name}' Analysis ---")

            # Use 'databases_queried_for_this_item' from LLM's plan
            databases_planned_str = ", ".join(common_items_info.get('databases_queried_for_this_item', ['relevant databases']))
            summary_parts.append(f"  Data for '{item_type_name}' commonality check was requested from: {databases_planned_str}.")

            if common_items_info.get("note"):
                summary_parts.append(f"  Note on data collection for '{item_type_name}': {common_items_info['note']}")

            distinct_values_map = common_items_info.get("distinct_values_per_database_for_this_item")
            if distinct_values_map:
                summary_parts.append(f"  Distinct '{item_type_name}' values found per database (for Stage 2 LLM to analyze for commonality using semantic hints):")
                found_any_distinct_values_for_current_item_type = False
                for db_name_in_map, distinct_list_val in distinct_values_map.items():
                    if db_name_in_map not in common_items_info.get('databases_queried_for_this_item', []):
                        continue  # Only show data for explicitly planned DBs

                    if isinstance(distinct_list_val, (list, set)) and distinct_list_val:
                        sorted_distinct_list = sorted(list(distinct_list_val))
                        example_items = ', '.join(f"'{v}'" for v in sorted_distinct_list[:15])
                        ellipsis = '...' if len(sorted_distinct_list) > 15 else ''
                        summary_parts.append(f"    - {db_name_in_map} ({len(sorted_distinct_list)} distinct): {example_items}{ellipsis}")
                        found_any_distinct_values_for_current_item_type = True
                    elif isinstance(distinct_list_val, (list, set)) and not distinct_list_val:
                        summary_parts.append(f"    - {db_name_in_map}: No distinct '{item_type_name}' values found.")
                    elif isinstance(distinct_list_val, str) and "Error:" in distinct_list_val:
                        summary_parts.append(f"    - {db_name_in_map}: Retrieval Issue: {distinct_list_val[:200]}")
                    else:
                        summary_parts.append(f"    - {db_name_in_map}: Data not retrieved or none found for '{item_type_name}'.")

                if not found_any_distinct_values_for_current_item_type and \
                   not any("Error:" in str(v) or "Issue:" in str(v) for v in distinct_values_map.values()):
                    summary_parts.append(f"    No specific distinct '{item_type_name}' values were retrieved from any of the targeted databases.")
            
            elif not common_items_info.get("note"):
                summary_parts.append(f"  No information on distinct '{item_type_name}' values was processed from the databases for this check.")

    if executed_queries_stats_list:
        summary_parts.append("\n--- Individual Query Part Summaries ---")
        for i, stats_item in enumerate(executed_queries_stats_list):
            # Skip queries used solely for common item checks
            purpose_type_for_skip_check = stats_item.get('purpose_type', '')
            if purpose_type_for_skip_check.startswith("list_distinct_values_for_common_check_"):
                continue

            part_summary = [f"\nQuery Part {i+1} (Description: '{stats_item.get('description_from_llm', 'N/A')}')"]
            if stats_item.get('original_sql'):
                part_summary.append(f"  SQL Executed: `{stats_item.get('original_sql')}`")
            part_summary.append(f"  Target: {stats_item.get('database_conceptual_name', 'Unknown DB')} (Table: {stats_item.get('target_table', 'N/A')})")
            purpose_type = stats_item.get('purpose_type', 'N/A')
            part_summary.append(f"  Purpose: {purpose_type}")

            if stats_item.get('error_if_any'):
                part_summary.append(f"  Status: Error - {stats_item['error_if_any']}")
            else:
                total_rows = stats_item.get('total_rows_found', 0)
                part_summary.append(f"  Status: Success. Records Found: {total_rows}")

                if purpose_type == 'aggregation' and total_rows == 1 and stats_item.get('original_sql') and 'COUNT(' in stats_item.get('original_sql','').upper():
                    count_col_name = next((col for col in (stats_item.get('key_column_statistics') or {}).keys() if 'COUNT(' in col.upper()), None)
                    if count_col_name and 'sum' in stats_item.get('key_column_statistics', {}).get(count_col_name, {}):
                        actual_count = int(stats_item['key_column_statistics'][count_col_name]['sum'])
                        part_summary.append(f"  Aggregate Result (e.g., Count): {actual_count}")

                if stats_item.get('statistics_based_on_sample'):
                    part_summary.append("  (Statistics below are based on a sample of a large dataset.)")

                if purpose_type == 'aggregation_categorical_expression' and total_rows > 0 and stats_item.get("preview_for_ui_only"):
                    part_summary.append("  Categorical Expression Counts:")
                    for row in stats_item["preview_for_ui_only"]:
                        group_desc_parts = []
                        if 'Tissue' in row:
                            group_desc_parts.append(f"Tissue='{row.get('Tissue', 'N/A')}'")
                        if 'Isomir_Type' in row:
                            group_desc_parts.append(f"Isomir_Type='{row.get('Isomir_Type', 'N/A')}'")
                        group_desc = ", ".join(group_desc_parts) if group_desc_parts else "Overall"
                        count_val = row.get('count_per_category', row.get('count', 'N/A'))
                        part_summary.append(f"    - For {group_desc}: {count_val} records")

                if total_rows > 0 and stats_item.get('key_column_statistics') and purpose_type not in ['aggregation_categorical_expression']:
                    part_summary.append("  Key Column Stats:")
                    for col, col_stats in stats_item['key_column_statistics'].items():
                        if 'COUNT(' in col.upper() and purpose_type == 'aggregation' and total_rows == 1:
                            continue
                        col_details = [f"    - '{col}':"]
                        if 'non_null_count' in col_stats and col_stats['non_null_count'] != total_rows:
                            col_details.append(f"Non-nulls: {col_stats['non_null_count']}/{total_rows}")
                        if 'distinct_count' in col_stats and col_stats['distinct_count'] < total_rows and col_stats['distinct_count'] > 0:
                            col_details.append(f"Distinct: {col_stats['distinct_count']}")
                        numeric_stats_added = False
                        if col == 'RPM' and 'mean' in col_stats:
                            col_details.append(f"Avg RPM: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        elif col == 'Non_coding_probability' and 'mean' in col_stats:
                            col_details.append(f"Avg Non-coding Probability: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        elif col == 'p_value' and 'mean' in col_stats:
                            col_details.append(f"Avg p-value: {col_stats['mean']:.4f}")
                            numeric_stats_added = True
                        elif col == 'allenscore' and 'mean' in col_stats:
                            col_details.append(f"Avg Allen Score: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        elif col == 'GC_content' and 'mean' in col_stats:
                            col_details.append(f"Avg GC Content: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        elif col == 'Length' and 'mean' in col_stats:
                            col_details.append(f"Avg Length: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        elif 'mean' in col_stats and isinstance(col_stats['mean'], (int, float)):
                            col_details.append(f"Avg: {col_stats['mean']:.2f}")
                            numeric_stats_added = True
                        if 'median' in col_stats and isinstance(col_stats['median'], (int, float)):
                            col_details.append(f"Median: {col_stats['median']:.2f}")
                            numeric_stats_added = True
                        if 'min' in col_stats and 'max' in col_stats and col_stats['min'] != col_stats['max']:
                            col_details.append(f"Range: {col_stats['min']}-{col_stats['max']}")
                            numeric_stats_added = True
                        elif 'min' in col_stats and not numeric_stats_added:
                            col_details.append(f"Value(s) include: {col_stats['min']}")
                        if 'top_values' in col_stats and col_stats['top_values']:
                            top_v_str = ", ".join([f"'{k}' ({v}x)" for k, v in list(col_stats['top_values'].items())[:3]])
                            col_details.append(f"Top Values: {top_v_str}")
                        if len(col_details) > 1:
                            part_summary.extend(col_details)

                elif total_rows == 0 and not stats_item.get('error_if_any'):
                    part_summary.append("  No data records found for this part. Try rephrasing to focus on miRNAs, lncRNAs, or RNA interactions.")

            summary_parts.extend(part_summary)

    return "\n".join(summary_parts)

# --- MODIFIED: Added PtRNAdb to the knowledge blob ---
DATABASE_METADATA_KNOWLEDGE_BLOB = """
**General Concepts:**
- **lncRNA (long non-coding RNA):** Non-coding RNAs >200 nt, involved in regulatory processes.
**Database Summaries:**
*   **Alnc Database (linc_data):** Long non-coding RNA data across multiple plant species. Key data: `Sequence`, `Species`, `Tissue`, `Non_coding_probability`, `Length`, `GC_content`. Focuses on lncRNA characterization.
"""

# ... (rest of the file from line 895 to 1109 is unchanged) ...
def execute_sql_query(sql_query: str, table_context: str) -> Tuple[List[Dict] | str, str]:
    try:
        sql_to_execute = sanitize_sql(sql_query)
        db_path_full = os.path.join(DB_PATH, DB_FILE)
        if not os.path.exists(db_path_full):
            logger.error(f"Database not found: {db_path_full}")
            return f"Error: Database {DB_FILE} not found.", table_context
        conn = sqlite3.connect(f"file:{db_path_full}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        logger.info(f"Executing SQL on {table_context}: {sql_to_execute}")
        cursor.execute(sql_to_execute)
        results = cursor.fetchall()
        conn.close()
        if not results:
            logger.info(f"No results for query on {table_context}")
            return [], table_context
        result_dicts = [dict(row) for row in results]
        logger.info(f"Retrieved {len(result_dicts)} rows from {table_context}")
        return result_dicts, table_context
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {sql_query} on {table_context}. Error: {str(e)}", exc_info=True)
        return f"Error executing SQL: {str(e)}", table_context
    except ValueError as e:
        logger.error(f"SQL sanitization error: {sql_query} on {table_context}. Error: {str(e)}", exc_info=True)
        return f"Error in SQL structure: {str(e)}", table_context
    except Exception as e:
        logger.error(f"Unexpected error: {sql_query} on {table_context}. Error: {str(e)}", exc_info=True)
        return f"Unexpected error: {str(e)}", table_context

def generate_csv(results: List[Dict], query_context: str) -> str:
    if not results:
        return ""
    try:
        df = pd.DataFrame(results)
        filename_prefix = re.sub(r'\W+', '_', query_context[:30]) if query_context else "data"
        filename = f"{filename_prefix}_{uuid.uuid4()}.csv"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        df.to_csv(filepath, index=False)
        download_url = f"{BASE_URL.rstrip('/')}/{filename}"
        logger.info(f"Generated CSV: {filepath}, URL: {download_url}")
        return download_url
    except Exception as e:
        logger.error(f"Failed to generate CSV: {str(e)}", exc_info=True)
        return ""

def sample_relevant_data_for_llm(results: List[Dict], query_context: str, max_rows: int = MAX_ROWS_FOR_LLM_SUMMARY) -> List[Dict]:
    """
    Sample relevant data for LLM summary when dataset is large, using knowledge graph context.
    """
    if len(results) <= max_rows:
        return results
    logger.warning(f"Sampling {max_rows} rows from {len(results)} for {query_context}")
    # Prioritize columns based on knowledge graph and query context
    priority_columns = ['Sequence', 'Isomir_Type', 'RPM', 'Parent', 'Species', 'Tissue',
                   'Non_coding_probability', 'GC_content', 'p_value', 'allenscore',
                   'interactor1_seq', 'interactor2_seq', 'rna_type1', 'rna_type2']
    df = pd.DataFrame(results)
    available_priority_cols = [col for col in priority_columns if col in df.columns]
    if available_priority_cols:
        # Sort by a relevant numerical column if available, else random sample
        numeric_cols = [col for col in available_priority_cols if pd.api.types.is_numeric_dtype(df[col].dropna())]
        if numeric_cols:
            sort_col = numeric_cols[0]  # Use first numeric column (e.g., rpm, log2fold_change)
            df = df.sort_values(by=sort_col, ascending=False, na_position='last')
        else:
            df = df.sort_values(by=available_priority_cols[0], na_position='last')
        sampled_df = df.head(max_rows)
    else:
        sampled_df = df.sample(n=max_rows, random_state=42)
    return sampled_df.to_dict('records')

def generate_statistics_from_results(results: List[Dict], query_context: str = "query") -> Dict:
    request_id_prefix = query_context.split('_query_')[0] if '_query_' in query_context else query_context
    logger.debug(f"[{request_id_prefix}] Generating stats for {query_context}, rows: {len(results)}")

    if not results:
        return {"count": 0, "column_stats": {}, "preview": [], "stats_based_on_sample": False}

    original_count = len(results)
    preview_for_display = results[:DISPLAY_ROW_LIMIT]
    stats_based_on_sample = False
    data_for_stats = results

    if original_count > MAX_ROWS_FOR_PANDAS_STATS:
        logger.warning(f"[{request_id_prefix}] Large result set ({original_count} rows). Sampling {MAX_ROWS_FOR_PANDAS_STATS} rows.")
        data_for_stats = random.sample(results, MAX_ROWS_FOR_PANDAS_STATS)
        stats_based_on_sample = True

    column_stats = {}
    if data_for_stats:
        df = pd.DataFrame(data_for_stats)
        for col in df.columns:
            series = df[col]
            stats = {"null_count": int(series.isnull().sum())}
            non_null = series.dropna()
            if pd.api.types.is_numeric_dtype(non_null) and not non_null.empty:
                stats.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if non_null.nunique() > 1 else 0.0,
                    "sum": float(non_null.sum()),
                    "non_null_count": int(len(non_null))
                })
            elif not non_null.empty:
                try:
                    value_counts = non_null.astype(str).value_counts()
                    stats.update({
                        "distinct_count": int(len(value_counts)),
                        "top_values": {str(k): int(v) for k, v in value_counts.head(10).items()},
                        "non_null_count": int(len(non_null))
                    })
                except Exception as e:
                    logger.warning(f"Could not compute categorical stats for {col}: {e}")
                    stats["categorical_stats_error"] = str(e)
            else:
                stats.update({"distinct_count": 0, "non_null_count": 0})
            column_stats[col] = stats

    return {
        "count": original_count,
        "column_stats": column_stats,
        "preview": preview_for_display,
        "stats_based_on_sample": stats_based_on_sample
    }

def summarize_conversation(conversation_id: Optional[str] = None, max_turns: int = 3, max_len_per_item: int = 100) -> str:
    if not conversation_id or conversation_id not in CONTEXT_CACHE:
        return "No prior conversation context available."
    context = CONTEXT_CACHE.get(conversation_id, {})
    history = context.get('history', [])
    if not history:
        return "No queries in this conversation yet."
    
    # Take last 'max_turns' and truncate content
    summaries = []
    for turn in history[-max_turns:]:
        user_q_preview = turn.get('query', 'User query unavailable')[:max_len_per_item]
        bot_s_preview = turn.get('summary_preview', 'No preview available')[:max_len_per_item]
        if len(turn.get('query', '')) > max_len_per_item:
            user_q_preview += "..."
        if len(turn.get('summary_preview', '')) > max_len_per_item:
            bot_s_preview += "..."
        summaries.append(f"User: \"{user_q_preview}\" -> Bot: \"{bot_s_preview}\"")
        
    if not summaries:
        return "No recent conversation context to summarize."
    return "Recent context:\n" + "\n".join(summaries)

@retry(tries=2, delay=5, backoff=2, logger=logger)
def invoke_groq_model(chain, input_data: Dict) -> str:
    try:
        logger.info("Invoking Groq model", model_input_keys=list(input_data.keys()))
        response_object = chain.invoke(input_data)
        raw_llm_text = response_object.content.strip() if hasattr(response_object, 'content') else str(response_object).strip()
        
        logger.debug(f"Raw LLM output (first 500 chars): {raw_llm_text[:500]}")
        
        # Attempt 1: Delimiter-based extraction
        start_token = "<JSON_START>"
        end_token = "<JSON_END>"
        start_index = raw_llm_text.find(start_token)
        end_index = raw_llm_text.rfind(end_token)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = raw_llm_text[start_index + len(start_token):end_index].strip()
            if json_str.startswith('{') and json_str.endswith('}'):
                try:
                    json.loads(json_str)
                    logger.info("Extracted JSON using delimiters.")
                    return json_str
                except json.JSONDecodeError as e:
                    logger.warning(f"Delimiter-extracted string not valid JSON: {json_str[:200]}. Error: {str(e)}")

        # Attempt 2: Markdown ```json ... ``` block
        match_markdown = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_llm_text, re.DOTALL | re.IGNORECASE)
        if match_markdown:
            extracted_json_str = match_markdown.group(1).strip()
            if extracted_json_str.startswith('{') and extracted_json_str.endswith('}'):
                try:
                    json.loads(extracted_json_str)
                    logger.info("Extracted JSON from markdown block.")
                    return extracted_json_str
                except json.JSONDecodeError as e:
                    logger.warning(f"Markdown-extracted string not valid JSON: {extracted_json_str[:200]}. Error: {str(e)}")

        # Attempt 3: Find the largest valid JSON object
        best_json_match = None
        for match in re.finditer(r"(\{[\s\S]*?\})", raw_llm_text):
            potential_json_str = match.group(1)
            try:
                json.loads(potential_json_str)
                if best_json_match is None or len(potential_json_str) > len(best_json_match):
                    best_json_match = potential_json_str
            except json.JSONDecodeError:
                continue
        
        if best_json_match:
            logger.info("Extracted largest valid JSON object found in raw text.")
            return best_json_match

        # Attempt 4: Fallback to first '{' and last '}'
        json_start = raw_llm_text.find('{')
        json_end = raw_llm_text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            extracted_json_str = raw_llm_text[json_start:json_end + 1].strip()
            try:
                json.loads(extracted_json_str)
                logger.warning(f"Falling back to brace-extraction for JSON. String: {extracted_json_str[:200]}")
                return extracted_json_str
            except json.JSONDecodeError as e:
                logger.warning(f"Brace-extracted string not valid JSON: {extracted_json_str[:200]}. Error: {str(e)}")

        # Fallback: Retry with simplified prompt
        logger.warning("No valid JSON found. Retrying with simplified prompt.")
        simplified_prompt = chain.prompt.template + "\n**Simplified Instruction**: Return only a valid JSON object wrapped in <JSON_START> and <JSON_END>. No additional text."
        simplified_chain = simplified_prompt | chain.llm
        response_object = simplified_chain.invoke(input_data)
        raw_llm_text = response_object.content.strip() if hasattr(response_object, 'content') else str(response_object).strip()
        start_index = raw_llm_text.find(start_token)
        end_index = raw_llm_text.rfind(end_token)
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = raw_llm_text[start_index + len(start_token):end_index].strip()
            if json_str.startswith('{') and json_str.endswith('}'):
                try:
                    json.loads(json_str)
                    logger.info("Extracted JSON from simplified prompt.")
                    return json_str
                except json.JSONDecodeError as e:
                    logger.error(f"Simplified prompt failed to produce valid JSON: {json_str[:200]}. Error: {str(e)}")

        logger.error(f"No valid JSON structure found in LLM output after all attempts. Raw output: {raw_llm_text[:500]}")
        raise ValueError("LLM output is not a valid JSON structure after multiple extraction attempts.")

    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429:
            error_message = e.response.json().get('error', {}).get('message', '')
            wait_match = re.search(r'Please try again in ([\d\.]+)s', error_message)
            wait_seconds = float(wait_match.group(1)) + random.uniform(1,3) if wait_match else 60
            logger.warning(f"Rate limit exceeded. Waiting {wait_seconds:.2f}s. Message: {error_message}")
            time.sleep(wait_seconds)
            raise
        raise
    except Exception as e:
        logger.error(f"Model invocation failed: {str(e)}", exc_info=True)
        raise
def parse_json_response(response_text: str) -> Optional[Dict]:
    """
    Tries to parse a string as JSON, with fallbacks for common LLM output issues.
    """
    if not isinstance(response_text, str):
        logger.error(f"parse_json_response received non-string input: {type(response_text)}")
        return None

    original_response_text_preview = response_text[:500] # For logging

    # First, try direct parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {str(e)}. Preview: {original_response_text_preview}")

    cleaned_text = response_text.strip()
    match_json_block = re.search(r"(\{[\s\S]*?\})", cleaned_text) # Find first complete JSON block
    if match_json_block:
        cleaned_text = match_json_block.group(1)
    else:
        # If no clear block, try to find first { and last } again
        # This might happen if invoke_groq_model returned raw text
        start_brace = cleaned_text.find('{')
        end_brace = cleaned_text.rfind('}')
        if start_brace != -1 and end_brace > start_brace:
            cleaned_text = cleaned_text[start_brace : end_brace + 1]
        else:
            logger.error(f"Could not find even basic JSON structure (curly braces). Preview: {original_response_text_preview}")
            return None 

    cleaned_text_for_parsing = cleaned_text
    
    # Remove trailing commas before } or ]
    cleaned_text_for_parsing = re.sub(r',\s*([}\]])', r'\1', cleaned_text_for_parsing)
    

    try:
        logger.debug(f"Attempting parse on cleaned text: {cleaned_text_for_parsing[:500]}")
        return json.loads(cleaned_text_for_parsing)
    except json.JSONDecodeError as e_cleaned:
        logger.warning(f"JSON parse failed on cleaned text: {str(e_cleaned)}. Preview of cleaned: {cleaned_text_for_parsing[:500]}")
        
        # Fallback: Try to parse as YAML, as it's more flexible
        try:
            yaml_data = yaml.safe_load(cleaned_text_for_parsing) # Use the potentially cleaned text
            if isinstance(yaml_data, dict):
                logger.info("Successfully parsed response as YAML after JSON failures.")
                return yaml_data
        except yaml.YAMLError as e_yaml:
            logger.error(f"YAML parse also failed: {str(e_yaml)}. Original preview: {original_response_text_preview}")
            
    logger.error(f"Could not parse response as JSON or YAML. Original preview: {original_response_text_preview}")
    return None

# --- MODIFIED: Updated Intent Classification Prompt ---
INTENT_CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "available_databases_summary", "database_content_summary", "conversation_summary"],
    template="""
You are an AI assistant responsible for classifying user queries directed at a suite of plant non-coding RNA (ncRNA) databases.
*** You are a specialized chatbot for answering queries related to lncRNA, catering to the following database Alnc Database ***
Your primary goal is to determine if the query is answerable using the provided databases and their described content, or if it's out of scope. Understand the query and its intent with clarity to identify the true meaning of what type of data or information the user is asking for.
****If a user asks a "what data do you have on [topic]?" or "show me data about [topic]" question, interpret this as a `data_preview` request*****

User Query: {user_query}
Conversation Summary (Recent turns, if any): {conversation_summary}
Available Databases (Name: Conceptual Name): {available_databases_summary}
Database Content Summary (Focus on types of molecules and data):
{database_content_summary}

Based on the user query and the nature of the available databases (which primarily focus on specific biomolecules and genomic features like miRNAs, isomiRs, lncRNAs, tRNAs, and RNA interactions as detailed in the Database Content Summary), classify the query into ONE of the following categories:

1. "METADATA_DIRECT_ANSWER_PREFERRED": The query asks for general knowledge or facts ABOUT the databases (e.g., "What is Alnc?", "What are lncRNAs?", "Do you have data on RNA interactions?", "What kind of data is in Alnc Database?"). These should be answered from a knowledge base.
2. "METADATA_SQL_FALLBACK_PREFERRED": Query asks for metadata *about the data within the databases* that might require a simple SQL query if not directly in a knowledge text, but still benefits from the knowledge text for context (e.g., "How many species are in PtRNAdb?", "List tissue types in Alnc Database?", "What lncRNA types are in Alnc?").
3. "DATA_RETRIEVAL": ***Query starting with ("what is the", "which", "fetch", "tell me about", "what can be", "give me", etc.)***, Query asks to find specific data records, perform calculations on data, or compare data sets *within the described scope of the databases* (e.g., "Find miRNAs in Arabidopsis with RPM > 100", "List lncRNAs in rice", "Find tRNA genes in rice"). **Crucially, this also includes vague follow-up commands (like 'ok do it', 'show me', 'run that', 'yes please') that directly relate to a data action proposed or described by the bot in the `conversation_summary`.** Use the context to see if the user is confirming a previous action.
4. "AMBIGUOUS": Query is related to the database domain but is unclear and needs clarification from the user.
5. "TYPO": Query seems related to the database domain but likely contains a typo.
6. "GENERAL_CONVERSATION": Query is conversational, a greeting, or a simple closing.
7. "OUT_OF_SCOPE": Query is clearly unrelated to plant ncRNAs, the specific databases, their content, or asks about entities explicitly NOT covered (e.g., "What is the time?", "tRF data", "siRNA data").

Output ONLY the classification string (e.g., "DATA_RETRIEVAL" or "OUT_OF_SCOPE"). Ensure no extra text or explanation.

Classification:
"""
)

# --- MODIFIED: Updated SQL Planning Prompt ---
SQL_PLAN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "classified_intent", "db_file_name", "schemas_json", "knowledge_graph",
        "database_table_mapping_info", "DISPLAY_ROW_LIMIT", "retrieved_knowledge_context",
        "conversation_summary", "previous_query_summary"
    ],
    template="""
You are an AI assistant designed to analyze user queries and plan data retrieval or information provision for plant non-coding RNA (ncRNA) databases. Your task is to generate precise, SQLite-compatible SQL queries or metadata responses based on the provided database schemas and knowledge graph.

*** You are a specialized chatbot for answering queries related to lncRNA, catering to the following database Alnc Database ***

Your *ONLY* output MUST be a single, valid JSON object, wrapped in <JSON_START> and <JSON_END>. No other text, explanations, or conversational filler is allowed before or after the JSON block. Ensure the JSON is well-formed and parseable without errors.

**User Input and Context:**
- User Query: {user_query}
- Pre-classified Intent: {classified_intent}
- Database File: {db_file_name}
- Schemas: {schemas_json}
- Knowledge Graph: {knowledge_graph}
- Database Mapping: {database_table_mapping_info}
- Conversation History: {previous_query_summary}; {conversation_summary}
- Retrieved Knowledge Context: {retrieved_knowledge_context}
- UI Preview Row Limit: {DISPLAY_ROW_LIMIT}

**Instructions:**
- For `DATA_RETRIEVAL` or `METADATA_SQL_FALLBACK_PREFERRED` intents, generate one or more SQLite-compatible SQL queries targeting the relevant tables (``linc_data`).
- Use only the column names provided in `schemas_json`. Do NOT use table names (e.g., 'linc_data') as column names.
- For joins, use the `common_joins` from the schema.
- For `METADATA_DIRECT_ANSWER_PREFERRED`, provide a direct answer based on `retrieved_knowledge_context` or `database_table_mapping_info`.
- For `OUT_OF_SCOPE`, return an empty queries list and a direct answer explaining the limitation.
- Include a `display_columns_hint` to prioritize columns for UI display.
- Ensure all queries are safe (SELECT or WITH statements only).
- Database Mapping (Conceptual Names): {database_table_mapping_info}
- Retrieved Knowledge Context: {retrieved_knowledge_context} (Relevant info from a knowledge base, used for metadata queries.)

**Database Overview (Derived from Schemas):**
- **Alnc Database (linc_data):** Contains long non-coding RNA (lncRNA) data across multiple species. Key columns include: `AlnC_ID`, `Species`, `Tissue`, `Non_coding_probability`, `Length`, `GC_content`, `Sequence`.
  - Notes: Multi-species data. Use `Non_coding_probability` for lncRNA classification, `Species` and `Tissue` for filtering.

**CRITICAL DATABASE-SPECIFIC RULES & DATA SEMANTICS:**
- **Cross-Database Joins:**
  - `anninter2` and `linc_data`: Join on `interactor1_seq` or `interactor2_seq` = `linc_data.Sequence`.
- **Species and Tissue Handling:**
  - Species: `linc_data` has a `Species` column.
  - Tissue: Only `linc_data` has a `Tissue` column.
- **Unsupported Molecules:** tRFs (tRNA-derived fragments) and siRNAs are NOT in these databases. Queries about them are out-of-scope. tRNA genes are supported via `PtRNAdb`.
- **Fuzzy Matching:** Use `LIKE` with wildcards (`%`) for text-based filters (e.g., `WHERE "Plant Source" LIKE '%Oryza sativa%'`).
- **Identifier Columns:**`AlnC_ID` (linc_data).
**Decision Process & Output Generation:**
1. **Confirm Intent & Scope:**
   - If `classified_intent` is "OUT_OF_SCOPE", set `analysis_plan` to "Query is unrelated to ncRNA data." and keep `queries` empty.
   - If the query mentions unsupported molecules (e.g., tRFs, siRNAs), set `analysis_plan` to "Query asks about [molecule], which is not covered."

2. **Database Targeting Strategy:**
   - **Default Grouping:**
     - "lncRNA" -> `linc_data`.
   - **Species Handling:** For `linc_data`, use `Species`.

3. **Response Strategy & SQL Planning:**
   - **METADATA_DIRECT_ANSWER_PREFERRED:**
     - `query_type`: "metadata_answered_from_blob".
     - `analysis_plan`: "Answering general information query..."
     - Populate `direct_answer_from_metadata` from `retrieved_knowledge_context`.
   - **METADATA_SQL_FALLBACK_PREFERRED:**
     - `query_type`: "metadata_sql_fallback".
     - `analysis_plan`: "Listing distinct '{{item_type}}'..."
     - `sql`: `SELECT DISTINCT "<column_name>" AS item_value FROM "<table>"...`.

4. **SQL Best Practices:**
   - Always quote table and column names (e.g., `SELECT "tRNA Sequence" FROM "PtRNAdb"`).

**Output JSON Structure:**
<JSON_START>
{{
  "query_type": "data_retrieval|metadata_answered_from_blob|metadata_sql_fallback|out_of_scope|ambiguous|typo|general_conversation",
  "analysis_plan": "Description of the AI's plan to address the query.",
  "direct_answer_from_metadata": "Answer from knowledge base for metadata queries, otherwise null.",
  "queries": [
    {{
      "sql": "SELECT ... FROM \"table_name\" ...",
      "target_table": "table_name",
      "database_conceptual_name": "Conceptual Name",
      "description": "Goal of this SQL query.",
      "purpose_type": "data_preview|entity_lookup_detail|aggregation|list_distinct_values|data_preview_ncRNA_intersection",
      "display_columns_hint": ["column1", "column2"]
    }}
  ]
}}
<JSON_END>

Generate ONLY the JSON object now, starting with <JSON_START> and ending with <JSON_END>.
"""
)

# --- MODIFIED: Updated Summary Interpretation Prompt ---
SUMMARY_INTERPRET_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "analysis_plan_from_stage1", "textual_data_summary_from_python",
        "knowledge_graph_snippet", "conversation_summary", "query_type_from_stage1",
        "DISPLAY_ROW_LIMIT"
    ],
    template="""
You are a friendly and expert bioinformatics assistant designed to provide concise, accurate, and user-friendly summaries for queries related to plant non-coding RNA (ncRNA) databases. Your entire response MUST be a single JSON object containing only the "summary" and "databases_conceptually_involved" fields. No additional text, explanations, or conversational filler is allowed.

*** You are a specialized chatbot for answering queries related to lncRNA, catering to the following database Alnc Database ***

**Objective**: Rephrase the provided "System-Generated Data Summary" into a concise, conversational paragraph that directly addresses the user's query, focusing on biological insights.

**Database Summaries (Based on Provided Sources):**
- **Alnc Database (linc_data)**: An extensive database of 10,855,598 lncRNAs across 682 flowering plant species. Key data includes `AlnC_ID`, `Species`, `Tissue`, and `Non_coding_probability`.

**Input Context:**
- User Query: {user_query}
- AI's Analysis Plan (from Stage 1): {analysis_plan_from_stage1}
- System-Generated Data Summary: {textual_data_summary_from_python} (Factual data report from SQL queries, to be rephrased.)
- Knowledge Graph Snippet: {knowledge_graph_snippet} (Contains semantic hints for grouping terms.)
- Recent Conversation Summary: {conversation_summary}
- Query Type (from Stage 1): {query_type_from_stage1}
- UI Display Row Limit: {DISPLAY_ROW_LIMIT}

**Task**: Generate a JSON response with a "summary" field that rephrases the "System-Generated Data Summary" into a conversational paragraph, and a "databases_conceptually_involved" field listing the conceptual database names involved.

**Instructions for "summary" Field:**
- Base the summary *solely* on the "System-Generated Data Summary". Do not invent facts.
- Use conceptual database names (e.g., AlnC) instead of internal table names.
- Avoid mentioning internal mechanics (e.g., query numbers, row counts unless directly relevant).
- For "No data records found" or "Error", state this factually (e.g., "No tRNA genes were found in PtRNAdb for this species.").
- Highlight key biological insights from "Key Column Stats" (e.g., dominant isoacceptors, significant scores).
- If the system summary includes a "Common 'ITEM_TYPE' Analysis", synthesize the findings (e.g., "The analysis identified 'oryza sativa' as a common species in both Alnc Database and PtRNAdb.").
- For lists of items (e.g., species from `AlnC`), present them clearly.
- If a preview is mentioned, note that a preview (up to {DISPLAY_ROW_LIMIT} rows) is available.
- For aggregations (COUNT, AVG), report the specific result (e.g., "The system counted 150 tRNA genes for the specified criteria.").

**Instructions for "databases_conceptually_involved" Field:**
- List unique conceptual database names (e.g., "AlnC") explicitly mentioned in the "System-Generated Data Summary".

**CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:**
- **Strictly Data-Driven**: Base the summary *only* on the "System-Generated Data Summary".
- **Professional Tone**: Use clear, formal language.
- **Formatting**: Replace underscores with spaces for readability (e.g., `Plant_Source` -> `Plant Source`).

**Output JSON Structure:**
{{
  "summary": "Rephrased conversational summary based on the System-Generated Data Summary.",
  "databases_conceptually_involved": ["Alnc Database", "Alnc"]
}}

Produce the JSON object now.
"""
)
def process_query(query: str, conversation_id: Optional[str] = None, model_name: str = "openai/gpt-oss-120b") -> Dict:
    processing_start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting query processing", extra={"query": query, "conversation_id": conversation_id})

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Generated conversation_id: {conversation_id}")

    try:
        sanitized_user_query = sanitize_query_input(query)
    except ValueError as e:
        logger.warning(f"[{request_id}] Invalid query: {str(e)}")
        return {
            "summary": f"Invalid query: {str(e)}",
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "error": str(e)
            }
        }

    # Initialize LLMs
    llm = ChatGroq(
        api_key=API_KEY, model_name=model_name, temperature=0.05,
        max_tokens=4096, request_timeout=60.0
    )
    # --- MODIFIED: Prepare for Stage 0 Intent Classification with PtRNAdb ---
    DATABASE_MAPPING = {
        "linc_data": "Alnc Database",
    }
    available_databases_summary_str = "\n".join([f"- {name}: {desc}" for name, desc in DATABASE_MAPPING.items()])
    database_content_summary_for_intent = """
    - Alnc Database: Contains lncRNAs across 682 plant species. Key columns: `AlnC_ID`, `Species`, `Tissue`, `Non_coding_probability`.
    - General: These databases focus on miRNAs, lncRNAs, tRNAs, and RNA interactions. tRFs and siRNAs are NOT included.
    """
    conversation_summary_for_intent = summarize_conversation(conversation_id, max_turns=1, max_len_per_item=50)

    intent_input_data = {
        "user_query": sanitized_user_query,
        "conversation_summary": conversation_summary_for_intent,
        "available_databases_summary": available_databases_summary_str,
        "database_content_summary": database_content_summary_for_intent
    }
    # ... (rest of the file from line 1832 to 1860 is unchanged) ...
    # Execute Stage 0 Intent Classification
    classified_intent = "UNKNOWN"
    try:
        logger.info(f"[{request_id}] Invoking LLM Stage 0 for Intent Classification...")
        s0_invoke_start = time.time()
        raw_intent_response = invoke_groq_model(INTENT_CLASSIFICATION_PROMPT_TEMPLATE | intent_llm, intent_input_data)
        s0_invoke_end = time.time()
        classified_intent = raw_intent_response.strip().upper().replace("\"", "")
        logger.info(f"[{request_id}] LLM Stage 0 Intent Classification completed in {s0_invoke_end - s0_invoke_start:.2f}s. Intent: {classified_intent}")
    except Exception as e_intent:
        logger.error(f"[{request_id}] Stage 0 Intent Classification error: {str(e_intent)}", exc_info=True)
        classified_intent = "DATA_RETRIEVAL"  # Fallback to data retrieval if intent classification fails

    # Conditional Vector DB Query
    retrieved_knowledge_context = ""
    if classified_intent in ["METADATA_DIRECT_ANSWER_PREFERRED", "METADATA_SQL_FALLBACK_PREFERRED"]:
        logger.info(f"[{request_id}] Intent is '{classified_intent}'. Querying vector DB for context.")
        retrieved_knowledge_context = get_relevant_context_from_vectordb(sanitized_user_query)
    else:
        logger.info(f"[{request_id}] Intent is '{classified_intent}'. Skipping vector DB search to prioritize SQL generation.")
        retrieved_knowledge_context = "Vector DB search was skipped for this query type. The query should be answered by generating SQL against the database schemas."

    # --- MODIFIED: Construct Knowledge Graph with PtRNAdb ---
    schemas_json_str = json.dumps(schema_manager.get_schema_for_prompt(), indent=2)
    database_table_mapping_info_str = "\n".join([f"- {name} (Conceptual: {desc})" for name, desc in DATABASE_MAPPING.items()])

    species_info_parts = []
    for table_name in schema_manager.get_tables():
        species_set = schema_manager.get_species(table_name)
        valid_species = {s for s in species_set if s and s.strip() and s.lower() not in ['na', '--na--', 'unknown', 'unspecified']}
        if valid_species:
            species_list = sorted(list(valid_species))
            display_species = species_list[:3]
            if len(species_list) > 3: display_species.append("...")
            conceptual_name = DATABASE_MAPPING.get(table_name, table_name)
            species_info_parts.append(f"- {conceptual_name} ({table_name}): {', '.join(display_species)}")
    species_info_str = "\n".join(species_info_parts) if species_info_parts else "No specific species data pre-loaded."

    tissues_info_parts = []
    for table_name in schema_manager.get_tables():
        tissues_set = schema_manager.get_tissues(table_name)
        valid_tissues = {t for t in tissues_set if t and t.strip() and t.lower() not in ['na', '--na--', 'unknown', 'unspecified']}
        if valid_tissues:
            tissues_list = sorted(list(valid_tissues))
            display_tissues = tissues_list[:3]
            if len(tissues_list) > 3: display_tissues.append("...")
            conceptual_name = DATABASE_MAPPING.get(table_name, table_name)
            tissues_info_parts.append(f"- {conceptual_name} ({table_name}): {', '.join(display_tissues)}")
    tissues_info_str = "\n".join(tissues_info_parts) if tissues_info_parts else "No specific tissue data pre-loaded. Only Alnc Database has a Tissue column."

    knowledge_graph_str = f"""
*** You are a specialized chatbot for answering ncRNA-related queries, focusing on miRNA, lncRNA, tRNA, and RNA interactions, using the available databases. ***
AlnC is an extensive database of long non-coding RNAs (lncRNAs) in Angiosperms. Here, we have incorporated a total of 10,855,598 lncRNAs, annotated in 809 RNA-seq samples of 682 flowering plants by using machine learning approach previously described by Singh et al., 2017. All transcriptome data available at 1KP project initiative was processed and used to annotate the lncRNAs. At AlnC, different kind of modules are available to search and download various kind (e.g. sample, tissue, sequence, structure, length etc.) of information related to lncRNAs in each plant species.
Concise Knowledge Graph for SQLite DB ({DB_FILE}):
- Tables & Primary Use (Conceptual Names from DATABASE_MAPPING):
  - {DATABASE_MAPPING.get("linc_data", "Alnc Database")}: Long non-coding RNAs across many plant species. Species col: `Species`. Tissue col: `Tissue`. Key cols: `AlnC_ID`, `Sequence`, `Non_coding_probability`.

- **Handling of Complex Concepts:**
  - **lncRNA Queries:** Use `linc_data`.
  - **Cross-Database Joins:** Join `athisomir` and `anninter2` on `Sequence` = `interactor1_seq`; join `anninter2` and `linc_data` on `interactor1_seq` = `Sequence`.
  - **Unsupported molecules:** tRFs (tRNA-derived fragments) and siRNAs are NOT in the databases.

- **Default Database Grouping for General Queries:**
  - "lncRNA" queries: Use `linc_data`.

- **Categories for Common Item Checks:**
  - "species": linc_data (`Species`)
  - "tissue": linc_data (`Tissue`); other tables have no tissue data unless joined

- **Species Name Hint:** For `linc_data`, use `LIKE 'Oryza sativa%'` for rice. For `PtRNAdb`, use `LIKE '%Oryza sativa%'` for rice.
- **Sample Species:** {species_info_str}
- **Sample Tissues:** {tissues_info_str}

- **Semantic Equivalence Hints for Common Item Analysis:**
  - **Species:** 'Arabidopsis thaliana', 'Arabidopsis' are equivalent. 'Oryza sativa', 'Oryza sativa Japonica Group' map to 'rice'.
  - **Tissues:** 'leaf', 'rosette leaf' map to 'leaf'. 'flower', 'inflorescence' map to 'flower/floral organs'.
"""
# ... (rest of the file from line 1970 to the end is unchanged, as the logic is generic enough to handle the new database via the schema and prompts) ...
    # Prepare Stage 1 Input
    context = CONTEXT_CACHE.get(conversation_id, {})
    DISPLAY_ROW_LIMIT = 10  # Example value, adjust as needed
    previous_query_summary_str = json.dumps(context.get('last_response_summary_for_llm', {}))
    conversation_summary_str = summarize_conversation(conversation_id, max_turns=2, max_len_per_item=100)

    stage1_input_data = {
        "user_query": sanitized_user_query,
        "classified_intent": classified_intent,
        "db_file_name": DB_FILE,
        "schemas_json": schemas_json_str,
        "knowledge_graph": knowledge_graph_str,
        "previous_query_summary": previous_query_summary_str,
        "conversation_summary": conversation_summary_str,
        "database_table_mapping_info": database_table_mapping_info_str,
        "DISPLAY_ROW_LIMIT": DISPLAY_ROW_LIMIT,
        "retrieved_knowledge_context": retrieved_knowledge_context
    }

    # Execute Stage 1 (SQL Planning)
    parsed_stage1_response = None
    try:
        logger.info(f"[{request_id}] Invoking LLM Stage 1 for planning...")
        s1_invoke_start = time.time()
        raw_stage1_response_str = invoke_groq_model(SQL_PLAN_PROMPT_TEMPLATE | llm, stage1_input_data)
        s1_invoke_end = time.time()
        logger.info(f"[{request_id}] LLM Stage 1 invocation completed in {s1_invoke_end - s1_invoke_start:.2f}s.")

        parsed_stage1_response = parse_json_response(raw_stage1_response_str)
        if parsed_stage1_response is None:
            logger.error(f"[{request_id}] Stage 1 LLM response could not be parsed into JSON.", extra={"raw_response": raw_stage1_response_str})
            return {
                "summary": "I encountered an issue while planning how to answer your query. Please try rephrasing or ask a simpler question.",
                "executed_queries_details": [],
                "databases_conceptually_involved": [],
                "metadata": {
                    "execution_time": round(time.time() - processing_start_time, 3),
                    "total_rows_retrieved_by_sql_across_all_queries": 0,
                    "conversation_id": conversation_id,
                    "error": "LLM Stage 1 response format error (not JSON)",
                    "classified_intent": classified_intent
                }
            }
        if "query_type" not in parsed_stage1_response or "analysis_plan" not in parsed_stage1_response:
            logger.error(f"[{request_id}] Stage 1 response missing 'query_type' or 'analysis_plan'", extra={"response_obj": parsed_stage1_response})
            return {
                "summary": "The AI's plan for your query was incomplete. Please try rephrasing your query.",
                "executed_queries_details": [],
                "databases_conceptually_involved": [],
                "metadata": {
                    "execution_time": round(time.time() - processing_start_time, 3),
                    "total_rows_retrieved_by_sql_across_all_queries": 0,
                    "conversation_id": conversation_id,
                    "error": "Stage 1 planning response incomplete fields",
                    "classified_intent": classified_intent
                }
            }
    except Exception as e:
        logger.error(f"[{request_id}] Stage 1 processing error: {str(e)}", exc_info=True)
        summary_msg = (
            "The system is currently experiencing high load for query planning. Please try again in a few moments."
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e)
            else f"Sorry, an error occurred during query planning ({type(e).__name__}). Please try rephrasing."
        )
        return {
            "summary": summary_msg,
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "error": f"Stage 1 error: {str(e)}",
                "classified_intent": classified_intent
            }
        }

    query_type_from_stage1 = parsed_stage1_response.get("query_type", classified_intent)
    analysis_plan_from_stage1 = parsed_stage1_response.get("analysis_plan", "No analysis plan from AI.")
    direct_answer_from_metadata = parsed_stage1_response.get("direct_answer_from_metadata")
    llm_generated_queries = parsed_stage1_response.get("queries", [])

    # Handle Metadata Direct Answer
    if classified_intent == "METADATA_DIRECT_ANSWER_PREFERRED" and \
       query_type_from_stage1 not in ["metadata_answered_from_blob", "metadata_sql_fallback", "DATA_RETRIEVAL", "entity_lookup_detail"]:
        query_type_from_stage1 = "metadata_answered_from_blob"

    if query_type_from_stage1 == "metadata_answered_from_blob":
        final_summary_text = ""
        if direct_answer_from_metadata and direct_answer_from_metadata.strip():
            final_summary_text = direct_answer_from_metadata
            logger.info(f"[{request_id}] Using direct_answer_from_metadata from Stage 1.")
        else:
            logger.warning(f"[{request_id}] 'metadata_answered_from_blob' planned but 'direct_answer_from_metadata' missing/empty. Python fallback for: {sanitized_user_query}")
            extracted_metadata_fallback = ""
            user_query_lower = sanitized_user_query.lower()

            db_keywords_found_in_query = {}
            for db_code, db_name_conceptual in DATABASE_MAPPING.items():
                if db_code.lower() in user_query_lower or db_name_conceptual.lower() in user_query_lower:
                    db_keywords_found_in_query[db_code] = db_name_conceptual

            concept_keywords_map = {
                "lncrna": ["lncRNA (long non-coding RNA)", "Alnc Database"],
            }
            found_concept_blob_keywords = []
            for query_kw, blob_hints in concept_keywords_map.items():
                if query_kw in user_query_lower:
                    found_concept_blob_keywords.extend(blob_hints)

            relevant_blob_parts_dict = {}
            DATABASE_METADATA_KNOWLEDGE_BLOB = """
*   **Alnc Database (linc_data):** Contains 10,855,598 lncRNAs across 682 plant species from 809 RNA-seq samples. Key data: `AlnC_ID`, `Species`, `Tissue`, `Non_coding_probability`, `Length`, `GC_content`.
"""
            blob_sections = DATABASE_METADATA_KNOWLEDGE_BLOB.split('\n\n*   **')
            blob_sections = ["*   **" + s for s in blob_sections if s.strip()]

            if db_keywords_found_in_query:
                for db_code, db_name in db_keywords_found_in_query.items():
                    pattern_str = rf"^\*\s+\*\*{re.escape(db_name)}\s*\({re.escape(db_code)}\):?\*\*"
                    for section in blob_sections:
                        if re.search(pattern_str, section, re.IGNORECASE | re.MULTILINE):
                            section_content = section.split("Key data:")[0].strip()
                            section_content = section_content.replace("*   **", "").replace("**:", ":").replace("**", "").strip()
                            relevant_blob_parts_dict[db_code] = section_content
                            break

            if not relevant_blob_parts_dict and found_concept_blob_keywords:
                for concept_hint in found_concept_blob_keywords:
                    for section in blob_sections:
                        if concept_hint.lower() in section.lower():
                            first_sentence_match = re.search(r"^\-\s+\*\*(.*?)\*\*:\s*(.*?)\.", section, re.IGNORECASE | re.MULTILINE)
                            if first_sentence_match and concept_hint.lower() in first_sentence_match.group(1).lower():
                                relevant_blob_parts_dict[concept_hint] = first_sentence_match.group(0).replace("- **","").replace("**","").strip()
                                break
                            elif concept_hint.lower() in section.lower():
                                relevant_blob_parts_dict[concept_hint] = section.split('.')[0].replace("*   **", "").replace("**:", ":").replace("**", "").strip() + "."
                                break
                    if relevant_blob_parts_dict: break

            if relevant_blob_parts_dict:
                extracted_metadata_fallback = " ".join(list(set(relevant_blob_parts_dict.values())))
                logger.info(f"[{request_id}] Python fallback provided metadata snippet: {extracted_metadata_fallback[:300]}")
                final_summary_text = extracted_metadata_fallback
            else:
                final_summary_text = f"I understood you were asking for general information (plan: '{analysis_plan_from_stage1}'), but I couldn't find a specific pre-compiled answer. Could you try asking for specific data, like miRNAs or lncRNAs?"

        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": analysis_plan_from_stage1[:150],
                "agent_summary_of_findings": final_summary_text[:350],
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": bool(final_summary_text and "couldn't find" not in final_summary_text.lower())
            }
            context_data['history'].append({"query": sanitized_user_query, "summary_preview": final_summary_text[:200] + "..."})
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": final_summary_text.strip(),
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "llm_analysis_plan": analysis_plan_from_stage1,
                "query_type_processed": "metadata_answered_from_blob",
                "classified_intent_s0": classified_intent
            }
        }

    # Handle Ambiguous or Typo Queries
    if query_type_from_stage1 in ["ambiguous_query", "typo_clarification", "AMBIGUOUS", "TYPO"]:
        final_summary_text = analysis_plan_from_stage1
        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": f"{query_type_from_stage1.replace('_', ' ').title()}: " + analysis_plan_from_stage1[:150],
                "agent_summary_of_findings": final_summary_text[:350],
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": False
            }
            context_data['history'].append({"query": sanitized_user_query, "summary_preview": final_summary_text[:200] + "..."})
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": final_summary_text.strip(),
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "llm_analysis_plan": analysis_plan_from_stage1,
                "query_type_processed": query_type_from_stage1,
                "classified_intent_s0": classified_intent
            }
        }

    # Handle Out-of-Scope Queries
    if classified_intent == "OUT_OF_SCOPE":
        logger.info(f"[{request_id}] Query classified as OUT_OF_SCOPE.")
        summary_text = "I specialize in plant genomics data related to miRNAs, lncRNAs, and RNA interactions. I'm unable to answer questions about tRFs, siRNAs, or other topics outside this domain."
        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": "Query out of scope",
                "agent_summary_of_findings": summary_text,
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": False
            }
            context_data['history'].append({"query": sanitized_user_query, "summary_preview": "Out of scope..."})
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": summary_text,
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "classified_intent": classified_intent,
                "llm_analysis_plan": "Query determined out of scope."
            }
        }

    # Handle General Conversation
    if classified_intent == "GENERAL_CONVERSATION":
        logger.info(f"[{request_id}] Query classified as GENERAL_CONVERSATION.")
        bot_response = "Hello! How can I help you with miRNA, lncRNA, or RNA interaction data today?"
        if any(kw in sanitized_user_query.lower() for kw in ["thank", "bye", "that's all"]):
            bot_response = "You're welcome! Feel free to ask more questions about miRNAs or lncRNAs."
        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": "General conversation",
                "agent_summary_of_findings": bot_response,
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": False
            }
            context_data['history'].append({"query": sanitized_user_query, "summary_preview": bot_response[:100] + "..."})
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": bot_response,
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "classified_intent": classified_intent,
                "llm_analysis_plan": "General conversation handling."
            }
        }

    # Execute SQL Queries and Process Results
    if not llm_generated_queries:
        logger.error(f"[{request_id}] No SQL queries generated by Stage 1. S0 Intent: {classified_intent}, S1 query_type: {query_type_from_stage1}, Plan: '{analysis_plan_from_stage1}'")
        summary_msg = f"I understood your query as: '{analysis_plan_from_stage1}', but I couldn't formulate a specific database search. Could you try rephrasing?"
        if "trf" in sanitized_user_query.lower() or "sirna" in sanitized_user_query.lower():
            summary_msg = "I focus on miRNAs, lncRNAs, and RNA interactions. Data for tRFs or siRNAs is not available in these databases."

        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": analysis_plan_from_stage1[:150],
                "agent_summary_of_findings": summary_msg[:350],
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": False
            }
            context_data['history'].append({"query": sanitized_user_query, "summary_preview": summary_msg[:200] + "..."})
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": summary_msg,
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {
                "execution_time": round(time.time() - processing_start_time, 3),
                "total_rows_retrieved_by_sql_across_all_queries": 0,
                "conversation_id": conversation_id,
                "llm_analysis_plan": analysis_plan_from_stage1,
                "error": "SQL planned but not generated",
                "query_type_processed": query_type_from_stage1,
                "classified_intent_s0": classified_intent
            }
        }

    logger.info(f"[{request_id}] Stage 1 Plan (Type: {query_type_from_stage1}): '{analysis_plan_from_stage1}'. Generated {len(llm_generated_queries)} SQL queries.")

    concise_executed_queries_stats_for_py_summary = []
    all_results_for_display_and_csv = []
    grand_total_rows_retrieved_by_sql = 0
    any_query_produced_data = False
    common_items_data_store = {"species": {}, "tissue": {}}
    requested_common_item_checks = {"species": False, "tissue": False}

    for q_info_loop in llm_generated_queries:
        pt = q_info_loop.get("purpose_type", "")
        if pt == "list_distinct_values_for_common_check_species":
            requested_common_item_checks["species"] = True
        elif pt == "list_distinct_values_for_common_check_tissue":
            requested_common_item_checks["tissue"] = True

    for i, query_info_from_llm in enumerate(llm_generated_queries):
        sql_statement = query_info_from_llm.get("sql")
        target_table_from_llm = query_info_from_llm.get("target_table")
        db_conceptual_name_from_llm = query_info_from_llm.get("database_conceptual_name")

        if not target_table_from_llm or target_table_from_llm == "UnknownTable":
            logger.warning(f"[{request_id}] Query {i+1} missing 'target_table'. SQL: {sql_statement}")
            target_table_from_llm = "UnknownTable"

        if not db_conceptual_name_from_llm or db_conceptual_name_from_llm in ["Unknown Database", target_table_from_llm]:
            temp_conceptual_name = DATABASE_MAPPING.get(target_table_from_llm, target_table_from_llm)
            if temp_conceptual_name == target_table_from_llm and target_table_from_llm not in DATABASE_MAPPING.values():
                db_conceptual_name_from_llm = "Unknown Database"
            else:
                db_conceptual_name_from_llm = temp_conceptual_name
            logger.info(f"[{request_id}] Query {i+1} 'database_conceptual_name' resolved to '{db_conceptual_name_from_llm}'.")

        ui_card_target_name = db_conceptual_name_from_llm
        if ui_card_target_name in ["Unknown Database", "UnknownTable"]:
            ui_card_target_name = DATABASE_MAPPING.get(target_table_from_llm, "Target N/A")

        query_description_from_llm = query_info_from_llm.get("description", "No description provided by AI plan.")
        purpose_type = query_info_from_llm.get("purpose_type", "general_query")
        display_cols_hint_from_query_plan = query_info_from_llm.get("display_columns_hint", [])
        overall_display_cols_hint = parsed_stage1_response.get("display_columns_hint", [])

        current_query_display_details = {
            "sql": sql_statement,
            "target_for_ui": ui_card_target_name,
            "_internal_table_sqlite_name": target_table_from_llm,
            "_internal_database_conceptual_name": db_conceptual_name_from_llm,
            "description": query_description_from_llm,
            "results_preview": [],
            "preview_table": [],
            "download_url": "",
            "error": None,
            "row_count_from_sql": 0,
            "purpose_type_from_llm": purpose_type
        }
        stats_item_for_py_summary = {
            "original_sql": sql_statement,
            "target_table": target_table_from_llm,
            "database_conceptual_name": db_conceptual_name_from_llm,
            "description_from_llm": query_description_from_llm,
            "purpose_type": purpose_type,
            "total_rows_found": 0,
            "key_column_statistics": {},
            "preview_for_ui_only": [],
            "statistics_based_on_sample": False,
            "error_if_any": None
        }

        if not sql_statement:
            error_msg = "Empty SQL from AI planner."
            current_query_display_details["error"] = error_msg
            stats_item_for_py_summary["error_if_any"] = error_msg
            item_type_for_failed_check = None
            if purpose_type == "list_distinct_values_for_common_check_species":
                item_type_for_failed_check = "species"
            elif purpose_type == "list_distinct_values_for_common_check_tissue":
                item_type_for_failed_check = "tissue"
            if item_type_for_failed_check and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                common_items_data_store[item_type_for_failed_check][db_conceptual_name_from_llm] = f"Error: {error_msg}"
        else:
            actual_results = []
            try:
                actual_results, _ = execute_sql_query(sql_statement, target_table_from_llm)
                if isinstance(actual_results, str):
                    current_query_display_details["error"] = actual_results
                    stats_item_for_py_summary["error_if_any"] = actual_results
                    item_type_for_failed_check = None
                    if purpose_type == "list_distinct_values_for_common_check_species":
                        item_type_for_failed_check = "species"
                    elif purpose_type == "list_distinct_values_for_common_check_tissue":
                        item_type_for_failed_check = "tissue"
                    if item_type_for_failed_check and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                        common_items_data_store[item_type_for_failed_check][db_conceptual_name_from_llm] = f"Error: {actual_results}"
                else:
                    any_query_produced_data = True if actual_results else any_query_produced_data
                    full_stats_obj = generate_statistics_from_results(actual_results, f"[{request_id}]_query_{i+1}_{target_table_from_llm}")

                    current_query_display_details["row_count_from_sql"] = full_stats_obj["count"]
                    current_query_display_details["results_preview"] = full_stats_obj["preview"]

                    stats_item_for_py_summary["total_rows_found"] = full_stats_obj["count"]
                    stats_item_for_py_summary["statistics_based_on_sample"] = full_stats_obj["stats_based_on_sample"]
                    stats_item_for_py_summary["preview_for_ui_only"] = full_stats_obj["preview"]

                    concise_col_stats = {}
                    if full_stats_obj["column_stats"]:
                        for col, S_stats in full_stats_obj["column_stats"].items():
                            col_stat_summary = {}
                            if 'non_null_count' in S_stats:
                                col_stat_summary['non_null_count'] = S_stats['non_null_count']
                            if 'null_count' in S_stats and S_stats['null_count'] > 0:
                                col_stat_summary['null_count'] = S_stats['null_count']
                            if 'distinct_count' in S_stats:
                                col_stat_summary['distinct_count'] = S_stats['distinct_count']
                            if 'mean' in S_stats and S_stats['mean'] is not None:
                                col_stat_summary['mean'] = round(S_stats['mean'], 2) if isinstance(S_stats['mean'], float) else S_stats['mean']
                            if 'median' in S_stats and S_stats['median'] is not None:
                                col_stat_summary['median'] = round(S_stats['median'], 2) if isinstance(S_stats['median'], float) else S_stats['median']
                            if 'min' in S_stats and S_stats['min'] is not None:
                                col_stat_summary['min'] = S_stats['min']
                            if 'max' in S_stats and S_stats['max'] is not None:
                                col_stat_summary['max'] = S_stats['max']
                            if 'sum' in S_stats and S_stats['sum'] is not None and purpose_type == 'aggregation' and full_stats_obj["count"] == 1 and 'COUNT(' in col.upper():
                                col_stat_summary['sum'] = S_stats['sum']
                            if 'top_values' in S_stats and S_stats['top_values']:
                                col_stat_summary['top_values'] = dict(list(S_stats['top_values'].items())[:3])
                            if col_stat_summary:
                                concise_col_stats[col] = col_stat_summary
                    stats_item_for_py_summary["key_column_statistics"] = concise_col_stats
                    grand_total_rows_retrieved_by_sql += full_stats_obj["count"]

                    item_type_being_checked = None
                    if purpose_type == "list_distinct_values_for_common_check_species":
                        item_type_being_checked = "species"
                    elif purpose_type == "list_distinct_values_for_common_check_tissue":
                        item_type_being_checked = "tissue"

                    if item_type_being_checked and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                        distinct_values_for_this_db = set()
                        if actual_results:
                            for row_dict in actual_results:
                                val = row_dict.get('item_value')
                                if val is not None and str(val).strip().lower() not in ["", "na", "--na--", "unknown", "unspecified"]:
                                    distinct_values_for_this_db.add(str(val).strip())
                        common_items_data_store[item_type_being_checked][db_conceptual_name_from_llm] = distinct_values_for_this_db

                    if full_stats_obj["preview"]:
                        preview_df = pd.DataFrame(full_stats_obj["preview"])
                        if not preview_df.empty:
                            cols_to_show_final = []
                            active_display_cols_hint = display_cols_hint_from_query_plan if display_cols_hint_from_query_plan else overall_display_cols_hint

                            if active_display_cols_hint:
                                cols_to_show_final = [col_df for col_df in active_display_cols_hint if col_df in preview_df.columns]
                            if not cols_to_show_final:
                                cols_to_show_final = list(preview_df.columns[:min(5, len(preview_df.columns))])

                            if cols_to_show_final:
                                current_query_display_details["preview_table"] = preview_df[cols_to_show_final].to_dict('records')
                            elif not preview_df.empty:
                                current_query_display_details["preview_table"] = preview_df[[preview_df.columns[0]]].to_dict('records')

                    if actual_results:
                        csv_filename_context = ui_card_target_name if ui_card_target_name not in ["Target N/A", "Unknown Database"] else target_table_from_llm
                        csv_url = generate_csv(actual_results, f"query_{i+1}_{csv_filename_context.replace(' ', '_')}_{purpose_type}")
                        if csv_url:
                            current_query_display_details["download_url"] = csv_url

            except Exception as e_exec_stats:
                error_msg = f"System error during execution/stats: {str(e_exec_stats)}"
                logger.error(f"[{request_id}] Error in SQL exec/stats loop for query {i}: {error_msg}", exc_info=True)
                current_query_display_details["error"] = error_msg
                stats_item_for_py_summary["error_if_any"] = error_msg
                item_type_for_failed_check = None
                if purpose_type == "list_distinct_values_for_common_check_species":
                    item_type_for_failed_check = "species"
                elif purpose_type == "list_distinct_values_for_common_check_tissue":
                    item_type_for_failed_check = "tissue"
                if item_type_for_failed_check and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                    common_items_data_store[item_type_for_failed_check][db_conceptual_name_from_llm] = f"Error: {error_msg}"

        concise_executed_queries_stats_for_py_summary.append(stats_item_for_py_summary)
        all_results_for_display_and_csv.append(current_query_display_details)

    logger.info(f"[{request_id}] Finished SQL execution. Total rows from successful queries: {grand_total_rows_retrieved_by_sql}")

    # Common Item Analysis
    final_common_items_reports_for_summary = []
    athisomir_conceptual_name = DATABASE_MAPPING.get("athisomir")
    anninter2_conceptual_name = DATABASE_MAPPING.get("anninter2")

    for item_type_key, collected_data_for_item_type in common_items_data_store.items():
        if not requested_common_item_checks.get(item_type_key):
            continue

        all_dbs_planned_for_this_item_check = set()
        for q_info_loop_2 in llm_generated_queries:
            if q_info_loop_2.get("purpose_type") == f"list_distinct_values_for_common_check_{item_type_key}":
                planned_target_table_2 = q_info_loop_2.get("target_table", "UnknownTable")
                planned_conceptual_name_2 = q_info_loop_2.get("database_conceptual_name")

                if not planned_conceptual_name_2 or planned_conceptual_name_2 in ["Unknown Database", planned_target_table_2]:
                    planned_conceptual_name_2 = DATABASE_MAPPING.get(planned_target_table_2, planned_target_table_2)
                    if planned_conceptual_name_2 == planned_target_table_2 and planned_target_table_2 == "UnknownTable":
                        planned_conceptual_name_2 = "Unknown Database"

                if planned_conceptual_name_2 and planned_conceptual_name_2 not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                    all_dbs_planned_for_this_item_check.add(planned_conceptual_name_2)

        if item_type_key == "species" and athisomir_conceptual_name and athisomir_conceptual_name in all_dbs_planned_for_this_item_check:
            if athisomir_conceptual_name not in collected_data_for_item_type or not collected_data_for_item_type[athisomir_conceptual_name]:
                collected_data_for_item_type[athisomir_conceptual_name] = {'Arabidopsis thaliana'}
        if item_type_key == "species" and anninter2_conceptual_name and anninter2_conceptual_name in all_dbs_planned_for_this_item_check:
            if anninter2_conceptual_name not in collected_data_for_item_type or not collected_data_for_item_type[anninter2_conceptual_name]:
                collected_data_for_item_type[anninter2_conceptual_name] = {'Arabidopsis thaliana'}

        report = {
            "item_type": item_type_key,
            "note": None,
            "databases_queried_for_this_item": sorted(list(all_dbs_planned_for_this_item_check)),
            "distinct_values_per_database_for_this_item": {}
        }

        num_succeeded_for_this_item_type = 0
        num_errors_or_unavailable_for_this_item_type = 0
        any_distinct_values_found_across_dbs = False

        for db_name in report["databases_queried_for_this_item"]:
            db_values_or_error = collected_data_for_item_type.get(db_name)

            if isinstance(db_values_or_error, set):
                report["distinct_values_per_database_for_this_item"][db_name] = sorted(list(db_values_or_error))
                if db_values_or_error:
                    any_distinct_values_found_across_dbs = True
                num_succeeded_for_this_item_type += 1
            elif isinstance(db_values_or_error, str) and ("Error:" in db_values_or_error or "Issue:" in db_values_or_error):
                report["distinct_values_per_database_for_this_item"][db_name] = db_values_or_error
                num_errors_or_unavailable_for_this_item_type += 1
            else:
                report["distinct_values_per_database_for_this_item"][db_name] = f"Data not retrieved or none found for {db_name}."
                num_errors_or_unavailable_for_this_item_type += 1

        total_planned_dbs_for_item = len(report["databases_queried_for_this_item"])
        if not report["databases_queried_for_this_item"] and requested_common_item_checks.get(item_type_key):
            report["note"] = f"A commonality check for '{item_type_key}' was intended, but no specific databases were successfully identified or targeted in the AI's plan."
        elif num_errors_or_unavailable_for_this_item_type > 0:
            if num_succeeded_for_this_item_type > 0:
                report["note"] = (
                    f"Analysis for '{item_type_key}' commonality involved {total_planned_dbs_for_item} database(s). "
                    f"Data was successfully processed for {num_succeeded_for_this_item_type}. "
                    f"However, issues or no data occurred for {num_errors_or_unavailable_for_this_item_type} database(s). Commonality assessment by the AI will be based on available data."
                )
            else:
                report["note"] = (
                    f"Analysis for '{item_type_key}' commonality involved {total_planned_dbs_for_item} database(s). "
                    f"Unfortunately, data retrieval failed or no specific data was found for all of them. Cannot determine common '{item_type_key}'s."
                )
        elif total_planned_dbs_for_item > 0 and not any_distinct_values_found_across_dbs:
            report["note"] = f"No specific distinct '{item_type_key}' values were found in any of the {total_planned_dbs_for_item} targeted database(s) to compare for commonality."

        if report["databases_queried_for_this_item"] or (requested_common_item_checks.get(item_type_key) and report["note"]):
            final_common_items_reports_for_summary.append(report)

    # Generate Textual Summary for Stage 2
    textual_data_summary_for_stage2 = generate_textual_summary_from_stats(
        concise_executed_queries_stats_for_py_summary,
        sanitized_user_query,
        analysis_plan_from_stage1,
        common_items_info_list=final_common_items_reports_for_summary if final_common_items_reports_for_summary else None
    )
    logger.debug(f"[{request_id}] Textual summary for Stage 2 LLM (length {len(textual_data_summary_for_stage2)}):\n{textual_data_summary_for_stage2[:1000]}...")

    # Prepare Stage 2 Input
    stage2_input_data = {
        "user_query": sanitized_user_query,
        "analysis_plan_from_stage1": analysis_plan_from_stage1,
        "textual_data_summary_from_python": textual_data_summary_for_stage2,
        "knowledge_graph_snippet": knowledge_graph_str[:3000],
        "conversation_summary": conversation_summary_str,
        "query_type_from_stage1": query_type_from_stage1,
        "DISPLAY_ROW_LIMIT": DISPLAY_ROW_LIMIT
    }

    # Execute Stage 2 (Summary Interpretation)
    final_summary_text_from_stage2 = "I processed your request."
    final_databases_involved_from_stage2 = set()

    for item_stat in concise_executed_queries_stats_for_py_summary:
        db_name_for_involvement = item_stat.get("database_conceptual_name")
        matching_detail = next((detail for detail in all_results_for_display_and_csv
                               if detail["_internal_table_sqlite_name"] == item_stat.get("target_table") and
                                  detail["_internal_database_conceptual_name"] == db_name_for_involvement and
                                  detail.get("sql") == item_stat.get("original_sql")), None)
        if matching_detail and matching_detail.get("target_for_ui") not in ["Target N/A", "Unknown Database"]:
            db_name_for_involvement = matching_detail.get("target_for_ui")
        elif db_name_for_involvement in ["UnknownTable", "Unknown Database", None]:
            continue
        if not item_stat.get("error_if_any"):
            final_databases_involved_from_stage2.add(db_name_for_involvement)

    if final_common_items_reports_for_summary:
        for report_item in final_common_items_reports_for_summary:
            for db_name, data in report_item.get("distinct_values_per_database_for_this_item", {}).items():
                if isinstance(data, list) and db_name not in ["Target N/A", "Unknown Database"]:
                    final_databases_involved_from_stage2.add(db_name)

    if not any_query_produced_data and not final_common_items_reports_for_summary and llm_generated_queries:
        final_summary_text_from_stage2 = f"No specific data records were found for your query: '{sanitized_user_query}'. My plan was: '{analysis_plan_from_stage1}'."
    elif not llm_generated_queries and not final_common_items_reports_for_summary and query_type_from_stage1 != "metadata_answered_from_blob":
        final_summary_text_from_stage2 = f"I was unable to perform specific data lookups or comparisons based on your query: '{sanitized_user_query}'. My plan was: '{analysis_plan_from_stage1}'."
    elif query_type_from_stage1 == "metadata_answered_from_blob" and final_summary_text_from_stage2 == "I processed your request.":
        final_summary_text_from_stage2 = f"I understood you were asking for information, but I couldn't formulate a direct response."

    if llm_generated_queries or final_common_items_reports_for_summary:
        try:
            logger.info(f"[{request_id}] Invoking LLM Stage 2 for rephrasing textual stats summary...")
            s2_invoke_start = time.time()
            raw_stage2_response_str = invoke_groq_model(SUMMARY_INTERPRET_PROMPT_TEMPLATE | llm, stage2_input_data)
            s2_invoke_end = time.time()
            logger.info(f"[{request_id}] LLM Stage 2 invocation completed in {s2_invoke_end - s2_invoke_start:.2f}s.")

            parsed_stage2_response = parse_json_response(raw_stage2_response_str)
            if parsed_stage2_response is None:
                logger.error(f"[{request_id}] Stage 2 LLM response could not be parsed into JSON.", extra={"raw_response": raw_stage2_response_str})
                final_summary_text_from_stage2 = "I've processed the data. Here's a direct summary of the findings:\n" + textual_data_summary_for_stage2
                if len(final_summary_text_from_stage2) > 1500:
                    final_summary_text_from_stage2 = final_summary_text_from_stage2[:1497] + "..."
            elif "summary" in parsed_stage2_response:
                final_summary_text_from_stage2 = parsed_stage2_response["summary"]
                db_list_from_llm_s2 = parsed_stage2_response.get("databases_conceptually_involved", [])
                if db_list_from_llm_s2:
                    final_databases_involved_from_stage2.update(d for d in db_list_from_llm_s2 if d and d not in ["Target N/A", "Unknown Database"])
            else:
                logger.error(f"[{request_id}] Stage 2 response missing 'summary' field.", extra={"response_obj": parsed_stage2_response})
                final_summary_text_from_stage2 = f"I can see the data based on your query, but I'm having trouble creating a natural language summary. Here's a structured view of what was found:\n{textual_data_summary_for_stage2[:500]}..."
        except Exception as e_stage2:
            logger.error(f"[{request_id}] Stage 2 summarization processing error: {str(e_stage2)}", exc_info=True)
            final_summary_text_from_stage2 = (
                f"The system is currently experiencing high load for summarization. Here's a structured view of what was found:\n{textual_data_summary_for_stage2[:500]}..."
                if "rate_limit_exceeded" in str(e_stage2).lower() or "429" in str(e_stage2)
                else f"An error occurred during final summary generation ({type(e_stage2).__name__}). The system found: {textual_data_summary_for_stage2[:300]}..."
            )

    if conversation_id:
        context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
        context_data['last_query'] = sanitized_user_query
        overall_data_found_for_context = any_query_produced_data
        if not overall_data_found_for_context and final_common_items_reports_for_summary:
            for report in final_common_items_reports_for_summary:
                if any(isinstance(val_list, list) and val_list for val_list in report.get("distinct_values_per_database_for_this_item", {}).values()):
                    overall_data_found_for_context = True
                    break

        context_data['last_response_summary_for_llm'] = {
            "user_query_that_led_to_this_summary": sanitized_user_query,
            "agent_analysis_plan_executed": analysis_plan_from_stage1[:300],
            "agent_summary_of_findings": final_summary_text_from_stage2[:400],
            "databases_queried_in_this_turn": sorted(list(d for d in final_databases_involved_from_stage2 if d and d not in ["Target N/A", "Unknown Database"])),
            "data_found_overall_in_this_turn": overall_data_found_for_context
        }
        context_data['history'].append({"query": sanitized_user_query, "summary_preview": final_summary_text_from_stage2[:250] + "..."})
        if len(context_data['history']) > 5: context_data['history'].pop(0)
        CONTEXT_CACHE[conversation_id] = context_data

    final_response = {
        "summary": final_summary_text_from_stage2.strip(),
        "executed_queries_details": all_results_for_display_and_csv,
        "databases_conceptually_involved": sorted(list(d for d in final_databases_involved_from_stage2 if d and d not in ["Target N/A", "Unknown Database"])),
        "metadata": {
            "execution_time": round(time.time() - processing_start_time, 3),
            "total_rows_retrieved_by_sql_across_all_queries": grand_total_rows_retrieved_by_sql,
            "conversation_id": conversation_id,
            "llm_analysis_plan": analysis_plan_from_stage1,
            "query_type_processed": query_type_from_stage1,
            "classified_intent_s0": classified_intent
        }
    }
    logger.info(f"[{request_id}] Query processing finished for '{sanitized_user_query}'. Exec time: {final_response['metadata']['execution_time']}s. S0 Intent: {classified_intent}, S1 Type: {query_type_from_stage1}")
    return final_response

@app.route('/query', methods=['POST'])
@limiter.limit("100 per hour")
def query_endpoint():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("Missing 'query' key in JSON payload")
            return jsonify({"error": "Missing 'query' parameter in JSON payload"}), 400
        
        user_query = data['query']
        conversation_id = data.get('conversation_id')

        if not isinstance(user_query, str) or not user_query.strip():
            logger.warning("Invalid query type or empty query")
            return jsonify({"error": "Query must be a non-empty string"}), 400
        
        response_data = process_query(user_query, conversation_id=conversation_id)
        clean_response_data = clean_nan_and_inf(response_data)
        return jsonify(clean_response_data)
        
    except Exception as e:
        logger.error(f"Unhandled error in /query endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected internal server error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    db_status = "disconnected"
    db_error = "Health check not fully completed."
    try:
        db_path_full = os.path.join(DB_PATH, DB_FILE)
        if not os.path.exists(db_path_full):
            raise FileNotFoundError(f"Database file not found at {db_path_full}")
        conn = sqlite3.connect(f"file:{db_path_full}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        cursor.fetchone()
        conn.close()
        db_status = "connected"
        db_error = None
        return jsonify({
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        })
    except FileNotFoundError as e:
        db_status = "file_not_found"
        db_error = str(e)
        logger.error(f"Health check DB file error: {db_error}", exc_info=True)
    except sqlite3.Error as e:
        db_status = "sqlite_error"
        db_error = f"SQLite error: {str(e)}"
        logger.error(f"Health check DB error: {db_error}", exc_info=True)
    except Exception as e:
        db_status = "unexpected_error"
        db_error = f"Unexpected error: {str(e)}"
        logger.error(f"Health check unexpected error: {db_error}", exc_info=True)
    
    return jsonify({
        "status": "unhealthy",
        "database_status": db_status,
        "error_details": db_error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

