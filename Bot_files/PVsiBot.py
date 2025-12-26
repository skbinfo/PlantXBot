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
DB_FILE = 'all_databases.db' # Assumes PVsiRNAdb is in this file
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

# <-- REPLACED: Updated Database to Table Mapping for PVsiRNAdb -->
DATABASE_MAPPING = {
    "PVsiRNAdb": "Plant Virus siRNA Database",
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
        self.schema = {
            'all_database_2.db': {
                'tables': {
                    # <-- REPLACED: Swapped plantpepdb schema with PVsiRNAdb schema -->
                    'PVsiRNAdb': {
                        'columns': [
                            {'name': 'PVsiRNAdb Id', 'sqlite_type': 'TEXT', 'semantic_type': 'primary_identifier', 'description': 'Unique identifier for the siRNA record.'},
                            {'name': 'Plant', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'The common name of the plant host.'},
                            {'name': 'Scientific Name', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'The scientific name of the plant host.'},
                            {'name': 'Virus', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'The abbreviated name of the infecting virus.'},
                            {'name': 'Virus Name', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'The full name of the infecting virus.'},
                            {'name': 'PMID', 'sqlite_type': 'TEXT', 'semantic_type': 'publication_identifier', 'description': 'PubMed ID of the source publication.'},
                            {'name': 'Tissue', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_tissue', 'description': 'The plant tissue from which the siRNA was sequenced.'},
                            {'name': 'siRNA Sequence', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_sequence', 'description': 'The sequence of the small interfering RNA.'},
                            {'name': 'Virus Genome Type', 'sqlite_type': 'TEXT', 'semantic_type': 'category', 'description': 'The type of nucleic acid making up the viral genome (e.g., ssRNA, dsDNA).'},
                            {'name': 'Genus', 'sqlite_type': 'TEXT', 'semantic_type': 'taxonomy_rank', 'description': 'The taxonomic genus of the virus.'},
                            {'name': 'Family', 'sqlite_type': 'TEXT', 'semantic_type': 'taxonomy_rank', 'description': 'The taxonomic family of the virus.'},
                            {'name': 'Viral Strain', 'sqlite_type': 'TEXT', 'semantic_type': 'category', 'description': 'The specific strain of the virus.'},
                            {'name': 'Map Position', 'sqlite_type': 'TEXT', 'semantic_type': 'genomic_position', 'description': 'The position of the siRNA mapped to the viral genome.'},
                            {'name': 'Length', 'sqlite_type': 'INTEGER', 'semantic_type': 'quantity', 'description': 'The length of the siRNA sequence in nucleotides.'},
                            {'name': 'Coordinates', 'sqlite_type': 'TEXT', 'semantic_type': 'genomic_position', 'description': 'The start and end coordinates of the siRNA on the viral genome.'},
                            {'name': 'Strand Information', 'sqlite_type': 'TEXT', 'semantic_type': 'sequence_feature', 'description': 'The strand (+ or -) from which the siRNA originates.'}
                        ],
                        'description': 'PVsiRNAdb: A database of plant-derived viral small interfering RNAs (vsiRNAs) from various plants and viruses, detailing sequences, genomic context, and experimental source.',
                        'primary_keys': ['PVsiRNAdb Id'],
                        'common_joins': {},
                        'notes': ["Focuses on vsiRNAs from plant-virus interactions. Contains sequences, host/virus info, and genomic mapping data."]
                    },
                }
            }
        }
        self.species = {table_name: set() for table_name in self.schema['all_database_2.db']['tables']}
        self.tissues = {table_name: set() for table_name in self.schema['all_database_2.db']['tables']}
        self.update_schema_from_db()

    def _enrich_schema_with_dynamic_info(self):
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for table_name, table_data in self.schema['all_database_2.db']['tables'].items():
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
            for table_name in self.schema['all_database_2.db']['tables']:
                self.species.setdefault(table_name, set())
                self.tissues.setdefault(table_name, set())
                table_columns = {col['name'] for col in self.schema['all_database_2.db']['tables'][table_name]['columns']}

                # <-- MODIFIED: Updated to handle PVsiRNAdb's specific column names -->
                species_col = None
                if table_name == 'PVsiRNAdb':
                    species_col = 'Scientific Name'
                # ... (keep other elif conditions for other dbs if they exist in the file)

                if species_col and species_col in table_columns:
                    cursor.execute(f'SELECT DISTINCT "{species_col}" FROM "{table_name}" WHERE "{species_col}" IS NOT NULL AND "{species_col}" != ""')
                    self.species[table_name].update(row[0] for row in cursor.fetchall())

                # <-- MODIFIED: Tissue column determination for PVsiRNAdb -->
                tissue_col = None
                if table_name == 'PVsiRNAdb':
                    tissue_col = 'Tissue'
                # ... (keep other tissue logic for other dbs)

                if tissue_col and tissue_col in table_columns:
                    cursor.execute(f'SELECT DISTINCT "{tissue_col}" FROM "{table_name}" WHERE "{tissue_col}" IS NOT NULL AND "{tissue_col}" != ""')
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
        return list(self.schema['all_database_2.db']['tables'].keys())

    def get_species(self, table: str) -> Set[str]:
        return self.species.get(table, set())

    def get_tissues(self, table: str) -> Set[str]:
        return self.tissues.get(table, set())

schema_manager = SchemaManager(os.path.join(DB_PATH, DB_FILE))

# <-- The rest of the script from this point largely remains the same, except for the prompts -->
# <-- I will include the full script for completeness, with prompts modified below -->

logger.info("Initializing ChromaDB vector store...")
vector_db = None
try:
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(f"ChromaDB persistence directory not found at: {CHROMA_PERSIST_DIR}")
    
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )
    logger.info(f"ChromaDB loaded successfully from {CHROMA_PERSIST_DIR}. Collection has {vector_db._collection.count()} documents.")
except Exception as e:
    logger.error(f"FATAL: Failed to load ChromaDB. The app may not be able to answer informational queries. Error: {e}", exc_info=True)

def get_relevant_context_from_vectordb(query: str, k: int = 4) -> str:
    if vector_db is None:
        logger.warning("Vector DB is not available, returning empty context.")
        return "No vector context available due to a system error."
    logger.info(f"Querying vector DB for: '{query}'")
    try:
        relevant_docs = vector_db.similarity_search(query, k=k)
        if not relevant_docs:
            logger.info("No relevant documents found in vector DB for the query.")
            return "No specific context found in the knowledge base for this query."
        context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"Retrieved {len(relevant_docs)} relevant context chunks from vector DB.")
        return context_str
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        return "An error occurred while retrieving context from the knowledge base."

# ... (Sanitize functions, clean_nan_inf, etc. remain unchanged) ...
def sanitize_query_input(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string")
    return query

def sanitize_sql(sql_query: str) -> str:
    sql_query = sql_query.strip()
    if any(token in sql_query.upper() for token in ['DROP', 'DELETE FROM', 'TRUNCATE', 'INSERT INTO', 'UPDATE ']) and not sql_query.upper().startswith(("SELECT", "WITH")):
        raise ValueError(f"Potentially malicious SQL: {sql_query}")
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
        return (f"For your query: '{original_user_query}', the AI's plan was: '{analysis_plan}'. "
                "However, no data summaries from SQL queries or common item checks are available.")

    summary_parts = ["System-Generated Data Summary:"]

    if common_items_info_list:
        for common_items_info in common_items_info_list:
            if not common_items_info: continue

            item_type_name = common_items_info.get('item_type', 'unknown items')
            summary_parts.append(f"\n--- Common '{item_type_name}' Analysis ---")
            
            # Use the more reliable 'databases_queried_for_this_item' which is based on LLM's plan
            databases_planned_str = ", ".join(common_items_info.get('databases_queried_for_this_item', ['relevant databases']))
            summary_parts.append(f"  Data for '{item_type_name}' commonality check was requested from: {databases_planned_str}.")

            if common_items_info.get("note"): # e.g. "could not be fully determined due to errors"
                summary_parts.append(f"  Note on data collection for '{item_type_name}': {common_items_info['note']}")

            distinct_values_map = common_items_info.get("distinct_values_per_database_for_this_item")
            if distinct_values_map:
                summary_parts.append(f"  Distinct '{item_type_name}' values found per database (for Stage 2 LLM to analyze for commonality using semantic hints):")
                found_any_distinct_values_for_current_item_type = False
                for db_name_in_map, distinct_list_val in distinct_values_map.items():
                    # Ensure db_name_in_map is one of the planned dbs for this item type check
                    if db_name_in_map not in common_items_info.get('databases_queried_for_this_item', []):
                        continue # Only show data for explicitly planned DBs for this check

                    if isinstance(distinct_list_val, (list, set)) and distinct_list_val:
                        # Sort for consistent output, show more examples
                        sorted_distinct_list = sorted(list(distinct_list_val))
                        example_items = ', '.join(f"'{v}'" for v in sorted_distinct_list[:15]) # Increased examples
                        ellipsis = '...' if len(sorted_distinct_list) > 15 else ''
                        summary_parts.append(f"    - {db_name_in_map} ({len(sorted_distinct_list)} distinct): {example_items}{ellipsis}")
                        found_any_distinct_values_for_current_item_type = True
                    elif isinstance(distinct_list_val, (list, set)) and not distinct_list_val:
                         summary_parts.append(f"    - {db_name_in_map}: No distinct '{item_type_name}' values found.")
                    elif isinstance(distinct_list_val, str) and "Error:" in distinct_list_val: # Error string
                        summary_parts.append(f"    - {db_name_in_map}: Retrieval Issue: {distinct_list_val[:200]}") # Increased error visibility
                    elif isinstance(distinct_list_val, str): # Other string, possibly "Data not available..."
                        summary_parts.append(f"    - {db_name_in_map}: {distinct_list_val[:200]}")
                    else: # Fallback for unexpected type
                        summary_parts.append(f"    - {db_name_in_map}: Unexpected data format for distinct '{item_type_name}'s: {str(distinct_list_val)[:100]}")

                if not found_any_distinct_values_for_current_item_type and \
                   not any("Error:" in str(v) or "Issue:" in str(v) or "failed" in str(v) for v in distinct_values_map.values()):
                     summary_parts.append(f"    No specific distinct '{item_type_name}' values were retrieved from any of the targeted databases in this group.")
            
            # If distinct_values_map is empty but databases were planned, it might be covered by the note.
            # If the map is empty AND no note exists, it implies an issue or no data.
            elif not common_items_info.get("note"):
                 summary_parts.append(f"  No information on distinct '{item_type_name}' values was processed from the databases for this check.")
    
    if executed_queries_stats_list:
        summary_parts.append("\n--- Individual Query Part Summaries ---")
        for i, stats_item in enumerate(executed_queries_stats_list):
            # Skip if this query part was solely for common item check, as it's covered above
            purpose_type_for_skip_check = stats_item.get('purpose_type', '')
            if purpose_type_for_skip_check.startswith("list_distinct_values_for_common_check_"):
                continue

            part_summary = [f"\nQuery Part {i+1} (Description: '{stats_item.get('description_from_llm', 'N/A')}')"]
            # MODIFICATION START: Add the SQL query to the summary for the next AI stage.
            if stats_item.get('original_sql'):
                part_summary.append(f"  SQL Executed: `{stats_item.get('original_sql')}`")
            # MODIFICATION END
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
                    if count_col_name and 'sum' in stats_item.get('key_column_statistics',{}).get(count_col_name, {}):
                        actual_count = int(stats_item['key_column_statistics'][count_col_name]['sum'])
                        part_summary.append(f"  Aggregate Result (e.g., Count): {actual_count}")
                
                if stats_item.get('statistics_based_on_sample'):
                    part_summary.append("  (Statistics below are based on a sample of a large dataset.)")

                if purpose_type == 'aggregation_categorical_expression' and total_rows > 0 and stats_item.get("preview_for_ui_only"):
                    part_summary.append("  Categorical Expression Counts:")
                    for row in stats_item["preview_for_ui_only"]:
                        group_desc_parts = []
                        if 'Tissue' in row: group_desc_parts.append(f"Tissue='{row.get('Tissue', 'N/A')}'")
                        if 'Expression' in row: group_desc_parts.append(f"Expression='{row.get('Expression', 'N/A')}'")
                        group_desc = ", ".join(group_desc_parts) if group_desc_parts else "Overall"
                        count_val = row.get('count_per_category', row.get('count', 'N/A'))
                        part_summary.append(f"    - For {group_desc}: {count_val} records")
                
                if total_rows > 0 and stats_item.get('key_column_statistics') and purpose_type not in ['aggregation_categorical_expression']:
                    part_summary.append("  Key Column Stats:")
                    for col, col_stats in stats_item['key_column_statistics'].items():
                        if 'COUNT(' in col.upper() and purpose_type == 'aggregation' and total_rows == 1 : continue
                        col_details = [f"    - '{col}':"]
                        if 'non_null_count' in col_stats and col_stats['non_null_count'] != total_rows : col_details.append(f"Non-nulls: {col_stats['non_null_count']}/{total_rows}")
                        if 'distinct_count' in col_stats and col_stats['distinct_count'] < total_rows and col_stats['distinct_count'] > 0: col_details.append(f"Distinct: {col_stats['distinct_count']}")
                        numeric_stats_added = False
                        if 'avg_log2fc' == col and 'avg_log2fc' in col_stats:
                             col_details.append(f"Avg Log2FC: {col_stats['avg_log2fc']:.2f}"); numeric_stats_added=True
                        elif 'avg_rpm' == col and 'avg_rpm' in col_stats:
                             col_details.append(f"Avg RPM: {col_stats['avg_rpm']:.2f}"); numeric_stats_added=True
                        elif 'mean' in col_stats: col_details.append(f"Avg: {col_stats['mean']:.2f}" if isinstance(col_stats['mean'], float) else f"Avg: {col_stats['mean']}"); numeric_stats_added=True
                        if 'median' in col_stats: col_details.append(f"Med: {col_stats['median']}"); numeric_stats_added=True
                        if 'min_log2fc' == col and 'min_log2fc' in col_stats: col_details.append(f"Min Log2FC: {col_stats['min_log2fc']}")
                        elif 'min_rpm' == col and 'min_rpm' in col_stats: col_details.append(f"Min RPM: {col_stats['min_rpm']}")
                        elif 'min' in col_stats and 'max' in col_stats and col_stats['min'] != col_stats['max']:
                            col_details.append(f"Range: {col_stats['min']}-{col_stats['max']}"); numeric_stats_added=True
                        elif 'min' in col_stats and not numeric_stats_added: # If only min is present and no other numeric summary
                             col_details.append(f"Value(s) include: {col_stats['min']}")
                        elif 'min' in col_stats and numeric_stats_added and 'max' not in col_stats : # If other numeric stats but no max for range
                             col_details.append(f"Min value: {col_stats['min']}")


                        if 'max_log2fc' == col and 'max_log2fc' in col_stats: col_details.append(f"Max Log2FC: {col_stats['max_log2fc']}")
                        elif 'max_rpm' == col and 'max_rpm' in col_stats: col_details.append(f"Max RPM: {col_stats['max_rpm']}")
                        elif 'max' in col_stats and numeric_stats_added and 'min' not in col_stats: # If other numeric stats but no min for range
                             col_details.append(f"Max value: {col_stats['max']}")


                        if 'num_records' == col and 'num_records' in col_stats : col_details.append(f"Record Count for Group: {col_stats['num_records']}")
                        
                        if 'top_values' in col_stats and col_stats['top_values']:
                            top_v_str = ", ".join([f"'{k}' ({v}x)" for k, v in list(col_stats['top_values'].items())[:3]]) # Show top 3
                            col_details.append(f"Top: {top_v_str}")
                        if len(col_details) > 1: part_summary.extend(col_details)

                elif total_rows == 0 and not stats_item.get('error_if_any'):
                    part_summary.append("  No data records found for this part.")
            summary_parts.extend(part_summary)
            
    return "\n".join(summary_parts)

# DATABASE_METADATA_KNOWLEDGE_BLOB (Global Scope - place near other constants)
DATABASE_METADATA_KNOWLEDGE_BLOB = """
**Database Summaries:**
*   **PVsiRNAdb, is the plant exclusive database dedicated to virus-derived small interfering RNAs (vsiRNA), harboring information related to vsiRNA sequences from different viruses infecting different plants collected from published literature so far. This database contains a total of 322,214 entries and 282,549 unique vsiRNAs sequences derived from 20 different viral strains infecting 12 plants. This database is believed to be beneficial for exploring the complex domain of viral genetics and genomics, plant viral interactions and very useful for scientific community working on various silencing pathways. It will prove to be highly advantageous resource in the agricultural field for producing viral resistant transgenic crops.
"""


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
    priority_columns = ['sequence', 'tRF_type', 'tncRNA_Type', 'organism', 'Tissue', 'tissue', 'deg_tissue', 'rpm', 'log2fold_change', 'Expression']
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

@retry(tries=3, delay=2, backoff=2, logger=logger)
def invoke_groq_model(chain, input_data: Dict) -> str:
    try:
        logger.info("Invoking Groq model", model_input_keys=list(input_data.keys()))
        response_object = chain.invoke(input_data)
        raw_llm_text = response_object.content.strip() if hasattr(response_object, 'content') else str(response_object).strip()
        
        logger.debug(f"Raw LLM output received (first 500 chars): {raw_llm_text[:500]}")

        # Attempt 1: Delimiter-based extraction (most reliable if LLM follows it)
        start_token = "<JSON_START>"
        end_token = "<JSON_END>"
        start_index = raw_llm_text.find(start_token)
        end_index = raw_llm_text.rfind(end_token)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = raw_llm_text[start_index + len(start_token):end_index].strip()
            # Basic validation before returning
            if json_str.startswith('{') and json_str.endswith('}'):
                logger.info("Extracted JSON using delimiters.")
                return json_str
            else:
                logger.warning(f"Delimiter-extracted string not valid JSON structure: {json_str[:200]}")

        # Attempt 2: Markdown ```json ... ``` block
        match_markdown = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_llm_text, re.DOTALL | re.IGNORECASE)
        if match_markdown:
            extracted_json_str = match_markdown.group(1).strip()
            if extracted_json_str.startswith('{') and extracted_json_str.endswith('}'):
                logger.info("Extracted JSON from markdown block.")
                return extracted_json_str
            else:
                 logger.warning(f"Markdown-extracted string not valid JSON structure: {extracted_json_str[:200]}")


        # Attempt 3: Find the largest valid JSON object in the text
        # This is more robust against leading/trailing garbage text if delimiters are missed.
        best_json_match = None
        for match in re.finditer(r"(\{[\s\S]*?\})", raw_llm_text): # finditer to get all potential JSON objects
            potential_json_str = match.group(1)
            try:
                # Try to parse it to see if it's valid JSON
                json.loads(potential_json_str)
                # If it's valid, and longer than previous best match, keep it
                if best_json_match is None or len(potential_json_str) > len(best_json_match):
                    best_json_match = potential_json_str
            except json.JSONDecodeError:
                continue
        
        if best_json_match:
            logger.info("Extracted largest valid JSON object found in raw text.")
            return best_json_match

        # Attempt 4: Fallback to first '{' and last '}' if no valid JSON object found by regex
        # This is less reliable but can sometimes salvage partial JSONs or JSONs with minor issues
        json_start = raw_llm_text.find('{')
        json_end = raw_llm_text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            extracted_json_str = raw_llm_text[json_start:json_end + 1].strip()
            logger.warning(f"Falling back to brace-extraction for JSON. String: {extracted_json_str[:200]}")
            return extracted_json_str # parse_json_response will try to clean this

        logger.warning("No clear JSON structure found in LLM output using any extraction method. Returning raw text.")
        return raw_llm_text # Fallback to raw text if all else fails

    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 429: # Rate limit
            error_message = e.response.json().get('error', {}).get('message', '')
            wait_match = re.search(r'Please try again in ([\d\.]+)s', error_message)
            wait_seconds = float(wait_match.group(1)) + random.uniform(1,3) if wait_match else 60 # add jitter
            logger.warning(f"Rate limit exceeded. Waiting {wait_seconds:.2f}s. Message: {error_message}")
            time.sleep(wait_seconds)
            raise # Re-raise to trigger retry
        raise # Re-raise other HTTP errors
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
# <-- REPLACED: Updated Intent Classification Prompt for PVsiRNAdb -->
INTENT_CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "available_databases_summary", "database_content_summary", "conversation_summary"],
    template="""
You are an AI assistant for PVsiRNAdb, a plant genomics database.
Your primary goal is to determine if a query is answerable using the PVsiRNAdb, which focuses on virus-derived small interfering RNAs (vsiRNAs) in plants.

*** KEY CONTEXT: PVsiRNAdb is a plant-exclusive database for vsiRNAs, with 322,214 entries from 20 viral strains infecting 12 plants. It is used for exploring plant-viral interactions and silencing pathways. It contains vsiRNA sequences, genomic mapping data, and source information (Plant, Virus, Tissue, PMID). The database also features tools like BLAST and sequence mapping. ***

If a user asks "what data do you have on [topic]?" or "show me data about [topic]", interpret this as a `data_preview` request.

User Query: {user_query}
Conversation Summary (Recent turns, if any): {conversation_summary}
Available Databases (Name: Conceptual Name): {available_databases_summary}
Database Content Summary (Focus on types of molecules and data):
{database_content_summary}

Based on the user query and the nature of PVsiRNAdb, classify the query into ONE of the following categories:

1.  "METADATA_DIRECT_ANSWER_PREFERRED": The query asks for general knowledge ABOUT the PVsiRNAdb or the concepts it contains. Examples: "What is a vsiRNA?", "What kind of tools does PVsiRNAdb have?", "Do you have data from grapevine fanleaf virus?".
2.  "METADATA_SQL_FALLBACK_PREFERRED": The query asks for metadata *about the data within the database* that might require a simple SQL query. Examples: "how many plant species are in PVsiRNAdb?", "list all viruses covered in the database".
3.  "DATA_RETRIEVAL": The query asks to find specific data records, perform calculations, or filter data. Examples: "find vsiRNAs from rice in leaf tissue", "list all siRNAs from tomato with length 21". This also includes vague follow-ups like 'show me' or 'run that' if the context implies a data action.
4.  "AMBIGUOUS": The query is related to the database domain but is unclear and needs clarification.
5.  "TYPO": The query seems related but likely contains a typo.
6.  "GENERAL_CONVERSATION": A greeting, closing, or other conversational filler.
7.  "OUT_OF_SCOPE": The query is unrelated to plant genomics, vsiRNAs, or the PVsiRNAdb. This includes requests for peptides, tRFs, fusion transcripts, or any molecule not a vsiRNA.

Output ONLY the classification string.

Classification:
"""
)

# <-- REPLACED: Updated SQL Planning Prompt for PVsiRNAdb -->
SQL_PLAN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "classified_intent", "db_file_name", "schemas_json", "knowledge_graph",
        "previous_query_summary", "conversation_summary", "database_table_mapping_info",
        "DISPLAY_ROW_LIMIT", "retrieved_knowledge_context"
    ],
    template="""
You are an AI assistant that plans data retrieval for the PVsiRNAdb database.
Your *ONLY* output MUST be a single, valid JSON object, wrapped in <JSON_START> and <JSON_END>.

*** KEY CONTEXT: You serve PVsiRNAdb, a database of plant-derived virus-derived small interfering RNAs (vsiRNAs). It contains 322,214 entries from 20 viral strains infecting 12 plants. Key data includes vsiRNA sequences, host plant, infecting virus, tissue of origin, and genomic mapping information. ***

User Query: {user_query}
Pre-classified Intent: {classified_intent}

Context Provided:
- Database File: {db_file_name}
- Schemas: {schemas_json} (Use for exact table and column names)
- Knowledge Graph: {knowledge_graph}
- Retrieved Knowledge Context: {retrieved_knowledge_context}
- UI Preview Row Limit: {DISPLAY_ROW_LIMIT}

**CRITICAL DATABASE-SPECIFIC RULES for PVsiRNAdb:**
- The primary table is `PVsiRNAdb`.
- Species/Plant information is in the `"Scientific Name"` and `"Plant"` columns.
- Virus information is in the `"Virus Name"` and `"Virus"` columns.
- Use `LIKE '%keyword%'` for flexible text matching in these columns.
- The `"Length"` column is INTEGER and can be used for direct numeric comparison.

**Decision Process & JSON Output Generation:**

1.  **Confirm Intent & Scope:**
    * If {classified_intent} is "OUT_OF_SCOPE", "AMBIGUOUS", etc., `analysis_plan` should reflect this and the `queries` array MUST be empty.
    * If the query asks about molecules NOT in PVsiRNAdb (peptides, tRFs, fusion transcripts), state this in the `analysis_plan` and generate NO SQL.

2.  **Response Strategy & SQL Planning:**
    * **IF {classified_intent} is "METADATA_DIRECT_ANSWER_PREFERRED" (e.g., "what is vsiRNA?", "describe PVsiRNAdb"):**
        * `analysis_plan`: "Answering general information query about '{user_query}' using retrieved knowledge."
        * **CRITICAL: `direct_answer_from_metadata` MUST be populated with an answer synthesized from `retrieved_knowledge_context`.**
        * **CRITICAL: `queries` array MUST be empty. `query_type` MUST be "metadata_answered_from_blob".**

    * **IF {classified_intent} is "METADATA_SQL_FALLBACK_PREFERRED" (e.g., "list all plants in the database"):**
        * `analysis_plan`: "Listing distinct 'plants' from the Plant Virus siRNA Database."
        * `purpose_type`: `list_distinct_values`
        * `sql`: `SELECT DISTINCT "Scientific Name" FROM "PVsiRNAdb" WHERE "Scientific Name" IS NOT NULL AND TRIM("Scientific Name") != '' ORDER BY 1;`

    * **IF {classified_intent} is "DATA_RETRIEVAL":**
        * `query_type: data_retrieval`.
        * **FOR AGGREGATIONS/COUNTING (e.g., "how many vsiRNAs from rice?"):**
            * `analysis_plan`: "Counting vsiRNA entries for rice in the Plant Virus siRNA Database."
            * `purpose_type`: `aggregation_count`.
            * `sql`: `SELECT COUNT(*) as vsirna_count FROM "PVsiRNAdb" WHERE "Scientific Name" LIKE '%Oryza sativa%';`
        * **FOR Specific Entity Lookup (e.g., "details for PVsiRNAdb_123"):**
            * `analysis_plan`: "Retrieving detailed information for ID 'PVsiRNAdb_123' from the Plant Virus siRNA Database."
            * `purpose_type`: `entity_lookup_detail`.
            * `sql`: `SELECT * FROM "PVsiRNAdb" WHERE "PVsiRNAdb Id" = 'PVsiRNAdb_123';`
        * **FOR General Data Preview / Listing (e.g., "show me vsiRNAs from tomato"):**
            * `purpose_type: data_preview`.
            * `analysis_plan`: "Retrieving a sample of vsiRNA data from tomato."
            * `sql`: `SELECT * FROM "PVsiRNAdb" WHERE "Scientific Name" LIKE '%Solanum lycopersicum%' LIMIT {DISPLAY_ROW_LIMIT};`

3.  **ALWAYS Quote Identifiers:** Wrap all table and column names in double quotes (`"`). Example: `SELECT "siRNA Sequence" FROM "PVsiRNAdb"`.

4.  **JSON Output Structure (MUST be followed precisely):**
    <JSON_START>
    {{
      "query_type": "...",
      "analysis_plan": "...",
      "direct_answer_from_metadata": "...",
      "queries": [
        {{
          "sql": "SELECT ... FROM \\"PVsiRNAdb\\" ...",
          "target_table": "PVsiRNAdb",
          "database_conceptual_name": "Plant Virus siRNA Database",
          "description": "Brief description of this query's goal.",
          "purpose_type": "data_preview OR aggregation_count OR ...",
          "display_columns_hint": ["colA", "colB"]
        }}
      ]
    }}
    <JSON_END>

Generate ONLY the JSON object now.
"""
)

# <-- REPLACED: Updated Summary Interpretation Prompt for PVsiRNAdb -->
SUMMARY_INTERPRET_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "analysis_plan_from_stage1",
        "textual_data_summary_from_python",
        "knowledge_graph_snippet", "conversation_summary",
        "query_type_from_stage1", "DISPLAY_ROW_LIMIT"
    ],
    template="""
You are a friendly and expert bioinformatics assistant for PVsiRNAdb. Your entire response MUST be a single JSON object.

*** KEY CONTEXT: You serve PVsiRNAdb, a database of plant-derived virus-derived small interfering RNAs (vsiRNAs). The database contains information on vsiRNA sequences, their host plants, infecting viruses, and genomic mapping. It is a valuable resource for research in plant-viral interactions and gene silencing. ***

Your primary task is to rephrase the provided "System-Generated Data Summary" into a concise, user-friendly paragraph. Your summary MUST be based *solely* on the facts presented in the summary. DO NOT invent information.

User Query: {user_query}
AI's Analysis Plan: {analysis_plan_from_stage1}
System-Generated Data Summary (Factual data report to rephrase):
{textual_data_summary_from_python}
Relevant Knowledge Graph Snippet: {knowledge_graph_snippet}
Recent Conversation Summary: {conversation_summary}
Query Type Determined by System: {query_type_from_stage1}

**Instructions for "summary" FIELD:**

1.  **Synthesize Findings Coherently:** Do not mention internal mechanics like query numbers or table names. Focus on the biological insights.

2.  **HANDLING `entity_lookup_detail`:**
    * The system summary will contain the full record. Narrate the key details in a readable paragraph.
    * **Example:** "The search for PVsiRNAdb_2500 found one record in the database. This 22-nucleotide vsiRNA is from *Oryza sativa* (Rice) infected with the Rice stripe virus. It was identified in leaf tissue and maps to the positive strand of the viral genome at coordinates 1050-1071."

3.  **HANDLING `data_preview` / `list_distinct_values`:**
    * State the total number of records found. If a preview is mentioned, state that a sample is available.
    * Highlight key findings from "Key Column Stats" if significant. Example: "The search for vsiRNAs in rice returned 15,230 records. These records come from several different viral strains, most commonly the Rice stripe virus."

4.  **HANDLING `aggregation` (e.g., COUNT):**
    * Report the specific aggregate result directly. Example: "The database contains a total of 8,450 vsiRNA entries for *Vitis vinifera* (Grapevine)."

5.  **Handling Errors or No Data:**
    * If the summary states "Error" or "No data records found", report this plainly. Example: "No vsiRNA data was found for infections by the Tomato yellow leaf curl virus in the database."

**Output JSON Structure:**
{{
  "summary": "[Your rephrased, conversational summary of the findings. No 'Bot:' prefix.]",
  "databases_conceptually_involved": ["Plant Virus siRNA Database"]
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
            "metadata": {"execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "error": str(e)}
        }
    
    # Initialize LLMs
    llm = ChatGroq(
    api_key=API_KEY, model_name=model_name, temperature=0.05,
    max_tokens=4096, request_timeout=60.0
    )
    intent_llm = ChatGroq(api_key=API_KEY, model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=500)

    # Prepare for Stage 0 Intent Classification
    available_databases_summary_str = "\n".join([f"- {name}: {desc}" for name, desc in DATABASE_MAPPING.items()])
    database_content_summary_for_intent = """
    -PVsiRNAdb: is the plant exclusive database dedicated to virus-derived small interfering RNAs (vsiRNA), harboring information related to vsiRNA sequences from different viruses infecting different plants collected from published literature so far. This database contains a total of 322,214 entries and 282,549 unique vsiRNAs sequences derived from 20 different viral strains infecting 12 plants. This database is believed to be beneficial for exploring the complex domain of viral genetics and genomics, plant viral interactions and very useful for scientific community working on various silencing pathways. It will prove to be highly advantageous resource in the agricultural field for producing viral resistant transgenic crops.
    """
    conversation_summary_for_intent = summarize_conversation(conversation_id, max_turns=1, max_len_per_item=50)

    intent_input_data = {
        "user_query": sanitized_user_query,
        "conversation_summary": conversation_summary_for_intent,
        "available_databases_summary": available_databases_summary_str,
        "database_content_summary": database_content_summary_for_intent
    }
    
    # Execute Stage 0
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
        classified_intent = "DATA_RETRIEVAL"  # Fallback if Stage 0 fails

    # --- THIS IS THE CORE FIX: CONDITIONALLY QUERY THE VECTOR DB ---
    retrieved_knowledge_context = ""  # Initialize with an empty string
    
    # Only query the vector DB if the intent is informational.
    if classified_intent in ["METADATA_DIRECT_ANSWER_PREFERRED", "METADATA_SQL_FALLBACK_PREFERRED"]:
        logger.info(f"[{request_id}] Intent is '{classified_intent}'. Querying vector DB for context.")
        retrieved_knowledge_context = get_relevant_context_from_vectordb(sanitized_user_query)
    else:
        logger.info(f"[{request_id}] Intent is '{classified_intent}'. Skipping vector DB search to prioritize SQL generation.")
        # Provide a clear message to the next stage LLM that this was intentional.
        retrieved_knowledge_context = "Vector DB search was skipped for this query type. The query should be answered by generating SQL against the database schemas."

    if classified_intent == "OUT_OF_SCOPE":
        logger.info(f"[{request_id}] Query classified as OUT_OF_SCOPE.")
        # Modified out-of-scope message
        summary_text = "I specialize in plant genomics data related to plant peptides. I'm unable to answer questions outside this domain, such as about general knowledge, current events, or other types of biological molecules like siRNAs."
        
        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {"user_query_that_led_to_this_summary": sanitized_user_query, "agent_analysis_plan_executed": "Query out of scope", "agent_summary_of_findings": summary_text, "databases_queried_in_this_turn": [], "data_found_overall_in_this_turn": False }
            context_data['history'].append({ "query": sanitized_user_query, "summary_preview": "Out of scope..." })
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": summary_text,
            "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {"execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "classified_intent": classified_intent, "llm_analysis_plan": "Query determined out of scope."}
        }
    if classified_intent == "GENERAL_CONVERSATION":
        logger.info(f"[{request_id}] Query classified as GENERAL_CONVERSATION.")
        bot_response = "Hello! How can I help you with plant peptides ncRNA data today?"
        if any(kw in sanitized_user_query.lower() for kw in ["thank", "bye", "that's all"]):
            bot_response = "You're welcome! Feel free to ask more questions."
        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {"user_query_that_led_to_this_summary": sanitized_user_query, "agent_analysis_plan_executed": "General conversation", "agent_summary_of_findings": bot_response, "databases_queried_in_this_turn": [], "data_found_overall_in_this_turn": False }
            context_data['history'].append({ "query": sanitized_user_query, "summary_preview": bot_response[:100] + "..." })
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": bot_response, "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": {"execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "classified_intent": classified_intent, "llm_analysis_plan": "General conversation handling."}
        }

    schemas_json_str = json.dumps(schema_manager.get_schema_for_prompt(), indent=2)
    database_table_mapping_info_str = "\n".join([f"- {name} (Conceptual: {desc})" for name, desc in DATABASE_MAPPING.items()])
    
    # **IMPROVED KNOWLEDGE GRAPH CONSTRUCTION with new DB**
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
    
    # Logic to add notes about Oryza subspecies remains the same
    updated_species_info_parts = []
    for part in species_info_parts:
        is_pbt_rf = DATABASE_MAPPING.get("PbtRF", "PbtRF Database") in part
        is_ptrfdb = DATABASE_MAPPING.get("PtRFdb", "PtRFdb") in part
        is_pfusiondb = DATABASE_MAPPING.get("PFusionDB", "Plant Fusion Database") in part
        is_ptncrnadb = DATABASE_MAPPING.get("PtncRNAdb", "PtncRNAdb") in part
        if (is_pbt_rf or is_ptrfdb or is_pfusiondb or is_ptncrnadb) and \
           "Oryza sativa" in part and "Japonica" not in part and "Indica" not in part:
            part_suffix = " (note: this DB may also contain 'Oryza sativa Japonica Group' and 'Oryza sativa Indica Group'. If user asks for 'rice' generally, consider all.)"
            if part_suffix not in part:
                 part += part_suffix
        updated_species_info_parts.append(part)
    species_info_str = "\n".join(updated_species_info_parts) if updated_species_info_parts else "No specific species data pre-loaded."

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
    tissues_info_str = "\n".join(tissues_info_parts) if tissues_info_parts else "No specific tissue data pre-loaded."

    # KNOWLEDGE GRAPH with CoNCRAtlasdb integrated
    # In process_query, when constructing knowledge_graph_str
    # KNOWLEDGE GRAPH with PVsiRNAdb as the primary focus
    # In process_query, when constructing knowledge_graph_str
    knowledge_graph_str = f"""
    Concise Knowledge Graph for SQLite DB ({DB_FILE}):
    - Tables & Primary Use (Conceptual Names from DATABASE_MAPPING):
      - {DATABASE_MAPPING.get("PVsiRNAdb", "PVsiRNAdb")}: Plant-derived virus small interfering RNAs (vsiRNAs). Species col: `Scientific Name`. Numeric cols like `Length` are TEXT, require CAST.

    - **Default Database Grouping for General Queries (IMPORTANT):**
      - For general "siRNA", "vsiRNA", or "plant virus" queries: Use ONLY PVsiRNAdb.

    - Categories for Common Item Checks (when user asks "what is common across [category]?"):
      - "siRNA database": PVsiRNAdb

    - Species Name Hint: For "rice", use `LIKE 'Oryza sativa%'`. For "cotton", use `LIKE 'Gossypium%'`. For "Arabidopsis", use `LIKE 'Arabidopsis thaliana%'` (except for AtFusionDB which is implicitly Arabidopsis).
    - Sample Species: {species_info_str}
    - Sample Tissues: {tissues_info_str}
    - Explicit Scope: The PVsiRNAdb contains 322,214 entries of plant virus-derived siRNAs (vsiRNAs) from 20 viral strains infecting 12 plants. For miRNA and lncRNA, data is ONLY available for cotton species in the CoNCRAtlasdb.

    - **Semantic Equivalence Hints for Common Item Analysis (Stage 2 LLM uses this for interpreting distinct lists):**
      - **Species:** 'Oryza sativa' and its subspecies (Indica, Japonica) are 'rice'. 'Arabidopsis thaliana' and 'Arabidopsis' are the same. 'Gossypium hirsutum', 'Gossypium barbadense' are types of 'cotton'.
      - **Tissues:** 'leaf', 'rosette leaf', 'cauline leaf', 'leaves', 'leaf part' map to 'leaf'. 'flower', 'inflorescence', 'floral bud', 'stamen', 'petal' map to 'flower/floral organs'. 'root', 'root tip' map to 'root'. 'seedling', 'young seedling' map to 'seedling'.
      - **General Principle:** Apply semantic grouping to find meaningful conceptual overlaps, not just exact string matches. Normalize text (lowercase, remove underscores, trim whitespace) before comparing.
    """
    # ... rest of the process_query function

    context = CONTEXT_CACHE.get(conversation_id, {})
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
        "retrieved_knowledge_context": retrieved_knowledge_context # <-- MODIFIED: Pass the new context here
    }
    
    # --- The rest of the process_query function from this point onward remains the same. ---
    # The existing logic is generic and will correctly handle the new plans,
    # SQL queries, and results generated by the LLM based on the updated context.
    # For brevity, the remainder of the function is omitted here, as it is unchanged.
    # You should keep the rest of your original `process_query` function.
    # ... (existing code from line `logger.info(f"[{request_id}] Prepared Stage 1 input...")`
    # ... all the way to the end of the function.

    # NOTE: The following is the remainder of the original function. You do not need to copy this
    # if you are just replacing the top part of your `process_query` function. This is provided
    # for completeness to show where the new code ends.

    logger.info(f"[{request_id}] Prepared Stage 1 input. KG len: {len(knowledge_graph_str)}, Schema len: {len(schemas_json_str)}")

    parsed_stage1_response = None
    try:
        logger.info(f"[{request_id}] Invoking LLM Stage 1 for planning...")
        s1_invoke_start = time.time()
        # Use the corrected SQL_PLAN_PROMPT_TEMPLATE
        raw_stage1_response_str = invoke_groq_model(SQL_PLAN_PROMPT_TEMPLATE | llm, stage1_input_data)
        s1_invoke_end = time.time()
        logger.info(f"[{request_id}] LLM Stage 1 invocation completed in {s1_invoke_end - s1_invoke_start:.2f}s.")

        parsed_stage1_response = parse_json_response(raw_stage1_response_str)
        if parsed_stage1_response is None:
            logger.error(f"[{request_id}] Stage 1 LLM response could not be parsed into JSON.", extra={"raw_response": raw_stage1_response_str})
            return {
                "summary": "I encountered an issue while planning how to answer your query. Please try rephrasing or ask a simpler question.",
                "executed_queries_details": [], "databases_conceptually_involved": [],
                "metadata": { "execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "error": "LLM Stage 1 response format error (not JSON)", "classified_intent": classified_intent}
            }
        if "query_type" not in parsed_stage1_response or "analysis_plan" not in parsed_stage1_response:
            logger.error(f"[{request_id}] Stage 1 response missing 'query_type' or 'analysis_plan'", extra={"response_obj": parsed_stage1_response})
            return {
                "summary": "The AI's plan for your query was incomplete. Please try rephrasing your query.",
                "executed_queries_details": [], "databases_conceptually_involved": [],
                "metadata": { "execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "error": "Stage 1 planning response incomplete fields", "classified_intent": classified_intent}
            }
    except Exception as e:
        logger.error(f"[{request_id}] Stage 1 processing error: {str(e)}", exc_info=True)
        if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
             summary_msg = "The system is currently experiencing high load for query planning. Please try again in a few moments."
        elif isinstance(e, requests.exceptions.ConnectionError): # Added specific connection error handling
             summary_msg = "I'm having trouble connecting to the AI services. Please check your network or try again later."
        else:
            summary_msg = f"Sorry, an error occurred during query planning ({type(e).__name__}). Please try rephrasing."
        return {
            "summary": summary_msg,
            "executed_queries_details": [], "databases_conceptually_involved": [],
            "metadata": {"execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "error": f"Stage 1 error: {str(e)}", "classified_intent": classified_intent}
        }

    query_type_from_stage1 = parsed_stage1_response.get("query_type", classified_intent)
    analysis_plan_from_stage1 = parsed_stage1_response.get("analysis_plan", "No analysis plan from AI.")
    direct_answer_from_metadata = parsed_stage1_response.get("direct_answer_from_metadata")
    llm_generated_queries = parsed_stage1_response.get("queries", [])

    if classified_intent == "METADATA_DIRECT_ANSWER_PREFERRED" and \
       query_type_from_stage1 not in ["metadata_answered_from_blob", "metadata_sql_fallback", "DATA_RETRIEVAL", "entity_lookup_detail"]:
        query_type_from_stage1 = "metadata_answered_from_blob"

    if query_type_from_stage1 == "metadata_answered_from_blob":
        final_summary_text = ""
        if direct_answer_from_metadata and direct_answer_from_metadata.strip():
            final_summary_text = direct_answer_from_metadata
            logger.info(f"[{request_id}] Using direct_answer_from_metadata from Stage 1.")
        elif not llm_generated_queries: # Only attempt Python fallback if LLM didn't provide direct answer AND no SQL
            logger.warning(f"[{request_id}] 'metadata_answered_from_blob' planned but 'direct_answer_from_metadata' missing/empty and no SQL. Python fallback for: {sanitized_user_query}")
            extracted_metadata_fallback = ""
            user_query_lower = sanitized_user_query.lower()

            db_keywords_found_in_query = {}
            for db_code, db_name_conceptual in DATABASE_MAPPING.items():
                if db_code.lower() in user_query_lower or db_name_conceptual.lower() in user_query_lower:
                    db_keywords_found_in_query[db_code] = db_name_conceptual

            concept_keywords_map = {
                "trf": ["tRFs (tRNA-derived fragments):", "PbtRF Database", "PtRFdb", "PtncRNAdb"],
                "tncrna": ["tncRNAs (transfer RNA-derived non-coding RNAs):", "PtncRNAdb"],
                "fusion": ["fusion transcripts", "AtFusionDB", "Plant Fusion Database", "fusion gene data"],
                "peptide": ["plant-derived peptides", "Plant Peptide Database"],
                "mirna": ["miRNA (microRNA)", "Cotton Non-Coding RNA Atlas Database"] # Added miRNA
            }
            found_concept_blob_keywords = []
            for query_kw, blob_hints in concept_keywords_map.items():
                if query_kw in user_query_lower:
                    found_concept_blob_keywords.extend(blob_hints)

            relevant_blob_parts_dict = {}
            blob_sections = DATABASE_METADATA_KNOWLEDGE_BLOB.split('\n\n*   **')
            blob_sections = ["*   **" + s for s in blob_sections if s.strip()]

            if db_keywords_found_in_query:
                for db_code, db_name in db_keywords_found_in_query.items():
                    pattern_str = rf"^\*\s+\*\*{re.escape(db_name)}\s*\({re.escape(db_code)}\):?\*\*"
                    for section in blob_sections:
                        if re.search(pattern_str, section, re.IGNORECASE | re.MULTILINE):
                            section_content = section.split("Key data:")[0].strip() # Get the description part
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
                            elif concept_hint.lower() in section.lower(): # broader match
                                relevant_blob_parts_dict[concept_hint] = section.split('.')[0].replace("*   **", "").replace("**:", ":").replace("**", "").strip() + "."
                                break
                    if relevant_blob_parts_dict : break # one concept match is enough

            if relevant_blob_parts_dict:
                extracted_metadata_fallback = " ".join(list(set(relevant_blob_parts_dict.values())))
                logger.info(f"[{request_id}] Python fallback provided metadata snippet: {extracted_metadata_fallback[:300]}")
                final_summary_text = extracted_metadata_fallback
            else:
                final_summary_text = f"I understood you were asking for general information (plan: '{analysis_plan_from_stage1}'), but I couldn't find a specific pre-compiled answer or a direct snippet from my knowledge base. Could you try asking for specific data, or rephrase your information request?"
                logger.warning(f"[{request_id}] Python fallback for metadata also failed for: {sanitized_user_query}")

        if conversation_id:
            context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
            context_data['last_query'] = sanitized_user_query
            context_data['last_response_summary_for_llm'] = {
                "user_query_that_led_to_this_summary": sanitized_user_query,
                "agent_analysis_plan_executed": analysis_plan_from_stage1[:150],
                "agent_summary_of_findings": final_summary_text[:350],
                "databases_queried_in_this_turn": [],
                "data_found_overall_in_this_turn": bool(final_summary_text and "couldn't find" not in final_summary_text.lower() and "couldn't pinpoint" not in final_summary_text.lower())
            }
            context_data['history'].append({ "query": sanitized_user_query, "summary_preview": final_summary_text[:200] + "..." })
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": final_summary_text.strip(), "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": { "execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "llm_analysis_plan": analysis_plan_from_stage1, "query_type_processed": "metadata_answered_from_blob", "classified_intent_s0": classified_intent }
        }

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
            context_data['history'].append({ "query": sanitized_user_query, "summary_preview": final_summary_text[:200] + "..." })
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": final_summary_text.strip(), "executed_queries_details": [],
            "databases_conceptually_involved": [],
            "metadata": { "execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "llm_analysis_plan": analysis_plan_from_stage1, "query_type_processed": query_type_from_stage1, "classified_intent_s0": classified_intent }
        }

    if not llm_generated_queries:
        logger.error(f"[{request_id}] No SQL queries were generated by Stage 1 for a non-metadata-blob intent. S0 Intent: {classified_intent}, S1 query_type: {query_type_from_stage1}, Plan: '{analysis_plan_from_stage1}'")
        summary_msg = f"I understood your query as: '{analysis_plan_from_stage1}', but I couldn't formulate a specific database search. Could you try rephrasing?"
        if "mirna" in sanitized_user_query.lower() and "cotton" not in sanitized_user_query.lower(): # Added specific response
            summary_msg = "I can search for miRNA data, but only for cotton. Please specify if your query relates to cotton."
        elif "siRNA" in sanitized_user_query.lower():
            summary_msg = "I can search for various ncRNAs, but data for siRNAs is not available in these databases."

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
            context_data['history'].append({ "query": sanitized_user_query, "summary_preview": summary_msg[:200] + "..." })
            if len(context_data['history']) > 5: context_data['history'].pop(0)
            CONTEXT_CACHE[conversation_id] = context_data
        return {
            "summary": summary_msg,
            "executed_queries_details": [], "databases_conceptually_involved": [],
            "metadata": { "execution_time": round(time.time() - processing_start_time, 3), "total_rows_retrieved_by_sql_across_all_queries": 0, "conversation_id": conversation_id, "llm_analysis_plan": analysis_plan_from_stage1, "error": "SQL planned but not generated or other unhandled plan", "query_type_processed": query_type_from_stage1, "classified_intent_s0": classified_intent}
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
            logger.warning(f"[{request_id}] Query {i+1} from LLM plan missing 'target_table' or is 'UnknownTable'. SQL: {sql_statement}. Plan: {query_info_from_llm}")
            target_table_from_llm = "UnknownTable"

        if not db_conceptual_name_from_llm or db_conceptual_name_from_llm in ["Unknown Database", target_table_from_llm]:
            temp_conceptual_name = DATABASE_MAPPING.get(target_table_from_llm, target_table_from_llm)
            if temp_conceptual_name == target_table_from_llm and target_table_from_llm not in DATABASE_MAPPING.values():
                 db_conceptual_name_from_llm = "Unknown Database"
            else:
                 db_conceptual_name_from_llm = temp_conceptual_name
            if query_info_from_llm.get("database_conceptual_name") != db_conceptual_name_from_llm:
                 logger.info(f"[{request_id}] Query {i+1} 'database_conceptual_name' resolved to '{db_conceptual_name_from_llm}'. Original in plan: '{query_info_from_llm.get('database_conceptual_name')}'. SQL: {sql_statement}")

        ui_card_target_name = db_conceptual_name_from_llm
        if ui_card_target_name in ["Unknown Database", "UnknownTable"] or ui_card_target_name == target_table_from_llm:
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
            "results_preview": [], "preview_table": [],
            "download_url": "", "error": None, "row_count_from_sql": 0,
            "purpose_type_from_llm": purpose_type
        }
        stats_item_for_py_summary = {
            "original_sql": sql_statement, "target_table": target_table_from_llm,
            "database_conceptual_name": db_conceptual_name_from_llm,
            "description_from_llm": query_description_from_llm, "purpose_type": purpose_type,
            "total_rows_found": 0, "key_column_statistics": {},
            "preview_for_ui_only": [],
            "statistics_based_on_sample": False, "error_if_any": None
        }

        if not sql_statement:
            error_msg = "Empty SQL from AI planner."
            current_query_display_details["error"] = error_msg
            stats_item_for_py_summary["error_if_any"] = error_msg
            item_type_for_failed_check = None
            if purpose_type == "list_distinct_values_for_common_check_species": item_type_for_failed_check = "species"
            elif purpose_type == "list_distinct_values_for_common_check_tissue": item_type_for_failed_check = "tissue"
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
                    if purpose_type == "list_distinct_values_for_common_check_species": item_type_for_failed_check = "species"
                    elif purpose_type == "list_distinct_values_for_common_check_tissue": item_type_for_failed_check = "tissue"
                    if item_type_for_failed_check and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                        common_items_data_store[item_type_for_failed_check][db_conceptual_name_from_llm] = f"Error: {actual_results}"
                else:
                    any_query_produced_data = True if actual_results else any_query_produced_data
                    full_stats_obj = generate_statistics_from_results(actual_results, f"[{request_id}]_query_{i+1}_{target_table_from_llm}")

                    current_query_display_details["row_count_from_sql"] = full_stats_obj["count"]
                    current_query_display_details["results_preview"] = full_stats_obj["preview"]

                    stats_item_for_py_summary["total_rows_found"] = full_stats_obj["count"]
                    stats_item_for_py_summary["statistics_based_on_sample"] = full_stats_obj["stats_based_on_sample"]

                    if purpose_type == 'aggregation_categorical_expression' and actual_results:
                        stats_item_for_py_summary["preview_for_ui_only"] = actual_results
                    else:
                        stats_item_for_py_summary["preview_for_ui_only"] = full_stats_obj["preview"]

                    concise_col_stats = {}
                    if full_stats_obj["column_stats"]:
                        for col, S_stats in full_stats_obj["column_stats"].items():
                            col_stat_summary = {}
                            if 'non_null_count' in S_stats: col_stat_summary['non_null_count'] = S_stats['non_null_count']
                            if 'null_count' in S_stats and S_stats['null_count'] > 0: col_stat_summary['null_count'] = S_stats['null_count']
                            if 'distinct_count' in S_stats: col_stat_summary['distinct_count'] = S_stats['distinct_count']
                            if 'mean' in S_stats and S_stats['mean'] is not None: col_stat_summary['mean'] = round(S_stats['mean'], 2) if isinstance(S_stats['mean'], float) else S_stats['mean']
                            if 'median' in S_stats and S_stats['median'] is not None: col_stat_summary['median'] = round(S_stats['median'], 2) if isinstance(S_stats['median'], float) else S_stats['median']
                            if 'min' in S_stats and S_stats['min'] is not None: col_stat_summary['min'] = S_stats['min']
                            if 'max' in S_stats and S_stats['max'] is not None: col_stat_summary['max'] = S_stats['max']
                            if 'sum' in S_stats and S_stats['sum'] is not None and purpose_type == 'aggregation' and full_stats_obj["count"] == 1 and 'COUNT(' in col.upper():
                                col_stat_summary['sum'] = S_stats['sum']
                            if 'top_values' in S_stats and S_stats['top_values']: col_stat_summary['top_values'] = dict(list(S_stats['top_values'].items())[:3])
                            if col_stat_summary: concise_col_stats[col] = col_stat_summary
                    stats_item_for_py_summary["key_column_statistics"] = concise_col_stats
                    grand_total_rows_retrieved_by_sql += full_stats_obj["count"]

                    item_type_being_checked = None
                    if purpose_type == "list_distinct_values_for_common_check_species": item_type_being_checked = "species"
                    elif purpose_type == "list_distinct_values_for_common_check_tissue": item_type_being_checked = "tissue"

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
                if purpose_type == "list_distinct_values_for_common_check_species": item_type_for_failed_check = "species"
                elif purpose_type == "list_distinct_values_for_common_check_tissue": item_type_for_failed_check = "tissue"
                if item_type_for_failed_check and db_conceptual_name_from_llm and db_conceptual_name_from_llm not in ["Unknown Database", "UnknownTable", "Target N/A"]:
                    common_items_data_store[item_type_for_failed_check][db_conceptual_name_from_llm] = f"Error: {error_msg}"

        concise_executed_queries_stats_for_py_summary.append(stats_item_for_py_summary)
        all_results_for_display_and_csv.append(current_query_display_details)

    logger.info(f"[{request_id}] Finished SQL execution. Total rows from successful queries: {grand_total_rows_retrieved_by_sql}")

    final_common_items_reports_for_summary = []
    atfusion_conceptual_name = DATABASE_MAPPING.get("AtFusionDB")

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

        if item_type_key == "species" and atfusion_conceptual_name and atfusion_conceptual_name in all_dbs_planned_for_this_item_check:
            if atfusion_conceptual_name not in collected_data_for_item_type or not collected_data_for_item_type[atfusion_conceptual_name]:
                 collected_data_for_item_type[atfusion_conceptual_name] = {'Arabidopsis thaliana'}

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
                num_errors_or_unavailable_for_this_item_type +=1

        total_planned_dbs_for_item = len(report["databases_queried_for_this_item"])
        if not report["databases_queried_for_this_item"] and requested_common_item_checks.get(item_type_key):
             report["note"] = f"A commonality check for '{item_type_key}' was intended, but no specific databases were successfully identified or targeted in the AI's plan."
        elif num_errors_or_unavailable_for_this_item_type > 0:
            if num_succeeded_for_this_item_type > 0:
                report["note"] = (f"Analysis for '{item_type_key}' commonality involved {total_planned_dbs_for_item} database(s)/group(s). "
                                  f"Data was successfully processed for {num_succeeded_for_this_item_type}. "
                                  f"However, issues or no data occurred for {num_errors_or_unavailable_for_this_item_type} database(s). Commonality assessment by the AI will be based on available data.")
            else:
                report["note"] = (f"Analysis for '{item_type_key}' commonality involved {total_planned_dbs_for_item} database(s)/group(s). "
                                  f"Unfortunately, data retrieval failed or no specific data was found for all of them. Cannot determine common '{item_type_key}'s.")
        elif total_planned_dbs_for_item > 0 and not any_distinct_values_found_across_dbs:
            report["note"] = f"No specific distinct '{item_type_key}' values were found in any of the {total_planned_dbs_for_item} targeted database(s) to compare for commonality."

        if report["databases_queried_for_this_item"] or (requested_common_item_checks.get(item_type_key) and report["note"]):
            final_common_items_reports_for_summary.append(report)

    textual_data_summary_for_stage2 = generate_textual_summary_from_stats(
        concise_executed_queries_stats_for_py_summary,
        sanitized_user_query,
        analysis_plan_from_stage1,
        common_items_info_list=final_common_items_reports_for_summary if final_common_items_reports_for_summary else None
    )
    logger.debug(f"[{request_id}] Textual summary for Stage 2 LLM (length {len(textual_data_summary_for_stage2)}):\n{textual_data_summary_for_stage2[:1000]}...")

    stage2_input_data = {
        "user_query": sanitized_user_query,
        "analysis_plan_from_stage1": analysis_plan_from_stage1,
        "textual_data_summary_from_python": textual_data_summary_for_stage2,
        "knowledge_graph_snippet": knowledge_graph_str[:3000],
        "conversation_summary": conversation_summary_str,
        "query_type_from_stage1": query_type_from_stage1,
        "DISPLAY_ROW_LIMIT": DISPLAY_ROW_LIMIT
    }
    final_summary_text_from_stage2 = "I processed your request."
    final_databases_involved_from_stage2 = set()

    for item_stat in concise_executed_queries_stats_for_py_summary:
        db_name_for_involvement = item_stat.get("database_conceptual_name")
        matching_detail = next((detail for detail in all_results_for_display_and_csv
                                if detail["_internal_table_sqlite_name"] == item_stat.get("target_table") and
                                   detail["_internal_database_conceptual_name"] == db_name_for_involvement and
                                   detail.get("sql") == item_stat.get("original_sql")
                                   ), None)
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
        final_summary_text_from_stage2 = "I understood you were asking for information, but I couldn't formulate a direct response."

    if (llm_generated_queries or final_common_items_reports_for_summary) and \
       final_summary_text_from_stage2 == "I processed your request.":
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
                if len(final_summary_text_from_stage2) > 1500: final_summary_text_from_stage2 = final_summary_text_from_stage2[:1497] + "..."
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
            if "rate_limit_exceeded" in str(e_stage2).lower() or "429" in str(e_stage2):
                final_summary_text_from_stage2 = "The system is currently experiencing high load for summarization. Here's a structured view of what was found:\n" + textual_data_summary_for_stage2[:500]+"..."
            elif isinstance(e_stage2, requests.exceptions.ConnectionError):
                 final_summary_text_from_stage2 = "I'm having trouble connecting to the AI services for summarization. Here's a structured view of what was found:\n" + textual_data_summary_for_stage2[:500]+"..."
            else:
                final_summary_text_from_stage2 = f"An error occurred during final summary generation ({type(e_stage2).__name__}). The system found: {textual_data_summary_for_stage2[:300]}..."

    if conversation_id:
        context_data = CONTEXT_CACHE.get(conversation_id, {'history': []})
        context_data['last_query'] = sanitized_user_query
        overall_data_found_for_context = any_query_produced_data
        if not overall_data_found_for_context and final_common_items_reports_for_summary:
            for report in final_common_items_reports_for_summary:
                if any(isinstance(val_list, list) and val_list for val_list in report.get("distinct_values_per_database_for_this_item", {}).values()):
                    overall_data_found_for_context = True; break

        context_data['last_response_summary_for_llm'] = {
            "user_query_that_led_to_this_summary": sanitized_user_query,
            "agent_analysis_plan_executed": analysis_plan_from_stage1[:300],
            "agent_summary_of_findings": final_summary_text_from_stage2[:400],
            "databases_queried_in_this_turn": sorted(list(d for d in final_databases_involved_from_stage2 if d and d not in ["Target N/A", "Unknown Database"])),
            "data_found_overall_in_this_turn": overall_data_found_for_context
        }
        context_data['history'].append({ "query": sanitized_user_query, "summary_preview": final_summary_text_from_stage2[:250] + "..." })
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



