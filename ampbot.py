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
from langchain.vectorstores import Chroma  # <-- ADD THIS
from langchain.embeddings import SentenceTransformerEmbeddings # <-- ADD THIS
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

#API_KEY = os.getenv('GROQ_API_KEY', '')
DB_PATH = os.getenv('DB_PATH', '/var/www/html/PlantXBot')
DB_FILE = 'all_database_2.db'
DOWNLOAD_DIR = os.path.join(DB_PATH, 'public/downloads')
BASE_URL = 'http://14.139.61.8/PlantXBot/public/downloads'
CACHE = TTLCache(maxsize=1000, ttl=3600)
CONTEXT_CACHE = TTLCache(maxsize=100, ttl=3600)
DISPLAY_ROW_LIMIT = 10
MAX_ROWS_FOR_PANDAS_STATS = 50000
MAX_ROWS_FOR_LLM_SUMMARY = 1000

# --- ADD THIS SECTION for ChromaDB Configuration ---
CHROMA_PERSIST_DIR = os.path.join(DB_PATH, 'Chroma_db')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Database to Table Mapping
DATABASE_MAPPING = {
"plantpepdb": "Plant Peptide Database",
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
                    'plantpepdb': { # NEW TABLE SCHEMA
                        'columns': [
                            {'name': 'PPepDB_ID', 'sqlite_type': 'TEXT', 'semantic_type': 'primary_identifier', 'description': 'Unique identifier for the peptide entry in PlantPeptideDB.'},
                            {'name': 'Peptide_Name', 'sqlite_type': 'TEXT', 'semantic_type': 'name', 'description': 'Name of the peptide.'},
                            {'name': 'PMID', 'sqlite_type': 'TEXT', 'semantic_type': 'publication_id', 'description': 'PubMed ID(s) for references, often HTML formatted.'},
                            {'name': 'Plant_Source_Scientific', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_name', 'description': 'Scientific name of the plant source for the peptide.'},
                            {'name': 'Plant_Source_Common', 'sqlite_type': 'TEXT', 'semantic_type': 'common_name', 'description': 'Common name of the plant source.'},
                            {'name': 'Plant_Family', 'sqlite_type': 'TEXT', 'semantic_type': 'organism_taxonomy', 'description': 'Family of the source plant.'},
                            {'name': 'Peptide_Family', 'sqlite_type': 'TEXT', 'semantic_type': 'category', 'description': 'Family to which the peptide belongs (e.g., Defensin, Cyclotide).'},
                            {'name': 'Peptide_Function', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_function', 'description': 'General biological function of the peptide (e.g., Antimicrobial, Antihypertensive).'},
                            {'name': 'Function_Description', 'sqlite_type': 'TEXT', 'semantic_type': 'description', 'description': 'Detailed description of the peptide\'s function.'},
                            {'name': 'Activity_Against', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_target', 'description': 'Specifies the target organism or condition against which the peptide shows activity.'},
                            {'name': 'IC50_Value', 'sqlite_type': 'TEXT', 'semantic_type': 'measurement_value', 'description': 'IC50 value, often includes units or notes (CAST to REAL if strictly numeric).'},
                            {'name': 'Sequence', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_sequence_peptide', 'description': 'Amino acid sequence of the peptide.'},
                            {'name': 'Sequence_Length', 'sqlite_type': 'TEXT', 'semantic_type': 'length_value', 'description': 'Length of the peptide sequence (CAST to INTEGER).'},
                            {'name': 'Validation', 'sqlite_type': 'TEXT', 'semantic_type': 'metadata', 'description': 'Validation status or experimental evidence level.'},
                            {'name': 'Avg_Molecular_Weight', 'sqlite_type': 'TEXT', 'semantic_type': 'molecular_weight', 'description': 'Average molecular weight (CAST to REAL).'},
                            {'name': 'Monoisotopic_MW', 'sqlite_type': 'TEXT', 'semantic_type': 'molecular_weight', 'description': 'Monoisotopic molecular weight (CAST to REAL).'},
                            {'name': 'Isoelectric_Point', 'sqlite_type': 'TEXT', 'semantic_type': 'biophysical_property', 'description': 'Isoelectric point (pI) of the peptide (CAST to REAL).'},
                            {'name': 'Method_Extraction', 'sqlite_type': 'TEXT', 'semantic_type': 'experimental_method', 'description': 'Method used for peptide extraction.'},
                            {'name': 'Uniprot', 'sqlite_type': 'TEXT', 'semantic_type': 'accession_id', 'description': 'UniProt accession ID, if available.'},
                            {'name': 'NCBI', 'sqlite_type': 'TEXT', 'semantic_type': 'accession_id', 'description': 'NCBI accession ID, if available.'},
                            {'name': 'EMBL', 'sqlite_type': 'TEXT', 'semantic_type': 'accession_id', 'description': 'EMBL accession ID, if available.'},
                            {'name': 'Source_DB_Links', 'sqlite_type': 'TEXT', 'semantic_type': 'url', 'description': 'Links to source databases, often HTML formatted.'},
                            {'name': 'Additional_Info', 'sqlite_type': 'TEXT', 'semantic_type': 'metadata', 'description': 'Any other relevant information.'},
                            {'name': 'Classification', 'sqlite_type': 'TEXT', 'semantic_type': 'category', 'description': 'Peptide classification (e.g., AMP, AHTP).'},
                            {'name': 'Peptide_Activity', 'sqlite_type': 'TEXT', 'semantic_type': 'biological_activity', 'description': 'Observed biological activity of the peptide.'},
                            {'name': 'Physico_Chemical_Properties', 'sqlite_type': 'TEXT', 'semantic_type': 'metadata_complex', 'description': 'A pipe-separated string containing multiple physico-chemical properties; not easily queryable for individual values directly.'}
                        ],
                        'description': 'PlantPeptideDB: Contains information about plant-derived peptides, including their sequences, sources, functions, activities, and physico-chemical properties.',
                        'primary_keys': ['PPepDB_ID'],
                        'common_joins': {}, # Explicit joins with tRF/Fusion DBs are unlikely, but can be linked via organism.
                        'notes': ["Focuses on plant peptides. Some fields like PMID and Source_DB_Links might contain HTML. Numerical fields like IC50_Value, Sequence_Length, Avg_Molecular_Weight, Monoisotopic_MW, Isoelectric_Point are stored as TEXT and need CASTing for numerical operations. 'Physico_Chemical_Properties' is a complex packed string."]
                    },
                }
            }
        }
        # Initialize species and tissues for all tables, including the new one
        self.species = {table_name: set() for table_name in self.schema['all_database_2.db']['tables']}
        self.tissues = {table_name: set() for table_name in self.schema['all_database_2.db']['tables']}

        # This will now also process 'CoNCRAtlasdb'
        self.update_schema_from_db()

    def _enrich_schema_with_dynamic_info(self): # No changes needed here, it's generic
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

    def update_schema_from_db(self): # Modified to handle new table's species column
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for table_name in self.schema['all_database_2.db']['tables']:
                self.species.setdefault(table_name, set())
                self.tissues.setdefault(table_name, set())
                table_columns = {col['name'] for col in self.schema['all_database_2.db']['tables'][table_name]['columns']}

                # Determine species column based on table
                species_col = None
                if table_name == 'plantpepdb':
                    species_col = 'Plant_Source_Scientific' if 'Plant_Source_Scientific' in table_columns else None
                elif 'species' in table_columns: # For PFusionDB
                    species_col = 'species'
                elif 'Species' in table_columns: # For CoNCRAtlasdb (NEW)
                    species_col = 'Species'
                elif 'Organism' in table_columns: # For PbtRF
                    species_col = 'Organism'
                elif 'organism' in table_columns: # For PtRFdb, PtncRNAdb
                    species_col = 'organism'

                if species_col:
                    cursor.execute(f"SELECT DISTINCT \"{species_col}\" FROM \"{table_name}\" WHERE \"{species_col}\" IS NOT NULL AND \"{species_col}\" != ''")
                    self.species[table_name].update(row[0] for row in cursor.fetchall())

                # Tissue column determination (CoNCRAtlasdb and plantpepdb do not have a direct tissue column)
                if table_name not in ['plantpepdb', 'CoNCRAtlasdb_mi', 'CoNCRAtlasdb_lnc']:
                    tissue_col = next((col for col in ['Tissue', 'tissue', 'deg_tissue'] if col in table_columns), None)
                    if tissue_col:
                        cursor.execute(f"SELECT DISTINCT \"{tissue_col}\" FROM \"{table_name}\" WHERE \"{tissue_col}\" IS NOT NULL AND \"{tissue_col}\" != ''")
                        self.tissues[table_name].update(row[0] for row in cursor.fetchall())
            conn.close()
            logger.info("Schema updated with species and tissues")
            self._enrich_schema_with_dynamic_info()
        except Exception as e:
            logger.error(f"Failed to update schema: {str(e)}", exc_info=True)

    def get_schema_for_prompt(self) -> Dict: # No changes needed here, it's generic
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

    def get_tables(self) -> List[str]: # No changes needed here
        return list(self.schema['all_database_2.db']['tables'].keys())

    def get_species(self, table: str) -> Set[str]: # No changes needed here
        return self.species.get(table, set())

    def get_tissues(self, table: str) -> Set[str]: # No changes needed here
        return self.tissues.get(table, set())

schema_manager = SchemaManager(os.path.join(DB_PATH, DB_FILE))
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
*   **Plant Peptide Database (plantpepdb):** Plant-derived peptides. Key data: peptide `Sequence`, `Plant_Source_Scientific` (species), `Peptide_Family`, `Peptide_Function`, `Activity_Against`, `IC50_Value`. Numerical fields are TEXT, need CAST.
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
    priority_columns = [
    "Peptide_Family",
    "Peptide_Function",
    "Function_Description",
    "Activity_Against",
    "IC50_Value",
    "Sequence",
    "Sequence_Length",
    "Validation",
    "Avg_Molecular_Weight",
    "Monoisotopic_MW",
    "Isoelectric_Point",
    "Method_Extraction",
    "Uniprot",
    "NCBI",
    "EMBL",
    "Source_DB_Links",
    "Additional_Info",
    "Classification",
    "Peptide_Activity",
    "Physico_Chemical_Properties"
]
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





# INTENT_CLASSIFICATION_PROMPT_TEMPLATE (Further Refined for Metadata)
INTENT_CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_query", "available_databases_summary", "database_content_summary"],
    template="""
You are an AI assistant responsible for classifying user queries directed at a suite of plant genomics databases.
*** You are an AI agent that serves PlantPepDB database of plant peptides having different functions and therapeutic activities. It contains 3848 peptide entries collected from 11 databases and 835 published research articles.***
Your primary goal is to determine if the query is answerable using the provided databases and their described content, or if it's out of scope.
Understand the querry and it's intent with clarity and identify the true meaning of the user querry that user is asking for what type of data or information.
****If a user asks a "what data do you have on [topic]?" or "show me data about [topic]" question, interpret this as a `data_preview` request*****
User Query: {user_query}
Conversation Summary (Recent turns, if any): {conversation_summary}
Available Databases (Name: Conceptual Name): {available_databases_summary}
Database Content Summary (Focus on types of molecules and data):
{database_content_summary}

Based on the user query and the nature of the available databases (which primarily focus on specific biomolecules and genomic features like tRFs, tncRNAs, fusion transcripts, plant peptides, and certain ncRNAs as detailed in the Database Content Summary), classify the query into ONE of the following categories:

1.  "METADATA_DIRECT_ANSWER_PREFERRED": The query asks for general knowledge or facts ABOUT the databases or the concepts they contain. This includes questions of existence like "What is peptide?", "What are therapeutic peptides?", "Do you have data on antimicrobial peptides?", or "What kind of data is in PlantPepDB?". These should be answered from a knowledge base.
2.  "METADATA_SQL_FALLBACK_PREFERRED": Query asks for metadata *about the data within the databases* that might require a simple SQL query if not directly in a knowledge text, but still benefits from the knowledge text for context (e.g., "how many species are in PlantPeppDB?", "list peptide types in PlantPepDB having protein level experimental evidence", "what peptide functions are listed in plantpepdb?").
3.  "DATA_RETRIEVAL": ***Any Query starting with ("what is the the", "which","fetch","tell me about","what can be", "give me" etc.)***, Query asks to find specific data records, perform calculations on data, or compare data sets *within the described scope of the databases* (e.g., "find peptides for antifungal activity", "list peptides with IC50 value > - 2").**Crucially, this also includes vague follow-up commands (like 'ok do it', 'show me', 'run that', 'yes please') that directly relate to a data action proposed or described by the bot in the `conversation_summary`.** Use the context to see if the user is confirming a previous action.
4.  "AMBIGUOUS": Query is related to the database domain but is unclear and needs clarification from the user.
5.  "TYPO": Query seems related to the database domain but likely contains a typo.
6.  "GENERAL_CONVERSATION": Query is conversational, a greeting, or a simple closing.
7.  "OUT_OF_SCOPE": Query is clearly unrelated to plant genomics, the specific databases, their content, or asks about entities explicitly NOT covered (e.g., "what is the time?", "miRNA data for non-covered species", "siRNA data").

Output ONLY the classification string (e.g., "DATA_RETRIEVAL" or "OUT_OF_SCOPE"). Ensure no extra text or explanation.

Classification:
"""
)

SQL_PLAN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "classified_intent", "db_file_name", "schemas_json", "knowledge_graph",
        "previous_query_summary", "conversation_summary", "database_table_mapping_info",
        "DISPLAY_ROW_LIMIT", "retrieved_knowledge_context"  # <-- MODIFIED: Replaced metadata_knowledge_blob
    ],
    template="""
You are an AI assistant that analyzes user queries and plans data retrieval or information provision for plant genomics databases.
Your *ONLY* output MUST be a single, valid JSON object, wrapped in <JSON_START> and <JSON_END>. No other text, explanations, or conversational filler before or after the JSON block.
*** You are an AI agent that serves PlantPepDB database of plant peptides having different functions and therapeutic activities. It contains 3848 peptide entries collected from 11 databases and 835 published research articles.***
User Query: {user_query}
Pre-classified Intent (from Stage 0): {classified_intent}

Context Provided:
- Database File: {db_file_name}
- Schemas: {schemas_json} (Use this for exact table and column names)
- Knowledge Graph: {knowledge_graph}
    (This graph contains:
        - Table purposes and conceptual names (use `database_table_mapping_info` for conceptual names in output).
        - Mappings of semantic concepts (e.g., 'species', 'tissue', 'expression value') to actual column names for each table.
        - Database-specific characteristics and default groupings.
        - Hints for common name mappings (e.g., 'rice' to 'Oryza sativa%').
        - Key columns for functional searches.
    )
- Database Mapping (Conceptual Names): {database_table_mapping_info}
- Retrieved Knowledge Context (Relevant info from a knowledge base based on the query): 
  {retrieved_knowledge_context}  # <-- MODIFIED: Using the dynamically retrieved context
- UI Preview Row Limit: {DISPLAY_ROW_LIMIT} (UI hint ONLY, NOT for SQL `LIMIT` on data_preview/list_distinct/entity_lookup_detail unless it's a generic unbounded list.)


**CRITICAL DATABASE-SPECIFIC RULES & DATA SEMANTICS (Supplemented by Schemas & Knowledge Graph):**
- **plantpepdb Numeric Fields:** Numeric fields in `plantpepdb` (e.g., IC50_Value, Sequence_Length, Avg_Molecular_Weight) are stored as TEXT. You MUST CAST them to REAL or INTEGER for numeric operations (e.g., `CAST(IC50_Value AS REAL)`).
- **Species/Tissue Columns:** These vary per table. Consult `schemas_json` or specific column mappings in `knowledge_graph`.
    - Example Species Columns (refer to KG/schemas for authoritative list): PbtRF.Organism, PtRFdb.organism, PtncRNAdb.organism, PFusionDB.species, plantpepdb.Plant_Source_Scientific, CoNCRAtlasdb.Species.
    - Example Tissue Columns (refer to KG/schemas for authoritative list): (CoNCRAtlasdb and plantpepdb generally do not have a standard 'tissue' column for broad searches).
- **plantpepdb Numeric Fields:** Columns like `IC50_Value` and `Sequence_Length` are TEXT and MUST be CAST for numeric operations (e.g., `CAST("Sequence_Length" AS INTEGER)`).


Decision Process & Output Generation (Strictly adhere to the JSON output format defined at the end):
0.  **Handle Vague Follow-up Commands:** If the `user_query` is vague (e.g., 'ok do it', 'run it', 'yes', 'show me') AND the `classified_intent` is `DATA_RETRIEVAL`, this indicates the user is confirming an action from the previous turn. You **MUST IGNORE** the text of the current vague query. Instead, you **MUST re-create the plan based on the USER'S QUERY FROM THE PREVIOUS TURN**, Your `analysis_plan` should still reflect the original goal (e.g., "Finding common plant source...").

1.  **Confirm Intent & Scope:**
    *   If {classified_intent} is "OUT_OF_SCOPE", "AMBIGUOUS", "TYPO", or non-data "GENERAL_CONVERSATION", `analysis_plan` should reflect this and `queries` array MUST be empty.
    *   If the query mentions unsupported molecules (e.g., siRNA) and Stage 0 missed it, `analysis_plan` must state "Query asks about [molecule], which is not covered by these databases." and generate NO SQL. (Note: miRNA/lncRNA ARE supported via CoNCRAtlasdb for cotton, as per `knowledge_graph`).

2.  **Database Targeting Strategy (Guided by `knowledge_graph` and above CRITICAL RULES):**
    *   **Default Grouping (if no specific DB named by user):** Use groupings specified in `knowledge_graph`.
        *   Example: "peptides" -> ONLY plantpepdb.
    *   **Peptide Function/Activity:** For general queries on `plantpepdb`, search keywords in multiple relevant text columns (e.g., `Peptide_Function`, `Activity_Against`, etc., as identified from `schemas_json` or `knowledge_graph`) using `LIKE '%keyword%'` and OR conditions.

3.  **Response Strategy & SQL Planning:**

    *   **IF {classified_intent} is "METADATA_DIRECT_ANSWER_PREFERRED" (e.g., "describe [DB_name]", "what is [concept]?"):**
        Your primary goal is to answer from the `retrieved_knowledge_context`. # <-- MODIFIED
        `analysis_plan`: "Answering general information query about '{user_query}' using retrieved knowledge." # <-- MODIFIED
        **CRITICAL: `direct_answer_from_metadata` field MUST be populated with a comprehensive answer synthesized from `retrieved_knowledge_context`.** # <-- MODIFIED
        **CRITICAL: `queries` array MUST be empty. `query_type` MUST be "metadata_answered_from_blob".**
        (Exception: If query is "how many species in PbtRF?" AND the `retrieved_knowledge_context` doesn't state it, then `query_type` becomes `metadata_sql_fallback`, `analysis_plan` becomes "Counting/Listing distinct items...", `direct_answer_from_metadata` is null/empty, and generate SQL.)

    *   **IF ({classified_intent} is "METADATA_SQL_FALLBACK_PREFERRED" OR {classified_intent} is "DATA_RETRIEVAL") AND (query asks for simple counts/listings of metadata attributes like species/tissues):**
        `analysis_plan`: "Checking for common '{{item_type}}' across {{databases}}." OR "Listing distinct '{{item_type}}' from {{Database_Conceptual_Name}}."
        Identify ALL relevant tables using `knowledge_graph` and `schemas_json`.
        **CRITICAL: For EACH relevant table/database, generate a SEPARATE query object in `queries`.**
        `purpose_type`: `list_distinct_values_for_common_check_species` (or `_tissue`, etc.).
        `sql`: `SELECT DISTINCT "<column_name_for_item_type>" AS item_value FROM "<ActualTableName>" WHERE "<column_name_for_item_type>" IS NOT NULL AND TRIM("<column_name_for_item_type>") != '' AND LOWER(TRIM("<column_name_for_item_type>")) NOT IN ('na', '--na--', 'unknown', 'unspecified', 'not available', 'none');`. (Get `<column_name_for_item_type>` from `schemas_json` / `knowledge_graph`).
        If a table has no relevant column for the item type (e.g., 'Tissue' in plantpepdb), DO NOT generate a query for it for that check.
     
    *   IF the query asks to **list distinct items from a SINGLE table** (and is not a commonality check):
        `analysis_plan`: "Listing distinct '{{item_type}}' from {{Database_Conceptual_Name}}."
        `purpose_type`: `list_distinct_values`
        Generate one query object.
        `sql`: `SELECT DISTINCT "<column_name_for_item_type>" FROM "<ActualTableName>" WHERE "<column_name_for_item_type>" IS NOT NULL AND TRIM("<column_name_for_item_type>") != '' ORDER BY 1;`

    *   **IF {classified_intent} is "DATA_RETRIEVAL" (and not common item check / simple metadata listing):**
        `query_type: data_retrieval`.

        *   **FOR AGGREGATIONS / COUNTING / FINDING MOST COMMON:**
            `analysis_plan`: "Performing aggregation: '{{description_of_aggregation}}' on {{Database_Conceptual_Name(s)}}."
            `purpose_type`: `aggregation` (or `aggregation_count`, `aggregation_most_frequent_value`).
            Generate query objects. Use correct table/column names from `schemas_json` and `knowledge_graph`.
            Example for "how many arabidopsis fusions are present in pfusiondb?":
              Query (PFusionDB):
                `sql`: "SELECT COUNT(*) as fusion_count FROM \\"PFusionDB\\" WHERE species LIKE 'Arabidopsis thaliana%';" (Confirm 'species' column name from schema/KG for PFusionDB)
                `display_columns_hint`: ["fusion_count"]
            Example for "In which tissue are most fusions found in AtFusionDB?":
              Query (AtFusionDB):
                `sql`: "SELECT Tissue, COUNT(*) as fusion_count FROM \\"AtFusionDB\\" WHERE Tissue IS NOT NULL AND TRIM(Tissue) != '' GROUP BY Tissue ORDER BY fusion_count DESC LIMIT 1;" (Confirm 'Tissue' column name from schema/KG for AtFusionDB)
                `display_columns_hint`: ["Tissue", "fusion_count"]

        *   **FOR Specific Entity Lookup:**
            `analysis_plan`: "Retrieving detailed information for ID/entity '{{identifier_value_or_description}}' from {{Database_Conceptual_Name(s)}}."
            `purpose_type`: `entity_lookup_detail`.
            Determine relevant table(s) and identifier column(s) from `knowledge_graph`/`schemas_json`.
            Example for "do AT1G65440 make fusions?":
              Query 1 (AtFusionDB): `sql`: "SELECT * FROM \\"AtFusionDB\\" WHERE gene1 = 'AT1G65440' OR gene2 = 'AT1G65440';" (Confirm gene columns for AtFusionDB)
              Query 2 (PFusionDB): `sql`: "SELECT * FROM \\"PFusionDB\\" WHERE (Gene1 = 'AT1G65440' OR Gene2 = 'AT1G65440') AND species LIKE 'Arabidopsis thaliana%';" (Confirm gene & species columns for PFusionDB)
            Provide comprehensive `display_columns_hint`.

        *   **FOR COMPARATIVE QUERIES:**
            `analysis_plan`: "Comparing '{{measure_being_compared}}' for '{{group_A}}' versus '{{group_B}}' in {{Database_Conceptual_Name}}."
            Generate **MULTIPLE SEPARATE query objects**.
            Example for "Compare tRF expression in Leaf and Root in PbtRF":
              Use `PbtRF.Expression` for categorical counts and `PbtRF.log2fold_change` (CAST AS REAL if needed) for numerical averages, as per CRITICAL RULES.
              Query 1 (Leaf - Categorical): `sql`: "SELECT Expression, COUNT(*) ... FROM PbtRF WHERE Tissue LIKE '%Leaf%' ... GROUP BY Expression;"
              Query 2 (Leaf - Numerical): `sql`: "SELECT AVG(CAST(log2fold_change AS REAL)) ... FROM PbtRF WHERE Tissue LIKE '%Leaf%' AND log2fold_change IS NOT NULL ...;"

        *   **FOR General Data Preview / Listing:**
            `purpose_type: data_preview`.
            `analysis_plan`: "Retrieving a general preview/list for '{user_query}' from {{Database_Conceptual_Name(s)}}."
            Construct SQL using `LIKE '%keyword%'` for text searches on relevant columns (identified via schema/KG).
       

4.  **ALWAYS Quote Identifiers:** To prevent syntax errors, you **MUST** wrap all table and column names in double quotes (`"`). Example: `SELECT "Organism" FROM "PbtRF" WHERE "tRF_type" = 'tRF-5';`. This is mandatory. Column names like `"05_heat"` require it.

6.  **Handling Vague Data Requests (e.g., "what data on peptide?"):**
    - If a user asks a "what data do you have on [topic]?" or "show me data about [topic]" question, interpret this as a `data_preview` request.
    - First, identify the most relevant database for the [topic] using the Knowledge Graph (e.g., 'peptides' -> `plantpepdb`).
    - Then, generate a simple `SELECT * FROM "RelevantTable" LIMIT {DISPLAY_ROW_LIMIT};` query to provide the user with a sample of the data. If the topic implies a filter (e.g., `...on peptide`), add the appropriate `WHERE` clause: `... FROM "PlantPepDB" WHERE "Species" LIKE 'Undaria%' LIMIT {DISPLAY_ROW_LIMIT};`.

7.  **Fuzzy Matching and Name Variations:**
    - User input for species, tissues, or genes may be misspelled, partial, or a common name (e.g., "solanaum", "tomato", "arabidopsis").
    - **Always use `LIKE` with wildcards (`%`)** for these text-based filters to maximize the chance of a match.
    - **Example:** For a query on "solanaum", you should generate `WHERE "species_column" LIKE '%Solanum%'`. The knowledge graph indicates 'Solanum lycopersicum' is in the DB, so this will match.

8.  **Numeric Columns Stored as TEXT:**
    - In `plantpepdb`, columns like `"IC50_Value"`, `"Sequence_Length"`, `"Avg_Molecular_Weight"` are TEXT. You **MUST** `CAST` them for any numerical operations (e.g., `WHERE CAST("Sequence_Length" AS INTEGER) > 10`).


10.  **Analyze Intent & Scope:**
    *   If `classified_intent` is "OUT_OF_SCOPE", "AMBIGUOUS", etc., the `analysis_plan` should state this and the `queries` array must be empty.
    *   If the query asks for something explicitly not in the databases (e.g., "siRNA data","lncRNA data"), state this in the `analysis_plan` and generate no SQL, even if the intent was misclassified.

12.  **Formulate a Plan:** Based on the user query and the rules above, decide on a strategy. Will you query one DB? Multiple? Will you perform an aggregation? This becomes your `analysis_plan`.
14.  **JSON Output Structure (MUST be followed precisely):**
    <JSON_START>
    {{
      "query_type": "...", // e.g., "data_retrieval", "metadata_answered_from_blob", "metadata_sql_fallback"
      "analysis_plan": "...", // User-friendly description of what the AI plans to do
      "direct_answer_from_metadata": "...", // Populated ONLY if query_type is "metadata_answered_from_blob", otherwise null or empty string
      "queries": [ // Empty if direct_answer_from_metadata is used, or for out_of_scope/ambiguous etc.
        {{
          "sql": "SELECT ... FROM \\"ActualTableName\\" ...", // Valid SQLite
          "target_table": "ActualTableName", // Actual SQLite table name from schemas_json
          "database_conceptual_name": "Conceptual Name for ActualTableName", // From database_table_mapping_info
          "description": "Brief description of this specific SQL query's goal.",
          "purpose_type": "data_preview OR entity_lookup_detail OR aggregation OR list_distinct_values_for_common_check_species OR ...", // Describes the query's role
          "display_columns_hint": ["colA", "colB"] // Optional: Suggested columns for UI display
        }}
        // ... more query objects if needed
      ]
    }}
    <JSON_END>

Generate ONLY the JSON object now, starting with <JSON_START> and ending with <JSON_END>.
"""
)

SUMMARY_INTERPRET_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "user_query", "analysis_plan_from_stage1",
        "textual_data_summary_from_python",
        "knowledge_graph_snippet", "conversation_summary",
        "query_type_from_stage1", "DISPLAY_ROW_LIMIT"
    ],
    template="""
You are a friendly and expert bioinformatics assistant. Your entire response MUST be a single JSON object.
*** You are an AI agent that serves PlantPepDB database of plant peptides having different functions and therapeutic activities. It contains 3848 peptide entries collected from 11 databases and 835 published research articles.***
Your primary task is to rephrase the provided "System-Generated Data Summary" into a concise, user-friendly, and conversational paragraph.
Accuracy is paramount:
- Your summary MUST be based *solely* on the facts and data presented in the "System-Generated Data Summary".
- DO NOT invent information, statistics, or interpretations not explicitly supported by the "System-Generated Data Summary".
- Use conceptual database names (e.g., Plant peptide database or PlantPepDB) if available in the "System-Generated Data Summary", instead of internal table names (e.g., PFusionDB).
- DO NOT describe your internal reasoning or the query processing steps, only the findings.
- DO NOT include a "suggestions" field or ask follow-up questions.
- DO NOT start your summary with "Bot:".

User Query: {user_query}
AI's Analysis Plan (from Stage 1): {analysis_plan_from_stage1}
System-Generated Data Summary (This is the factual data report you must rephrase. It includes results from SQL queries or common item analyses):
{textual_data_summary_from_python}
Relevant Knowledge Graph Snippet (Use this for "Semantic Equivalence Hints" to understand how different terms for species, tissues, etc., relate to each other. This is CRITICAL for accurate common item analysis): {knowledge_graph_snippet}
Recent Conversation Summary: {conversation_summary}
Query Type Determined by System (from Stage 1): {query_type_from_stage1}
UI Display Row Limit Constant: {DISPLAY_ROW_LIMIT} (Used for interpreting statements about data previews)

Task: Generate a JSON response with ONLY "summary" and "databases_conceptually_involved" fields.

**Instructions for "summary" FIELD (Derived from "System-Generated Data Summary" and contextualized by "Knowledge Graph Snippet"):**
0.  **Synthesize Findings Coherently:**
    *   **Do not mention the internal mechanics** such as the query number, the target table, or the exact number of rows found for each query part unless it is a specific count that answers the user's question (e.g., "how many..."). Focus on the biological insights.



**2. CRITICAL NORMALIZATION AND GROUPING RULES:**
Before performing any analysis or writing your summary, you MUST mentally apply these normalization rules to the data items:
*   **Lowercase and Trim:** Convert all items to lowercase and trim whitespace for comparison.
*   **Species Normalization:**
    *   'oryza sativa japonica' and 'oryza sativa indica' both map to 'oryza sativa'.
    *   'gh' maps to 'gossypium hirsutum'.
    *   'gb' maps to 'gossypium barbadense'.
*   **Tissue Normalization:**
    *   'rosette' and 'rosette leaf' both map to 'leaf'.
    *   'plant' maps to 'whole plant'.

3.  **APPLY NORMALIZATION AND GROUPING:**
    *   Before summarizing, you MUST apply the normalization and grouping rules found in the "**Semantic Equivalence Hints**" section of the provided `knowledge_graph_snippet`. This is your primary source for understanding how to group related terms (e.g., abbreviations, synonyms).
    *   All comparisons must be case-insensitive. For readability in the final text, replace underscores with spaces.

4.  **TASK: HANDLING "COMMON ITEM" ANALYSIS (VERY IMPORTANT):**
    *   **IF the `System-Generated Data Summary` contains a "Common 'ITEM_TYPE' Analysis" section, you MUST follow this logic precisely:**
        a.  **Extract Lists:** For each database, identify its list of distinct values from the summary.
        b.  **Normalize Each Item:** For every single item in every list, you must first convert it to lowercase and trim all leading/trailing whitespace.
        c.  **Group Semantically:** Apply the "Semantic Equivalence Hints" from the `knowledge_graph_snippet` to group related, normalized terms (e.g., after normalization, both 'rosette' and 'rosette leaf' become 'leaf').
        d.  **Find Intersection:** After normalization and grouping, determine the set of items that are present in **ALL** of the processed lists. These are the common items.
        e.  **Find Differences:** For each database, identify the items that are unique to it after the grouping process.
        f.  **Synthesize Report:** Clearly state the common items found. Then, for each database, list its unique items. **You must be precise and account for every item in the original lists.**
5.  **TASK: HANDLING INDIVIDUAL ENTITY LOOKUP (`entity_lookup_detail`):**
    *   The `System-Generated Data Summary` will contain the full record. Your task is to narrate the key details from that record in a readable paragraph.
    *   **Example:** "The query for PPepDB_1629 returned one record from the Plant Peptide Database. The peptide, named Gamma1-hordothionin, is sourced from *Hordeum vulgare* (Barley). It is classified as a Defensin with antimicrobial functions..."
5.  **TASK: HANDLING INDIVIDUAL ENTITY LOOKUP (`entity_lookup_detail`):**
    *   The `System-Generated Data Summary` will contain the full record. Your task is to narrate the key details from that record in a readable paragraph.
    *   **Example:** "The query for PPepDB_1629 returned one record from the Plant Peptide Database. The peptide, named Gamma1-hordothionin, is sourced from *Hordeum vulgare* (Barley). It is classified as a Defensin with antimicrobial functions..."

6.  **TASK: HANDLING INDIVIDUAL ENTITY LOOKUP (`entity_lookup_detail`):**
    *   The `System-Generated Data Summary` will contain the full record. Your task is to narrate the key details from that record in a readable paragraph.
    *   **Example:** "The query for PPepDB_1629 returned one record from the Plant Peptide Database. The peptide, named Gamma1-hordothionin, is sourced from *Hordeum vulgare* (Barley). It is classified as a Defensin with antimicrobial functions..."




7.  **Interpreting "--- Individual Query Part Summaries ---" from System Summary:**
    *   For each "Query Part":
        *   Refer to its "Purpose", "Records Found", or "Aggregate Result" from the system summary.
        *   **'data_preview', 'list_distinct_values':** State total "Records Found" (N). If N > 0 and the system summary mentions a preview, state that a preview (up to {DISPLAY_ROW_LIMIT} rows) is available. Highlight key findings from "Key Column Stats" if significant and present in the system summary (e.g., "top species was...", "values ranged from...").
        *   **'entity_lookup_detail':** If "Records Found" > 0, synthesize the key information from the record(s) into 1-2 sentences, directly answering the user's question about the entity, using only data from the system summary.
        *   **'aggregation' (e.g., COUNT, AVG):** Report the "Aggregate Result". E.g., "The system counted X items..."
        *   **Error/No Data:** If the system summary states an "Error" or "No data records found" for a query part, report this.

8.  **Overall Summary Construction:**
    *   Begin by acknowledging the user's query if appropriate, then present the findings.
    *   If the `analysis_plan_from_stage1` indicates the query was out-of-scope, ambiguous, or a general conversation handled without data retrieval, the summary should reflect that concisely (this scenario implies `textual_data_summary_from_python` might be minimal or state this directly).
    *   Focus on directly answering the user's query using the synthesized information.

**Instructions for "databases_conceptually_involved" FIELD:**
- List unique conceptual database names that are explicitly mentioned in the "System-Generated Data Summary" as having been queried or contributing information (even if that contribution was 'no data found' for a specific part). Prioritize conceptual names.
9.  **Interpreting "--- Individual Query Part Summaries ---" from System Summary:**
    *   For each "Query Part":
        *   Refer to its "Purpose", "Records Found", or "Aggregate Result" from the system summary.
        *   **'data_preview', 'list_distinct_values':** State total "Records Found" (N). If N > 0 and the system summary mentions a preview, state that a preview (up to {DISPLAY_ROW_LIMIT} rows) is available. Highlight key findings from "Key Column Stats" if significant and present in the system summary (e.g., "top species was...", "values ranged from...").
        *   **'entity_lookup_detail':** If "Records Found" > 0, synthesize the key information from the record(s) into 1-2 sentences, directly answering the user's question about the entity, using only data from the system summary.
        *   **'aggregation' (e.g., COUNT, AVG):** Report the "Aggregate Result". E.g., "The system counted X items..."
        *   **Error/No Data:** If the system summary states an "Error" or "No data records found" for a query part, report this.

10.  **Overall Summary Construction:**
    *   Begin by acknowledging the user's query if appropriate, then present the findings.
    *   If the `analysis_plan_from_stage1` indicates the query was out-of-scope, ambiguous, or a general conversation handled without data retrieval, the summary should reflect that concisely (this scenario implies `textual_data_summary_from_python` might be minimal or state this directly).
    *   Focus on directly answering the user's query using the synthesized information.

**Instructions for "databases_conceptually_involved" FIELD:**
- List unique conceptual database names that are explicitly mentioned in the "System-Generated Data Summary" as having been queried or contributing information (even if that contribution was 'no data found' for a specific part). Prioritize conceptual names.


**CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:**

11.  **Strictly Data-Driven:** Your summary **MUST** be based *only* on the information in the "System-Generated Data Summary". DO NOT invent facts, statistics, or interpretations. If the summary says "No data found" or "Error", you must report that.

12.  **Professional Tone & Formatting:**
    *   Write in clear, formal language. Avoid conversational filler like "Well," or "As you can see...".
    *   **Replace underscores (`_`)** in column names or values with spaces (e.g., `avg_log2fold_change` becomes `average log2 fold change`).
    *   Use conceptual database names (e.g., "Plant Fusion Database") found in the system summary.

13.  **How to Interpret "--- Common 'ITEM_TYPE' Analysis ---":**
    Your goal is to determine and report what is common for an `ITEM_TYPE` (e.g., species, tissue) across the involved databases. Follow this precise procedure:
    
    A. **Identify Involved Databases:** Note all databases listed for this common item check in the system summary.
    B. **Process Each Database's List:** For each database that provided a list of values:
        i. **Normalize:** For each value, convert to lowercase, trim whitespace, and replace underscores with spaces (e.g., 'Rosette Leaf' -> 'rosette leaf').
        ii. **Group Semantically:** Use the "Semantic Equivalence Hints" from the `knowledge_graph_snippet` to group related terms. For example, if the hint says `'leaf', 'rosette leaf'` map to 'leaf', then treat them as one conceptual item: 'leaf'.
        iii. **Use General Reasoning:** If a term isn't in the knowledge graph, use general biological reasoning to group obvious synonyms (e.g., 'flower' and 'floral organs').
    C. **Find Intersection:** After normalization and grouping, find the set of conceptual items that are present in **ALL** of the databases that successfully returned data.
    D. **Report the Findings:**
        - **State common items clearly:** "After analysis, 'leaf' and 'root' were identified as common tissues across the queried databases."
        - **Mention unique items:** "Additionally, the PbtRF Database uniquely listed 'gall', while the Plant Fusion Database contained 'seedling' and 'flower'."
        - **Acknowledge failures:** "A complete comparison was not possible as data retrieval from [Database Name] encountered an error." or "No tissues were found in [Database Name]."
        - **Explain your reasoning if needed:** "The analysis grouped 'rosette leaf' and 'leaf' as 'leaf' based on the provided semantic context."

14.  **How to Interpret "--- Individual Query Part Summaries ---":**
    *   **Data Previews/Lookups:** State the number of records found. Briefly highlight key findings from the "Key Column Stats" (e.g., "The search for antifungal peptides in the Plant Peptide Database returned 530 records. The peptides are active against a range of fungi, including several *Fusarium* species.").
    *   **Aggregations (COUNT, AVG):** Report the specific result directly. (e.g., "A total of 6,663 tRFs for *Oryza sativa* were found in the PbtRF Database.").
    *   **Errors/No Data:** State it plainly. (e.g., "The query for fusions in *Solanum* did not return any results from the Plant Fusion Database.").

15. **How to Handle `Tissue_Specificity_Index` Summaries:**
    *   If the user asked to "list" or "count" unique tissues from the Cotton Atlas, and the system summary provides data from the `Tissue_Specificity_Index` column, you MUST NOT present this as a simple list of unique tissues.
    *   You MUST explain the format. Start your summary by saying: "The Cotton Atlas stores tissue information as expression profiles, where each record lists multiple tissues and their specificity scores."
    *   If the system provides a COUNT, clarify it: "The system found 357 unique *tissue expression profiles*."
    *   If the system provides a data preview, give examples: "For example, one profile shows high specificity in Stigma (0.33) and Leaf (0.15), while another profile shows high specificity in root."

Output JSON Structure:
{{
  "summary": "[Your rephrased, conversational summary. Adhere strictly to the facts in the System-Generated Data Summary, contextualized by the Knowledge Graph Snippet. No 'Bot:' prefix. Underscores replaced by spaces.]",
  "databases_conceptually_involved": [ ... ] // List of conceptual names
}}

Produce the JSON object now.
"""
)
def process_query(query: str, conversation_id: Optional[str] = None, model_name: str = 'gpt-oss-120b', user_api_key: Optional[str] = None) -> Dict:
    processing_start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting query processing", extra={"query": query, "conversation_id": conversation_id})

    # This check is good.
    if not user_api_key:
        logger.error(f"[{request_id}] Critical error: API key not provided to process_query.")
        return {
            "summary": "Configuration error: The server is missing the API key needed to process the request.",
            "executed_queries_details": [], "databases_conceptually_involved": [],
            "metadata": {"error": "API key not provided."}
        }

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
    
    # Initialize LLMs with the user's key
    llm = ChatGroq(
        api_key=user_api_key,
        model_name=model_name,
        temperature=0.05,
        max_tokens=4096, 
        request_timeout=60.0
    )
    # --- STEP 2: USE THE CORRECT, EFFICIENT MODEL FOR INTENT CLASSIFICATION ---
    intent_llm = ChatGroq(
        api_key=user_api_key,
        model_name="llama-3.1-8b-instant", # Reverted to the faster model
        temperature=0.0, 
        max_tokens=500
    )
    # Prepare for Stage 0 Intent Classification
    available_databases_summary_str = "\n".join([f"- {name}: {desc}" for name, desc in DATABASE_MAPPING.items()])
    database_content_summary_for_intent = """
    - plantpepdb: Contains plant-derived peptides.
    - General: The databases DO NOT contain siRNA or other non-tRNA-derived lncRNAs outside of the cotton-specific CoNCRAtlasdb.
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

    intent_input_data = {
        "user_query": sanitized_user_query,
        "conversation_summary": conversation_summary_for_intent,
        "available_databases_summary": available_databases_summary_str,
        "database_content_summary": database_content_summary_for_intent
    }
    classified_intent = "UNKNOWN" # This is Stage 0 intent
    try:
        logger.info(f"[{request_id}] Invoking LLM Stage 0 for Intent Classification...")
        s0_invoke_start = time.time()
        raw_intent_response = invoke_groq_model(INTENT_CLASSIFICATION_PROMPT_TEMPLATE | intent_llm, intent_input_data)
        s0_invoke_end = time.time()
        classified_intent = raw_intent_response.strip().upper().replace("\"", "")
        logger.info(f"[{request_id}] LLM Stage 0 Intent Classification completed in {s0_invoke_end - s0_invoke_start:.2f}s. Intent: {classified_intent}")
    except Exception as e_intent:
        logger.error(f"[{request_id}] Stage 0 Intent Classification error: {str(e_intent)}", exc_info=True)
        classified_intent = "DATA_RETRIEVAL" # Fallback if Stage 0 fails

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
    knowledge_graph_str = f"""
Concise Knowledge Graph for SQLite DB ({DB_FILE}):
- Tables & Primary Use (Conceptual Names from DATABASE_MAPPING):
  - {DATABASE_MAPPING.get("plantpepdb", "plantpepdb")}: Plant peptides. Species col: `Plant_Source_Scientific`. Numeric cols are TEXT, require CAST.


- **Default Database Grouping for General Queries (IMPORTANT):**
  - For general "peptide" queries: Use ONLY plantpepdb.

- Categories for Common Item Checks (when user asks "what is common across [category]?"):
  - "peptide database": plantpepdb

- Species Name Hint: For "rice", use `LIKE 'Oryza sativa%'`. For "cotton", use `LIKE 'Gossypium%'`. For "Arabidopsis", use `LIKE 'Arabidopsis thaliana%'` (except for AtFusionDB which is implicitly Arabidopsis).
- Sample Species: {species_info_str}
- Sample Tissues: {tissues_info_str}
- Explicit Scope: The databases DO NOT contain siRNA. For miRNA and lncRNA, data is ONLY available for cotton species in the CoNCRAtlasdb.

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
             summary_msg = "The API key used for this request has exceeded its rate limit. Please use a new API key to continue accessing the service. You can manage your API keys at https://console.groq.com/keys."
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


