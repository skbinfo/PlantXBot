from flask import Flask, send_from_directory, request, jsonify
from werkzeug.serving import run_simple
import os
import math

# Import the process_query function from each bot's file
from trfbot import process_query as trf_process_query
from fusionbot import process_query as fusion_process_query
from ampbot import process_query as amp_process_query
from cottonbot import process_query as cotton_process_query
from ncRNA_bot import process_query as ncrna_process_query
from PVsiBot import process_query as pvsibot_process_query
from AtFusionBot import process_query as atfusion_process_query
from pfusionbot import process_query as pfusion_process_query
from ptrfbot import process_query as ptrf_process_query
from pbtrfbot import process_query as pbtrf_process_query
from ptncbot import process_query as ptnc_process_query
from athisomirbot import process_query as athisomir_process_query
from anninterbot import process_query as anninter_process_query
from alncbot import process_query as alncbot_process_query
from ptrnabot import process_query as ptrnabot_process_query

app = Flask(__name__, static_folder='public')

# A dictionary to map bot names to their processing functions
BOT_PROCESSORS = {
    'trf': trf_process_query,
    'fusion': fusion_process_query,
    'amp': amp_process_query,
    'cotton': cotton_process_query,
    'ncrna': ncrna_process_query,
    'pvsi': pvsibot_process_query,
    'atfusion': atfusion_process_query,
    'pfusion': pfusion_process_query,
    'ptrfbot': ptrf_process_query,
    'pbtrfbot': pbtrf_process_query,
    'ptncbot': ptnc_process_query,
    'athisomirbot': athisomir_process_query,
    'anninterbot': anninter_process_query,
    'alncbot': alncbot_process_query,
    'ptrnabot': ptrnabot_process_query,
}

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

@app.route('/query/<bot_name>', methods=['POST'])
def query(bot_name):
    if bot_name not in BOT_PROCESSORS:
        return jsonify({"error": "Invalid bot name"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    # EXTRACT THE API KEY FROM THE REQUEST
    user_api_key = data.get('api_key')
    if not user_api_key:
        return jsonify({"error": "Missing 'api_key' parameter in JSON payload"}), 400
        
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter in JSON payload"}), 400

    user_query = data['query']
    conversation_id = data.get('conversation_id')
    
    # Get the model from the request, with a default fallback
    model_name = data.get('model', 'gpt-oss-120b') # Default to gpt-oss-120b if not provided

    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"error": "Query must be a non-empty string"}), 400

    # Get the appropriate processing function from the dictionary
    process_function = BOT_PROCESSORS[bot_name]

    # CALL THE PROCESSING FUNCTION, PASSING THE API KEY
    response_data = process_function(
        user_query, 
        conversation_id=conversation_id, 
        model_name=model_name,
        user_api_key=user_api_key
    )
    
    # A simple way to clean NaN/Infinity if they occur for JSON serialization
    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_json(elem) for elem in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    clean_response_data = clean_json(response_data)
    
    return jsonify(clean_response_data)

if __name__ == '__main__':
    # Use werkzeug's run_simple to handle the proxying correctly
    run_simple('127.0.0.1', 5001, app, use_reloader=True)
