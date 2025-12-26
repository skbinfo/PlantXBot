from flask import Flask, send_from_directory, request, jsonify
from werkzeug.serving import run_simple
import os
import math

# Import the process_query function from your generic bot file
# Assumes the previous file was saved as 'demo_bot.py'
try:
    from demo_bot import process_query as demo_process_query
except ImportError:
    print("CRITICAL ERROR: Could not import 'demo_bot.py'. Make sure the file exists in the same directory.")
    # Define a dummy function to prevent immediate crash, though app won't work correctly
    def demo_process_query(*args, **kwargs):
        return {"error": "demo_bot.py not found on server."}

app = Flask(__name__, static_folder='public')

# A dictionary to map bot names (from the URL) to their processing functions.
# Since this is a demo app, we map 'demo' and 'generic' to your new bot.
BOT_PROCESSORS = {
    'demo': demo_process_query,
    'generic': demo_process_query,
    'custom': demo_process_query
}

@app.route('/')
def index():
    # Ensure you have a 'public' folder with an index.html, 
    # otherwise create a simple one or remove this route.
    if os.path.exists(os.path.join('public', 'index.html')):
        return send_from_directory('public', 'index.html')
    return "Demo App Running. Send POST requests to /query/demo"

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

@app.route('/query/<bot_name>', methods=['POST'])
def query(bot_name):
    # Check if the requested bot exists in our mapping
    if bot_name not in BOT_PROCESSORS:
        return jsonify({
            "error": f"Invalid bot name '{bot_name}'. Available bots: {list(BOT_PROCESSORS.keys())}"
        }), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    # EXTRACT THE API KEY FROM THE REQUEST
    # The demo_bot requires this to function
    user_api_key = data.get('api_key')
    if not user_api_key:
        return jsonify({"error": "Missing 'api_key' parameter in JSON payload"}), 400
        
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' parameter in JSON payload"}), 400

    user_query = data['query']
    conversation_id = data.get('conversation_id')
    
    # Note: The demo_bot.py defined in the previous step accepts:
    # process_query(query, conversation_id, user_api_key)
    # It does not strictly require 'model_name' as an argument in the signature,
    # but strictly passing what matches the function definition prevents TypeErrors.

    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"error": "Query must be a non-empty string"}), 400

    # Get the appropriate processing function from the dictionary
    process_function = BOT_PROCESSORS[bot_name]

    try:
        # CALL THE PROCESSING FUNCTION
        response_data = process_function(
            user_query, 
            conversation_id=conversation_id, 
            user_api_key=user_api_key
        )
    except TypeError as e:
        # Fallback if the bot function signature expects different arguments
        return jsonify({"error": f"Server Configuration Error: Function signature mismatch. {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Internal Processing Error: {str(e)}"}), 500
    
    # A simple way to clean NaN/Infinity if they occur for JSON serialization
    # (Pandas often produces NaNs which are invalid in standard JSON)
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
    print(f"Starting Demo App...")
    print(f"Registered Bots: {list(BOT_PROCESSORS.keys())}")
    # Run the app on port 5001
    run_simple('0.0.0.0', 5001, app, use_reloader=True)
