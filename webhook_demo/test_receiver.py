from flask import Flask, request, jsonify
import logging

# Disable Flask's default verbose logging for a cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook_receiver():
    """
    This endpoint listens for POST requests, prints the JSON data it receives,
    and returns a success response.
    """
    print("✅ --- Webhook Received! --- ✅")

    # Get the JSON data sent by your subscription API
    data = request.get_json()

    # Pretty-print the received JSON data to the console
    import json
    print(json.dumps(data, indent=2))

    print("-----------------------------\n")

    # Return a 200 OK response to let the sender know we received it
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    print("🚀 Starting simple webhook receiver on http://127.0.0.1:5000/webhook")
    app.run(port=5000)