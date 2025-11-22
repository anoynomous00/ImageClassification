from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import logging
import traceback
from flask_cors import CORS

# Ensure static_folder uses absolute path so Flask reliably finds files
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
CORS(app)

# Load model & processor
# Load model & processor (Switching to Large model for better accuracy)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
clip_model.eval()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    # serve the main index from the configured static folder
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files if requested, otherwise fall back to index.html (useful for SPA and prevents directory listing)
@app.route("/<path:path>")
def catch_all(path):
    static_path = os.path.join(app.static_folder, path)
    if os.path.exists(static_path) and os.path.isfile(static_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

# Change the route to accept GET as well as POST and show instructions on GET
@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "GET":
        # brief HTML so visiting /classify in a browser is informative
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>/classify</title></head><body style='font-family:Arial,Helvetica,sans-serif;"
            "padding:24px;color:#222;'>"
            "<h2>/classify endpoint</h2>"
            "<p>This endpoint accepts <strong>POST</strong> requests with form fields:"
            "<ul><li><code>image</code> — uploaded image file</li>"
            "<li><code>classes</code> — comma-separated class names (e.g. <code>cat,dog,bird</code>)</li></ul>"
            "Use the web UI at <a href='/'>/</a> or submit a POST with curl or your app.</p>"
            "</body></html>"
        ), 200, {"Content-Type": "text/html"}

    try:
        logger.info("Received /classify request")
        if "image" not in request.files:
            logger.warning("Missing image in request")
            return jsonify({"error": "Missing image"}), 400

        image_file = request.files["image"]
        logger.info("Image file name: %s, size: %s", getattr(image_file, 'filename', None), request.content_length)
        image = Image.open(image_file).convert("RGB")

        prompts = []
        # Check for raw prompts first (newline separated)
        if "prompts" in request.form and request.form["prompts"].strip():
            raw_prompts = [p.strip() for p in request.form["prompts"].split('\n') if p.strip()]
            # Deduplicate while preserving order
            seen = set()
            prompts = []
            for p in raw_prompts:
                if p not in seen:
                    prompts.append(p)
                    seen.add(p)
        # Fallback to classes if prompts not provided
        elif "classes" in request.form and request.form["classes"].strip():
            class_names = [c.strip() for c in request.form["classes"].split(',') if c.strip()]
            prompts = [f"This is an image of a {c}." for c in class_names]
        
        if not prompts:
            logger.warning("No valid prompts or classes provided")
            return jsonify({"error": "No valid prompts provided"}), 400

        # Preprocess inputs
        logger.info("Preprocessing inputs with %d prompts", len(prompts))
        
        # Preprocess inputs
        logger.info("Preprocessing inputs with %d prompts", len(prompts))
        
        # If only one prompt is provided, add contrastive prompts
        # to prevent 100% confidence on a single class.
        # We track the indices of these auto-added prompts to penalize them slightly.
        auto_negative_indices = []
        if len(prompts) == 1:
            # Add specific negatives
            negatives = [
                "This is an image of a different object.",
                "This is an image of the background."
            ]
            for neg in negatives:
                prompts.append(neg)
                auto_negative_indices.append(len(prompts) - 1)
            logger.info("Added contrastive prompts for single-input mode")

        image_inputs = clip_processor(images=image, return_tensors="pt")
        text_inputs = clip_processor(text=prompts, padding=True, return_tensors="pt")

        # Move tensors to device
        def move_inputs(inp_dict, dev):
            return {k: v.to(dev) for k, v in inp_dict.items()}

        image_inputs = move_inputs(image_inputs, device)
        text_inputs = move_inputs(text_inputs, device)

        def run_inference(dev):
            logger.info("Running inference on device: %s", dev)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                
                # Normalize features for cosine similarity
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate raw cosine similarity (range -1 to 1)
                # This gives an absolute measure of how well the image matches the text
                raw_similarity = (image_features @ text_features.T).cpu().numpy().flatten()
                
                # Calculate probabilities using logit_scale (standard CLIP behavior)
                logit_scale = clip_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.T
                
                # Apply penalty to auto-added negative prompts to prefer user's prompt
                if auto_negative_indices:
                    penalty = 2.0
                    for idx in auto_negative_indices:
                        logits_per_image[0, idx] -= penalty
                
                probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
            return probs, raw_similarity

        try:
            probs, raw_scores = run_inference(device)
        except RuntimeError as e:
            # Fallback to CPU on OOM or other CUDA errors
            errmsg = str(e).lower()
            logger.error("Inference failed on %s: %s", device, errmsg)
            if "out of memory" in errmsg or "cuda" in errmsg:
                logger.info("Attempting fallback to CPU")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # move model and inputs to CPU and retry
                clip_model.to("cpu")
                image_inputs = move_inputs(image_inputs, "cpu")
                text_inputs = move_inputs(text_inputs, "cpu")
                probs, raw_scores = run_inference("cpu")
                # move model back to original device if it was CUDA and available
                if device.type == "cuda":
                    try:
                        clip_model.to(device)
                    except Exception:
                        logger.warning("Could not move model back to CUDA device")
            else:
                raise

        # Format results
        result = [{"label": p, "probability": float(prob), "similarity": float(sim)} for p, prob, sim in zip(prompts, probs, raw_scores)]
        # Sort by probability descending
        result.sort(key=lambda x: x["probability"], reverse=True)
        
        best = result[0]
        
        # Threshold check: If the best similarity is too low, reject the prediction.
        # Adjusted for ViT-Large model which is more robust.
        similarity_threshold = 0.23
        if best["similarity"] < similarity_threshold:
            best_label_display = "Uncertain / None of the above"
            explanation = f"No strong match found (Best similarity {best['similarity']:.4f} is below threshold {similarity_threshold})."
            # We don't change the probabilities list, just the top-level prediction display
            prediction_output = best_label_display
        else:
            prediction_output = best["label"]
            explanation = f"The image matches best with the prompt '{best['label']}' (Confidence: {best['probability']:.2%}, Similarity: {best['similarity']:.4f})."
        
        logger.info("Prediction: %s (%.4f)", prediction_output, best["probability"])
        return jsonify(
            prediction=prediction_output, 
            probabilities=result,
            explanation=explanation
        )

    except Exception as e:
        logger.error("Error in /classify: %s", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Disable caching for static assets (helps ensure frontend updates are loaded)
@app.after_request
def add_no_cache_headers(response):
    try:
        # only disable cache for static files and index page
        if request.path.startswith("/static/") or request.path == "/" or request.path.endswith("index.html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    except Exception:
        pass
    return response

if __name__ == "__main__":
    # Run on port 5001 to avoid conflict with VS Code Live Server (5500)
    logger.info("Starting Flask server at http://127.0.0.1:5001 (serving static files from %s)", app.static_folder)
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
