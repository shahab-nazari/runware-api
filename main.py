from flask import Flask, request, jsonify
from runware import Runware, IImageInference
import asyncio

app = Flask(__name__)

RUNWARE_API_KEY = "aLKwRIWrs3lNob740aORVoaX8O4QTZbf"

@app.route("/")
def home():
    return "Runware API is running."

@app.route("/image", methods=["GET"])
def image():
    prompt = request.args.get("prompt", default="a futuristic city")
    image_url = asyncio.run(generate_image(prompt))
    if image_url:
        return jsonify({"image_url": image_url})
    return jsonify({"error": "Failed to generate"}), 500

async def generate_image(prompt):
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()
    request_image = IImageInference(
        positivePrompt=prompt,
        model="civitai:101055@128078",
        numberResults=1,
        height=512,
        width=512,
    )
    images = await runware.imageInference(requestImage=request_image)
    if images:
        return images[0].imageURL
    return None
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
