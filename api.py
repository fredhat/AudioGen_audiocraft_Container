from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
from os import environ, path
from azure.storage.blob import BlobServiceClient
import uuid
from quart import Quart, jsonify, request, json
from awaits.awaitable import awaitable
import requests
import asyncio
import pathlib

app = Quart(__name__)

MAX_TOKENS = 768
MAIN_CONTAINER = 'music'
DOWNLOAD_URL = 'https://audiocraftgen.blob.core.windows.net/'
blob_service_client = BlobServiceClient.from_connection_string(environ.get("AZURE_STORAGE_CONNECTION_STRING"))
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

@awaitable
def background_gen(prompt, uid):
    url = "https://audiocraft-llama2.azurewebsites.net/gen"
    headers = {"Content-Type":"application/json"}
    data = {"prompt":prompt}
    response = requests.post(url, headers=headers, json=data)
    data = json.loads(response.text)
    if 'prompt' in data:
        inputs = processor(
            text=" ".join(data['prompt']),
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        sampling_rate = model.config.audio_encoder.sampling_rate
        wav_id = uid + '.wav'
        scipy.io.wavfile.write(wav_id, rate=sampling_rate, data=audio_values[0, 0].numpy())
        with open(file=path.join('./', wav_id), mode="rb") as data:
            blob_client = blob_service_client.get_blob_client(container=MAIN_CONTAINER, blob=wav_id)
            blob_client.upload_blob(data)
            data.close()
        file_to_rem = pathlib.Path("./" + wav_id)
        file_to_rem.unlink()

async def gen_music(prompt, uid):
    await background_gen(prompt, uid)

@app.route('/', methods=['GET'])
async def test():
    return jsonify({'message': 'Running'})

@app.route('/gen', methods=['POST'])
async def generate():
    try:
        data = json.loads(await request.data)
        if 'prompt' in data:
            id = uuid.uuid1().hex
            app.add_background_task(gen_music, data['prompt'], id)
            return jsonify({'id':id})
        else:
            return jsonify({"error": "Missing required parameters"}), 400
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@app.route('/music', methods=['GET'])
async def get_all():
    try:
        blobs = blob_service_client.get_container_client(container=MAIN_CONTAINER).list_blob_names()
        return jsonify({"music":list(blobs)})
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@app.route('/music/<uid>', methods=['GET'])
async def get_id(uid):
    try:
        blob_client = blob_service_client.get_blob_client(container=MAIN_CONTAINER, blob=uid+'.wav')
        if blob_client.exists():
            file_path = DOWNLOAD_URL + MAIN_CONTAINER + '/' + uid+'.wav'
            return jsonify({"link":file_path})
        else:
            return ('', 204)
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)