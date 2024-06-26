import os
import requests
import base64


def GPT_4V(image_path, prompt="What’s in this image?"):
    GPT4V_KEY = "XXX"
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    payload = {
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
                }
            ]
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    GPT4V_ENDPOINT = "https://llm-testing-vision.openai.azure.com/openai/deployments/gpt4-vision/chat/completions?api-version=2023-07-01-preview"

    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    return response.json()['choices'][0]['message']['content'].strip().replace('\n', ' ') + '\n'
