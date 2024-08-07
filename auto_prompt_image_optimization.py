import base64
import requests
import argparse

from openai import OpenAI
from openai_api import OPENAI_API_KEY


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == "__main__":

  # OpenAI API Key
  api_key = OPENAI_API_KEY

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  # Getting the base64 string
  base64_image = encode_image(image_path)


  # Input original image, user query and prompt below
  image_path = ''
  user_query = ''
  prompt = ''


  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"Do you think this picture correctly depicts {user_query}? Please briefly explain your answer and if not, please help me modify the prompt {prompt}. Please return the explanation and your modification in a json."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  print(response.json())