import openai
import base64
import requests
import argparse
import os
import json

from openai import OpenAI
from openai_api import OPENAI_API_KEY

# Set image directory and original image name
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Experimental folder directory.', default=f'{os.getenv("HOME")}/LLM-as-a-blackbox-optimizer/Images')
    parser.add_argument('--original_image_name', type=str, help="Name of original image", default="OriginalImage")
    args = parser.parse_args()
    return args

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to ask chatgpt for a prompt
def ask_chatgpt_for_prompt(image_folder_dir, original_image_name, iter, original_image_base64, last_prompt):

    prompt = f'''
    The first attached image is the original image, and the second attached image is the generated image. \
      Conduct a thorough comparison, emphasizing discrepancies in elements such as lighting, textures, perspective, facial expressions, \
        object interactions, and background elements, in addition to content, style, details, composition, color, and mood. The original \
          prompt is: {last_prompt}. Use this analysis to articulate two targeted modifications to the text prompt that would rectify \
            identified issues and bring the generated image into closer alignment with the original. Ensure these modifications are \
              precise, likely to be interpreted correctly by an AI, and pertain to aspects like descriptive adjectives, spatial \
                relations, color, and lighting terms. Slightly alter the original prompt directly to include these modifications. \
                  Respond only with the revised text prompt and exclude any additional commentary.
    '''

    original_image_base64 = original_image_base64
    candidate_image = f'{image_folder_dir}/from_{original_image_name}_iter_{iter}.png'
    candidate_image_base64 = encode_image(candidate_image)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{original_image_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{candidate_image_base64}"
                }
            }
            ]
        }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    temp_prompt = response.json()["choices"][0]["message"]["content"]

    return temp_prompt

def ask_dalle_for_image(image_folder_dir, original_image_name, iter, prompt, client):

    instruction_temp_image = prompt

    response = client.images.generate(
        model="dall-e-3",
        prompt=instruction_temp_image,
        size="1024x1024",
        quality="standard",
        n=1,
        response_format='b64_json'
    )

    temp_image_base64 = response.data[0].b64_json

    # Convert base 64 image into a png file
    with open(f'{image_folder_dir}/from_{original_image_name}_iter_{iter+1}.png', 'wb') as f:
        f.write(base64.b64decode(temp_image_base64))


if __name__ == "__main__":
  args = parse_args()
  print(args)

  # Save Text Prompts for Reference
  text_prompts_and_responses = dict()

  # OpenAI API Key
  api_key = OPENAI_API_KEY

  # Initialize the OpenAI client
  client = OpenAI(api_key=OPENAI_API_KEY)

  print('------------------')
  print('----Init------')
  print('------------------')
  ## Generate text prompt for an initial candidate image
  instruction_init_prompt = 'Generate a detailed text prompt that can be used to recreate the attached image using an image generator. \
    Please only reply with a text prompt, and do not include any other text in your response.'
  text_prompts_and_responses['initial_prompt_inversion'] = instruction_init_prompt
  image_prefix = f'{args.image_dir}/{args.original_image_name}'
  if os.path.isfile(f'{image_prefix}.jpg'):
    original_image = f'{image_prefix}.jpg'
  elif os.path.isfile(f'{image_prefix}.png'):
     original_image = f'{image_prefix}.png' 
  else:
      print("Image not found (must be .jpg or .png).")
      exit()
  original_image_base64 = encode_image(original_image)

  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": instruction_init_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{original_image_base64}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  init_prompt = response.json()["choices"][0]["message"]["content"]
  text_prompts_and_responses['initial_prompt'] = init_prompt

  print("Initial Prompt from GPT4-V: \n" + init_prompt)

  ## Use the initial prompt to generate a new image
  instruction_init_image = 'Create this exact image (ensure only one is generated) without any changes to the prompt: ' + init_prompt
  text_prompts_and_responses['initial_image_generation_prompt'] = instruction_init_image

  response = client.images.generate(
    model="dall-e-3",
    prompt=instruction_init_image,
    size="1024x1024",
    quality="standard",
    n=1,
    response_format='b64_json'
  )

  init_image_base64 = response.data[0].b64_json

  # Convert base 64 image into a png file
  with open(f'{args.image_dir}/from_{args.original_image_name}_iter_0.png', 'wb') as f:
      f.write(base64.b64decode(init_image_base64))

  temp_prompt = init_prompt

  for i in range(5):
      print('------------------')
      print('----Temp------')
      print('------------------')

      prompt = ask_chatgpt_for_prompt(args.image_dir, args.original_image_name, i, original_image_base64, temp_prompt)
      text_prompts_and_responses[f'prompt_iter_{i}'] = prompt

      temp_prompt = prompt

      print(f"Refined Prompt from GPT4-V [iteration {i}]: \n" + temp_prompt)

      ask_dalle_for_image(args.image_dir, args.original_image_name, i, prompt, client)

  with open(f'{args.image_dir}/{args.original_image_name}.json', 'w') as outfile:
    json.dump(text_prompts_and_responses, outfile)