
import openai
import base64
import requests
import argparse
import os

from openai import OpenAI
from openai_api import OPENAI_API_KEY

# Set Image Directory
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
    The first attached image is the original image, and the second attached image is the generated image. 
    Perform a detailed comparison between the original and generated images, focusing on differences in content, style, details, \
      composition, color, and mood, and articulate three precise alterations to the text prompt that could enhance the accuracy of \
        the recreated image. Be highly specific and actionable, and make unambiguous suggestions. 
    Then mofify the original prompt to incorporate your suggestions. 
    The original prompt is: 
    {last_prompt}

    Please only reply with a text prompt, and do not include any other text in your response.
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

def ask_dalle_for_image(image_folder_dir, original_image_name, iter, prompt, client, args):

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

    # make base 64 image into a png file
    with open(f'{image_folder_dir}/from_{original_image_name}_iter_{iter+1}.png', 'wb') as f:
        f.write(base64.b64decode(temp_image_base64))


if __name__ == "__main__":
  # Get Image Directory
  args = parse_args()
  print(args)

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
  image_prefix = f'{args.image_dir}/{args.original_image_name}'
  if os.path.isfile(f'{image_prefix}.jpg'):
    original_image = f'{image_prefix}.jpg'
  elif os.path.isfile(f'{image_prefix}.png'):
     original_image = f'{image_prefix}.png' 
  else:
      print("Image not found (must be .jpg or .png).")
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

  print("This is init prompt from GPT4-V: " + init_prompt)        

  ## Use the initial prompt to generate a new image

  instruction_init_image = 'Use the detailed prompt provided to generate a single image that aims to replicate the attached \
    image as closely as possible, without deviating from the details specified in the prompt: ' + init_prompt

  response = client.images.generate(
    model="dall-e-3",
    prompt=instruction_init_image,
    size="1024x1024",
    quality="standard",
    n=1,
    response_format='b64_json'
  )

  init_image_base64 = response.data[0].b64_json

  # make base 64 image into a png file
  with open(f'{args.image_dir}/from_{args.original_image_name}_iter_0.png', 'wb') as f:
      f.write(base64.b64decode(init_image_base64))

  temp_prompt = init_prompt

  for i in range(5):

      print('------------------')
      print('----Temp------')
      print('------------------')

      prompt = ask_chatgpt_for_prompt(args.image_dir, args.original_image_name, i, original_image_base64, temp_prompt)

      ask_dalle_for_image(args.image_dir, args.original_image_name, i, prompt, client)

      temp_prompt = prompt

      print("This is temp prompt from GPT4-V: " + temp_prompt)
