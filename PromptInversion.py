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
def ask_chatgpt_for_prompt(image_folder_dir, original_image_name, iter, original_image_base64, last_prompt, text_prompts_and_responses):

    prompt = f'''
    The statement in triple quotes is extra important: """The first attached image is the original image, and the second attached image is the generated image."""
    Act as a prompt inversion tool. Compare the original and generated images and analyze: Subject Matter, Color Scheme, Composition, Detail Level, Style and Aesthetic, Perspective and Depth, Lighting and Shadow, Scale and Proportions, Emotional Tone, Context and Environment, Texture and Material, Facial Features and Expressions (if applicable), Movement and Dynamics, Symbolism and Metaphors, Cultural and Historical Accuracy, Consistency, Uniqueness and Creativity, Anatomical Accuracy (if applicable), Alignment with Theme or Concept, and Localization of Objects and Their Pose.
    Think to yourself three important augmentations to the text prompt to be more similar to the original image.
    Think to yourself three important augmentations to the text prompt to be more dissimilar to the generated image.
    Be actionable, specific, and unambiguous in the changes and do not directly refer to the original image in the text prompt.
    First, incorporate all 4 changes together to update the original prompt in-place: {last_prompt}
    Then evaluate yourself if the new text prompt has improved to be more representative of the original image, otherwise, repeat the process until you are satisfied with the improvements.
    Respond only with the revised text prompt and exclude any additional commentary.
    '''
    text_prompts_and_responses['iterative_comparison_prompt'] = prompt

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
        "max_tokens": 300,
        "temperature": 0.7
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
  instruction_init_prompt = 'Generate a semi-detailed text prompt to recreate the attached image using an image generator using only 4 sentences. Please only reply with a text prompt, and do not include any other text in your response.'
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
      "max_tokens": 300,
      "temperature": 0.7
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  init_prompt = response.json()["choices"][0]["message"]["content"]
  text_prompts_and_responses['initial_prompt'] = init_prompt

  print("Initial Prompt from GPT4-V: \n" + init_prompt)

  ## Use the initial prompt to generate a new image
  instruction_init_image = 'Create this exact image without any changes to the prompt: ' + init_prompt
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
      prompt = ask_chatgpt_for_prompt(args.image_dir, args.original_image_name, i, original_image_base64, temp_prompt, text_prompts_and_responses)

      text_prompts_and_responses[f'prompt_iter_{i+1}'] = prompt
      with open(f'{args.image_dir}/{args.original_image_name}.json', 'w') as outfile:
        json.dump(text_prompts_and_responses, outfile)

      temp_prompt = prompt

      print(f"Refined Prompt from GPT4-V [iteration {i+1}]: \n" + temp_prompt)
      try:
        ask_dalle_for_image(args.image_dir, args.original_image_name, i, prompt, client)
      except Exception as e:
        print(f"An error occurred: {e}")
        text_prompts_and_responses[f'error_message'] = str(e)
        with open(f'{args.image_dir}/{args.original_image_name}.json', 'w') as outfile:
          json.dump(text_prompts_and_responses, outfile)
        exit()
