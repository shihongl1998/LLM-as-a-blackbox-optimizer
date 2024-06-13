def get_message(prompt_method, good_templates_str, bad_templates_str, random_templates_str ,num_templates_from_gpt, dataset):
    if prompt_method == 'method_1':
        p = f"""
Hi ChatGPT, I have two lists of templates: one with good templates and the other with bad templates. There are characteristics that make a template good or bad. Based on these characteristics, give me {num_templates_from_gpt} better template. 
Here is the list of good templates:
{good_templates_str}

Here is the list of bad templates: 
{bad_templates_str}

Here are my requirements:
- Please only reply the template.
- The template should be less than 15 words.
- The template should have similar structure to the above template.
- Only the template should start with '- ' in a separate line.    
"""

    elif prompt_method == 'method_x':
        # You can design your prompt here
        p = f""" 
"""

    return p
    
