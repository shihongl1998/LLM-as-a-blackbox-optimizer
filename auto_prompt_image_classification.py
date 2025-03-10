import os
import torch
import argparse
import openai
from tqdm import tqdm

from engine.datasets import dataset_classes
from engine.tools.utils import makedirs, set_random_seed
from engine import clip
from features import prepare_image_dataset
from prompt_pool import get_message
from utils import get_experiment_name, validate_message, getTopBotK, get_best5_average, \
                        record_request_response, plot_tendency, write_templates_to_excel, eval

from init_templates import INIT_TEMPLATES
from openai_api import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

parser = argparse.ArgumentParser()

###########################
# Prompting Config
###########################
parser.add_argument(
    "--gpt_model", type=str, default="gpt-3.5-turbo-0301", choices=["gpt-3.5-turbo-0301", "gpt-4-0314"],
    help="the model used for prompting"
)
parser.add_argument(
    "--prompt_method", type=str, default="good_bad_better",
    help="the prompting method"
)
parser.add_argument(
    "--num_iters", type=int, default=10,
    help="number of iterations for prompting",
)
parser.add_argument(
    "--temperature", type=float, default=1,
    help="default temperature for ChatGPT",
)
parser.add_argument(
    '--init_templates', type=str, default='laioncoco', choices=INIT_TEMPLATES.keys(),
    help="the initial templates for prompting",
)
parser.add_argument(
    '--laion_seed', type=int, default=0,
    help="seed for selecting laion templates",
)
parser.add_argument(
    '--sample_size', type=int, default=80,
    help="number of sampling size of selecting laion templates",
)
parser.add_argument(
    "--template_pool_size", type=int, default=3,
    help="the size of template pool to show for ChatGPT (default: 3)",
)
parser.add_argument(
    "--eval", type=str, default='train', choices=['train', 'val', 'train_val', 'test'],
    help="the split used for evaluating the templates",
)
parser.add_argument(
    "--num_templates_from_gpt", type=int, default=1,
    help="the number of templates we asked from chatgpt",
)
parser.add_argument(
    "--run", type=int, default=0,
    help="the i'th run of the experiment",
)

###########################
# Directory Config (modify if using your own paths)
###########################
parser.add_argument(
    "--data_dir", type=str, default="./data", help="where the dataset is saved",
)
parser.add_argument(
    "--indices_dir", type=str, default="./indices", help="where the (few-shot) indices is saved",
)
parser.add_argument(
    "--feature_dir", type=str, default="./features", help="where to save pre-extracted features",
)
parser.add_argument(
    "--result_dir", type=str, default="./experiments", help="where to save experiment results",
)

###########################
# Dataset Config (few_shot_split.py)
###########################
parser.add_argument(
    "--dataset", type=str, default="imagenet", choices=dataset_classes.keys(),
    help="dataset name",
)
parser.add_argument(
    "--shot", type=int, default=1, choices=[1, 2, 4, 8, 16],
    help="train shot number. note that val shot is automatically set to min(4, shot)",
)
parser.add_argument(
    "--seed", type=int, default=1, help="seed number",
)

###########################
# Feature Extraction Config (features.py)
###########################
parser.add_argument(
    "--clip_encoder", type=str, default="RN50", choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101"],
    help="specify the clip encoder to use",
)
args = parser.parse_args()


def main(args):
    if args.seed is not None:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # get experiment name
    # CAUTION: Please modify this if you add any new arguments
    config = get_experiment_name(args)
    save_dir = os.path.join(args.result_dir, config['experiment_name'], config['prompt'], config['run'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # check if the experiment have been conducted
    responses_dir = os.path.join(save_dir, 'responses.txt')
    if os.path.exists(responses_dir) and os.path.getsize(responses_dir) > 0:
        print("The experiment have been conducted in " + responses_dir)
        return 
    
    # load dataset
    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval().cuda()
    
    train_set, val_set, test_set, lab2cname = prepare_image_dataset(args, clip_model)

    # L2 normalize train/val/test image features
    train_set['features'] = torch.nn.functional.normalize(train_set['features'], dim=1).cuda()
    val_set['features'] = torch.nn.functional.normalize(val_set['features'], dim=1).cuda()
    test_set['features'] = torch.nn.functional.normalize(test_set['features'], dim=1).cuda()
    train_set['labels'] = train_set['labels'].cuda()
    val_set['labels'] = val_set['labels'].cuda()
    test_set['labels'] = test_set['labels'].cuda()
    
    print("Acc List Initialization...")
    init_train_acc_list = []
    init_val_acc_list = []
    init_train_val_acc_list = []
    init_test_acc_list = []

    train_acc_list = []
    val_acc_list = []
    train_val_acc_list = []
    test_acc_list = []

    best5_average_train_acc_list = []
    best5_average_val_acc_list = []
    best5_average_train_val_acc_list = []
    best5_average_test_acc_list = []

    print("Template Pool Constructing...")
    # get initial templates
    def get_init_templates(init_templates, dataset, seed, sample_size):
        if init_templates == 'laioncoco': # Please run load_laioncoco_templates.py to sample from 1M laioncoco templates first. Data will be saved to laion_coco_results
            LAION_COCO_DIR = './laion_coco_results/'
            with open(os.path.join(LAION_COCO_DIR, f'laion_coco_samples_size_{sample_size}_seed_{seed}.txt'), 'r') as f:
                lines = f.readlines()
                templates = [line.strip() for line in lines]
            return templates
        elif init_templates == 'openai' or init_templates == 'coop':
            return INIT_TEMPLATES[init_templates][dataset]
        else:
            return INIT_TEMPLATES[init_templates]
        
    init_templates = get_init_templates(args.init_templates, args.dataset, args.laion_seed, args.sample_size)

    # evaluate initial templates performance on train/val/test set
    prompt_dict = {}
    for template in tqdm(init_templates):
        try:
            prompt_dict[template] = eval(train_set, val_set, test_set, template, lab2cname, clip_model)
            init_train_acc_list.append(prompt_dict[template]['train'])
            init_val_acc_list.append(prompt_dict[template]['val'])
            init_train_val_acc_list.append(prompt_dict[template]['train_val'])
            init_test_acc_list.append(prompt_dict[template]['test'])
        except:
            pass

    print("Auto Prompting Starts...")
    responses = []
    for iter in range(args.num_iters):
        
        prompts_with_acc_sorted_by_eval = sorted(prompt_dict.items(), key=lambda kv: kv[1][args.eval], reverse=True)
        prompts_sorted_by_eval = list(prompts_with_acc_sorted_by_eval)
        
        # update top-k best templates and bottom-k worst templates performance on train/val/test set only when the number of templates is larger than 5
        if (len(prompts_sorted_by_eval)) >= 5:
            best5_average_train_acc_list.append(get_best5_average(prompts_sorted_by_eval, 'train'))
            best5_average_val_acc_list.append(get_best5_average(prompts_sorted_by_eval, 'val'))
            best5_average_train_val_acc_list.append(get_best5_average(prompts_sorted_by_eval, 'train_val'))
            best5_average_test_acc_list.append(get_best5_average(prompts_sorted_by_eval, 'test'))

        top_k_good_templates, top_k_bad_templates, random_k_templates = getTopBotK(prompts_sorted_by_eval, args.template_pool_size)
        good_templates_str = '\n'.join(top_k_good_templates)
        bad_templates_str = '\n'.join(top_k_bad_templates)
        random_templates_str = '\n'.join(random_k_templates)

        p = get_message(args.prompt_method, good_templates_str, bad_templates_str, random_templates_str,args.num_templates_from_gpt, args.dataset)
        
        templates = None
        num_tries = 0
        while templates is None and num_tries < 15:
            try:
                num_tries += 1
                completion = openai.ChatCompletion.create(
                    model=args.gpt_model,
                    messages=[
                        {"role": "user", "content": p},
                        ],

                    temperature=args.temperature,
                )
                response = completion.choices[0].message.content.strip()

                responses.append(p + '\n' + response + '\n')
                templates = validate_message(response)
                if templates is None:
                    print("Invalid response.")
            except:
                print("ChatGPT is not responding. Please try again later.")
                continue

        # evaluate templates performance on train/val/test set
        for template in templates:
            
            # to avoid eval templates that have been evaluated before
            if template in prompt_dict:
                continue
            prompt_dict[template] = eval(train_set, val_set, test_set, template, lab2cname, clip_model)
            train_acc_list.append(prompt_dict[template]['train'])
            val_acc_list.append(prompt_dict[template]['val'])
            train_val_acc_list.append(prompt_dict[template]['train_val'])
            test_acc_list.append(prompt_dict[template]['test'])

            print("This is the template: " + template)
            print(prompt_dict[template])
        
    # plot the figure
    init_acc_by_iter_plot_dir = os.path.join(save_dir, 'init_acc_by_iteration.png')
    acc_by_iter_plot_dir = os.path.join(save_dir, 'acc_by_iteration.png')
    best_5_acc_by_iter_plot_dir = os.path.join(save_dir, 'average5acc_by_iteration.png')
    plot_tendency(init_train_acc_list, init_val_acc_list, init_train_val_acc_list, init_test_acc_list, args.eval, init_acc_by_iter_plot_dir, 'Acc of new generated templates by iterations')
    plot_tendency(train_acc_list, val_acc_list, train_val_acc_list, test_acc_list, args.eval, acc_by_iter_plot_dir, 'Acc of new generated templates by iterations')
    plot_tendency(best5_average_train_acc_list, best5_average_val_acc_list ,best5_average_train_val_acc_list, best5_average_test_acc_list, args.eval, best_5_acc_by_iter_plot_dir, 'Average Acc of 5 Best Templates by iterations')

    # write to excel
    templates_by_iter = prompt_dict.items()
    templates_by_test_acc = sorted(prompt_dict.items(), key=lambda kv: kv[1]['test'], reverse=True) # display templates sorted by their test acc
    templates_by_iter_excel_dir = os.path.join(save_dir, 'results_by_iteration.xlsx')
    templates_by_test_acc_excel_dir = os.path.join(save_dir, 'results_by_test_acc.xlsx')
    write_templates_to_excel(templates_by_iter, templates_by_iter_excel_dir)
    write_templates_to_excel(templates_by_test_acc, templates_by_test_acc_excel_dir)

    # write responses to txt
    responses_dir = os.path.join(save_dir, 'responses.txt')
    with open(responses_dir, 'w') as f:
        for response in responses:
            f.write(response)


if __name__ == "__main__":
    main(args)
