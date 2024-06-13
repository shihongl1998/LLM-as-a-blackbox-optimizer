import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from engine import clip
import random

# Function to construct an experiment name based on various arguments
def get_experiment_name(args):

    experiment_config_str = f"{args.clip_encoder.replace('/', '-')}_{args.dataset}_shot_{args.shot}_seed_{args.seed}"
    prompt_config_str = f"{args.gpt_model}_prompt_{args.prompt_method}_iter_{args.num_iters}_t_{args.temperature}_init_{args.init_templates}_laion_seed_{args.laion_seed}_pool_{args.template_pool_size}_numtemplates_{args.num_templates_from_gpt}_eval_{args.eval}"
    run_str = f"_run_{args.run}"
    return {
        'experiment_name': experiment_config_str,
        'prompt': prompt_config_str,
        'run': run_str,
    }

# Function to extract text features using a specified template and text encoder
def extract_text_features(dataset, template_to_test, text_encoder, lab2cname):

    templates = [template_to_test]
    
    features_dict = {
        'features': None,
        'labels': None,
        'eot_indices': None,
        'prompts': {},
        'lab2cname': lab2cname,
    }
    text_encoder.feature_extractor.eval()
    with torch.no_grad():
        for label, cname in lab2cname.items():
            str_prompts = [template.format(cname.replace("_", " ")) for template in templates]
            if len(str_prompts) >= 1000:
                # break into chunks of 1000
                str_prompts_chunk = []
                features = None
                eot_indices = None
                for i in range(0, len(str_prompts), 1000):
                    str_prompts_chunk = str_prompts[i:i+1000]
                    prompts_chunk = torch.cat([clip.tokenize(p) for p in str_prompts_chunk]).cuda()
                    features_chunk, eot_indices_chunk = text_encoder.feature_extractor(prompts_chunk)
                    if features is None:
                        features = features_chunk
                        eot_indices = eot_indices_chunk
                    else:
                        features = torch.cat((features, features_chunk), 0)
                        eot_indices = torch.cat((eot_indices, eot_indices_chunk), 0)
            else:
                prompts = torch.cat([clip.tokenize(p) for p in str_prompts]).cuda()
                features, eot_indices = text_encoder.feature_extractor(prompts)
            features = features.cpu()
            eot_indices = eot_indices.cpu()
            labels = torch.Tensor([label for _ in str_prompts]).long()
            if features_dict['features'] is None:
                features_dict['features'] = features
                features_dict['labels'] = labels
                features_dict['eot_indices'] = eot_indices
            else:
                features_dict['features'] = torch.cat((features_dict['features'], features), 0)
                features_dict['labels'] = torch.cat((features_dict['labels'], labels))
                features_dict['eot_indices'] = torch.cat((features_dict['eot_indices'], eot_indices))
            features_dict['prompts'][label] = str_prompts
    return features_dict

# Function to get initial templates based on a given method
def get_init_templates(init_templates, dataset, seed):
        if init_templates == 'laioncoco':
            # Specify path to the laion coco samples here
            LAION_COCO_DIR = ''
            with open(os.path.join(LAION_COCO_DIR, f'laion_coco_samples_seed_{seed}.txt'), 'r') as f:
                lines = f.readlines()
                templates = [line.strip() for line in lines]
            return templates
        else:
            return INIT_TEMPLATES[init_templates][dataset]
        
# Function to validate the performance of the model on the validation set
def validate(logit_head, image_encoder, val_loader, device="cuda"):
    with torch.no_grad():
        logit_head.eval()
        image_encoder.eval()
        val_acc = 0
        val_count = 0.
        for image, image_label in val_loader:
            image = image.to(device)
            image_label = image_label.to(device)
            image_feature = image_encoder(image)
            logit = logit_head(image_feature)
            pred = torch.argmax(logit, dim=1)
            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)
            image.cpu()
        val_acc /= val_count
    return val_acc

# Function to validate the structure and content of a message template
def validate_message(message):
    lines = message.splitlines()
    templates = []
    for line in lines:
        if line.startswith("- "):
            template = line[2:]
            if '{}' not in template:
                return None
            try:
                template.format("test")
            except:
                return None
            templates.append(template)
    if templates == []:
        return None
    return templates

# Function to record request and response pairs during the experiment
def record_request_response(p, response, save_dir, iter):
    iter_path = os.path.join(save_dir, 'request_response')
    if not os.path.exists(iter_path):
        os.makedirs(iter_path)
    with open(os.path.join(iter_path, f'iter_{iter}.txt'), 'w') as f:
        f.write(f"Request: {p} \nResponse: {response} \n\n")

# Function to get top, bottom, and random templates from sorted prompts
def getTopBotK(sorted_prompts, k):
    good_templates = []
    bad_templates = []
    for i in range(0, k):
        good_templates.append(sorted_prompts[i][0])
        bad_templates.append(sorted_prompts[-i-1][0])

    random_templates = []
    for i in range(0, k):
        random_templates.append(sorted_prompts[random.randint(0, len(sorted_prompts)-1)][0])
    return good_templates, bad_templates,random_templates

# Function to calculate the average of the best 5 prompts based on a specific criteria
def get_best5_average(prompts_sorted_by_eval, criteria):
    acc = 0
    for i in range(5):
        acc += prompts_sorted_by_eval[i][1][criteria]
    acc /= 5
    return acc

# Function to calculate the mean and standard deviation of a list of accuracies
def cal_mean_std(acc_list):
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    return acc_mean, acc_std

# Function to plot the accuracy tendency over training iterations
def plot_tendency(train_acc_list, val_acc_list, train_val_acc_list, test_acc_list, eval_criteria, filename, figure_title):

    train_acc_mean, train_acc_std = cal_mean_std(train_acc_list)
    val_acc_mean, val_acc_std = cal_mean_std(val_acc_list)
    train_val_acc_mean, train_val_acc_std = cal_mean_std(train_val_acc_list)
    test_acc_mean, test_acc_std = cal_mean_std(test_acc_list)
 
    best_index = None
    if eval_criteria == 'train':
        best_index = np.argmax(train_acc_list)
    elif eval_criteria == 'val':
        best_index = np.argmax(val_acc_list)
    elif eval_criteria == 'train_val':
        best_index = np.argmax(train_val_acc_list)
    elif eval_criteria == 'test':
        best_index = np.argmax(test_acc_list)

    epoch_count = len(train_acc_list)
    epochs = range(1, epoch_count + 1)
    plt.plot(epochs, train_acc_list, 'b', label='Train Accuracy')
    plt.plot(epochs, val_acc_list, 'y', label='Val Accuracy')
    plt.plot(epochs, train_val_acc_list, 'g', label='Train Val Accuracy')
    plt.plot(epochs, test_acc_list, 'r', label='Test Accuracy')
    plt.axhline(test_acc_list[best_index], linestyle=':', color='grey')
    plt.text(0.5, test_acc_list[best_index], f'Highest test acc by {eval_criteria} acc: {test_acc_list[best_index]:.5f} at Iter {best_index+1}', fontsize=10)
    plt.title(figure_title)
    plt.xlabel('Hill Climbing Iterations')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.2, 1, 0.05))
    plt.xticks(np.arange(0, epoch_count + 1, 10))
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(filename)
    plt.clf()

# Function to write template results to an Excel file
def write_templates_to_excel(templates, filename):
    def format_percent(x):
        if isinstance(x, (float)):
            return '{:.6%}'.format(x)
        else:
            return x
    columns = ['prompt', 'train', 'val', 'train_val', 'test']
    results = []
    for prompt, acc in templates:
        results.append([prompt, acc['train'], acc['val'], acc['train_val'], acc['test']])
    df = pd.DataFrame(results, columns=columns)
    df.applymap(format_percent)
    df.to_excel(filename, index=False)

@torch.no_grad()
# Function to evaluate the model performance on different sets using a given template
def eval(train_set, val_set, test_set, template, lab2cname, clip_model):
    num_classes = len(lab2cname)
    prompts = [template.format(lab2cname[label]) for label in range(num_classes)]
    clip_model.eval()
    prompts = clip.tokenize(prompts).cuda()
    text_features = clip_model.encode_text(prompts)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    head = torch.nn.Linear(text_features.shape[1], num_classes, bias=False)
    head.weight.data = text_features
    head = head.cuda().eval()
    
    results = {}
    train_logit = head(train_set['features'])
    val_logit = head(val_set['features'])
    train_val_logit = torch.cat([train_logit, val_logit], dim=0)
    test_logit = head(test_set['features'])
    train_acc = torch.mean((torch.argmax(train_logit, dim=1) == train_set['labels']).float())
    val_acc = torch.mean((torch.argmax(val_logit, dim=1) == val_set['labels']).float())
    train_val_labels = torch.cat([train_set['labels'], val_set['labels']], dim=0)
    train_val_acc = torch.mean((torch.argmax(train_val_logit, dim=1) == train_val_labels).float())
    test_acc = torch.mean((torch.argmax(test_logit, dim=1) == test_set['labels']).float())
    results['train'] = float(train_acc)
    results['val'] = float(val_acc)
    results['train_val'] = float(train_val_acc)
    results['test'] = float(test_acc)
    return results