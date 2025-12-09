import os
import time
import random
import openai
import argparse

# import clip
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

import sys
sys.path.append(r"/root/tar_att_lvlm/")
from utils.seed import seedEverything
from data_provider.data_loader import ImageFolderWithPaths, Caption

sys.path.append(r"/root/tar_att_lvlm/EVA-CLIP/rei/")
from eva_clip import create_model_and_transforms, get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# time.sleep(5000)

openai.api_type = "azure"
openai.api_base = "https://llm-testing-ca.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "7907e4397bf5457aa14401b2319a1423"


def rephrase(text):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages = [{"role": "user", "content": 'paraphrase this sentence: "{}"'.format(text)}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    text = response.choices[0].message.content.strip()
    return text


def get_rephrase(ins_dic, texts, n=10):
    instructions = []
    for t in texts:
        if t not in ins_dic:
            ins_dic[t] = [t]
            ins_dic[t].append(rephrase(t))
        elif len(ins_dic[t]) < n:
            ins_dic[t].append(rephrase(t))
        else: pass
        r = random.randint(0, len(ins_dic[t])-1)
        instructions.append(ins_dic[t][r])
    return instructions


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="EVA01-CLIP-g-14", type=str)
    parser.add_argument("--llm", default='vicuna7b', type=str)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--output", default="../result/adv_images/", type=str, help='the folder name of output')
    parser.add_argument("--cle_data_path", default='../data/ImageNet-1K/10K', type=str, help='path of the clean images')
    parser.add_argument("--ins_path", default='../data/instruction_g_4_reasoning.txt', type=str, help='instruction for the lvlm')
    parser.add_argument("--tgt_img_path", default='../data/target_images_instruct_reasoning', type=str, help='path of the target images')
    parser.add_argument("--tgt_cap_path", default='../data/answers_reasoning.txt', type=str, help='caption of the target text')
    args = parser.parse_args()

    folder_to_save = os.path.join(args.output, 'reasoning_mfitevaclip_instructblip_{}_gpt_{}_{}_{}'.format(args.llm, args.clip_encoder.replace('/', ''), args.batch_size, args.epsilon))
    # folder_to_save = os.path.join(args.output, 'mfitevaclip_instructblip_{}_{}_{}'.format(args.llm, args.clip_encoder.replace('/', ''), args.batch_size))
    
    alpha = args.alpha
    epsilon = args.epsilon

    # clip_model, preprocess = clip.load(args.clip_encoder, device=device)
    # clip_model.eval()
    clip_model = create_model_and_transforms(args.clip_encoder, 'eva_clip', force_custom_clip=True)[0]
    tokenizer = get_tokenizer(args.clip_encoder)
    clip_model = clip_model.to(device)
    clip_model.eval()

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=args.llm, is_eval=True, device=device)
    # model.llm_model.to('cpu')
    model.eval()
    
    # ------------- pre-processing images/text ------------- #
    clean_data = ImageFolderWithPaths(args.cle_data_path)
    ins_data = Caption(args.ins_path)
    target_img_data = ImageFolderWithPaths(args.tgt_img_path)
    target_cap_data = Caption(args.tgt_cap_path)

    data_loader_imagenet = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_instruction = DataLoader(ins_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_img_target = DataLoader(target_img_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_cap_target = DataLoader(target_cap_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    instruct_dic = {}
    # start attack
    for i, ((image_org, path), (image_tgt, _), txt_ins, txt_tgt) in enumerate(zip(data_loader_imagenet, data_loader_img_target, data_loader_instruction, data_loader_cap_target)):
        if i+1 < 279: continue
        if args.batch_size * i + image_org.size(0) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)
        
        # get tgt featutres
        with torch.no_grad():
            # tgt_image_features = clip_model.encode_image(clip_preprocess(image_tgt))
            # tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
            tgt_txt_features = clip_model.encode_text(tokenizer(txt_tgt).to(device))
            tgt_txt_features /= tgt_txt_features.norm(dim=-1, keepdim=True)

        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            with torch.no_grad():
                instruction = get_rephrase(instruct_dic, txt_ins)
                tgt_image_features = model({"image": clip_preprocess(image_tgt), "text_input": instruction, 'text_output': ''})

            adv_image = image_org + delta
            adv_image = clip_preprocess(adv_image)

            adv_image_features = clip_model.encode_image(adv_image)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_txt_features, dim=1))
            
            instruction = get_rephrase(instruct_dic, txt_ins)
            blip_outputs = model({"image": adv_image, "text_input": instruction, 'text_output': ''})
            blip_embed_distance = torch.mean((blip_outputs - tgt_image_features)**2)

            (embedding_sim - blip_embed_distance).backward()
            
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            # d = (image_org.data + d).clamp(0, 255) - image_org.data
            delta.data = d
            delta.grad.zero_()

            if (j+1) % 20 == 0 or j == 0:
                print(f"iter {i+1}/{args.num_samples//args.batch_size} step:{j+1:3d}, clip_sim={embedding_sim.item():.5f}, instructblip_sim={blip_embed_distance:.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")

        # save imgs
        adv_image = image_org + delta
        # outputs = model.generate({"image": clip_preprocess(adv_image[0].unsqueeze(0)), "prompt": txt_ins[0]})
        print(get_rephrase(instruct_dic, txt_ins))
        # print(outputs)
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        for path_idx in range(len(path)):
            name = os.path.splitext(os.path.basename(path[path_idx]))[0]
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name + '.png'))
