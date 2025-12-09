import os
import random
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import sys
sys.path.append(r"/root/tar_att_lvlm/")
from utils.gpt import rephrase
from utils.seed import seedEverything
from data_provider.data_loader import ImageFolderWithPaths, Caption
from lavis.models import load_model_and_preprocess


device = "cuda" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="../result/adv_images/", type=str, help='the folder name of output')
    parser.add_argument("--cle_data_path", default='../data/ImageNet-1K/10K', type=str, help='path of the clean images')
    parser.add_argument("--ins_path", default='../data/instruction_g_4.txt', type=str, help='instruction for the lvlm')
    parser.add_argument("--tgt_img_path", default='../data/target_images_instruct_d', type=str, help='path of the target images')
    args = parser.parse_args()

    folder_to_save = os.path.join(args.output, 'instructblip_{}_gpt_{}_{}'.format(args.llm, args.clip_encoder.replace('/', ''), args.batch_size))
    # folder_to_save = os.path.join(args.output, 'instructblip_{}_{}_{}'.format(args.llm, args.clip_encoder.replace('/', ''), args.batch_size))
    
    alpha = args.alpha
    epsilon = args.epsilon

    # clip_model = create_model_and_transforms(args.clip_encoder, 'eva_clip', force_custom_clip=True)[0]
    # tokenizer = get_tokenizer(args.clip_encoder)
    # clip_model = clip_model.to(device)
    # clip_model.eval()

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=args.llm, is_eval=True, device=device)
    # model.llm_model.to('cpu')
    model.eval()
    
    # ------------- pre-processing images/text ------------- #
    clean_data = ImageFolderWithPaths(args.cle_data_path)
    ins_data = Caption(args.ins_path)
    target_img_data = ImageFolderWithPaths(args.tgt_img_path)
    # target_cap_data = Caption(args.tgt_cap_path)

    data_loader_imagenet = DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_instruction = DataLoader(ins_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_img_target = DataLoader(target_img_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    # data_loader_cap_target = DataLoader(target_cap_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    clip_preprocess = T.Compose(
        [
            T.Resize(args.input_res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            T.CenterCrop(args.input_res),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    instruct_dic = {}
    # start attack
    for i, ((image_org, path), (image_tgt, _), txt_ins) in enumerate(zip(data_loader_imagenet, data_loader_img_target, data_loader_instruction)):
        if args.batch_size * i + image_org.size(0) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)

        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            with torch.no_grad():
                instruction = get_rephrase(instruct_dic, txt_ins)
                tgt_image_features = model({"image": clip_preprocess(image_tgt), "text_input": instruction, 'text_output': ''})

            adv_image = image_org + delta
            adv_image = clip_preprocess(adv_image)
            
            instruction = get_rephrase(instruct_dic, txt_ins)
            blip_outputs = model({"image": adv_image, "text_input": instruction, 'text_output': ''})
            blip_embed_distance = torch.mean((blip_outputs - tgt_image_features)**2)

            blip_embed_distance.backward()
            
            grad = delta.grad.detach()
            d = torch.clamp(delta - alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            # d = (image_org.data + d).clamp(0, 255) - image_org.data
            delta.data = d
            delta.grad.zero_()

            if (j+1) % 20 == 0 or j == 0:
                print(f"iter {i+1}/{args.num_samples//args.batch_size} step:{j+1:3d}, instructblip_d={blip_embed_distance:.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")

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
