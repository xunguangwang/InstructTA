import os
import torch
import argparse
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

import sys
sys.path.append(r"/root/tar_att_lvlm/")
from data_provider.data_loader import ImageFolderWithPaths, Caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="blip2_t5")
    parser.add_argument("--model_type", type=str, default='pretrain_flant5xxl')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--instruction", type=str, default='../data/instruction_reasoning.txt')
    # parser.add_argument("--adv_image", type=str, default='../data/ImageNet-1K/10K')
    # parser.add_argument("--adv_image", type=str, default='../result/adv_images/reasoning_mfiiclip_ViT-B32_50')
    # parser.add_argument("--adv_image", type=str, default='../result/adv_images/reasoning_mfiievaclip_EVA01-CLIP-g-14_10')
    # parser.add_argument("--adv_image", type=str, default='../result/adv_images/instructblip_vicuna7b_gpt_EVA01-CLIP-g-14_1')
    parser.add_argument("--adv_image", type=str, default='../result/adv_images/reasoning_mfiievaclip_instructblip_vicuna7b_gpt_EVA01-CLIP-g-14_1')
    # parser.add_argument("--adv_image", type=str, default='../result/adv_images/mfitevaclip_instructblip_vicuna7b_gpt_EVA01-CLIP-g-14_1_64')
    parser.add_argument("--out_path", type=str, default='../result/')
    args = parser.parse_args()

    attack_method = args.adv_image.split('/')[-1]
    if '10K' in attack_method: attack_method = 'reasoning_clean'
    out_path = os.path.join(args.out_path, attack_method + '_blip2.txt')
    # out_path = os.path.join(args.out_path, 'instructblip_vicuna7b_EVA01-CLIP-g-14_1' + '_blip2.txt')

    inst_data = Caption(args.instruction)
    adv_data = ImageFolderWithPaths(args.adv_image)
    
    data_loader_ins = torch.utils.data.DataLoader(inst_data, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    data_loader_adv = torch.utils.data.DataLoader(adv_data, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    model, vis_processors, _ = load_model_and_preprocess(name=args.name, model_type=args.model_type, is_eval=True, device=args.device)

    results = []
    for i, ((_, path), ins) in enumerate(zip(data_loader_adv, data_loader_ins)):
        if i + 1 > args.num_samples:
            break

        adv_image = Image.open(path[0]).convert('RGB')
        adv_image = vis_processors["eval"](adv_image).unsqueeze(0).to(args.device)
        prompt = 'Question: {} Answer:'.format(ins[0])

        out = model.generate({"image": adv_image, "prompt": prompt})[0]
        results.append(out)
        print(f'{i}:', out)
    
np.savetxt(out_path, results, fmt='%s', delimiter='\n')
