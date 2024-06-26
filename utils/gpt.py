import openai

openai.api_type = "azure"
openai.api_base = "https://llm-testing-ca.openai.azure.com/"
# openai.api_base = 'https://llm-testing-east-us.openai.azure.com/'
openai.api_version = "2023-07-01-preview"
openai.api_key = "XXX"



def gpt4(text):
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        messages = [{"role": "user", "content": text}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response.choices[0].message.content


def inst_guess(answer):
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        # messages = [{"role": "user", "content": 'Give the most reasonable and reliable question or prompt for this answer:\n"""\n{}\n"""'.format(answer)}],
        messages = [{"role": "user", "content": 'What is the question or prompt for the following answer? You just provide a question or prompt.\n"""\n{}\n"""'.format(answer)}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response.choices[0].message.content


def rephrase(text, n=10):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages = [{"role": "user", "content": 'Rephrase the following sentence without changing its original meaning. Please give {} examples of paraphrasing that look different and begin the statement with "-".\n"""\n{}\n"""'.format(n, text)}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    text = response.choices[0].message.content.strip()
    return text


def target_attack_success(text_a, text_b):
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        messages = [{"role": "user", "content": 'Determine whether these two texts describe the same objects or things, you only need to answer yes or no:\n1. {}\n2. {}'.format(text_a, text_b)}],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    text = response.choices[0].message.content.strip()
    return text


# print(hw('SneakerNets are traditional networks that suffer from a very long propagation delaybut have a huge (virtually unlimited) bandwidth. This problem is meant to show us whySncakerNets are still a profitable alternative to the Internet via a case study. Assume thatBooble inc, is a company that has detailed images of the space, obtained via the Hubbletelescope. Booble inc, wants to transfer these images from its California data centre to itsSingapore data centre. The total volume of data available is 120Terabytes stored on a diskarray. Assuming the distance between the two sites is roughly 10.000Km and that the Internetlink between the two data centres can carry data at 10Gbps. (1). Explain why, instead of using the Internet, Booble would still prefer to use FeDexNet, aa)version of SneakerNets, that guarantees data delivery within 24 hours. (2). At what link speed would the Internet become a better alternative, if the additional packethcaders account for a 4% extra overhead, and to ensure reliability 20% extra traffic iscarried. Explain.'))

# import time
# import numpy as np

# with open('../data/instruction_g_4.txt', mode='r') as f: prompts = f.read().splitlines()
# results = []
# for i, p in enumerate(prompts):
#     # time.sleep(0.1)
#     if i+1>1000: break
#     out = rephrase(p)
#     results.append(out)
#     print(f'{i}:', out)

# np.savetxt('../data/instruction_r.txt', results, fmt='%s', delimiter='\n')

# print(rephrase('Describe this image.'))
# x = rephrase('Describe this image.')
# x = x.split('\n')
# x = [t[2:] for t in x]
# print(x)

# print(gpt4('以下是计算机图形学的作业，请用英文帮我回答。Describe briefly the explicit and implicit smoothing methods, assuming uniform weighting. Describe using your own words, no equations needed.'))
# print(rephrase("Can you describe what's in the picture you're looking at related to food?"))

# y = gpt4('Given a cubic Bezier curve defined by the control points P_i, i = 0, 1, 2, 3. Explain how you would find the four Bspline control points of the same polynomial curve.')
# print(y)
print(gpt4('huggingface中有一个peft库，请用其中的prefix tuning方法训练llama2-7b-hf'))
