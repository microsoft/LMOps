import argparse
import trlx
import json
import pytorch_lightning as pl
import clip
import numpy as np
import torch
import os
import torch.distributed as dist

from torch import autocast, nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from trlx.data.configs import TRLConfig


class AestheticMlp(pl.LightningModule):

  def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
    super().__init__()
    self.input_size = input_size
    self.xcol = xcol
    self.ycol = ycol
    self.layers = nn.Sequential(
      nn.Linear(self.input_size, 1024),
      #nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(1024, 128),
      #nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, 64),
      #nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(64, 16),
      #nn.ReLU(),
      nn.Linear(16, 1))

  def forward(self, x): return self.layers(x)


class PromptScorer:

  def __init__(self, sdmodel_name):
    
    # init scorer hparams
    self.lambda_aes = 0.05
    self.lambda_clip = 5.0
    self.num_images_per_prompt = 2

    # init models
    self.sdmodel_name = sdmodel_name
    self.init_clip_model()
    self.init_aesthetic_model()
    self.init_diffusion_model()

    self.eval_data_res = []
  
  def init_diffusion_model(self):
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    access_token = ""   # TODO Please provide the access token
    dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(self.sdmodel_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(self.sdmodel_name, use_auth_token=access_token, revision="fp16", 
    torch_dtype=torch.float16, scheduler=dpm_scheduler)
    # Disable NSFW detect
    pipe.safety_checker = None
    pipe = pipe.to(device)
    self.diffusion_pipe = pipe

  def init_clip_model(self):
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

  def init_aesthetic_model(self):
    model = AestheticMlp(768)
    s = torch.load("./aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    model.to(device)
    model.eval()
    self.aes_model = model
  
  def get_clip_features(self, pil_image, is_batched=False):
    if not is_batched:
      image = self.clip_preprocess(pil_image).unsqueeze(0)
    else:
      images = [self.clip_preprocess(i) for i in pil_image]
      image = torch.stack(images)
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    image = image.to(device)
    with torch.no_grad():
      image_features = self.clip_model.encode_image(image)
    return image_features
  
  def get_clip_score(self, image_features, prompt):
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    tokens = clip.tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
      text_features = self.clip_model.encode_text(tokens)
      image_features = image_features / image_features.norm(dim=1, keepdim=True)
      text_features = text_features / text_features.norm(dim=1, keepdim=True)
      logit_scale = self.clip_model.logit_scale.exp()
      # logit = logit_scale * image_features @ text_features.t()
      logit = image_features @ text_features.t()
      score = logit.item()
    return score
  
  def get_clip_score_batched(self, image_features, prompts):
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    tokens = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
      if len(image_features) != len(prompts):
        assert len(image_features) % len(prompts) == 0
        tokens = tokens.unsqueeze(1).expand(-1, self.num_images_per_prompt, -1).reshape(-1, tokens.shape[-1])
      
      text_features = self.clip_model.encode_text(tokens)
      image_features = image_features / image_features.norm(dim=1, keepdim=True)
      text_features = text_features / text_features.norm(dim=1, keepdim=True)
      # logit_scale = self.clip_model.logit_scale.exp()
      logit = image_features @ text_features.t()
    scores = logit.diag().tolist()
    return scores

  def get_aesthetic_score(self, image_features, is_batched=False):
    features = image_features.cpu().detach().numpy()
    order = 2
    axis = -1
    l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
    l2[l2 == 0] = 1
    im_emb_arr = features / np.expand_dims(l2, axis)
    prediction = self.aes_model(torch.from_numpy(im_emb_arr).to(f"cuda:{os.environ['LOCAL_RANK']}").type(torch.cuda.FloatTensor))
    if is_batched:
      return prediction[:, 0].tolist()
    else:
      return prediction.item()
  
  def gen_image(self, prompt):
    with autocast("cuda"):
      images = self.diffusion_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images
    return images[0]
  
  def gen_image_batched(self, prompts):
    images = []
    bsz = 1
    for i in range(0, len(prompts), bsz):
      pmpts = prompts[i: i + bsz]
      with autocast("cuda"):
        sub_images = self.diffusion_pipe(pmpts, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=20).images
        images.extend(sub_images)
    return images

  def get_score(self, prompt, plain_text):
    image = self.gen_image(prompt)
    image_features = self.get_clip_features(image)
    aes_score = self.get_aesthetic_score(image_features)
    clip_score = self.get_clip_score(image_features, plain_text)
    final_score = aes_score * self.lambda_aes + clip_score * self.lambda_clip
    return aes_score, clip_score, final_score
  
  def get_score_batched(self, prompts, plain_texts, plain_aes_score=None):
    images = self.gen_image_batched(prompts)
    image_features = self.get_clip_features(images, is_batched=True)
    aes_scores = self.get_aesthetic_score(image_features, is_batched=True)

    if plain_aes_score is None:
      images_plain = self.gen_image_batched(plain_texts)
      images_plain_features = self.get_clip_features(images_plain, is_batched=True)
      aes_scores_plain = self.get_aesthetic_score(images_plain_features, is_batched=True)
    else:
      aes_scores_plain = plain_aes_score.tolist()

    clip_scores = self.get_clip_score_batched(image_features, plain_texts)
    clip_scores = torch.Tensor(clip_scores)
    clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))
    clip_scores_plain = self.get_clip_score_batched(images_plain_features, plain_texts)
    clip_scores_plain = torch.Tensor(clip_scores_plain)
    clip_scores_plain = torch.maximum(clip_scores_plain, torch.zeros_like(clip_scores_plain))

    aes_scores = torch.Tensor(aes_scores)
    if len(aes_scores_plain) != len(aes_scores):
      aes_scores_plain = torch.Tensor(aes_scores_plain).unsqueeze(1).expand(-1, self.num_images_per_prompt).flatten()
    else:
      aes_scores_plain = torch.Tensor(aes_scores_plain)
    aes_scores_plain = aes_scores_plain.reshape(-1, self.num_images_per_prompt).mean(-1, keepdim=True).expand(-1, self.num_images_per_prompt).flatten()
    clip_scores_plain = clip_scores_plain.reshape(-1, self.num_images_per_prompt).mean(-1, keepdim=True).expand(-1, self.num_images_per_prompt).flatten()

    final_scores = (aes_scores-aes_scores_plain) + torch.where(clip_scores>0.28, 0, 20*clip_scores-5.6)
    final_scores = final_scores.reshape(-1, self.num_images_per_prompt).mean(1)

    return final_scores

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", type=str, default="/data_path")
  parser.add_argument("--sdmodel_name", type=str, default="CompVis/stable-diffusion-v1-4")
  parser.add_argument("--gpt_path", type=str, default="/gpt_model_path")
  parser.add_argument("--trl_config", type=str, default="./diff_prompter/configs/ppo_config.yml")
  parser.add_argument("--checkpoint_dir", type=str, default="/ckpt_dir")
  parser.add_argument("--ckpt_path", type=str, default=None)
  parser.add_argument("--eval_data_name", type=str, default="sentence_mix_valid")
  parser.add_argument("--max_new_tokens", type=int, default=-1)
  args = parser.parse_args()
  return args

def load_mix_dataset(data_dir, eval_data_name):
  # TODO please prepare your user data
  def _load(fn):
    texts = []
    with open(fn) as fp:
      for line in fp:
        texts.append(line.strip() + " Rephrase:")
    return texts
  
  valid500_fn = os.path.join(data_dir, f"{eval_data_name}.txt")
  train_fn = os.path.join(data_dir, "filtered_mix_train.txt")

  train_prompts = _load(train_fn)
  valid_prompts = _load(valid500_fn)
  print("train_prompts=")
  print(train_prompts[:5])
  print("valid_prompts=")
  print(valid_prompts[:5])
  return train_prompts, valid_prompts


def main(args):
  scorer = PromptScorer(args.sdmodel_name)

  # TODO maybe shard the data for parallel training
  train_plain_aes, valid_plain_aes = None, None, None
  train_prompts, valid_prompts = load_mix_dataset(args.data, args.eval_data_name)

  def reward_fn(samples, plain_aes_score=None):
    scores = []

    # TODO use plain texts here
    # diffuser_prompts = plain_texts = samples
    diffuser_prompts = []
    plain_texts = []
    for i, sample in enumerate(samples):
      _split = sample.split(" Rephrase:")
      if len(_split) == 1:
        print("[W] maybe ` Rephrase:` is truncated from the input, walkaround when computing socres")
        plain = diff = _split[0]
      elif len(_split) == 2:
        plain, diff = _split
      else:
        print("[W] multiple ` Rephrase:` in generated text")
        plain, diff = _split[0], _split[1]
      diffuser_prompts.append(diff)
      plain_texts.append(plain)

    scores = scorer.get_score_batched(diffuser_prompts, plain_texts, plain_aes_score)

    return scores

  trl_config = TRLConfig.load_yaml(args.trl_config)
  trl_config.train.checkpoint_dir = args.checkpoint_dir
  trl_config.model.ckpt_path = args.ckpt_path
  if args.max_new_tokens > 0:
    trl_config.method.gen_kwargs["max_new_tokens"] = args.max_new_tokens

  model = trlx.train(
    args.gpt_path,
    reward_fn=reward_fn,
    prompts=train_prompts,
    train_plain_aes=train_plain_aes,
    config=trl_config,
    eval_prompts=valid_prompts,
    eval_plain_aes=valid_plain_aes)
    

if __name__ == "__main__":
  args = get_args()
  main(args)
