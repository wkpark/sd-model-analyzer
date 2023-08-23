#
# MIT License
#
# merged model merger/analyzer using optimizer by wkpark at gmail.com
#
# - ASimilarity Calculator source used in this code.
#
from safetensors.torch import load_file, save_file
import sys
import os
import copy
import json
import time
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import argparse

from scipy.optimize import minimize, basinhopping, shgo, Bounds

models = {}
qkvs = {}
attn_a = {}
rand_inputs = {}
block = "input_blocks"
n = 1
sd_version = None
seed = 114514
debug = False

# for minimizer
opt_methods = [ 'powell', 'nelder-mead' ]
opt_method = 'nelder-mead'
xtol = None

def cal_cross_attn(q, k, v, input):
    hidden_dim, embed_dim = q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": q})
    attn_to_k.load_state_dict({"weight": k})
    attn_to_v.load_state_dict({"weight": v})

    return torch.einsum(
        "ik, jk -> ik",
        F.softmax(torch.einsum("ij, kj -> ik", attn_to_q(input), attn_to_k(input)), dim=-1),
        attn_to_v(input)
    )

def model_hash_old(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

# from webui/modules/hashes with minor fixes
#def calculate_sha256(filename):
def model_hash(filename):
    try:
        with open(filename, "rb") as f:
            import hashlib
            hash_sha256 = hashlib.sha256()
            blksize = 1024 * 1024

            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()[0:10]
    except FileNotFoundError:
        return 'NOFILE'

def prune_model(model):
    keys = list(model.keys())
    for k in keys:
        if "diffusion_model." not in k and "first_stage_model." not in k and "cond_stage_model." not in k:
            model.pop(k, None)
    return model

def to_half(tensor, enable):
    if enable and type(tensor) is dict:
        for key in tensor.keys():
            if 'model' in key and tensor[key].dtype == torch.float:
                tensor[key] = tensor[key].half()
    elif enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor

def load_model(path):
    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

def read_metadata_from_safetensors(filename):
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res

def eval(model, block, n, input):
    if block ==  "middle_block":
        keybase = f"model.diffusion_model.middle_block.1.transformer_blocks"
    else:
        keybase = f"model.diffusion_model.{block}.{n}.1.transformer_blocks"

    qk = f"{keybase}.0.attn1.to_q.weight"
    uk = f"{keybase}.0.attn1.to_k.weight"
    vk = f"{keybase}.0.attn1.to_v.weight"

    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)

    return attn

def load_qkvs(model, block, n, base_model = None, float32 = False):
    if block ==  "middle_block":
        keybase = f"model.diffusion_model.middle_block.1.transformer_blocks"
    else:
        keybase = f"model.diffusion_model.{block}.{n}.1.transformer_blocks"

    qkvs = {}
    for qkv in "to_q", "to_k", "to_v":
        key = f"{keybase}.0.attn1.{qkv}.weight"
        if base_model is not None and base_model[key] is not None:
            base_qkvs = to_half(base_model[key], not float32)
            qkvs[key] = model[key] - base_qkvs
        else:
            qkvs[key] = model[key]
        qkvs[key] = to_half(qkvs[key], not float32)

    return qkvs

def init_rand(model, block, n):
    if block ==  "middle_block":
        keybase = f"model.diffusion_model.middle_block.1.transformer_blocks"
    else:
        keybase = f"model.diffusion_model.{block}.{n}.1.transformer_blocks"

    hidden_dim, embed_dim = model[f"{keybase}.0.attn1.to_q.weight"].shape

    rand_input = torch.randn([embed_dim, hidden_dim])

    return rand_input

def func(x, block, n, attn_a, rand_inputs, models):

    if block ==  "middle_block":
        keybase = f"model.diffusion_model.middle_block.1.transformer_blocks"
    else:
        keybase = f"model.diffusion_model.{block}.{n}.1.transformer_blocks"

    qk = f"{keybase}.0.attn1.to_q.weight"
    kk = f"{keybase}.0.attn1.to_k.weight"
    vk = f"{keybase}.0.attn1.to_v.weight"

    theta_0 = {}

    base_model = models["base_model"]

    y = x
    # custom func for example
    #if len(x) == 2:
    #    # merge1 = model_a x (1 - alpha) + base_model x alpha
    #    # merge2 = merge1 x (1 - beta) + model_b x beta
    #    y[0] = (1.0 - x[0]) * (1.0 - x[1])
    #    y[1] = x[1]

    theta_0[qk] = base_model[qk]
    theta_0[kk] = base_model[kk]
    theta_0[vk] = base_model[vk]
    #theta_0[qk] = base_model.get(qk, torch.zeros_like(base_model[qk]))
    #theta_0[kk] = base_model.get(kk, torch.zeros_like(base_model[kk]))
    #theta_0[vk] = base_model.get(vk, torch.zeros_like(base_model[vk]))

    for i in range(0, len(x)):
        model_name = f"model_{chr(97+i)}"
        theta_0[qk] = theta_0[qk] + models[model_name][qk] * y[i]
        theta_0[kk] = theta_0[kk] + models[model_name][kk] * y[i]
        theta_0[vk] = theta_0[vk] + models[model_name][vk] * y[i]

    '''
    # for example (each mode_x term is 'add-difference' result of mode_a as 'modelA - base_model')
        model_a = models["model_a"]
        model_b = models["model_b"]
        model_c = models["model_c"]
        model_d = models["model_d"]
        theta_0[qk] = base_model[qk] + model_a[qk] * x[0] + model_b[qk] * x[1] + model_c[qk] * x[2] + model_d[qk] * x[3]
        theta_0[kk] = base_model[kk] + model_a[kk] * x[0] + model_b[kk] * x[1] + model_c[kk] * x[2] + model_d[kk] * x[3]
        theta_0[vk] = base_model[vk] + model_a[vk] * x[0] + model_b[vk] * x[1] + model_c[vk] * x[2] + model_d[vk] * x[3]
    '''

    attn_b = eval(theta_0, block, n, rand_inputs[f"{block}.{n}"])

    sim = -torch.mean(torch.cosine_similarity(attn_a, attn_b))

    print(f"{x} : {-sim*1e2:.4f}%")
    val = sim.detach().numpy()

    return val

def fit(files, selected=None, inits=None, method='nelder-mead', xtol=0.00001, float32=False, evaluate=False, use_global=None):
    global models

    file1 = Path(files[0])
    model_files = files[1:]

    if len(models) == 0 or "fit_model" not in models:
        fit_model = load_model(file1)

        print()
        print(f"- model: {file1.name} [{model_hash(file1)}]")

        w = fit_model["model.diffusion_model.input_blocks.1.1.proj_in.weight"]
        if len(w.shape) == 4:
            sd_version = "1.5"
        else:
            sd_version = "2.1"
        models["fit_model"] = fit_model
        print(f"- sd_version = {sd_version}")
    else:
        fit_model = models["fit_model"]

    # load the BASE model
    if "base_model" not in models.keys():
        if sd_version == "1.5":
            sdbase = Path("v1-5-pruned-emaonly.safetensors")
            base_model = load_model(sdbase)
        else:
            sdbase = Path("v2-1_768-nonema-pruned.safetensors")
            base_model = load_model(sdbase)
        models["base_model"] = base_model

        #print(f"- model: {sdbase.name} [{model_hash(sdbase)}]")
        print(f"- model: {sdbase.name}")
    else:
        base_model = models["base_model"]

    ret = {}
    # optimize mode. only cross-attention capable blocks
    for i,x in enumerate(sel_blocks):
        if "input_blocks." in x:
            n = int(x[13:len(x)-1])
            block = "input_blocks"
        elif "middle_block." in x:
            n = 1
            block = "middle_block"
        elif "output_blocks." in x:
            n = int(x[14:len(x)-1])
            block = "output_blocks"

        qkvs = load_qkvs(fit_model, block, n, float32=float32)
        if f"{block}.{n}" in rand_inputs.keys():
            rand_input = rand_inputs[f"{block}.{n}"]
        else:
            rand_input = init_rand(qkvs, block, n)
            rand_inputs[f"{block}.{n}"] = rand_input

        # load the original model
        attn_a = eval(qkvs, block, n, rand_input)

        arr = []
        # bounds
        ub = []
        lb = []
        for j, file in enumerate(model_files):
            file = Path(file)
            name = chr(97 + j)
            model_name = f"model_{name}"
            if f"{model_name}_orig" not in models:
                # make mode_a, model_b, ...
                print(f" - load {model_name}: {file.name}")
                #print(f" - load {model_name}: {file.name} [{model_hash(file)}]")
                model = load_model(file)
                models[f"{model_name}_orig"] = model
            else:
                model = models[f"{model_name}_orig"]

            qkvs = load_qkvs(model, block, n, base_model=models["base_model"], float32=float32)
            models[model_name] = qkvs

            arr.append(0.1)
            # upper bound
            ub.append(1.5)
            # lower bound
            lb.append(-0.5)
            #del model

        print(f" - block = {block}, n = {n}")
        bounds = Bounds(lb, ub)

        # some initial weights
        if inits is not None:
            for k, v in enumerate(inits[:len(arr)]):
                arr[k] = v[i]
        else:
            arr[0] = [0.4]
        print(f"initial weights = {arr}")

        options={'gtol': 1e-8, 'xatol': xtol, 'xtol': xtol, 'disp': True}

        if evaluate: # only evaluate
            res = func(arr, block, n, attn_a, rand_inputs, models)
        else:
            if use_global is not None:
                if use_global == "shgo":
                    res = shgo(func, bounds, args=(block, n, attn_a, rand_inputs, models), minimizer_kwargs={"method": method, "xtol":xtol},
                          options={ "disp": True })
                else:
                    res = basinhopping(func, arr, niter=10, stepsize=0.01, minimizer_kwargs={"args":(block, n, attn_a, rand_inputs, models), "method": method},
                          seed=seed, disp=True)
            else:
                res = minimize(func, arr, args=(block, n, attn_a, rand_inputs, models), method=method,
                      options=options, bounds=bounds)

            # print results
            print(f"seed = {seed}")
            print(res)
        ret[f"{block}.{n}"] = res

    return ret

def all_blocks():
    # return all blocks
    blocks = [ "cond_stage_model." ]
    for i in range(0,12):
        blocks.append(f"input_blocks.{i}.")
    blocks.append("middle_block.1.")
    for i in range(0,12):
        blocks.append(f"output_blocks.{i}.")
    return blocks

def print_blocks(blocks):
    str = []
    for i,x in enumerate(blocks):
        if "input_blocks." in x:
            n = int(x[13:len(x)-1])
            block = f"IN{n:02d}"
            str.append(block)
        elif "middle_block." in x:
            block = "MID00"
            str.append(block)
        elif "output_blocks." in x:
            n = int(x[14:len(x)-1])
            block = f"OUT{n:02d}"
            str.append(block)
        elif "cond_stage_model" in x:
            block = f"BASE"
            str.append(block)
    return ','.join(str)

def merge(files, mode="sum", selected=None, weights=None, prune=True, float32=False):
    '''Merge checkpoint files'''
    file1 = Path(files[0])
    model_files = files[1:]

    t0 = time.time()
    print(f"Loading model_a {file1}...")
    model_a = load_model(file1)
    if prune:
        model_a = prune_model(model_a)

    # get keylist of all selected
    keys = []
    keyremains = []
    theta_0 = {}
    for k in model_a.keys():
        keyadded = False
        for s in selected:
            if s in k:
                keys.append(k)
                theta_0[k] = model_a[k]
                keyadded = True
        if not keyadded:
            keyremains.append(k)

    # preserve some dicts
    checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    for k in checkpoint_dict_skip_on_merge:
        if k in keys:
            keys.remove(k)
            item = theta_0.pop(k)
            keyremains.append(k)

    model_base = {}
    weight_start = 0

    modes = [ mode ] if type(mode) is str else mode

    # prepare metadata
    metadata = { "format": "pt" }
    merge_recipe = {
        "type": "sd-model-analyzer",
        "blocks": print_blocks(selected),
        "weights_alpha": weights,
        "model_a": file1.name,
        "mode": modes,
        "mbw": True,
        "calcmode": "normal",
    }

    recipe_all = None
    if "add" in modes or "madd" in modes:
        # automatically detect sd_version and read the base model
        v15 = "v1-5-pruned-emaonly.safetensors"
        v21 = "v2-1_768-nonema-pruned.safetensors"
        dirs = [ ".", "..", "../..", "models", "Stable-diffusion/models" ]

        w = model_a["model.diffusion_model.input_blocks.1.1.proj_in.weight"]
        if len(w.shape) == 4:
            sd_base = v15
        else:
            sd_base = v21

        for d in dirs:
            path = os.path.join(d, sd_base)
            if Path(path).exists():
                print(f"Loading model_base {sd_base}...")
                file = Path(path)
                merge_recipe["model_base"] = file.name
                model_base = load_model(file)
                break
        if len(model_base) == 0:
            print(f"No {sd_base} found. aborting...")
            return False

        if modes[0] == 'madd':
            for key in (tqdm(keys, desc=f"Prepare theta_0 for madd mode...")):
                i = 0
                for j, sel in enumerate(selected):
                    if sel in key:
                        i = j
                        break
                base = model_base[key]
                theta_0[key] = base + (theta_0[key] - base) * weights[0][i]
            # first weights used
            weight_start = 1
            recipe_all = "model_base + (model_a - model_base) * alpha[0]"

    # merge main
    stages = len(model_files)
    for n, file in enumerate(model_files,start=weight_start):
        file = Path(file)
        print(f"Loading model {file.name}...")
        theta_1 = load_model(file)
        model_name = f"model_{chr(97+n+1-weight_start)}"
        merge_recipe[model_name] = file.name

        alpha = weights[n]
        print(f"mode = {modes[n]}, alpha = {alpha}")
        # Add the models together in specific ratio to reach final ratio
        for key in (tqdm(keys, desc=f"Stage #{n+1-weight_start}/{stages}")):
            if "model_" in key:
                continue
            if key in checkpoint_dict_skip_on_merge:
                continue
            if "model" in key and key in theta_0:
                i = 0
                for j, sel in enumerate(selected):
                    if sel in key:
                        i = j
                        break
                if modes[n] == "sum":
                    theta_0[key] = (1 - alpha[i]) * (theta_0[key]) + alpha[i] * theta_1[key]
                else:
                    theta_0[key] = theta_0[key] + (theta_1[key] - model_base[key]) * alpha[i]

        if modes[n] == "sum":
            if recipe_all is None:
                recipe_all = f"model_a * (1 - alpha[{n}]) + {model_name} * alpha[{n}]"
            else:
                recipe_all = f"({recipe_all}) * (1 - alpha[{n}]) + {model_name} * alpha[{n}]"
        elif modes[n] in [ "add", "madd" ]:
            if recipe_all is None:
                recipe_all = f"model_a + ({model_name} - model_base) * alpha[{n}]"
            else:
                recipe_all = f"{recipe_all} + ({model_name} - model_base) * alpha[{n}]"

        if n == weight_start:
            for key in (tqdm(keys, desc="Check uninitialized")):
                if "model" in key:
                    for s in selected:
                        if s in key and key not in theta_0 and key not in checkpoint_dict_skip_on_merge:
                            print(f" +{k}")
                            theta_0[key] = theta_1[key]

        del theta_1

    # store unmodified remains
    if len(keyremains) > 0:
        print("save not modified weights...")

    for key in (tqdm(keyremains, desc="Save unchanged remains")):
        theta_0[key] = model_a[key]
    t1 = time.time()
    print(f"- execution time: {t1 - t0:.4f}sec")

    merge_recipe["model_recipe"] = recipe_all
    metadata["sd_merge_recipe"] = json.dumps(merge_recipe)

    return to_half(theta_0, not float32), metadata

def load_saved(ret):
    nret = {}
    if "count" not in list(ret.keys())[0]:
        # convert old format to new
        for k in ret.keys():
            if type(ret[k]) is not dict:
                x = ret[k].x
                fun = float(ret[k].fun)
                nret[k] = {"fun": fun, "x": x, "success": ret[k].success, "count": 1, "mean": x, "sum": x}
            else:
                nret[k] = ret[k]
        ret = nret
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit merged models or merge models")
    parser.add_argument('-i', '--in', dest='inp', required=False, help='select input blocks')
    parser.add_argument('-m', '--mid', required=False, action='store_true', default=None, help='select middle block')
    parser.add_argument('-o', '--out', required=False, help='select output blocks')
    parser.add_argument('-a', '--all', required=False, action='store_true', help='select all blocks')
    parser.add_argument("--base", help="select base encoder", action='store_true', default=None, required=False)
    parser.add_argument("--vae", help="vae filename to bake in", default=None, required=False)
    parser.add_argument("--novae", help="no vae", action='store_true', default=False, required=False)
    parser.add_argument("-p", "--prune", help="Pruning before merge", action='store_true', default=False, required=False)
    parser.add_argument("--mode", type=str, help="Merge mode (weight sum: Sum, Add difference: Add, Multiple add-diff: Madd)", default=None, required=False)
    parser.add_argument("--sum", help="Weight sum merge mode", action='store_true', default=None, required=False)
    parser.add_argument("--add", help="Add difference merge mode", action='store_true', default=None, required=False)
    parser.add_argument("--fit", "--optimize", dest="optimize", help="Optimize mode", action='store_true', default=False, required=False)
    parser.add_argument("--global", dest="use_global", help="Global optimize (supported: 'shgo' or 'basin'hopping)", default=None, required=False)
    parser.add_argument('-s', '--seed', type=int, required=False, help='select seed')
    parser.add_argument('-w', '--alpha', dest='init', required=False, help='initial weights')
    parser.add_argument('-d', '--debug', required=False, action='store_true', help='debug some information')
    parser.add_argument('-e', '--eval', required=False, action='store_true', help='evaluate initial weights if available')
    parser.add_argument('--method', required=False, help='optimize method (available method Nelder-Mead:default, Powell)')
    parser.add_argument('-x', '--xtol', required=False, help='xtol option for optimize')
    parser.add_argument('-c', '--clear', required=False, action='store_true', help='clear saved file (tmp.npy)')
    parser.add_argument('--fp32', '--usefp32', dest='float32', required=False, action='store_true', help='Use float32')
    parser.add_argument('-O', dest='output', required=False, default='merged.safetensors', help='Merged output file')
    parser.add_argument('files', nargs='+', metavar='file', help='model file names')

    args = parser.parse_args()

    selected   = []
    base_block = []
    inp_blocks = []
    mid_block  = []
    out_blocks = []
    if args.inp is not None:
        inp =  args.inp.split(",")
        blk = []
        for x in inp:
            try:
                x = int(x)
                if x in range(0,12):
                    blk.append(f"input_blocks.{x}.")
            except ValueError:
                pass

        if len(blk) > 0:
            inp_blocks = blk

    if args.out is not None:
        out =  args.out.split(",")
        blk = []
        for x in out:
            try:
                x = int(x)
                if x in range(0,12):
                    blk.append(f"output_blocks.{x}.")
            except ValueError:
                pass

        if len(blk) > 0:
            out_blocks = blk

    if args.mid:
        mid_block = [ "middle_block.1." ]
    if args.base:
        base_block = [ "cond_stage_model." ]

    if args.inp is None and args.out is None and args.mid is None and args.base is None:
        # get all blocks
        sel_blocks = all_blocks()
    else:
        sel_blocks = base_block + inp_blocks + mid_block + out_blocks

    # support SD v1.5, v2.1
    INPBLOCKS = [1,2,4,5,7,8]
    OUTBLOCKS = [3,4,5,6,7,8,9,10,11]
    opt_sel_blocks = []
    if args.optimize:
        # optimize mode. only cross-attention capable blocks
        for i,x in enumerate(sel_blocks):
            if "input_blocks." in x:
                n = int(x[13:len(x)-1])
                if n in INPBLOCKS:
                    opt_sel_blocks.append(x)
            elif "middle_block." in x:
                opt_sel_blocks.append(x)
            elif "output_blocks." in x:
                n = int(x[14:len(x)-1])
                if n in OUTBLOCKS:
                    opt_sel_blocks.append(x)

    if len(inp_blocks)>0:
        print(f" * input blocks = {print_blocks(inp_blocks)}")
    if len(out_blocks)>0:
        print(f" * output blocks = {print_blocks(out_blocks)}")
    #print(f" * middle block = {mid_block}")
    #print(f" * base block = {base_block}")
    if len(sel_blocks) > 0:
        print(f" * selected blocks = {print_blocks(sel_blocks)}")
    if len(opt_sel_blocks) > 0:
        print(f" * optimize blocks = {print_blocks(opt_sel_blocks)}")

    # block level weights
    # normal sum-weights mode (sum)
    # - model_a + model_b merge case: only 1 alpha needed, for block-level weights blocks amount alphas needed
    # add-difference mode (add)
    # - model_a + model_b add-difference merge case: only 1 alpha needed
    # multiple add-difference mode
    # - a * model_a + b * model_b : a, b pair needed
    # - a * model_a + b * model_b + c * model_c : a,b,c alphas needed
    alpha_count = len(args.files) - 1

    mode = None
    if args.add:
        args.mode ='add'
    if args.sum:
        args.mode ='sum'

    modes = []
    alias = { "add+": "madd", "++": "madd" }
    if args.mode is not None:
        tmp = args.mode.lower().split(",")
        for k, m in enumerate(tmp):
           if m in alias:
               m = alias[m]
           if m in [ "sum", "add", "madd" ]:
               modes.append(m)
           if k == 0 and m == "madd":
               alpha_count += 1

        print(f"alpha_count = {alpha_count}")
        if len(modes) < alpha_count:
            for i in range(len(modes), alpha_count):
               lastmode = modes[len(modes)-1]
               modes.append(lastmode)

        print(f"modes = {modes}")

    inits = []
    if args.init is not None:
        blocks_size = len(opt_sel_blocks) if args.optimize else len(sel_blocks)
        tmp = args.init.split(":") # model

        for j, w in enumerate(tmp[:alpha_count]):
            init = [0] * blocks_size
            ws = tmp[j].split(",") # block level alpha
            for i, x in enumerate(ws[:blocks_size]):
                try:
                    x = x.replace("\\", "")
                    x = float(x)
                    init[i] = x
                except ValueError:
                    print(f"invalid initial weight {ws}")
                    exit(-1)
            if len(ws) < len(init):
                for i in range(len(ws), len(init)):
                    init[i] = init[len(ws)-1] # fill-up empty weights with the last one given
            inits.append(init)
        if len(tmp) < alpha_count:
            for i in range(len(tmp), alpha_count):
                inits.append(inits[len(inits)-1]) # fill-up empty weights

    if len(inits) > 0:
        print(f" * alpha = {inits}")
    if len(sel_blocks) > 0:
        blocks = all_blocks()
        for l, init in enumerate(inits):
            norm = [0.0]*len(blocks)
            for k,b in enumerate(blocks):
                for j,s in enumerate(sel_blocks):
                    if s in b:
                        norm[k] = init[j]
            normalout = f"({','.join('0' if s == 0.0 else str(round(s,5)) for s in norm)})"
            print(f"  > normalized block weights for merge #{l+1}: {normalout}")

    # set files
    files = args.files
    debug = args.debug

    ret = {}
    if args.optimize or args.eval:
        # set seed
        if args.seed is not None:
             seed = args.seed

        xtol = float(args.xtol) if args.xtol else 0.00001
        opt_method = args.method.lower() if args.method and args.method.lower() in opt_methods else "nelder-mead"

        print(f" * seed = {seed}")
        print(f" * optimize method = {opt_method}")

        torch.manual_seed(seed)

        # print options
        torch.set_printoptions(sci_mode=False, precision=6)

        r = fit(files, selected=opt_sel_blocks, inits=inits, method=opt_method, xtol=xtol, float32=args.float32, evaluate=args.eval, use_global=args.use_global)
        # load old results
        if not args.eval and Path("tmp.npy").exists() and not args.clear:
            tmp = np.load("tmp.npy", allow_pickle=True)
            ret = tmp.tolist()
            ret = load_saved(ret)
            # check size
            k1 = list(ret.keys())[0]
            k2 = list(r.keys())[0]
            if len(ret[k1]["x"]) != len(r[k2].x):
                # size mismatched. ignore
                ret = {}

        if args.optimize:
            for k in r.keys():
                x = r[k].x
                fun = float(r[k].fun)
                count = 1
                sum = [0.0]*len(x)
                if k in ret and "count" in ret[k]:
                    count = ret[k]["count"] + 1
                    sum = ret[k]["sum"] + x
                    mean = sum/count
                else:
                    mean = sum = x

                ret[k] = {"fun": fun, "x": x, "success": r[k].success, "count": count, "mean": mean, "sum": sum}

        # save results
        if not args.eval:
            np.save("tmp.npy", ret)

        print(f" * seed = {seed}")
        if args.eval:
            for k in ret.keys():
                print(f" - {k} : {-ret[k] * 1e2:.4f}%")
    elif len(modes) == 0 and args.vae is None and not args.novae and not args.prune and Path("tmp.npy").exists():
        print("Print saved info from tmp.npy...")
        tmp = np.load("tmp.npy", allow_pickle=True)
        ret = tmp.tolist()
        ret = load_saved(ret)

    if len(ret) > 0 and not args.eval:
        # print results
        np.set_printoptions(precision=6)

        alpha_out = []
        mean_out = []
        blocks = all_blocks()
        outs = []
        for k in blocks:
            k = k.rstrip(".")
            if k not in ret:
                continue
            outs.append(k)
            x = torch.Tensor(ret[k]["x"]).detach().numpy()
            x = x.tolist()
            xx = torch.Tensor(ret[k]["mean"]).detach().numpy()
            xx = xx.tolist()
            print_x = f"[ {', '.join(str(round(s,5)) for s in x)} ]"
            print(f"{k} : {print_x} / {-ret[k]['fun'] * 1e2:.4f}%")
            alpha_out.append(x)
            mean_out.append(xx)

        print("")
        coeff = [[]]*len(alpha_out[0])
        mean_coeff = [[]]*len(alpha_out[0])
        for j in range(0, len(alpha_out[0])):
            coeff[j] = [0]*len(alpha_out)
            mean_coeff[j] = [0]*len(alpha_out)
            for i in range(0, len(alpha_out)):
                coeff[j][i] = alpha_out[i][j]
                mean_coeff[j][i] = mean_out[i][j]

        for i in range(0, len(coeff)):
            print_out = f"alpha  = ({','.join('0' if abs(s) < 1e-4 else str(round(s,5)) for s in coeff[i])})"
            print(print_out)
        for i in range(0, len(coeff)):
            print_out = f"mean a = ({','.join('0' if abs(s) < 1e-4 else str(round(s,5)) for s in mean_coeff[i])})"
            print(print_out)

        for i in range(0, len(mean_coeff)):
            blocks = all_blocks()
            coeff = mean_coeff[i]
            norm = [0.0]*len(blocks)
            for k,b in enumerate(blocks):
                for j,s in enumerate(outs):
                    if s in b:
                        norm[k] = coeff[j]
            normalout = f"({','.join('0' if s == 0.0 else str(round(s,5)) for s in norm)})"
            print(f"  > normalized block weights for model #{i}: {normalout}")

    # merge models or manage model file
    theta_0 = {}
    save_needed = False
    if len(modes) > 0:
        theta_0, metadata = merge(files, mode=modes, selected=sel_blocks, weights=inits, prune=args.prune, float32=args.float32)
        save_needed = True
        if type(theta_0) is bool:
            print("Fail to merge")
            exit(-1)

    if args.optimize is False and args.eval is False and len(theta_0) == 0:
        # no operation
        if len(args.files) > 0 and Path(args.files[0]).exists():
            print(f"Loading model {args.files[0]}...")
            theta_0 = load_model(Path(args.files[0]))
            if args.prune:
                print(f"Pruning model {args.files[0]}...")
                theta_0 = prune_model(theta_0)
                save_needed = True
            metadata = read_metadata_from_safetensors(args.files[0])
            if len(metadata) == 0:
                metadata = { "format": "pt" }

    # fix/check bad clip
    position_id_key = 'cond_stage_model.transformer.text_model.embeddings.position_ids'
    if position_id_key in theta_0:
        correct = torch.tensor([list(range(77))], dtype=torch.int64, device="cpu")
        current = theta_0[position_id_key].to(torch.int64)
        broken = correct.ne(current)
        broken = [i for i in range(77) if broken[0][i]]
        if len(broken) != 0:
            if args.fixclip:
                theta_0[position_id_key] = correct
                print(f"Fixed broken clip\n{broken}")
                save_needed = True
            else:
                print(f"Broken clip!\n{broken}")
        else:
            print("Clip is fine")

    if args.vae is not None or args.novae:
        vae_dict = {}
        if args.vae is not None and Path(args.vae).exists():
            print(f"Loading vae file {args.vae}...")
            vae = load_model(Path(args.vae))
            # Replace VAE in model file with new VAE
            vae_dict = {k: v for k, v in vae.items() if k[0:4] not in ["loss", "mode"]}
        elif args.novae:
            vae_dict = {k: 1 for k in theta_0.keys() if "first_stage_model." in k and k[18:22] not in ["loss", "mode"]}

        if len(vae_dict) > 0:
            msg = "Replace VAE" if not args.novae else "Remove VAE"
            for k in (tqdm(vae_dict.keys(), desc=msg)):
                key_name = "first_stage_model." + k
                if not args.novae:
                    theta_0[key_name] = copy.deepcopy(vae[k])
                    theta_0[key_name] = to_half(theta_0[key_name], not args.float32)
                else:
                    theta_0.pop(key_name, None)
            save_needed = True

    if len(theta_0) > 0 and save_needed:
        output_file = args.output
        ext = Path(output_file).suffix
        if ext not in [ ".ckpt", ".safetensors" ]:
            output_file = output_file + ".safetensors"
            ext = Path(output_file).suffix
        print(f"saving {output_file}...")
        if ext == ".safetensors":
            save_file(theta_0, output_file, metadata)
        else:
            torch.save({"state_dict": theta_0}, output_file)
        print("Done.")

