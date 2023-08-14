#
# MIT License
#
# merged model analyzer using optimizer by wkpark at gmail.com
#
# - ASimilarity Calculator source used in this code.
#
from safetensors.torch import load_file
import sys
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import argparse

from scipy.optimize import minimize, Bounds

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

def to_half(tensor, enable):
    if enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor

def load_model(path):
    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

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

def find(init=None, method='nelder-mead', float32=False, evaluate=False):
    global block
    global n
    global attn_a
    global rand_inputs
    global models

    global sd_version
    global xtol

    global files

    file1 = Path(files[0])
    model_files = files[1:]

    if len(models) == 0 or "fit_model" not in models.keys():
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

    qkvs = load_qkvs(fit_model, block, n, float32=float32)
    if f"{block}.{n}" in rand_inputs.keys():
        rand_input = rand_inputs[f"{block}.{n}"]
    else:
        rand_input = init_rand(qkvs, block, n)
        rand_inputs[f"{block}.{n}"] = rand_input

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

    # load the original model
    attn_a = eval(qkvs, block, n, rand_input)

    j = 0
    arr = []
    # bounds
    ub = []
    lb = []
    for file in model_files:
        file = Path(file)
        model = load_model(file)
        qkvs = load_qkvs(model, block, n, base_model=models["base_model"], float32=float32)
        # make mode_a, model_b, ...
        name = chr(97 + j)
        model_name = f"model_{name}"
        print(f" - load {model_name}: {file.name}")
        #print(f" - load {model_name}: {file.name} [{model_hash(file)}]")
        models[model_name] = qkvs
        j += 1
        arr.append(0.1)
        # upper bound
        ub.append(1.5)
        # lower bound
        lb.append(-0.5)
        del model

    print(f" - block = {block}, n = {n}")
    bounds = Bounds(lb, ub)

    # some initial weights
    if init is not None:
        for i, v in enumerate(init):
            arr[i]  = v
    else:
        arr[0] = 0.4

    if evaluate: # only evaluate
        res = func(arr, block, n, attn_a, rand_inputs, models)
    else:
        res = minimize(func, arr, args=(block, n, attn_a, rand_inputs, models), method=opt_method,
              options={'gtol': 1e-8, 'xatol': xtol, 'xtol': xtol, 'disp': True}, bounds=bounds)

        # print results
        print(f"seed = {seed}")
        print(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSQ Fit merging models")
    parser.add_argument('-i', '--in', dest='inp', required=False, help='input blocks')
    parser.add_argument('-m', '--mid', required=False, action='store_true', help='middle block')
    parser.add_argument('-o', '--out', required=False, help='output blocks')
    parser.add_argument('-a', '--all', required=False, action='store_true', help='all blocks')
    parser.add_argument('-s', '--seed', type=int, required=False, help='seed')
    parser.add_argument('-w', dest='init', required=False, help='initial weights')
    parser.add_argument('-d', '--debug', required=False, action='store_true', help='debug some information')
    parser.add_argument('-e', '--eval', required=False, action='store_true', help='evaluate initial weights if available')
    parser.add_argument('--method', required=False, help='optimize method (available method Nelder-Mead:default, Powell)')
    parser.add_argument('-x', '--xtol', required=False, help='xtol option for minimize')
    parser.add_argument('-c', '--clear', required=False, action='store_true', help='clear saved file')
    parser.add_argument('-f32', '--float32', required=False, action='store_true', help='float32')
    parser.add_argument('files', nargs='+', metavar='file', help='model file names')

    # support SD v1.5, v2.1
    INPBLOCKS = [1,2,4,5,7,8]
    OUTBLOCKS = [3,4,5,6,7,8,9,10,11]
    BLOCKNAMES = [ "input_blocks", "middle_block", "output_blocks" ]

    args = parser.parse_args()
    xtol = float(args.xtol) if args.xtol else 0.00001
    opt_method = args.method.lower() if args.method and args.method.lower() in opt_methods else "nelder-mead"

    selected = []
    if args.inp is not None:
        inp =  args.inp.split(",")
        print(inp)
        blk = []
        for x in inp:
            try:
                x = int(x)
                if x in INPBLOCKS:
                    blk.append(x)
            except ValueError:
                pass

        if len(blk) > 0:
            print(f"selected input_blocks are {blk}")
            inp_blocks = blk
            selected.append("input_blocks")
    else:
        inp_blocks = []

    if args.out is not None:
        out =  args.out.split(",")
        blk = []
        for x in out:
            try:
                x = int(x)
                if x in OUTBLOCKS:
                    blk.append(x)
            except ValueError:
                pass

        if len(blk) > 0:
            print(f"selected output_blocks are {blk}")
            out_blocks = blk
            selected.append("output_blocks")
    else:
        out_blocks = []

    if args.mid:
        selected.append("middle_block")

    # all blocks?
    if args.all:
        inp_blocks = INPBLOCKS
        out_blocks = OUTBLOCKS
        sel_blocks = BLOCKNAMES
    else:
        # rearrange selected blocks
        sel_blocks = []
        for x in BLOCKNAMES:
            if x in selected:
                sel_blocks.append(x)

    # set seed
    if args.seed is not None:
        seed = args.seed

    init = None
    if args.init is not None:
        init = []
        tmp = args.init.split(",")
        for x in tmp:
            try:
                x = float(x)
                init.append(x)
            except ValueError:
                print(f"invalid initial weght {x}")
                exit(-1)

    # set files
    files = args.files

    debug = args.debug

    print(f" * input blocks = {inp_blocks}")
    print(f" * output blocks = {out_blocks}")
    print(f" * selected blocks = {sel_blocks}")
    print(f" * seed = {seed}")
    print(f" * optimize method = {opt_method}")

    '''
    # for example
    block = "output_blocks"
    n = 10
    seed = 114514
    r = find(init=None)
    print(f"block = {block}, n ={n}")

    torch.set_printoptions(sci_mode=False)
    print(torch.Tensor(r.x))
    exit(0)
    '''

    torch.manual_seed(seed)

    # print options
    torch.set_printoptions(sci_mode=False, precision=6)

    ret = {}
    # load old results
    if not args.eval and Path("tmp.npy").exists() and not args.clear:
        tmp = np.load("tmp.npy", allow_pickle=True)
        ret = tmp.tolist()

    for b in sel_blocks:
        if b == "input_blocks":
            for j in inp_blocks:
                n = j
                block = b
                r = find(init, evaluate=args.eval, float32=args.float32)
                ret[f"{b}.{n}"] = r
        elif b == "middle_block":
            n = 1
            block = b
            r = find(init, evaluate=args.eval, float32=args.float32)
            ret[f"{b}.{n}"] = r

        elif b == "output_blocks":
            for j in out_blocks:
                block = b
                n = j
                r = find(init, evaluate=args.eval, float32=args.float32)
                ret[f"{b}.{n}"] = r

    print(f" * seed = {seed}")

    # save results
    if not args.eval:
        np.save("tmp.npy", ret)

    # print results
    np.set_printoptions(precision=6)

    if args.eval:
        for k in ret.keys():
            print(f" - {k} : {-ret[k] * 1e2:.4f}%")
        exit(0)

    for k in ret.keys():
        print(f"{k} : {-ret[k].fun * 1e2:.4f}%")
        x = torch.Tensor(ret[k].x).detach().numpy()
        print(x)
