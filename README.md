# ad-model-analyzer
Merged model analyzer using scipy's [scipy.optimize.minimize()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

## usage
The following command tries to interpolate Anything model using `blessingMix_V1Fp16.safetensors` and `bp_mk5.safetensors` with `output_blocks` `4,5,6`
~~~bash
python sd-model-merger/fit.py Anything_v3Fixed-prunedFp16.safetensors blessingMix_V1Fp16.safetensors bp_mk5.safetensors --out 4,5,6
selected output_blocks are [4, 5, 6]
 * input blocks = []
 * output blocks = [4, 5, 6]
 * selected blocks = ['output_blocks']
 * seed = 114514
 * optimize method = nelder-mead

- model: Anything_v3Fixed-prunedFp16.safetensors [06ebba372a]
- sd_version = 1.5
- model: v1-5-pruned-emaonly.safetensors
 - load model_a: blessingMix_V1Fp16.safetensors
 - load model_b: bp_mk5.safetensors
 - block = output_blocks, n = 4
... (snip)
[0.4 0.1] : 69.8440%
[0.42 0.1 ] : 71.2073%
[0.4   0.105] : 69.9985%
[0.42  0.105] : 71.3305%
[0.43   0.1075] : 71.6405%
[0.45   0.1025] : 71.8553%
[0.475   0.10125] : 73.4433%
[0.485   0.10875] : 74.1305%
[0.5175   0.113125] : 76.0665%
[0.5625   0.106875] : 77.5666%
[0.62875   0.1065625] : 79.4849%
[0.67125   0.1184375] : 81.0828%
[0.769375   0.12703125] : 84.0874%
[0.880625   0.12046875] : 87.1003%
[1.0621875  0.12414062] : 88.5625%
...
Optimization terminated successfully.
         Current function value: -0.943215
         Iterations: 36
         Function evaluations: 85
seed = 114514
       message: Optimization terminated successfully.
       success: True
        status: 0
           fun: -0.943215012550354
             x: [ 9.558e-01  1.331e-01]
           nit: 36
          nfev: 85
 final_simplex: (array([[ 9.558e-01,  1.331e-01],
                       [ 9.558e-01,  1.331e-01],
                       [ 9.558e-01,  1.331e-01]]), array([-9.432e-01, -9.432e-01, -9.432e-01]))
 * seed = 114514
output_blocks.4 : 89.1261%
[0.996646 0.127445]
output_blocks.5 : 94.4880%
[0.481929 0.590231]
output_blocks.6 : 94.3215%
[0.955802 0.133139]
~~~

~~~bash
python sd-model-merger/fit.py  --help
usage: fit.py [-h] [-i INP] [-m] [-o OUT] [-a] [-s SEED] [-w INIT] [-d] [-e] [--method METHOD] [-x XTOL] [-c] [-f32] file [file ...]

LSQ Fit merging models

positional arguments:
  file                  model file names

options:
  -h, --help            show this help message and exit
  -i INP, --in INP      input blocks
  -m, --mid             middle block
  -o OUT, --out OUT     output blocks
  -a, --all             all blocks
  -s SEED, --seed SEED  seed
  -w INIT               initial weights
  -d, --debug           debug some information
  -e, --eval            evaluate initial weights if available
  --method METHOD       optimize method (available method Nelder-Mead:default, Powell)
  -x XTOL, --xtol XTOL  xtol option for minimize
  -c, --clear           clear saved file
  -f32, --float32       float32
~~~
