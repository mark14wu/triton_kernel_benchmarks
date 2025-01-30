commands = []
with open('model_list.txt', 'r') as f:
    model_list = f.read().splitlines()
    for model_name in model_list:
        commands.append(f"python -m sglang.bench_one_batch --model-path {model_name} --load-format dummy --trust-remote-code --disable-cuda-graph")
open('commands.txt', 'w').write('\n'.join(commands))