# HOW TO ...

## build

```bash
cd //openvino.genai

mkdir build
cd build

# or build with Debug mode
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=//local/path/openvino/runtime ..

# or build with Release mode
cmake -DCMAKE_BUILD_TYPE=Release-DCMAKE_PREFIX_PATH=//local/path/openvino/runtime ..

# build
make -j
```

The excutable binary will be generated in the build directory.

## model

Two ways to fetch the models of openvino format.

1. convert by command line
```bash
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0 --task text-generation-with-past --weight-format fp16
```

2. download from the cloud [recommended]
We have uploaded many optimized OpenVINO format models on the https://huggingface.co/OpenVINO. You can dowload what you want.

## run

### continunous_batching_accurary
```bash
./continuous_batching_accuracy -m //local/model/tiny-llama-1.1b-chat/pytorch/ov/FP16
```
