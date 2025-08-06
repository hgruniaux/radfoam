set shell := ["bash", "-uc"]

CC  := "/usr/bin/gcc-11"
CXX := "/usr/bin/g++-11"

TICK_BOX := "[" + GREEN + "✔" + NORMAL + "]"
EMPTY_BOX := "[ ]"

run target: build
    python viewer.py -c output/{{target}}/config.yaml

train target: build
    python train.py -c configs/{{target}}.yaml --downsample 8 --viewer

train-depth target: build
    python train.py -c configs/{{target}}.yaml --downsample 8 --viewer --depth_loss --depth_coeff 0.05 --depth_scale 20

train-error target: build
    python train.py -c configs/{{target}}.yaml --downsample 8 --viewer --error_sampling

build:
    ninja install -C build

clean:
    rm -rf build
    rm -rf radfoam

configure:
    cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER={{CC}} -DCMAKE_CXX_COMPILER={{CXX}} -DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_SQUARED_DENSITY=OFF

update-remote:
    ssh epic "cd radfoam && git pull"

@upload: build
    echo -en "{{EMPTY_BOX}} Uploading \{{BOLD}}train.py{{NORMAL}} to remote server..."
    rsync -az train.py epic:~/radfoam/train.py -p
    echo -e "\r{{TICK_BOX}} Uploaded \{{BOLD}}train.py{{NORMAL}} to remote server    "

    echo -en "{{EMPTY_BOX}} Uploading \{{BOLD}}configs directory{{NORMAL}} to remote server..."
    rsync -az configs/ epic:~/radfoam/configs -p --exclude "__pycache__"
    echo -e "\r{{TICK_BOX}} Uploaded \{{BOLD}}configs directory{{NORMAL}} to remote server     "

    echo -en "{{EMPTY_BOX}} Uploading \{{BOLD}}data_loader directory{{NORMAL}} to remote server..."
    rsync -az data_loader/ epic:~/radfoam/data_loader -p --exclude "__pycache__"
    echo -e "\r{{TICK_BOX}} Uploaded \{{BOLD}}data_loader directory{{NORMAL}} to remote server     "

    echo -en "{{EMPTY_BOX}} Uploading \{{BOLD}}radfoam_model directory{{NORMAL}} to remote server..."
    rsync -az radfoam_model/ epic:~/radfoam/radfoam_model -p --exclude "__pycache__"
    echo -e "\r{{TICK_BOX}} Uploaded \{{BOLD}}radfoam_model directory{{NORMAL}} to remote server     "

    echo -en "{{EMPTY_BOX}} Uploading \{{BOLD}}prebuilt radfoam binary{{NORMAL}} to remote server..."
    rsync -az radfoam/torch_bindings.cpython-312-x86_64-linux-gnu.so epic:~/radfoam/radfoam -p
    echo -e "\r{{TICK_BOX}} Uploaded \{{BOLD}}prebuilt radfoam binary{{NORMAL}} to remote server     "

    rsync -az error_sampling.py epic:~/radfoam/error_sampling.py -p
    rsync -az error_aware_data_handler.py epic:~/radfoam/error_aware_data_handler.py -p

    echo "The binary was uploaded with the given configuration:"
    echo "  - C compiler: $(cmake -LA -N -B build | grep CMAKE_C_COMPILER: | cut -d'=' -f2)"
    echo "  - C++ compiler: $(cmake -LA -N -B build | grep CMAKE_CXX_COMPILER: | cut -d'=' -f2)"
    echo "  - CUDA compiler: $(cmake -LA -N -B build | grep CMAKE_CUDA_COMPILER: | cut -d'=' -f2)"
    echo "  - Squared density: $(cmake -LA -N -B build | grep ENABLE_SQUARED_DENSITY: | cut -d'=' -f2)"

train-remote target: upload
    @echo -e "{{EMPTY_BOX}} Starting remote training on epic server for \{{BOLD}}{{target}}{{NORMAL}} target..."
    @time ssh epic "cd radfoam && source .venv/bin/activate && python train.py -c configs/{{target}}.yaml"

# @time ssh epic "cd radfoam && source .venv/bin/activate && python train.py -c configs/{{target}}.yaml --experiment_name --error_sampling"

@retrieve-results target test-name:
    if [ -d "tests/{{test-name}}" ]; then echo -e "{{BOLD+RED}}error{{NORMAL}}: The test {{BOLD}}{{test-name}}{{NORMAL}} already exist." && exit 1; fi
    mkdir -p tests/{{test-name}}

    echo -e "{{EMPTY_BOX}} Retrieving images for {{BOLD}}{{target}}{{NORMAL}} target from remote server..."
    scp -r epic:~/radfoam/output/{{target}}/test ./tests/{{test-name}}
    echo -e "{{TICK_BOX}} Retrieved images for \{{BOLD}}{{target}}{{NORMAL}} target from remote server"
    echo -e "Images are stored in the \{{BOLD}}tests/{{test-name}}/images{{NORMAL}} directory."

    echo -ne "{{EMPTY_BOX}} Retrieving config.yaml for {{BOLD}}{{target}}{{NORMAL}} target from remote server..."
    scp -r epic:~/radfoam/output/{{target}}/config.yaml ./tests/{{test-name}}
    echo -e "\r{{TICK_BOX}} Retrieved config.yaml for {{BOLD}}{{target}}{{NORMAL}} target from remote server       "

    echo -ne "{{EMPTY_BOX}} Retrieving stats.csv for {{BOLD}}{{target}}{{NORMAL}} target from remote server..."
    scp -r epic:~/radfoam/output/{{target}}/stats.csv ./tests/{{test-name}}
    echo -e "\r{{TICK_BOX}} Retrieved stats.csv for {{BOLD}}{{target}}{{NORMAL}} target from remote server       "

    echo -ne "{{EMPTY_BOX}} Generating plots for {{BOLD}}{{target}}{{NORMAL}} target..."
    python print_stats.py tests/{{test-name}}
    echo -e "\r{{TICK_BOX}} Generated plots for {{BOLD}}{{target}}{{NORMAL}} target          "

    echo -ne "{{EMPTY_BOX}} Retrieving scene.ply for {{BOLD}}{{target}}{{NORMAL}} target from remote server..."
    scp -r epic:~/radfoam/output/{{target}}/scene.ply ./tests/{{test-name}}
    echo -e "\r{{TICK_BOX}} Retrieved scene.ply for {{BOLD}}{{target}}{{NORMAL}} target from remote server          "

    echo -ne "{{EMPTY_BOX}} Retrieving model.pt for {{BOLD}}{{target}}{{NORMAL}} target from remote server..."
    scp -r epic:~/radfoam/output/{{target}}/model.pt ./tests/{{test-name}}
    echo -e "\r{{TICK_BOX}} Retrieved model.pt for {{BOLD}}{{target}}{{NORMAL}} target from remote server          "

retrieve-output file:
    scp epic:~/radfoam/output/{{file}} .