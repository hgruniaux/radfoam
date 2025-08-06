#!/bin/bash

TARGET="classroom"

just upload

test () {
    echo "========= Running test with depth loss scale $1 ========="
    time ssh epic "cd radfoam && source .venv/bin/activate && python train.py -c configs/${TARGET}.yaml --experiment_name ${TARGET}_depth_loss_scale_$1 --depth_loss --depth_scale 20 --depth_coeff $1 ${@:2}"
    just retrieve-results ${TARGET}_depth_loss_scale_$1 ${TARGET}_depth_loss_scale_$1
}

# test 0
# test 0.001
# test 0.01
# test 0.05
# test 0.1

test 0 --downsample 4 --iterations 5000
test 0.0001 --downsample 4 --iterations 5000
test 0.001 --downsample 4 --iterations 5000
test 0.01 --downsample 4 --iterations 5000
