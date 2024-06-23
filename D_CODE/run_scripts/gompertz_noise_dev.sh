
################################  Gompertz ODE Noise ################################

mkdir -p ../results/GompertzODE # SR with total variation regularized differentation (SR-T)

ode=GompertzODE
ode_param=1.5,1.5

seed_arr=( 100 )
n_seed=1
freq=10

#noise_arr=( 0.01 0.03 0.05 0.07 0.09 0.1 0.15 0.2 0.25 0.3 0.5 0.7 0.9 1.1 1.3 )
noise_arr=( 0.01 )


for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        echo "Method: SR-T"
        echo " "
        python -u ../run_simulation.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        # echo "Method: D-CODE"
        # echo " "
        # python -u ../run_simulation_vi.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        sleep 1
    done
done


## summarize noise

#noise_arr=( 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.15 0.2 0.25 0.3 0.5 0.7 0.9 1.1 1.3 )
# sample=50

# #rm ../results/GompertzODE-noise.txt

# for noise in "${noise_arr[@]}"
# do
#     echo "SR-T evaluation:"
#     python -u ../evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/GompertzODE-noise.txt
    
#     # echo "D-CODE evaluation:"
#     # python -u ../evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/GompertzODE-noise.txt

#     sleep 1
# done

# cat ../results/GompertzODE-noise.txt

