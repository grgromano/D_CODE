
################################  Logistic ODE  ################################
mkdir -p results/LogisticODE_k
ode=LogisticODE_k

seed_arr=( 100 )

noise_arr=( 0.01 )

n_seed=2
freq=5

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        # echo "Method: SR-T"
        # echo " "
        # python -u ../run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        
        echo "Method: D-CODE"
        echo " "
        python -u ../run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        sleep 1
    done
done


## summarize noise

# sample=50

# #rm ../results/LogisticODE_k-noise.txt

# for noise in "${noise_arr[@]}"
# do
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/LogisticODE_k-noise.txt
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/LogisticODE_k-noise.txt
# done

# cat ../results/LogisticODE_k-noise.txt


