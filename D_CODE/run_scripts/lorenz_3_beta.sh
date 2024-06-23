
################################  Lorenz ODE  ################################

# Third Equation

ode=Lorenz_beta
x_id=2

# change noise level

seed_arr=( 0 )

#noise_arr=( 0.09 0.15 0.2 0.25 0.3 )
noise_arr=( 0.01 )
n_seed=5 #50
freq=50
n_sample=50

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        echo " "
        echo "noise:"
        echo $noise
        echo " "
        
        echo "Method: SR-T"
        echo " "
        python -u ../run_simulation.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=${n_sample}
        
        echo "Method: D-CODE"
        echo " "
        python -u ../run_simulation_vi.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=${n_sample}
    done
done



## summarize noise

#rm ../results/Lorenz-noise-2_beta.txt

for noise in "${noise_arr[@]}"
do
    python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${n_sample} --freq=${freq} --x_id=${x_id} --alg=diff >> ../results/Lorenz-noise-2_beta.txt
    python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${n_sample} --freq=${freq} --x_id=${x_id} --alg=vi >> ../results/Lorenz-noise-2_beta.txt
done

cat ../results/Lorenz-noise-2_beta.txt
