
################################  Lorenz ODE  ################################

ode=Lorenz
x_id=0

# change noise level

seed_arr=( 0 )

#noise_arr=( 0.09 0.15 0.2 0.25 0.3 )
noise_arr=( 0.01 ) # 0.09
n_seed=5 #50
freq=25


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
        python -u ../run_simulation.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        
        echo "Method: D-CODE"
        echo " "
        python -u ../run_simulation_vi.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50    
    done
done


## summarize noise

sample=50

#rm results/Lorenz-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/Lorenz-noise.txt
    python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/Lorenz-noise.txt
done

cat ../results/Lorenz-noise.txt
