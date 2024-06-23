
################################  Selkov ODE  ################################

ode=SelkovODE_rho
x_id=1

# change noise level

seed_arr=( 0 )
n_seed=5

noise_arr=( 0.01 )
freq=10
sample=50

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        echo " "
        echo "noise:"
        echo $noise
        echo " "

        # echo "Method: SR-T"
        # echo " "
        # python -u ../run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --x_id=${x_id} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        
        echo "Method: D-CODE"
        echo " "
        python -u ../run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --x_id=${x_id} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        sleep 1
    done
done


#rm ../results/SelkovODE-noise-1_rho.txt

for noise in "${noise_arr[@]}"
do
    #python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/SelkovODE-noise-1_rho.txt 
    python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/SelkovODE-noise-1_rho.txt
done

cat ../results/SelkovODE-noise-1_rho.txt

## parameter accuracy

#x_id=1

#rm results/SelkovODE-noise-param-1.txt

#for noise in "${noise_arr[@]}"
#do
#    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=vi >> results/SelkovODE-noise-param-1.txt
#    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=diff >> results/SelkovODE-noise-param-1.txt
#    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=spline >> results/SelkovODE-noise-param-1.txt
#    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --x_id=${x_id} --n_sample=${sample} --freq=${freq} --alg=gp >> results/SelkovODE-noise-param-1.txt
#done

#cat results/SelkovODE-noise-param-1.txt

