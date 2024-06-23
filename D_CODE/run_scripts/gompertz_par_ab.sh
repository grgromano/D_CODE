
################################  Gompertz ODE - Parametric  ################################
mkdir -p ../results/GompertzODE_par_ab

ode=GompertzODE_par_ab

seed_arr=( 100 )
n_seed=1
freq=10

noise_arr=( 0.01 )

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do  
        echo "Method: SR-T"
        echo " "
        python -u ../run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        # echo "Method: D-CODE"
        # echo " "
        # python -u ../run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50

        sleep 1
    done
done


## summarize noise

# sample=50

# #rm ../results/GompertzODE_par_ab-noise.txt

# for noise in "${noise_arr[@]}"
# do
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/GompertzODE_par_ba-noise.txt 
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/GompertzODE_par_ba-noise.txt
# done

# cat ../results/GompertzODE_par_ba-noise.txt



#noise_arr_aux=( 0.01 0.1 0.2 )
# for noise in "${noise_arr[@]}"
# do
#     for aux in "${noise_arr_aux[@]}"
#     do
#         if ["$aux" = "$noise"]
#         then 
#             python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE_par_ab-noise.txt 
#             python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par_ab-noise.txt
#         else
#             python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par_ab-noise.txt
#         fi

    



#     if noise in "${noise_arr_aux[@]}"
#         python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE_par_ab-noise.txt 
#         python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par_ab-noise.txt
#     else
#         python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par_ab-noise.txt
    
    
#     #python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE_par_ab-noise.txt 
#     #python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE_par_ab-noise.txt 
# done

# cat results/GompertzODE_par_ab-noise.txt

## parameter accuracy


#rm results/GompertzODE_par-noise-param.txt

#for noise in "${noise_arr[@]}"
#do
    #python -u evaluation_param_selkov.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/SelkovODE-noise-param.txt 
    #python -u evaluation_param_selkov.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par-noise-param.txt
    #python -u evaluation_param_selkov.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE_par-noise-param.txt 
    #python -u evaluation_param_selkov.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE_par-noise-param.txt 
#done

#cat results/SelkovODE-noise-param.txt

