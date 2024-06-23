
################################  Gompertz ODE - Parametric  ################################
mkdir -p ../results/GompertzODE_par_b

ode=GompertzODE_par_b
#ode_param=1.5 
# RMK. per qualche motivo, chiamando run_simulation.py con l'argomento --ode_param, il programma dà errore, il problema è stato agggirato inserendo il parametro direttamente nel programma, tramite get_default_param()

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



# #rm ../results/GompertzODE_par_b-noise.txt

# for noise in "${noise_arr[@]}"
# do
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> ../results/GompertzODE_par_b-noise.txt 
#     python -u ../evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> ../results/GompertzODE_par_b-noise.txt
# done

# cat ../results/GompertzODE_par_b-noise.txt

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

