
################################  Gompertz ODE - Parametric  ################################
mkdir -p results/GompertzODE_par_a

ode=GompertzODE_par_a
#ode_param=1.5 
# RMK. per qualche motivo, chiamando run_simulation.py con l'argomento --ode_param, il programma dà errore, il problema è stato agggirato inserendo il parametro direttamente nel programma, tramite get_default_param()

seed_arr=( 0 50 )
n_seed=10
freq=10

#noise_arr=( 0.01 0.1 1.0 )
#noise_arr=( 0.011 ) # poi buttare la cartella!!!
noise_arr=( 0.01 )


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
        python -u run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
    done
done


## summarize noise

sample=50

rm results/GompertzODE_par_a-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE_par_a-noise.txt 
    #python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE_par-noise.txt
    #python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE_par-noise.txt 
    #python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE_par-noise.txt 
done

cat results/GompertzODE_par_a-noise.txt

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

