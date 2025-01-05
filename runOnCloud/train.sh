# sbatch baseline_fedavg.job
# for ps in server client
for ps in server
do 
    # Non-IID
    i=1
    for a in 0.1 0.5 0.8 1
    do
        for pr in 0.1 0.3 0.5 0.7 0.8 0.9 0.95 0.98
        do
            sbatch PaI_general.job $ps $i $pr $a
        done
    done
    # IID
    i=0
    a=10
    for pr in 0.1 0.3 0.5 0.7 0.8 0.9 0.95 0.98
    do
        sbatch PaI_general.job $ps $i $pr $a
    done
done
 
