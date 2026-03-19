echo ${MASTER_ADDR}
echo $OMPI_COMM_WORLD_RANK

ray stop --force

# if OMPI_COMM_WORLD_RANK is 0, then start the ray cluster, else print the value of MASTER_ADDR
if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    # Start Ray head node
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8
else
    echo ${MASTER_ADDR}
    ray start --address ${MASTER_ADDR}:6379  --num-gpus 8
fi