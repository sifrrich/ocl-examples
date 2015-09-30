kernel void sync(global float *src, global float *dest, local float *tmp) {
    int tid = get_local_id(0);
    int i = get_global_id(0);

    tmp[tid] = src[i];

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(int s=get_local_size(0)/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) dest[get_group_id(0)] = tmp[0];
}
