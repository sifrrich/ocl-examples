// Inspired by
// http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
//
kernel void reduce(global float *src, global float *dest, local float *tmp, int length) {
    int global_index = get_global_id(0);
    float accumulator = 0.f;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float element = src[global_index];
        accumulator += element;
        global_index += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    tmp[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
        if (local_index < offset) {
            float other = tmp[local_index + offset];
            float mine = tmp[local_index];
            tmp[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        dest[get_group_id(0)] = tmp[0];
    }
}
