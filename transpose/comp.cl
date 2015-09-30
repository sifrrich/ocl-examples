kernel void comp(global float *in, global float *out) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    size_t width  = get_global_size(0);
    size_t height = get_global_size(1);

    out[x*height+y] = in[y*width+x];
}
