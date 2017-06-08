constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant float mask[]={0.0625, 0.125, 0.0625,
                       0.125, 0.25, 0.125,
                       0.0625, 0.125, 0.0625};

kernel void gauss(read_only image2d_t in, global float *out) {
    const int2 pos = {get_global_id(0), get_global_id(1)};

    // Collect neighbor values and multiply with Gaussian
    float sum = 0.0f;
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            sum += mask[(y+1)*3+x+1]
                * read_imagef(in, sampler, pos + (int2)(x,y)).x;
        }
    }

    out[pos.x+pos.y*get_global_size(0)] = sum*255;
}
