const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_FILTER_LINEAR |
                            CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void interpolation(read_only image2d_t in, write_only image2d_t out) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float2 pos_norm = convert_float2(pos) / convert_float2(get_image_dim(out));

    float4 pix = read_imagef(in, sampler, pos_norm)*255;

    write_imagef(out, pos, pix);
}
