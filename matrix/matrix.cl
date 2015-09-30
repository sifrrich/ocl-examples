// Inspired by
// http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
//
kernel void matrix_mul1(global float *A, global float *B, global float *C, int M) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    for (int k=0; k<M; ++k){
        C[i*M+j] += A[i*M+k] * B[k*M+j];
    }
}

kernel void matrix_mul2(global float *A, global float *B, global float *C, int M) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float tmp = 0.0f;
    for (int k=0; k<M; ++k) {
        tmp += A[i*M+k] * B[k*M+j];
    }
    C[i*M+j] += tmp;
}

kernel void matrix_mul3(global float *A, global float *B, global float *C, int M) {
    int i = get_global_id(0);

    float tmp;
    for (int j=0; j<M; ++j) {
        tmp = 0.0f;
        for (int k=0; k<M; k++)
            tmp += A[i*M+k] * B[k*M+j];
        C[i*M+j] += tmp;
    }
}
#ifndef PRIVATE_BUF_SIZE
#define PRIVATE_BUF_SIZE 1024
#endif

kernel void matrix_mul4(global float *A, global float *B, global float *C, int M) {
    int i = get_global_id(0);

    float Arow[PRIVATE_BUF_SIZE];
    float tmp;
    for (int k=0; k<M; ++k) {
        Arow[k] = A[i*M+k];
    }

    for (int j=0; j<M; ++j){
        tmp = 0.0f;
        for (int k=0; k<M; ++k) {
            tmp += Arow[k] * B[k*M+j];
        }
        C[i*M+j] += tmp;
    }
}

#ifndef LOCAL_BUF_SIZE
#define LOCAL_BUF_SIZE 1024
#endif

kernel void matrix_mul5(global float *A, global float *B, global float *C, int M) {
    int i = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);

    float Arow[PRIVATE_BUF_SIZE];
    local float Brow[LOCAL_BUF_SIZE];
    float tmp;
    for (int k=0; k<M; ++k) {
        Arow[k] = A[i*M+k];
    }

    for (int j=0; j<M; ++j){
        for (int k=iloc; k<M; k+=nloc) {
            Brow[k] = B[k*M+j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = 0.0f;
        for (int k=0; k<M; ++k) {
            tmp += Arow[k] * Brow[k];
        }
        C[i*M+j] += tmp;
    }
}
