#include <iostream>
#include <fstream>
#include <cuComplex.h>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int outputToFile(int* image) {
    // Write the result to a file
    ofstream matrix_out;

    cout << "\t- Writing output to out_image.csv" << endl;

    matrix_out.open("out_image.csv", ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -1;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }

    matrix_out.close();
    return 0;
}

__global__ void computePixels(int* image, int width, int height, double step, double min_x, double min_y, int iterations) {
    // Computing i and j from block and thread number
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col >= width) || (row >= height)) return;

    cuDoubleComplex c = make_cuDoubleComplex(col * step + min_x, row * step + min_y);

    // z = z^2 + c
    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    for (int i = 1; i <= iterations; i++)
    {
        z = cuCadd(cuCmul(z, z), c);

        // If it is convergent
        if (cuCabs(z) >= 2)
        {
            image[(width * row) + col] = i;
            break;
        }
    }
}

int main(int argc, char** argv)
{
    const auto start = chrono::steady_clock::now();

    // Define image array
    const int size = WIDTH * HEIGHT * sizeof(int);
    int* image;
    cudaMallocManaged(&image, size);

    // Defining threads and blocks of threads
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH / threadsPerBlock.x) + 1, (HEIGHT / threadsPerBlock.y) + 1);

    // Starting kernel
    computePixels<<<blocksPerGrid, threadsPerBlock>>>(image, WIDTH, HEIGHT, STEP, MIN_X, MIN_Y, ITERATIONS);
    cudaDeviceSynchronize();

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
        << chrono::duration_cast<chrono::seconds>(end - start).count()
        << "." << chrono::duration_cast<chrono::milliseconds>(end - start).count() % 1000
        << " seconds." << endl;

    outputToFile(image);

    cudaFree(image);

    return 0;
}