#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <mpi.h>

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

int main(int argc, char** argv)
{
    const auto start = chrono::steady_clock::now();

    // MPI Section
    MPI_Init(&argc, &argv);

    // Retrieving program run infos
    int procRank, procCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    int iterations = HEIGHT * WIDTH;

    // Computing the integer division of the intervals and the remainder
    long int iterDiv = iterations / procCount;
    long int remainderIter = iterations % procCount;

    // Computing the various counts of the processes
    int* procIterStarts = new int[procCount];
    int* procIterCounts = new int[procCount];

    for (int proc = 0; proc < procCount; proc++) {
        procIterCounts[proc] = iterDiv + ((proc < remainderIter) ? 1 : 0);
        procIterStarts[proc] = (proc * (iterDiv)) + ((proc < remainderIter) ? proc : remainderIter);
    }

    // Getting the counts this processor will have to handle
    int procIterCount = procIterCounts[procRank];
    int procIterStart = procIterStarts[procRank];

    // Display info about what is going to be computed
    if (procRank == 0) {
        cout << "There are " << procCount << " processes and each will compute " << iterDiv;
        if (remainderIter > 0) {
            cout << " iterations with some computing 1 extra";
        }
        cout << endl;
    }

    //int* image = new int[HEIGHT * WIDTH];
    int* imageProcPortion = new int[iterDiv+1];

    // Distributing Iterations between the processes
    for (int pos = 0; pos < procIterCount; pos++)
    {
        imageProcPortion[pos] = 0;

        const int row = (pos + procIterStart) / WIDTH;
        const int col = (pos + procIterStart) % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                imageProcPortion[pos] = i;
                break;
            }
        }
    }

    int* image = nullptr;
    if (procRank == 0) {
        image = new int[HEIGHT * WIDTH];
    }

    // Gathering results on process 0
    MPI_Gatherv(
        &imageProcPortion[0], // Send buf
        procIterCount, // Send count
        MPI_INT, // Send type
        image, // Receive buf
        procIterCounts, // Receive count
        procIterStarts, // Receive displacements
        MPI_INT, // Receive type
        0, // Root process
        MPI_COMM_WORLD // comm
    );

    MPI_Finalize();

    if (procRank == 0) {
        const auto end = chrono::steady_clock::now();
        cout << "Time elapsed: "
            << chrono::duration_cast<chrono::seconds>(end - start).count()
            << "." << chrono::duration_cast<chrono::milliseconds>(end - start).count() % 1000
            << " seconds." << endl;

        outputToFile(image);
    }

    delete[] image; // It's here for coding style, but useless

    return 0;
}