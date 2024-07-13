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

int main(int argc, char** argv)
{
    const auto start = chrono::steady_clock::now();

    // MPI Section
    MPI_Init(&argc, &argv);

    // Retrieving program run infos
    int procRank, procCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    // Defining algo variables
    int* image = nullptr;
    int tasks = HEIGHT * WIDTH;
    int task, proc, output;
    int* payload = new int[2];
    MPI_Status status;

    if (procRank == 0) {
        // MASTER
        int currentTask = 0;

        image = new int[HEIGHT * WIDTH];

        // Indicate the algo is starting
        cout << "Starting to distribute pixels among " << procCount << " processes..." << endl;

        // First distribute tasks to all processes
        for (int i = 1; i < procCount; i++) {
            MPI_Send(
                &currentTask, // Send buf
                1, // Send count
                MPI_INT, // Send type
                i, // Destination
                0, // Tag
                MPI_COMM_WORLD // Comm
            );

            currentTask++;
        }

        // Then loop to receive back results and send new task
        for (int i = 0; i < tasks; i++) {
            MPI_Recv(
                payload, // Receive buf
                2, // Receive count
                MPI_INT, // Receive Type
                MPI_ANY_SOURCE, // Source
                0, // Tag
                MPI_COMM_WORLD, // Comm
                &status // Status
            );

            // Process output of slave
            proc = status.MPI_SOURCE; // Process from which we receive a Message
            task = payload[0];
            output = payload[1];

            // Writing in the output array
            image[task] = output;

            // DEBUG progress
            // cout << "Progress: " << ((currentTask * 100) / tasks) << "%\r";

            // Sending a new iter or a termination signal
            MPI_Send(
                &currentTask, // Send buf
                1, // Send count
                MPI_INT, // Send type
                proc, // Destination
                (currentTask < tasks)? 0: 1, // Tag (If 1, slave is done and can exit)
                MPI_COMM_WORLD // Comm
            );

            currentTask++;
        }
    }
    else {
        // SLAVE
        do {
            // Receive Task
            MPI_Recv(
                &task, // Receive buf
                1, // Receive count
                MPI_INT, // Receive Type
                0, // Source
                MPI_ANY_TAG, // Tag
                MPI_COMM_WORLD, // Comm
                &status // Status
            );

            // Dont do computation and break if tag 1 is received
            if (status.MPI_TAG == 1) break;

            // Computation
            const int row = task / WIDTH;
            const int col = task % WIDTH;
            const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);
            int output = 0;

            // z = z^2 + c
            complex<double> z(0, 0);
            for (int i = 1; i <= ITERATIONS; i++)
            {
                z = pow(z, 2) + c;

                // If it is convergent
                if (abs(z) >= 2)
                {
                    output = i;
                    break;
                }
            }
            // End computation

            payload[0] = task;
            payload[1] = output;

            MPI_Send(
                payload, // Send buf
                1, // Send count
                MPI_INT, // Send type
                0, // Destination
                0, // Tag
                MPI_COMM_WORLD // Comm
            );
        } 
        while (status.MPI_TAG == 0);
    }

    MPI_Finalize();

    if (procRank == 0) {
        const auto end = chrono::steady_clock::now();
        cout << "Time elapsed: "
            << chrono::duration_cast<chrono::seconds>(end - start).count()
            << " seconds." << endl;

        // Write the result to a file
        ofstream matrix_out;

        if (argc < 2)
        {
            cout << "Please specify the output file as a parameter." << endl;
        }
        else {
            matrix_out.open(argv[1], ios::trunc);
            if (!matrix_out.is_open())
            {
                cout << "Unable to open file." << endl;
            }
            else {
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
            }
        }
        matrix_out.close();
    }

    delete[] image; // It's here for coding style, but useless

    return 0;
}