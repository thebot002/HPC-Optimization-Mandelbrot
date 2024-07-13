#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <iomanip>
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

    // Get Arg to set block size
    int blockSize = 1;
    int tasks = HEIGHT * WIDTH;

    if (argc < 2) {
        if (procRank == 0) cout << "\t- No arguments found, setting the thread count to 1" << endl;
    }
    else {
        int enteredBlockSize;
        sscanf(argv[1], "%d", &enteredBlockSize);
        if (enteredBlockSize < 1) {
            if (procRank == 0) cout << "\t- Block size less than 1, will be set to 1 by default..." << endl;
        }
        else if (enteredBlockSize > tasks / (procCount - 1)) {
            int taskDiv = tasks / (procCount - 1);
            int taskRemainder = tasks % (procCount - 1);
            blockSize = taskDiv + ((taskRemainder > 0) ? 1 : 0);
            if (procRank == 0) cout << "\t- Block size too larger, defaulting to " << blockSize << " (size of Static behavior)." << endl;
        }
        else {
            blockSize = enteredBlockSize;
            if (procRank == 0) cout << "\t- Block size set to: " << blockSize << endl;
        }
    }

    // Defining algo variables
    int* image = nullptr;
    int task, proc, output;
    int* payload = new int[blockSize];
    MPI_Status status;

    if (procRank == 0) {
        // MASTER
        int currentTask = 0;
        int* sentTasks = new int[procCount];
        int terminatedCount = 0;

        image = new int[HEIGHT * WIDTH];

        // Indicate the algo is starting
        cout << "\t- Starting to distribute pixels among " << (procCount - 1) << " processes..." << endl;

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

            // Record what tasks have been sent and increase currentTask counter
            sentTasks[i] = currentTask;
            currentTask += blockSize;
        }

        // Then loop to receive back results and send new task
        for (int i = 0; i < ((tasks / blockSize) + 1); i++) {
            MPI_Recv(
                payload, // Receive buf
                blockSize, // Receive count
                MPI_INT, // Receive Type
                MPI_ANY_SOURCE, // Source
                0, // Tag
                MPI_COMM_WORLD, // Comm
                &status // Status
            );

            // Process output of slave
            proc = status.MPI_SOURCE; // Process from which we receive a Message
            task = sentTasks[proc];

            // Write output to image array
            for (int i = 0; i < blockSize; i++) {
                if ((task + i) < tasks)
                    image[task + i] = payload[i];
            }

            // DEBUG progress
            // cout << "\t- Progress: " << ((currentTask * 100) / tasks) << "%\r";
            // cout << "\t- Received [" << task << ", " << (task + blockSize) << "] and " <<
            //     "sending [" << currentTask << ", " << (currentTask + blockSize) << "] to process " << proc << 
            //     ((currentTask >= tasks) ? " (T)" : "") << endl;

            // Sending a new iter or a termination signal
            MPI_Send(
                &currentTask, // Send buf
                1, // Send count
                MPI_INT, // Send type
                proc, // Destination
                (currentTask < tasks) ? 0 : 1, // Tag (If 1, slave is done and can exit)
                MPI_COMM_WORLD // Comm
            );

            // In case termination signal is send, we record it and check if all have been terminated
            if (currentTask >= tasks) terminatedCount++;
            if (terminatedCount >= (procCount - 1)) {
                cout << "\t- All termination signals have been sent, breaking from Master." << endl;
                break;
            }

            // Record what tasks have been sent and increase currentTask counter
            sentTasks[proc] = currentTask;
            currentTask += blockSize;
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

            // DEBUG
            // if (status.MPI_TAG == 1) cout << "\t" << procRank << " process received termination signal" << endl;

            // Dont do computation and break if tag 1 is received
            if (status.MPI_TAG == 1) break;

            // Distributing Iterations between the processes
            for (int i = 0; i < blockSize; i++) {
                // Check if within image
                if (task + i >= tasks) break;

                // Computation
                const int row = (task + i) / WIDTH;
                const int col = (task + i) % WIDTH;
                const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);
                payload[i] = 0;

                // z = z^2 + c
                complex<double> z(0, 0);
                for (int j = 1; j <= ITERATIONS; j++)
                {
                    z = pow(z, 2) + c;

                    // If it is convergent
                    if (abs(z) >= 2)
                    {
                        payload[i] = j;
                        break;
                    }
                }
                // End computation
            }

            MPI_Send(
                payload, // Send buf
                blockSize, // Send count
                MPI_INT, // Send type
                0, // Destination
                0, // Tag
                MPI_COMM_WORLD // Comm
            );
        } while (status.MPI_TAG == 0);

        // cout << "Proc " << procRank << " shutting off..." << endl;
    }

    MPI_Finalize();

    if (procRank == 0) {
        const auto end = chrono::steady_clock::now();
        cout << "\t- Time elapsed: "
            << chrono::duration_cast<chrono::seconds>(end - start).count()
            << "." << chrono::duration_cast<chrono::milliseconds>(end - start).count() % 1000
            << " seconds." << endl;

        outputToFile(image);
    }

    delete[] image; // It's here for coding style, but useless

    return 0;
}