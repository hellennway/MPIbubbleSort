#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <mpi.h>

using namespace std;

int ProcNum = 0; // Number of available processes
int ProcRank = 0; // Rank of current process


// Function for copying the sorted data
void CopyData(double* pData, int Size, double* pDataCopy) {
	copy(pData, pData + Size, pDataCopy);
}
// Function for comparing the data
bool CompareData(double* pData1, double* pData2, int Size) {
	return equal(pData1, pData1 + Size, pData2);
}
// Serial bubble sort algorithm
void SerialBubbleSort(double* pData, int Size) {
	double Temp;
	for (int i = 1; i < Size; i++)
		for (int j = 0; j < Size - i; j++)
			if (pData[j] > pData[j + 1]) {
				Temp = pData[j];
				pData[j] = pData[j + 1];
				pData[j + 1] = Temp;
			}
}

// Function for formatted data output
void PrintData(double* pData, int Size) {
	for (int i = 0; i < Size; i++)
		printf("%7.4f ", pData[i]);
	printf("\n");
}

// Function for dummy definition of the array
void DummyDataInitialization(double*& pData, int& Size) {
	for (int i = 0; i < Size; i++)
		pData[i] = Size-1;
}

// Function for random definition of the array
void RandomDataInitialization(double*& pData, int& Size) {
	srand((unsigned)time(0));
	for (int i = 0; i < Size; i++)
		pData[i] = rand()/double(1000);
}


// Function for memory allocation and data initialization
void ProcessInitialization(double*& pData, int& Size, double
	*& pProcData, int& BlockSize) {
	setvbuf(stdout, 0, _IONBF, 0);
	if (ProcRank == 0) {
		do {
			printf("Enter the size of the array: ");
			scanf_s("%d", &Size);
			if (Size < ProcNum)
				printf("Size should be greater than number of processes\n");
		} while (Size < ProcNum);
		printf("\nChosen objects size = %d\n", Size);
	}
	// Broadcasting the data size
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int RestData = Size; // Number of elements, that haven’t been distributed yet
	for (int i = 0; i < ProcRank; i++)
		RestData -= RestData / (ProcNum - i);
	BlockSize = RestData / (ProcNum - ProcRank);
	pProcData = new double[BlockSize];
	if (ProcRank == 0) {
		pData = new double[Size];
		// Data initalization
		//RandomDataInitialization(pData, Size);
		DummyDataInitialization(pData, Size);
	}
}
// Function for computational process termination
void ProcessTermination(double* pData, double* pProcData) {
	if (ProcRank == 0)
		delete[]pData;
	delete[]pProcData;
}

//Function for distribution of the initial objects between the processes
void DataDistribution(double* pData, int Size, double* pProcData, int
	BlockSize) {
	//Allocate memory for temporary objects
	int* pSendInd = new int[ProcNum]; // the index of the first data element sent to the process
	int* pSendNum = new int[ProcNum]; // the number of elements sent to the process
	int RestData = Size; // Number of elements, that haven’t been distributed yet
	int CurrentSize = Size / ProcNum;
	pSendNum[0] = CurrentSize;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestData -= CurrentSize;
		CurrentSize = RestData / (ProcNum - i);
		pSendNum[i] = CurrentSize;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}
	MPI_Scatterv(pData, pSendNum, pSendInd, MPI_DOUBLE, pProcData,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
}

// Distribution test
void TestDistribution(double* pData, int Size, double* pProcData, int
	BlockSize) {
	if (ProcRank == 0) {
		printf("Unsorted array:\n");
		PrintData(pData, Size);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("ProcRank = %d\n", ProcRank);
			printf("Block:\n");
			PrintData(pProcData, BlockSize);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
} 
// Function for data collection
void DataCollection(double* pData, int DataSize, double* pProcData,
	int BlockSize) {
	// Allocate memory for temporary objects
	int* pReceiveNum = new int[ProcNum];
	int* pReceiveInd = new int[ProcNum];
	int RestData = DataSize;
	pReceiveInd[0] = 0;
	pReceiveNum[0] = DataSize / ProcNum;
	for (int i = 1; i < ProcNum; i++) {
		RestData -= pReceiveNum[i - 1];
		pReceiveNum[i] = RestData / (ProcNum - i);
		pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
	}
	// Gather the whole result on every processor
	MPI_Gatherv(pProcData, BlockSize, MPI_DOUBLE, pData,
		pReceiveNum, pReceiveInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Free the memory
	delete[]pReceiveNum;
	delete[]pReceiveInd;
}
// Function for data exchange between the neighboring processes
void ExchangeData(double* pProcData, int BlockSize, int DualRank,
	double* pDualData, int DualBlockSize) {
	MPI_Status status;
	MPI_Sendrecv(pProcData, BlockSize, MPI_DOUBLE, DualRank, 0,
		pDualData, DualBlockSize, MPI_DOUBLE, DualRank, 0,
		MPI_COMM_WORLD, &status);
}
// Parallel bubble sort algorithm
void ParallelBubble(double* pProcData, int BlockSize) {
	// Local sorting the process data
	SerialBubbleSort(pProcData, BlockSize);
	int Offset; //1 if it is necessary to exchange blocks with the following process, -1 if it necessary with the previous
	enum split_mode { KeepFirstHalf, KeepSecondHalf };
	split_mode SplitMode;
	for (int i = 0; i < ProcNum; i++) {
		if ((i % 2) == 1) {
			if ((ProcRank % 2) == 1) {
				Offset = 1;
				SplitMode = KeepFirstHalf;
			}
			else {
				Offset = -1;
				SplitMode = KeepSecondHalf;
			}
		}
		else {
			if ((ProcRank % 2) == 1) {
				Offset = -1;
				SplitMode = KeepSecondHalf;
			}
			else {
				Offset = 1;
				SplitMode = KeepFirstHalf;
			}
		}
		// Check the first and last processes
		if ((ProcRank == ProcNum - 1) && (Offset == 1)) continue;
		if ((ProcRank == 0) && (Offset == -1)) continue;
		MPI_Status status;
		int DualBlockSize;
		MPI_Sendrecv(&BlockSize, 1, MPI_INT, ProcRank + Offset, 0,
			&DualBlockSize, 1, MPI_INT, ProcRank + Offset, 0,
			MPI_COMM_WORLD, &status);
		double* pDualData = new double[DualBlockSize];
		double* pMergedData = new double[BlockSize + DualBlockSize];
		// Data exchange
		ExchangeData(pProcData, BlockSize, ProcRank + Offset, pDualData,
			DualBlockSize);
		//Data merging
		merge(pProcData, pProcData + BlockSize,
			pDualData, pDualData + DualBlockSize, pMergedData);
		// Data splitting
		if (SplitMode == KeepFirstHalf)
			copy(pMergedData, pMergedData + BlockSize, pProcData);
		else
			copy(pMergedData + DualBlockSize, pMergedData + BlockSize +
				DualBlockSize, pProcData);
		delete[]pDualData;
		delete[]pMergedData;
	}
}

// Fuction for testing the results
void TestParallelSort(double* pProcData, int BlockSize) {
	for (int i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("ProcRank = %d\n", ProcRank);
			printf("Proc sorted data: \n");
			PrintData(pProcData, BlockSize);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
// Function for testing the result of parallel bubble sort
void TestResult(double* pData, double* pSerialData, int Size) {
	if (ProcRank == 0) {
		SerialBubbleSort(pSerialData, Size);
		//SerialStdSort(pSerialData, Size);
		if (!CompareData(pData, pSerialData, Size))
			printf("The results of serial and parallel algorithms are "
				"NOT identical. Check your code\n");
		else
			printf("The results of serial and parallel algorithms are identical\n");
	}
}
int main(int argc, char* argv[]) {
	double* pData = 0; //Array
	double* pProcData = 0; //Part of the array on current process
	int Size = 0; //Size of initial array
	int BlockSize = 0; //Size of the part on current process
	double* pSerialData = 0;
	double Start, Finish;
	double Duration = 0.0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	if (ProcRank == 0)
		printf("Parallel bubble sort program\n");
	// Memory allocation and data initialization
	ProcessInitialization(pData, Size, pProcData, BlockSize);
	if (ProcRank == 0) {
		pSerialData = new double[Size];
		CopyData(pData, Size, pSerialData);
	}
	/*if (ProcRank == 0)
	{
		printf("Unsorted array:\n");
		PrintData(pData, Size);
	}*/
	Start = MPI_Wtime();
	// Distributing the initial data between processes 
	DataDistribution(pData, Size, pProcData, BlockSize);
	// Testing the data distribution
	//TestDistribution(pData, Size, pProcData, BlockSize);
	// Parallel bubble sort
	ParallelBubble(pProcData, BlockSize);
	// Fuction for testing the results
	//TestParallelSort(pProcData, BlockSize);
	// Execution of data collection
	DataCollection(pData, Size, pProcData, BlockSize);
	TestResult(pData, pSerialData, Size);
	Finish = MPI_Wtime();
	/*if (ProcRank == 0)
	{
		printf("Sorted array:\n");
		PrintData(pData, Size);
	}*/
	Duration = Finish - Start;
	if (ProcRank == 0)
		printf("Time of execution: %f\n", Duration);
	if (ProcRank == 0)
		delete[]pSerialData;
	// Process termination
	ProcessTermination(pData, pProcData);
	MPI_Finalize();

	return 0;
}