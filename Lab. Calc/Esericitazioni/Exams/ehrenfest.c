#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#define NTOT 100
#define NSTEP 10000


void update(int A[], int B[], int i){
    float pA = (float) A[i - 1]/NTOT;
    float pCalc = (double) rand()/RAND_MAX;
    if(pCalc < pA){
        A[i] = A[i - 1] - 1;
        B[i] = B[i - 1] + 1;
    } else{
        B[i] = B[i - 1] - 1;
        A[i] = A[i - 1] + 1;
    }
    // printf("pCalc= %f pA= %f\n", pCalc, pA);
}

void media(int A[], int B[], float *xA, float *xB){
    *xA = 0;
    *xB = 0;
    for(int i = 1; i <= NSTEP; i++){
        *xA += A[i];
        *xB += B[i];
    }
    *xA /= NSTEP;
    *xB /= NSTEP;
}


void stddev(int A[], int B[], float xA, float xB, float *devA, float *devB){
    *devA = 0;
    *devB = 0;

    for (int i = 0; i < NSTEP; i++){
        *devA += pow(A[i] - xA, 2);
        *devB += pow(B[i] - xB, 2);
    }
    *devA = sqrt(*devA / NSTEP);
    *devB = sqrt(*devB / NSTEP);
}

int isto(int A[]){
    int arr[NTOT] = {0};

    for (int i = 0; i < NTOT; i++){
        arr[A[i] + 1] += 1;
    }

    for (int i = 0; i < NTOT; i++){
        printf("%d %d\n", i, arr[i]);
    }
}

int main(){
    srand(time(NULL));

    int NA[NSTEP + 1] = {0};
    NA[0] = rand() % (NTOT + 1);
    int NB[NSTEP + 1] = {NTOT - NA[0]};

    float xA = 0;
    float xB = 0;
    float devA = 0;
    float devB = 0;

    // printf("Configurazione iniziale: NA= %d NB= %d NTOT= %d\n", NA[0], NB[0], NTOT);

    for(int i = 1; i <= NSTEP; i++){
        update(NA, NB, i);

        // printf("Passo %d: NA= %d NB= %d \n", i, NA[i], NB[i]);
    }

    media(NA, NB, &xA, &xB);
    stddev(NA, NB, xA, xB, &devA, &devB);
    // printf("xA/n: %f xB/n: %f\n", xA, xB);
    // printf("devA: %f devB: %f\n", devA, devB);
    // printf("Passi totali effettuti= %i NA= %i NB= %i xA= %f xB= %f devA= %f devB= %f \n", NSTEP, NA[NSTEP], NB[NSTEP], xA, xB, devA, devB);

    isto(NA);
}   