#include<stdio.h>
#include<time.h>
#include<random>

// SIMULAZIONE DEL PALLINOMETRO CON PROBABILITÀ DI RIMABALZO VERSO SINISTRA DI P E VERSO DESTRA DI Q; TALE PER CUI:
// P + Q = 1

#define NUM_PALLINE 1000
#define P 0.5
#define NUM_FILE 33

int main(){
    int res[NUM_PALLINE] = {};
    double n = 0;
    float pos = 0;

    FILE *fp = fopen("dati.txt", "w");

    for (int i = 0; i < NUM_PALLINE; i++){
        pos = 0;
        for (int j = 0; j < NUM_FILE; j++){
            n = drand48();
            if(n < P) pos += 1.0/2.0;
            else pos -= 1.0/2.0;
        }
        res[i] = pos + NUM_FILE/2;
        fprintf(fp, "%d \n", res[i]);
    }
}