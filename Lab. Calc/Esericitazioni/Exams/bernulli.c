#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#define A 2
#define B 5
#define NTOSS 7

void toss_dices(int array[]){
    for(int i = 0; i < NTOSS; i++){
        if(((float)(rand())/RAND_MAX) <= 1.0/6.0){
            array[i] = 1;
        }
        else{
            array[i] = 0;
        }
    }

    /*
    for (int i = 0; i < NTOSS; i++)
    {
        printf("outcomes[%d]: %d\n", i, array[i]);
    }
    */
    
}

float sim_bernulli(int numtry, int array[]){
    int succ = 0;
    for(int j = 0; j < numtry; j++){
        // printf("Tentativo numero: %d\n", j);
        toss_dices(array);
        int noscc = 0;

        for (int i = 0; i < A; i++){
            if(array[i] == 0){
                noscc = 1;
            }
        }
        if (!noscc){
            succ += 1;
        }
    }

    return succ;
}

int main(){
    int nevent = 0;
    int outcomes[NTOSS] = {0};
    float p = 0;
    float freq_arr[100] = {0.0};
    float success = 0;

    srand(time(NULL));

    p = pow(1.0/6.0, A) * pow(5.0/6.0, B);

    //printf("%f\n", p);

    /*do
    {
        printf("Inserisci il numero di lanci del dato (tra 1000 e 100000 estremi inclusi): \n");
        scanf("%d", &nevent);
    } while ((nevent < 1000) || (nevent > 100000));*/
    nevent = 1000;

    for (int i = 0; i < 100; i++){
        success = sim_bernulli(nevent, outcomes);

        freq_arr[i] = (float) (success/nevent);
        printf("%d %f\n", nevent, freq_arr[i]);
        
        nevent += 1000;
    }
}