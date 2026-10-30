#include <stdlib.h>
#include <stdio.h>

#define file "dati.txt"

int main(){
    FILE *filep;

    float x, y, maxY, maxXofY, sum = 0;
    int counter = 0;

    filep = fopen(file, "r");

    if (filep == NULL){
        printf("Il file non esite");
        return 1;
    }

    while (fscanf(filep, "%f %f", &x, &y) == 2){
        
        if (y > maxY || counter == 0){
            maxY = y;
            maxXofY = x;
        }

        sum += y;
        counter += 1;
    }

    fclose(filep);

    if (counter > 0)    printf("Righe: %d | Media delle y: %f | Max delle y: %f la cui x è: %f\n", counter, sum/(float)counter, maxY, maxXofY);    
    else printf("Il file è vuoto \n");
}