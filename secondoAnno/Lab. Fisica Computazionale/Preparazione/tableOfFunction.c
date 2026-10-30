#include <stdio.h>
#include <math.h>

#define STARTVALUE -10
#define ENDVALUE 10
#define STEP 0.1

int main(){
    float x = STARTVALUE;
    float value = 0;

    while (!(x > ENDVALUE + STEP/2)){
        if (x < STEP/2 && x > -STEP/2) value = 1;
        else value = sin(x);

        printf(" %6.4f | %6.4f \n", x, value);

        x += STEP;
    }
}