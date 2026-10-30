#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// \dot(v)(t) = g - \frac{k}{m}v

int main(int argc, char *argv[]){

    if (argc != 5){
        printf("Passare 4 argomenti (k, m, dt, t_max)"); 
        return 1;
    }

    double t, dt, v_0, t_max, v, g = 9.81, k, m, dv, v_esatta;

    dt = atof(argv[3]);
    t_max = atof(argv[4]);
    v = v_0;

    k = atof(argv[1]);
    m = atof(argv[2]);

    FILE *f = fopen("caduta.dat", "w");

    while(t < t_max){
        v_esatta = (g * m / k) * (1.0 - exp(-k / m *t));


        dv = g - k/m * v;
        t = t + dt;
        v = v + dv * dt;

        fprintf(f, "%g %g %g\n", t, v, v_esatta);
    }

    fclose(f);
    return 0;
}