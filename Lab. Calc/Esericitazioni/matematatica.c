#include <stdio.h> //libreria di I/O
#include <math.h> //ibreria matematica per le funzioni matematiche

int main(){
  printf("Ciao mondo!\n");
  printf("7 + 8 = %d\n", 8+7); //%d stampo int
  printf("3.2 + 6.4  = %.2f\n", 3.2 + 6.4); //%f stampo float, se prima di f metto m.n, stampo m cifre di cui  n cifre dopo la virgola, m > n
  printf("n1=%d n2=%d\n", 6,3); //in printf metto tanti quantificato tanti quanti elemento devo printare, quindi metto n elementi dopo la virgola separati da virgole
  printf("4^2 = %f  \n", pow(4,2)); //per il pow bisogna usare un formato float, per via di come è stata scritta la funzione
  printf("cos(pi/6) = %10.8f \n", cos(M_PI/6.0));
  printf("sin^2(pi/6) + cos^2(pi/6) = %f\n", pow(cos(M_PI/6.0), 2) + pow(sin(M_PI/6.0), 2));
}
